#################################################
# Fast-dLLM CACHE-ONLY reimplementation inside the dllm-meta framework --
# LLaDA runner. (arXiv 2505.22618, NVIDIA.)
#
# The paper has two components: (1) the DualCache block KV cache and (2)
# confidence-aware parallel decoding. This runner implements ONLY (1); the
# decode rule is the framework's uniform greedy confidence-argmax under the
# standard per-block quota (num_unmask_per_step=1 -> one token per step),
# identical to every other runner and the full-denoise baselines. So accuracy
# deltas are attributable to the cache alone, not to a different decode rule.
#
# DualCache, faithful to their design:
#   - block-wise decoding (paper: block length 32 -> num_blocks = len_target/32)
#   - at each BLOCK START, one full-canvas forward recomputes and caches KV for
#     EVERY position: prompt + finished blocks (prefix cache) + future masked
#     blocks (suffix cache). That forward's logits also serve as the block's
#     first decode step.
#   - WITHIN a block, only the current block's rows are forwarded; all other
#     rows attend from the cached KV (their block-frozen approximation). The
#     block's own K/V are re-merged into the cache each step.
#
# Threads: llada_base (plain prompts) and llada_instruct
# (use_official_gsm8k_prompt / truncate_at_eos=True via model_args).
#
# model_args example (block 32 at len 256):
#   ...,runner=run_llada_fastdllm,num_blocks=8,num_unmask_per_step=1,...
#################################################

import time

import torch

from components_llada import SimpleLogitsSnapshot
from tools_llada import (BlockDiffusionQuotaHelper, RunnerReport,
                         collect_ids_stop, truncate_text_at_stop)
from plugins_llada import SaveKVPreviousPlugin_Disabled, CachePastKVPlugin_Enabled,\
                            CacheAttnPlugin_Disabled, CacheVOPlugin_Disabled


class RunModel:

    DREAM_SHIFT = False    # run_dream_fastdllm overrides: token at p is predicted
                           # by output row p-1, so block forwards include the row
                           # BEFORE the block and logits are read shifted

    def __init__(self):
        self.report = RunnerReport()
        self.ids_stop = None
    # end

    def config_plugin_(self, config):
        config.klass_save_kv_previous = SaveKVPreviousPlugin_Disabled
        config.klass_cache_past_kv = CachePastKVPlugin_Enabled
        config.klass_cache_attn = CacheAttnPlugin_Disabled
        config.klass_cache_vo = CacheVOPlugin_Disabled

        return self
    # end

    def register_plugin_(self, model, config):
        model\
            .fill_plugin(config.klass_cache_past_kv)\
            .fill_plugin(config.klass_save_kv_previous)\
            .fill_plugin(config.klass_cache_attn)\
            .fill_plugin(config.klass_cache_vo)
        # end
    # end

    @torch.no_grad()
    def generate(self, model, tokenizer, config_diffusion, *args, **kwargs):
        num_blocks = config_diffusion.num_blocks
        size_block = config_diffusion.size_block
        step_per_block = config_diffusion.step_per_block
        id_mask = config_diffusion.id_mask
        sorter = config_diffusion.klass_sorter()
        collector = config_diffusion.klass_collector()

        words_stop = list(kwargs['until'])
        len_prompt = kwargs['len_prompt']
        x = kwargs['ids_input']
        assert x.shape[0] == 1, 'the fast-dllm runner is single-sample: size_batch=1'

        len_full = len_prompt + num_blocks * size_block
        assert x.shape[1] == len_full
        device = x.device

        idx_canvas = torch.arange(len_full, dtype=torch.long, device=device)
        shape_target = (x.shape[0], len_full, -1)

        snapshot = SimpleLogitsSnapshot(x, x, id_mask)

        for id_block in range(num_blocks):
            block_start = len_prompt + id_block * size_block
            block_end = block_start + size_block
            idx_block = idx_canvas[block_start:block_end]
            idx_block_2d = idx_block.unsqueeze(0)

            # standard quota decode over THIS block, like run_llada_instruct
            quota_helper = BlockDiffusionQuotaHelper(
                x[:, block_start:block_end] == id_mask, step_per_block)

            for step in range(step_per_block):
                if not bool((x[0, block_start:block_end] == id_mask).any()):
                    break    # block already fully unmasked
                # end

                if step == 0:
                    # DualCache refresh: full-canvas forward caches KV for the
                    # prefix AND the still-masked suffix, and decodes step 0
                    logits = model(x, idx_current=idx_canvas, shape_target=shape_target).logits
                    logits_block = logits[:, idx_block - 1] if self.DREAM_SHIFT else logits[:, idx_block]
                else:
                    # block-only forward against the frozen outside-KV
                    if self.DREAM_SHIFT:
                        idx_fwd = idx_canvas[block_start - 1:block_end]
                        logits = model(x[:, idx_fwd], idx_current=idx_fwd, shape_target=shape_target).logits
                        logits_block = logits[:, :-1]    # rows bs-1..be-2 predict tokens bs..be-1
                    else:
                        idx_fwd = idx_block
                        logits = model(x[:, idx_fwd], idx_current=idx_fwd, shape_target=shape_target).logits
                        logits_block = logits
                    # end
                # end

                snapshot.update_x0_(idx_block_2d, logits_block)
                conf_snapshot = snapshot.transform_logits(collector, logits_block, idx_transform=idx_block_2d)

                # confine to STILL-MASKED rows of the current block
                mask_no_cand = torch.ones(conf_snapshot.shape[-1], dtype=torch.bool, device=device)
                mask_no_cand[idx_block] = False
                mask_no_cand |= (x[0] != id_mask)
                conf_snapshot = conf_snapshot.masked_fill(
                    mask_no_cand.unsqueeze(0), torch.finfo(conf_snapshot.dtype).min)

                # uniform greedy decode: fixed quota (1/step), argmax by confidence
                num_unmask = quota_helper.get_quota(step)
                idx_sorted_by_conf = sorter.argsort(conf_snapshot, snapshot)
                idx_transform = idx_sorted_by_conf[:, :num_unmask]

                snapshot.materialize_by_idx_(idx_transform, conf_snapshot)
                snapshot.update_this(1, idx_transform, x0=x)
            # end for step
        # end for block

        '''assembly: instruct threads set truncate_at_eos; base threads decode plainly'''
        if getattr(config_diffusion, 'truncate_at_eos', None):
            if self.ids_stop is None:
                self.ids_stop = collect_ids_stop(tokenizer)
            # end
            sentence_all, has_done = truncate_text_at_stop(
                tokenizer, x[0, len_prompt:len_full], self.ids_stop, words_stop)
        else:
            sentence_all = tokenizer.decode(x[0, len_prompt:len_full], skip_special_tokens=True)
            has_done = any(word_stop in sentence_all for word_stop in words_stop)
            for word_stop in words_stop:
                if word_stop in sentence_all:
                    sentence_all = sentence_all.split(word_stop)[0]
                # end
            # end
        # end

        return sentence_all, has_done
    # end

    def run_one(self, model, tokenizer, config, *args, **kwargs):
        plugin_cache_past_kv = config.klass_cache_past_kv()
        plugin_cache_past_kv.clear(model)

        time_start = time.perf_counter()
        sentence_generated, has_done = self.generate(model, tokenizer, config, *args, **kwargs)
        duration_s = time.perf_counter() - time_start

        self.report.add_and_dump(config, kwargs['len_prompt'], has_done, duration_s)

        return sentence_generated, has_done
    # end
# end
