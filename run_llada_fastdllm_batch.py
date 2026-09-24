#################################################
# BATCHED Fast-dLLM cache-only runner (size_batch > 1) -- run_llada_fastdllm
# over a left-padded batch. LLaDA only (the experiment thread runs llada_base).
#
# Batching design:
#   - left-padded prompts (batch collater), response region column-aligned;
#     pad keys excluded via attention_mask -> additive bias
#   - fast-dllm batches TRIVIALLY well by construction: the block schedule is
#     canvas-aligned and identical for every sample, so each forward carries
#     exactly the same rows for all samples -- no union inflation at all
#     (contrast d2cache/dllm-cache whose per-sample selections union up).
#     This runner exists to measure that scaling curve.
#   - decode: uniform greedy 1-token/step per sample under the block quota
#     (num_unmask_per_step=1), same cache-only semantics as the bs-1 runner.
#
# model_args example:
#   ...,runner=run_llada_fastdllm_batch,size_batch=8,num_blocks=8,...
#################################################

import time

import torch

from components_llada import SimpleLogitsSnapshot
from tools_llada import (BlockDiffusionQuotaHelper, RunnerReport,
                         collect_ids_stop, truncate_text_at_stop)
from plugins_llada import SaveKVPreviousPlugin_Disabled, CachePastKVPlugin_Enabled,\
                            CacheAttnPlugin_Disabled, CacheVOPlugin_Disabled


class RunModel:

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
        len_prompt = kwargs['len_prompt']    # PADDED prompt length
        x = kwargs['ids_input']
        mask_attention = kwargs.get('attention_mask')
        if mask_attention is None:    # bs-1 collater path: no pads exist
            mask_attention = torch.ones_like(x)
        # end

        len_full = len_prompt + num_blocks * size_block
        assert x.shape[1] == len_full
        B = x.shape[0]
        device = x.device

        idx_canvas = torch.arange(len_full, dtype=torch.long, device=device)
        shape_target = (B, len_full, -1)

        snapshot = SimpleLogitsSnapshot(x, x, id_mask)

        for id_block in range(num_blocks):
            block_start = len_prompt + id_block * size_block
            block_end = block_start + size_block
            idx_block = idx_canvas[block_start:block_end]
            idx_block_2d = idx_block.unsqueeze(0).expand(B, -1)

            quota_helper = BlockDiffusionQuotaHelper(
                x[:, block_start:block_end] == id_mask, step_per_block)

            for step in range(step_per_block):
                if not bool((x[:, block_start:block_end] == id_mask).any()):
                    break    # every sample finished the block
                # end

                if step == 0:
                    # DualCache refresh: full-canvas forward caches KV for the
                    # prefix AND the still-masked suffix, and decodes step 0
                    logits = model(x, idx_current=idx_canvas, shape_target=shape_target,
                                   attention_mask=mask_attention).logits
                    logits_block = logits[:, idx_block]
                else:
                    idx_fwd = idx_block
                    logits = model(x[:, idx_fwd], idx_current=idx_fwd, shape_target=shape_target,
                                   attention_mask=mask_attention).logits
                    logits_block = logits
                # end

                snapshot.update_x0_(idx_block_2d, logits_block)
                conf_snapshot = snapshot.transform_logits(collector, logits_block, idx_transform=idx_block_2d)

                # confine to STILL-MASKED rows of the current block, PER SAMPLE
                mask_no_cand = torch.ones(B, conf_snapshot.shape[-1], dtype=torch.bool, device=device)
                mask_no_cand[:, idx_block] = False
                mask_no_cand |= (x != id_mask)
                conf_snapshot = conf_snapshot.masked_fill(
                    mask_no_cand, torch.finfo(conf_snapshot.dtype).min)

                num_unmask = quota_helper.get_quota(step)
                idx_sorted_by_conf = sorter.argsort(conf_snapshot, snapshot)
                idx_transform = idx_sorted_by_conf[:, :num_unmask]    # (B, u) per sample

                snapshot.materialize_by_idx_(idx_transform, conf_snapshot)
                snapshot.update_this(1, idx_transform, x0=x)
            # end for step
        # end for block

        '''assembly per sample'''
        sentences = [''] * B
        has_done = [False] * B
        for b in range(B):
            if getattr(config_diffusion, 'truncate_at_eos', None):
                if self.ids_stop is None:
                    self.ids_stop = collect_ids_stop(tokenizer)
                # end
                sentence_all, done_each = truncate_text_at_stop(
                    tokenizer, x[b, len_prompt:len_full], self.ids_stop, words_stop)
            else:
                sentence_all = tokenizer.decode(x[b, len_prompt:len_full], skip_special_tokens=True)
                done_each = any(word_stop in sentence_all for word_stop in words_stop)
                for word_stop in words_stop:
                    if word_stop in sentence_all:
                        sentence_all = sentence_all.split(word_stop)[0]
                    # end
                # end
            # end
            sentences[b] = sentence_all
            has_done[b] = done_each
        # end

        return sentences, has_done
    # end

    def run_one(self, model, tokenizer, config, *args, **kwargs):
        plugin_cache_past_kv = config.klass_cache_past_kv()
        plugin_cache_past_kv.clear(model)

        time_start = time.perf_counter()
        sentences, has_done = self.generate(model, tokenizer, config, *args, **kwargs)
        duration_s = time.perf_counter() - time_start

        len_prompts = kwargs.get('len_prompts', [kwargs['len_prompt']] * len(sentences))
        for len_real, done_each in zip(len_prompts, has_done):
            self.report.add_and_dump(config, len_real, done_each,
                                     duration_s / len(sentences))
        # end

        return sentences, has_done
    # end
# end
