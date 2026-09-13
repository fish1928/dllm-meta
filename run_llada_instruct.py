#################################################
# llada-instruct baseline runner (full denoising, no cache/router).
#
# Thread: GSAI-ML/LLaDA-8B-Instruct, NATIVE block diffusion (official gsm8k
# recipe: gen 256, block size 8 -> num_blocks = len_target / 8).
# Two instruct behaviors are ALWAYS on here (no flags):
#   - full-canvas windows: official generate() forwards the whole prompt+gen
#     canvas every step, future blocks visible as masks (a growing window is a
#     DIFFERENT decoding algorithm and costs real accuracy at small blocks)
#   - EOS truncation: instruct SFT EOS-fills the tail; without an ids-level cut
#     at the first EOS, later-block junk corrupts last-number extraction
# Prompting (chat template / official 4-shot CoT) lives in dataprocess_llada:
# pass use_chat_template=True or use_official_gsm8k_prompt=True in model_args.
#################################################

import time

import torch

from components_llada import SimpleLogitsSnapshot
from tools_llada import BlockDiffusionQuotaHelper, RunnerReport,\
                        collect_ids_stop, truncate_text_at_stop
from plugins_llada import SaveKVPreviousPlugin_Disabled, CachePastKVPlugin_Disabled,\
                            CacheAttnPlugin_Disabled, CacheVOPlugin_Disabled


class RunModel:

    def __init__(self):
        self.report = RunnerReport()
        self.ids_stop = None
    # end

    def config_plugin_(self, config):
        config.klass_save_kv_previous=SaveKVPreviousPlugin_Disabled
        config.klass_cache_past_kv=CachePastKVPlugin_Disabled
        config.klass_cache_attn=CacheAttnPlugin_Disabled
        config.klass_cache_vo=CacheVOPlugin_Disabled

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

    @ torch.no_grad()
    def generate(self, model, tokenizer, config_diffusion, *args, **kwargs):

        '''declare required variables'''
        num_blocks = config_diffusion.num_blocks
        step_per_block = config_diffusion.step_per_block
        size_block = config_diffusion.size_block
        id_mask = config_diffusion.id_mask
        sorter = config_diffusion.klass_sorter()
        collector = config_diffusion.klass_collector()

        words_stop = list(kwargs['until'])
        len_prompt = kwargs['len_prompt']
        x = kwargs['ids_input']

        position_start = 0
        len_full = len_prompt + num_blocks * size_block

        # official generate() semantics: the whole canvas is in context every
        # step, so window bounds and shape_target are block-invariant
        idx_denoising = torch.arange(position_start, len_full, dtype=torch.long).to(x.device)
        shape_target = (x.shape[0], len_full, -1)

        for id_block in range(num_blocks):
            block_end = len_prompt + (id_block + 1) * size_block
            block_start = block_end - size_block

            mask_mask_blk = x[:, block_start:block_end] == id_mask    # quota counts THIS block only

            idx_block = torch.arange(block_start, block_end, dtype=torch.long).to(x.device)
            quota_helper = BlockDiffusionQuotaHelper(mask_mask_blk, step_per_block)    # quotas spread over actual steps, not block size

            for step in range(step_per_block):
                x_denoising,  y_denoising= x[:, idx_denoising], x[:, idx_denoising]
                logits = model(x_denoising, idx_current=idx_denoising, shape_target=shape_target).logits

                # only the current block may be unmasked, so x0/conf are computed
                # on the block slice only (keeps softmax at (1, size_block, V));
                # window starts at 0 -> global positions == logits row positions
                snapshot = SimpleLogitsSnapshot(x_denoising, y_denoising, id_mask)
                snapshot.update_x0_(idx_block.unsqueeze(0), logits[:, idx_block])
                conf_snapshot = snapshot.transform_logits(collector, logits[:, idx_block], idx_transform=idx_block.unsqueeze(0))

                # future-block masks are in the window but must never be
                # unmasked; their conf is 0.0, which could tie with an
                # underflowed real confidence -> force them to -inf
                mask_outside = torch.ones_like(conf_snapshot, dtype=torch.bool)
                mask_outside[:, block_start:block_end] = False
                conf_snapshot = conf_snapshot.masked_fill(mask_outside, torch.finfo(conf_snapshot.dtype).min)

                idx_sorted_by_conf = sorter.argsort(conf_snapshot, snapshot)
                num_unmask = quota_helper.get_quota(step)
                idx_transform = idx_sorted_by_conf[:, :num_unmask]

                snapshot.materialize_by_idx_(idx_transform, conf_snapshot)
                snapshot.update_this(1, idx_transform, x0=x)
            # end for step
        # end for block

        if self.ids_stop is None:
            self.ids_stop = collect_ids_stop(tokenizer)
        # end
        sentence_all, has_done = truncate_text_at_stop(
            tokenizer, x[0, len_prompt:len_full], self.ids_stop, words_stop)

        return sentence_all, has_done
    # end

    def run_one(self, model, tokenizer, config, *args, **kwargs):

        time_start = time.perf_counter()
        sentence_generated, has_done = self.generate(
            model,
            tokenizer,
            config,
            *args,
            **kwargs
        )
        duration_s = time.perf_counter() - time_start

        self.report.add_and_dump(config, kwargs['len_prompt'], has_done, duration_s)

        return sentence_generated, has_done
    # end
# end
