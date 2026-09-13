#################################################
# dream-base baseline runner (full denoising, no cache/router).
#
# Thread: Dream-org/Dream-v0-Base-7B, ONE-BLOCK setting (num_blocks=1).
# Same skeleton as run_llada_semi.py with ONE model-family difference:
#   Dream predicts the token at position p from the OUTPUT ROW at position p-1
#   (AR-style shift inherited from Qwen2.5 init; see run_ppl_dream.py:
#   logits = cat([logits[:,:1], logits[:,:-1]])). The full window [0, block_end)
#   is forwarded anyway, so the shift is just a row re-index: the logits for
#   block positions idx_block are the rows at idx_block - 1 (block_start >= 1
#   always, the prompt is never empty).
# Base-only on purpose: no chat template, no EOS truncation -- instruct
# behaviors live in run_dream_instruct.py.
#################################################

import time

import torch

from components_llada import SimpleLogitsSnapshot
from tools_llada import BlockDiffusionQuotaHelper, RunnerReport
from plugins_llada import SaveKVPreviousPlugin_Disabled, CachePastKVPlugin_Disabled,\
                            CacheAttnPlugin_Disabled, CacheVOPlugin_Disabled


class RunModel:

    def __init__(self):
        self.report = RunnerReport()
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

        has_done = False
        position_start = 0

        for id_block in range(num_blocks):
            block_end = len_prompt + (id_block + 1) * size_block
            block_start = block_end - size_block
            position_end = block_end    # growing window: context ends at the current block

            mask_mask_blk = x[:, block_start:block_end] == id_mask

            idx_denoising = torch.arange(position_start, position_end, dtype=torch.long).to(x.device)
            idx_block = torch.arange(block_start, block_end, dtype=torch.long).to(x.device)
            idx_block_shifted = idx_block - 1    # dream shift: row p-1 carries the logits for p
            quota_helper = BlockDiffusionQuotaHelper(mask_mask_blk, step_per_block)    # quotas spread over actual steps, not block size
            shape_target = (x.shape[0], position_end, -1)

            for step in range(step_per_block):
                x_denoising,  y_denoising= x[:, idx_denoising], x[:, idx_denoising]
                logits = model(x_denoising, idx_current=idx_denoising, shape_target=shape_target).logits

                # only the current block may be unmasked, so x0/conf are computed
                # on the block slice only (keeps softmax at (1, size_block, V));
                # window starts at 0 -> global positions == logits row positions,
                # and the SHIFTED rows carry the block's predictions
                snapshot = SimpleLogitsSnapshot(x_denoising, y_denoising, id_mask)
                snapshot.update_x0_(idx_block.unsqueeze(0), logits[:, idx_block_shifted])
                conf_snapshot = snapshot.transform_logits(collector, logits[:, idx_block_shifted], idx_transform=idx_block.unsqueeze(0))

                idx_sorted_by_conf = sorter.argsort(conf_snapshot, snapshot)
                num_unmask = quota_helper.get_quota(step)
                idx_transform = idx_sorted_by_conf[:, :num_unmask]

                snapshot.materialize_by_idx_(idx_transform, conf_snapshot)
                snapshot.update_this(1, idx_transform, x0=x)
            # end for step

            sentence_block_current = tokenizer.batch_decode(x[:, block_start:block_end])[0]

            for word_stop in words_stop:
                if word_stop in sentence_block_current:
                    sentence_block_current = sentence_block_current.split(word_stop)[0]
                    has_done = True
                # end
            # end
        # end for block

        sentence_block_previous = tokenizer.batch_decode(x[:, len_prompt:position_end-size_block], skip_special_tokens=False)[0]
        sentence_all = sentence_block_previous + sentence_block_current
        sentence_all = tokenizer.decode(tokenizer(sentence_all)['input_ids'], skip_special_tokens=True)

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
