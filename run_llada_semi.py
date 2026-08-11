import json
import time

import torch
from tqdm import tqdm

from components_llada import SimpleLogitsSnapshot
from tools_llada import BlockDiffusionQuotaHelper
from plugins_llada import SaveKVPreviousPlugin_Disabled, SaveKVPreviousPlugin_Enabled,\
                            CachePastKVPlugin_Disabled, CachePastKVPlugin_Enabled,\
                            CacheAttnPlugin_Disabled, CacheAttnPlugin_Enabled,\
                            CacheVOPlugin_Disabled, CacheVOPlugin_Enabled

from tools_debug import jprint

# model runner
class RunModel:

    def __init__(self):
        self.report_rows = []
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

        words_stop = kwargs['until']
        len_prompt = kwargs['len_prompt']
        x = kwargs['ids_input']

        has_done = False
        position_start = 0        

        for id_block in range(num_blocks):
            position_end = position_start + len_prompt + (id_block+1) * size_block
            mask_mask_blk = x[:,position_start:position_end] == id_mask

            idx_denoising = torch.arange(position_start, position_end, dtype=torch.long).to(x.device)
            idx_block = torch.arange(position_end - size_block, position_end, dtype=torch.long).to(x.device)
            quota_helper = BlockDiffusionQuotaHelper(mask_mask_blk, step_per_block)    # quotas spread over actual steps, not block size
            shape_target = (x.shape[0], position_end, -1)

            for step in range(step_per_block):
                x_denoising,  y_denoising= x[:, idx_denoising], x[:, idx_denoising]
                logits = model(x_denoising, idx_current=idx_denoising, shape_target=shape_target).logits

                # only the current block can hold masked positions, so x0/conf are
                # computed on the block slice only (keeps softmax at (1, size_block, V));
                # window starts at 0 -> global positions == logits row positions
                snapshot = SimpleLogitsSnapshot(x_denoising, y_denoising, id_mask)
                snapshot.update_x0_(idx_block.unsqueeze(0), logits[:, idx_block])
                conf_snapshot = snapshot.transform_logits(collector, logits[:, idx_block], idx_transform=idx_block.unsqueeze(0))

                idx_sorted_by_conf = sorter.argsort(conf_snapshot, snapshot)
                num_unmask = quota_helper.get_quota(step)
                idx_transform = idx_sorted_by_conf[:, :num_unmask]

                snapshot.materialize_by_idx_(idx_transform, conf_snapshot)
                snapshot.update_this(1, idx_transform, x0=x)
            # end for step

            sentence_block_current = tokenizer.batch_decode(x[:, position_end-size_block:position_end])[0]

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

        # same per-sample report as run_llada_semi_cached_mlp, so baseline and
        # router runs produce directly comparable wall-clock artifacts
        path_report = getattr(config, 'path_report', None)
        if path_report:
            self.report_rows.append({
                'id_sample': len(self.report_rows),
                'len_prompt': kwargs['len_prompt'],
                'has_done': has_done,
                'duration_s': round(duration_s, 4),
            })
            with open(path_report, 'w') as file:    # rewritten per sample: crash-safe
                json.dump({
                    'path_router': None,
                    'num_samples': len(self.report_rows),
                    'duration_total_s': round(sum(r['duration_s'] for r in self.report_rows), 2),
                    'rows': self.report_rows,
                }, file, indent=2)
            # end
        # end

        return sentence_generated, has_done
    # end
# end

