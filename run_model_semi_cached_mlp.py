import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # put this at the very top of your script

import torch
from tqdm import tqdm

from components_llada import SimpleLogitsSnapshot
from tools_llada import BlockDiffusionQuotaHelper
from plugins_llada import SaveKVPreviousPlugin_Disabled, SaveKVPreviousPlugin_Enabled,\
                            CachePastKVPlugin_Disabled, CachePastKVPlugin_Enabled,\
                            CacheAttnPlugin_Disabled, CacheAttnPlugin_Enabled,\
                            CacheVOPlugin_Disabled, CacheVOPlugin_Enabled

from future_idx_selector import FutureIDXSelector, RandomModel, FutureIdxSelectorModelLoader

from tools_debug import jprint

from constants_llada import DTYPE_EVAL, NAME_MLP


class RunModel:

    def __init__(self):
        self.mlp = None
    # end

    def config_plugin_(self, config):
        config.klass_save_kv_previous=SaveKVPreviousPlugin_Disabled
        config.klass_cache_past_kv=CachePastKVPlugin_Enabled
        config.klass_cache_attn=CacheAttnPlugin_Enabled
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


    def generate(self, model, tokenizer, config_diffusion, *args, **kwargs):
        '''declare required variables'''
        num_blocks = config_diffusion.num_blocks
        step_per_block = config_diffusion.step_per_block
        size_block = config_diffusion.size_block
        id_mask = config_diffusion.id_mask
        sorter = config_diffusion.klass_sorter()
        collector = config_diffusion.klass_collector()

        step_refresh_remainder = config_diffusion.step_refresh_remainder
        
        words_stop = kwargs['until']
        len_prompt = kwargs['len_prompt']
        text_prompt = kwargs['text_prompt']
        x = kwargs['ids_input']

        plugin_cache_attn = kwargs['plugin_cache_attn']
        future_idx_selector = kwargs['future_idx_selector'] # budget is also here

        has_done = False

        idx_refresh = torch.tensor([], dtype=torch.long, device=x.device)

        position_start, position_end = 0, len_prompt
        idx_denoising = torch.arange(position_start, position_end, dtype=torch.long, device=x.device)
        idx_current = torch.cat([idx_refresh, idx_denoising])
        shape_target = (x.shape[0], position_end, -1)
        logits = model(x[:, idx_current], idx_current=idx_current, shape_target=shape_target).logits
        snapshot = SimpleLogitsSnapshot(logits, x[:, idx_current], x[:, idx_current], id_mask)

        for id_block in range(num_blocks):
            position_start = len_prompt + id_block * size_block
            position_end = position_start + size_block
            mask_mask_block = x[:,position_start:position_end] == id_mask
            quota_helper = BlockDiffusionQuotaHelper(mask_mask_block, size_block)

            idx_block = torch.arange(position_start, position_end, dtype=torch.long, device=x.device)
            shape_target = (x.shape[0], position_end, -1)

            for step in range(step_per_block):

                if step != 0 and step % step_refresh_remainder == 0:
                    idx_prompt = torch.arange(0, len_prompt, dtype=torch.long).to(x.device)
                    model(x[:, idx_prompt], idx_current=idx_prompt, shape_target=shape_target)
                # end

                if step == 0 or step % step_refresh_remainder == 0:
                    idx_denoising = idx_block

                    if step == 0:
                        idx_current = torch.cat([idx_refresh, idx_denoising])   # only the first time need refresh previous
                    else:
                        idx_current = idx_denoising
                    # end

                    logits = model(x[:, idx_current], idx_current=idx_current, shape_target=shape_target).logits
                    logits_denoising = logits[:, -idx_denoising.shape[-1]:]

                    logits_accumulated = torch.cat([snapshot.get_logits()[:, :position_start, :], logits_denoising], dim=1)
                    x_accumulated = x[:, :position_end]
                    snapshot = SimpleLogitsSnapshot(logits_accumulated, x_accumulated, x_accumulated, id_mask)
                    conf_snapshot = snapshot.transform_logits(collector)
                else:
                    score_attn = plugin_cache_attn.collect_attn_from_all_blocks(model)
                    idx_in_attn = torch.where(idx_transform_2d.squeeze(0) == idx_block)[0]    # idx_current is now last round
                    score_attn = score_attn[-1, idx_in_attn, -idx_block.shape[-1]:].squeeze(1)  # (1,64)
                    mask_mask_current_no = ~(x[:,position_start:position_end] == id_mask).view(1,-1)    # (B, K)
                    score_attn.masked_fill_(mask_mask_current_no, torch.finfo(score_attn.dtype).min)
                    idx_denoising = (future_idx_selector.select_future_by_attn(score_attn) + position_start).squeeze(0)
                    idx_current = torch.cat([idx_refresh, idx_denoising])

                    logits = model(x[:, idx_current], idx_current=idx_current, shape_target=shape_target).logits
                    logits_transform = logits[:, -idx_denoising.shape[-1]:]

                    # different here compared to step == 0
                    snapshot.update_logits_(idx_denoising.unsqueeze(0), logits_transform)
                    conf_snapshot = snapshot.transform_logits(collector)
                    # different ends

                    if future_idx_selector.select_only_in_h: #TODO: be careful of the use of scatter(shape)
                        mask_denoising_no = ~torch.isin(torch.arange(conf_snapshot.shape[-1], device=conf_snapshot.device), idx_denoising).unsqueeze(0)    # idx_denoising -> 
                        conf_snapshot.masked_fill_(mask_denoising_no, torch.finfo(conf_snapshot.dtype).min)
                    # end
                # end

                idx_sorted_by_conf = sorter.argsort(conf_snapshot, snapshot)    # truth
                num_unmask = quota_helper.get_quota(step)
                idx_transform_2d = idx_sorted_by_conf[:, :num_unmask]

                snapshot.materialize_by_idx_(idx_transform_2d, conf_snapshot) 
                snapshot.update_this(1, idx_src=idx_transform_2d, x0=x)
                idx_refresh = idx_transform_2d.squeeze(0)
            # end

            sentence_block_current = tokenizer.batch_decode(x[:, idx_block])[0]

            for word_stop in words_stop:
                if word_stop in sentence_block_current:
                    sentence_block_current = sentence_block_current.split(word_stop)[0]
                    has_done = True
                # end
            # end
        # end for

        sentence_block_previous = tokenizer.batch_decode(x[:, len_prompt:position_start], skip_special_tokens=False)[0]
        sentence_all = sentence_block_previous + sentence_block_current
        sentence_all = tokenizer.decode(tokenizer(sentence_all)['input_ids'], skip_special_tokens=True)

        return sentence_all, has_done
    # end function


    def run_one(self, model, tokenizer, config, *args, **kwargs):

        config.klass_cache_attn.set_size_block(config.size_block)
        config.klass_cache_attn.set_len_prompt(kwargs['len_prompt'])

        if self.mlp is None:
            loader_mlp = FutureIdxSelectorModelLoader(1, config.device)
            self.mlp = loader_mlp.load(NAME_MLP).to(DTYPE_EVAL)
            # self.mlp = RandomModel()
        # end

        kwargs_selector = {}
        if config.h:
            kwargs_selector['h'] = config.h
        # end

        if config.select_only_in_h:
            kwargs_selector['select_only_in_h'] = config.select_only_in_h
        # end

        future_idx_selector = FutureIDXSelector(self.mlp, **kwargs_selector)

        plugin_cache_past_kv = config.klass_cache_past_kv()
        plugin_cache_attn = config.klass_cache_attn()

        plugin_cache_past_kv.clear(model)
        plugin_cache_attn.clear(model)

        kwargs['future_idx_selector'] = future_idx_selector
        kwargs['plugin_cache_attn'] = plugin_cache_attn

        sentence_generated, has_done = self.generate(
            model,
            tokenizer,
            config,
            *args,
            **kwargs
        )

        return sentence_generated, has_done
    # end    
# end class