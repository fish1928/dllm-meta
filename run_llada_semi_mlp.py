#################################################
# llada-base router runner (KV/attn cache + sparse queries via router/MLP).
#
# Thread: GSAI-ML/LLaDA-8B-Base, ONE-BLOCK setting, growing window.
# Successor of run_llada_semi_cached_mlp.py: same validated decoding loop,
# with the shared scaffolding (plugin wiring, router loading, report) moved to
# runner_mlp_common.RunModelMLPBase. Base-only on purpose: no chat template,
# no EOS truncation -- instruct behavior lives in run_llada_instruct_mlp.py.
#
# Selection: refresh steps query the whole block; sparse steps query only
# [refresh | h candidates], candidates ranked by the router bundle
# (config.path_router) or the legacy scalar MLP when unset.
#################################################

import torch

from components_llada import SimpleLogitsSnapshot
from tools_llada import BlockDiffusionQuotaHelper
from router_deploy import select_topk_candidates
from runner_mlp_common import RunModelMLPBase


class RunModel(RunModelMLPBase):

    def generate(self, model, tokenizer, config_diffusion, *args, **kwargs):
        '''declare required variables'''
        num_blocks = config_diffusion.num_blocks
        step_per_block = config_diffusion.step_per_block
        size_block = config_diffusion.size_block
        id_mask = config_diffusion.id_mask
        sorter = config_diffusion.klass_sorter()
        collector = config_diffusion.klass_collector()

        step_refresh_remainder = config_diffusion.step_refresh_remainder

        words_stop = list(kwargs['until'])
        len_prompt = kwargs['len_prompt']
        x = kwargs['ids_input']

        plugin_cache_attn = kwargs['plugin_cache_attn']
        future_idx_selector = kwargs['future_idx_selector'] # budget is also here
        router_bundle = kwargs.get('router_bundle')    # (router, spec) or None (legacy MLP path)

        has_done = False

        idx_refresh = torch.tensor([], dtype=torch.long, device=x.device)

        position_start, position_end = 0, len_prompt
        idx_denoising = torch.arange(position_start, position_end, dtype=torch.long, device=x.device)
        idx_prompt = idx_denoising    # prompt positions, hoisted for the prompt-refresh forwards
        idx_current = torch.cat([idx_refresh, idx_denoising])
        shape_target = (x.shape[0], position_end, -1)
        model(x[:, idx_current], idx_current=idx_current, shape_target=shape_target, skip_logits=True)
        snapshot = SimpleLogitsSnapshot(x[:, idx_current], x[:, idx_current], id_mask)

        for id_block in range(num_blocks):
            position_start = len_prompt + id_block * size_block
            position_end = position_start + size_block
            mask_mask_block = x[:,position_start:position_end] == id_mask
            quota_helper = BlockDiffusionQuotaHelper(mask_mask_block, step_per_block)    # quotas spread over actual steps, not block size

            if future_idx_selector.select_only_in_h:
                assert future_idx_selector.h >= quota_helper.get_quota_max(),\
                    'horizon h must cover the per-step unmask quota: {} >= {}'.format(
                        future_idx_selector.h, quota_helper.get_quota_max())
            # end

            idx_block = torch.arange(position_start, position_end, dtype=torch.long, device=x.device)
            shape_target = (x.shape[0], position_end, -1)

            for step in range(step_per_block):

                if step != 0 and step % step_refresh_remainder == 0:
                    model(x[:, idx_prompt], idx_current=idx_prompt, shape_target=shape_target, skip_logits=True)
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

                    x_accumulated = x[:, :position_end]

                    conf_pad = torch.zeros(
                        (snapshot.conf.shape[0], x_accumulated.shape[1] - snapshot.conf.shape[1]),
                        dtype=snapshot.conf.dtype,
                        device=snapshot.conf.device)
                    conf_accumulated = torch.cat([snapshot.conf, conf_pad], dim=1)

                    x0_pad = torch.zeros(
                        (snapshot.x0.shape[0], x_accumulated.shape[1] - snapshot.x0.shape[1]),
                        dtype=snapshot.x0.dtype,
                        device=snapshot.x0.device)
                    x0_accumulated = torch.cat([snapshot.x0, x0_pad], dim=1)

                    snapshot = SimpleLogitsSnapshot(x_accumulated, x_accumulated, id_mask, x0_accumulated, conf_accumulated)
                    snapshot.update_x0_(idx_block.unsqueeze(0), logits_denoising)
                    conf_snapshot = snapshot.transform_logits(collector, logits_denoising, idx_transform=idx_block.unsqueeze(0))
                else:
                    score_attn_layers = plugin_cache_attn.collect_attn_from_all_blocks(model)    # (num_layers, size_block, size_block)
                    idx_in_attn = idx_transform_2d.squeeze(0) - position_start    # block is contiguous: global position -> block-local rows
                    mask_still = (x[0, position_start:position_end] == id_mask)

                    if router_bundle is not None:
                        # router path: online features (attention rows / geo / live conf)
                        # built exactly as at training time (router_deploy parity)
                        router, spec_router = router_bundle
                        attn_rows_all = score_attn_layers[:, idx_in_attn, -idx_block.shape[-1]:].mean(dim=1)    # (num_layers, size_block)
                        conf_block = snapshot.conf[0, position_start:position_end]
                        idx_local = select_topk_candidates(
                            router, spec_router,
                            attn_rows_all.float(), conf_block.float(), mask_still,
                            idx_in_attn[-1], future_idx_selector.h,
                        )
                        idx_denoising = idx_local + position_start
                    else:
                        # legacy scalar-MLP path
                        score_attn = score_attn_layers[-1, idx_in_attn, -idx_block.shape[-1]:]  # (num_unmask, size_block)
                        score_attn = score_attn.mean(dim=0, keepdim=True)  # aggregate the just-unmasked tokens' rows -> (1, size_block)
                        score_attn.masked_fill_(~mask_still.view(1, -1), torch.finfo(score_attn.dtype).min)
                        idx_denoising = (future_idx_selector.select_future_by_attn(score_attn) + position_start).squeeze(0)
                    # end
                    idx_current = torch.cat([idx_refresh, idx_denoising])

                    logits = model(x[:, idx_current], idx_current=idx_current, shape_target=shape_target).logits
                    logits_transform = logits[:, -idx_denoising.shape[-1]:]

                    # different here compared to step == 0
                    snapshot.update_x0_(idx_denoising.unsqueeze(0), logits_transform)
                    conf_snapshot = snapshot.transform_logits(collector, logits_transform, idx_transform=idx_denoising.unsqueeze(0))
                    # different ends

                    if future_idx_selector.select_only_in_h: #TODO: be careful of the use of scatter(shape)
                        mask_denoising_no = torch.ones(conf_snapshot.shape[-1], dtype=torch.bool, device=conf_snapshot.device)
                        mask_denoising_no[idx_denoising] = False    # True everywhere except the h selected positions
                        conf_snapshot.masked_fill_(mask_denoising_no.unsqueeze(0), torch.finfo(conf_snapshot.dtype).min)
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
# end class
