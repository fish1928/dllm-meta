#################################################
# ABLATION variant of run_llada_semi_mlp: refresh scope.
#
#   run_llada_semi_mlp (reference): every step_refresh_remainder steps,
#     re-forward the PROMPT and query the WHOLE block (all gen positions,
#     masked and unmasked) -> full KV re-sync + the whole block's conf/x0
#     refreshed periodically.
#
#   THIS variant: at refresh steps, query only the UNMASKED tokens of the
#     generation area (their KV re-syncs) -- the prompt is never re-forwarded
#     after init, and masked non-candidate positions never receive fresh
#     logits/conf between block start and their own unmasking (candidate
#     selection still refreshes the h candidates every step, as always).
#     Refresh steps therefore keep the normal router-driven sparse decode
#     shape; only the extra query rows change from [whole block]+[prompt] to
#     [unmasked gen tokens].
#
# The delta this measures: what does the periodic full-block + prompt re-sync
# buy (bounded staleness for masked conf and prompt KV), versus re-syncing
# only the tokens that actually changed -- the latter being one step toward
# the d2Cache never-refresh regime. Same config knobs as run_llada_semi_mlp;
# step 0 is unchanged (it seeds cache/attention/conf state).
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
        future_idx_selector = kwargs['future_idx_selector']
        router_bundle = kwargs.get('router_bundle')

        has_done = False

        idx_refresh = torch.tensor([], dtype=torch.long, device=x.device)

        position_start, position_end = 0, len_prompt
        idx_denoising = torch.arange(position_start, position_end, dtype=torch.long, device=x.device)
        idx_current = torch.cat([idx_refresh, idx_denoising])
        shape_target = (x.shape[0], position_end, -1)
        model(x[:, idx_current], idx_current=idx_current, shape_target=shape_target, skip_logits=True)
        snapshot = SimpleLogitsSnapshot(x[:, idx_current], x[:, idx_current], id_mask)

        for id_block in range(num_blocks):
            position_start = len_prompt + id_block * size_block
            position_end = position_start + size_block
            mask_mask_block = x[:,position_start:position_end] == id_mask
            quota_helper = BlockDiffusionQuotaHelper(mask_mask_block, step_per_block)

            if future_idx_selector.select_only_in_h:
                assert future_idx_selector.h >= quota_helper.get_quota_max(),\
                    'horizon h must cover the per-step unmask quota: {} >= {}'.format(
                        future_idx_selector.h, quota_helper.get_quota_max())
            # end

            idx_block = torch.arange(position_start, position_end, dtype=torch.long, device=x.device)
            shape_target = (x.shape[0], position_end, -1)

            for step in range(step_per_block):

                if step == 0:
                    # block init: unchanged from run_llada_semi_mlp -- seeds the
                    # KV cache, attention state and the block's conf/x0
                    idx_denoising = idx_block
                    idx_current = torch.cat([idx_refresh, idx_denoising])

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
                    score_attn_layers = plugin_cache_attn.collect_attn_from_all_blocks(model)
                    idx_in_attn = idx_transform_2d.squeeze(0) - position_start
                    mask_still = (x[0, position_start:position_end] == id_mask)

                    if router_bundle is not None:
                        router, spec_router = router_bundle
                        attn_rows_all = score_attn_layers[:, idx_in_attn, -idx_block.shape[-1]:].mean(dim=1)
                        conf_block = snapshot.conf[0, position_start:position_end]
                        idx_local = select_topk_candidates(
                            router, spec_router,
                            attn_rows_all.float(), conf_block.float(), mask_still,
                            idx_in_attn[-1], future_idx_selector.h,
                        )
                        idx_denoising = idx_local + position_start
                    else:
                        score_attn = score_attn_layers[-1, idx_in_attn, -idx_block.shape[-1]:]
                        score_attn = score_attn.mean(dim=0, keepdim=True)
                        score_attn.masked_fill_(~mask_still.view(1, -1), torch.finfo(score_attn.dtype).min)
                        idx_denoising = (future_idx_selector.select_future_by_attn(score_attn) + position_start).squeeze(0)
                    # end

                    if step % step_refresh_remainder == 0:
                        # THE ABLATION: refresh rows = every unmasked token in
                        # the generation area (their KV re-syncs); no prompt
                        # re-forward, no full-block query. Includes idx_refresh
                        # as a subset (just-unmasked tokens are unmasked).
                        mask_unmasked_gen = (x[0, len_prompt:position_end] != id_mask)
                        idx_unmasked_gen = mask_unmasked_gen.nonzero(as_tuple=True)[0] + len_prompt
                        idx_current = torch.cat([idx_unmasked_gen, idx_denoising])
                    else:
                        idx_current = torch.cat([idx_refresh, idx_denoising])
                    # end

                    logits = model(x[:, idx_current], idx_current=idx_current, shape_target=shape_target).logits
                    logits_transform = logits[:, -idx_denoising.shape[-1]:]

                    snapshot.update_x0_(idx_denoising.unsqueeze(0), logits_transform)
                    conf_snapshot = snapshot.transform_logits(collector, logits_transform, idx_transform=idx_denoising.unsqueeze(0))

                    if future_idx_selector.select_only_in_h:
                        mask_denoising_no = torch.ones(conf_snapshot.shape[-1], dtype=torch.bool, device=conf_snapshot.device)
                        mask_denoising_no[idx_denoising] = False
                        conf_snapshot.masked_fill_(mask_denoising_no.unsqueeze(0), torch.finfo(conf_snapshot.dtype).min)
                    # end
                # end

                idx_sorted_by_conf = sorter.argsort(conf_snapshot, snapshot)
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
