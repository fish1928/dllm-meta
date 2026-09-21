#################################################
# llada-instruct router runner (KV/attn cache + sparse queries via router/MLP).
#
# Thread: GSAI-ML/LLaDA-8B-Instruct, NATIVE block diffusion under FULL-CANVAS
# semantics (cached counterpart of run_llada_instruct.py):
#   - the initial forward covers the WHOLE prompt+gen canvas, so the KV cache
#     holds every position (future blocks as masks) from step 0 -- queries then
#     stay sparse but always attend to full-canvas context, mirroring official
#     generate() through the cache approximation
#   - shape_target is canvas-sized and block-invariant
#   - conf outside the current block is forced to -inf on EVERY selection step:
#     future-block masks are legitimate "still masked" positions to the sorter
#     but must never be unmasked early
#   - THREE refresh clocks, all independent:
#       step_refresh_remainder         (Kr)  current-block re-query
#       step_refresh_remainder_prompt  (Kp)  prompt KV re-forward; None/0 =
#                                            never (v2 semantics -- note the
#                                            historical behavior coupled the
#                                            prompt to Kr; pass Kp=Kr to
#                                            reproduce old runs)
#       step_refresh_remainder_surfix        future-block KV re-forward
#   - step_refresh_remainder_surfix (optional): every this many steps the
#     SUFFIX -- future, still-masked blocks after the current one -- is
#     re-forwarded KV-only, so its cache re-encodes the tokens unmasked since
#     prefill. None/0 = historical behavior (suffix stale until its block
#     becomes current). The forward runs under CacheAttnPlugin SKIP_SAVE:
#     suffix rows belong to a future block and would otherwise wipe the
#     current block's attention table.
#   - EOS truncation ALWAYS on (instruct SFT EOS-fills the tail)
# Depends on the block-based attention-column slice in CacheAttnPlugin_Enabled
# (under full canvas the current block sits mid-window, not at the end).
# Prompting (chat template / official 4-shot CoT) lives in dataprocess_llada.
#################################################

import torch

from components_llada import SimpleLogitsSnapshot
from tools_llada import BlockDiffusionQuotaHelper, collect_ids_stop, truncate_text_at_stop
from router_deploy import select_topk_candidates
from runner_mlp_common import RunModelMLPBase
from plugins_llada import CacheAttnPlugin_Enabled


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
        remainder_prompt = getattr(config_diffusion, 'step_refresh_remainder_prompt', None)
        remainder_surfix = getattr(config_diffusion, 'step_refresh_remainder_surfix', None)

        words_stop = list(kwargs['until'])
        len_prompt = kwargs['len_prompt']
        x = kwargs['ids_input']

        plugin_cache_attn = kwargs['plugin_cache_attn']
        future_idx_selector = kwargs['future_idx_selector'] # budget is also here
        router_bundle = kwargs.get('router_bundle')    # (router, spec) or None (legacy MLP path)

        idx_refresh = torch.tensor([], dtype=torch.long, device=x.device)

        # full-canvas init: forward prompt + ALL mask blocks once, so the KV
        # cache spans the canvas and the snapshot is canvas-sized from step 0
        len_full = len_prompt + num_blocks * size_block
        idx_prompt = torch.arange(0, len_prompt, dtype=torch.long, device=x.device)
        idx_canvas = torch.arange(0, len_full, dtype=torch.long, device=x.device)
        shape_target = (x.shape[0], len_full, -1)
        model(x[:, idx_canvas], idx_current=idx_canvas, shape_target=shape_target, skip_logits=True)
        snapshot = SimpleLogitsSnapshot(x, x, id_mask)

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

            # future-block masks are still id_mask (so "candidates" to the
            # sorter) but must not be unmasked before their block starts
            mask_outside_block = torch.ones((1, len_full), dtype=torch.bool, device=x.device)
            mask_outside_block[:, position_start:position_end] = False

            for step in range(step_per_block):

                # PROMPT clock (Kp), decoupled from the generation clock (v2
                # semantics: None/0 = the prompt is NEVER refreshed after the
                # initial canvas forward). NOTE behavior change: historically
                # the prompt refreshed on the Kr clock -- pass
                # step_refresh_remainder_prompt=Kr to reproduce old runs.
                if remainder_prompt and step != 0 and step % remainder_prompt == 0:
                    model(x[:, idx_prompt], idx_current=idx_prompt, shape_target=shape_target, skip_logits=True)
                # end

                # SUFFIX clock: re-encode the future still-masked blocks (KV
                # only) against the tokens unmasked since prefill. SKIP_SAVE
                # keeps the attention plugin's current-block table intact --
                # these rows belong to future blocks.
                if remainder_surfix and step != 0 and step % remainder_surfix == 0 \
                        and position_end < len_full:
                    idx_surfix = torch.arange(position_end, len_full, dtype=torch.long, device=x.device)
                    CacheAttnPlugin_Enabled.set_skip_save(True)
                    try:
                        model(x[:, idx_surfix], idx_current=idx_surfix,
                              shape_target=shape_target, skip_logits=True)
                    finally:
                        CacheAttnPlugin_Enabled.set_skip_save(False)
                    # end
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

                    # canvas-sized rebuild: x/x0/conf are len_full already, so the
                    # rebuild only refreshes the snapshot's unmask bookkeeping
                    snapshot = SimpleLogitsSnapshot(x, x, id_mask, snapshot.x0, snapshot.conf)
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

                conf_snapshot = conf_snapshot.masked_fill(mask_outside_block, torch.finfo(conf_snapshot.dtype).min)

                idx_sorted_by_conf = sorter.argsort(conf_snapshot, snapshot)    # truth
                num_unmask = quota_helper.get_quota(step)
                idx_transform_2d = idx_sorted_by_conf[:, :num_unmask]

                snapshot.materialize_by_idx_(idx_transform_2d, conf_snapshot)
                snapshot.update_this(1, idx_src=idx_transform_2d, x0=x)
                idx_refresh = idx_transform_2d.squeeze(0)
            # end
        # end for

        if self.ids_stop is None:
            self.ids_stop = collect_ids_stop(tokenizer)
        # end
        sentence_all, has_done = truncate_text_at_stop(
            tokenizer, x[0, len_prompt:len_full], self.ids_stop, words_stop)

        return sentence_all, has_done
    # end function
# end class
