#################################################
# BATCHED d2Cache runner (size_batch > 1) -- the in-framework d2Cache
# reimplementation (run_llada_d2cache) generalized to a left-padded batch.
#
# Batching design (mirrors run_llada_semi_mlp_v2_batch):
#   - left-padded prompts, generation area column-aligned; pad keys excluded
#     via attention_mask -> additive attention bias.
#   - the per-step CANDIDATE SETS (top-k by conf x certainty-density, rollout
#     nucleus extras, just-unmasked resync) are per-sample; one forward runs
#     the UNION of all samples' rows, then each sample gathers its own
#     candidate logits from the union via a position lookup.
#   - k_eff stays equal across the batch (all samples unmask the same quota
#     per step, so their masked counts match), which keeps the candidate
#     gather rectangular; only the rollout/resync extras are ragged, and
#     those live inside the shared union.
#   - rollout importance is (B, T) already; pad columns are excluded from
#     nucleus selection (they would otherwise soak up rollout mass).
#
# LLaDA only (DREAM_SHIFT not ported). conf_mode live/frozen as in bs-1.
# Usage (model_args): runner=run_llada_d2cache_batch,size_batch=4,...
#################################################

import time

import torch
import torch.nn.functional as F

from components_llada import SimpleLogitsSnapshot
from tools_llada import (BlockDiffusionQuotaHelper, RunnerReport,
                         certainty_density, nucleus_select, inflate_selection,
                         collect_ids_stop, truncate_text_at_stop)
from plugins_llada import SaveKVPreviousPlugin_Disabled, CachePastKVPlugin_Enabled,\
                            CacheAttnRolloutPlugin_Enabled, CacheAttnPlugin_Disabled,\
                            CacheVOPlugin_Disabled
# CachePastKVPlugin_Enabled.set_row_mask keeps each sample's KV staleness
# faithful to its bs-1 trajectory under union forwards (see plugin docstring)


class RunModel:

    def __init__(self):
        self.report = RunnerReport()
        self.ids_stop = None
    # end

    def config_plugin_(self, config):
        config.klass_save_kv_previous = SaveKVPreviousPlugin_Disabled
        config.klass_cache_past_kv = CachePastKVPlugin_Enabled
        rollout_p = config.d2c_rollout_p if config.d2c_rollout_p is not None else 0.1
        config.klass_cache_attn = CacheAttnRolloutPlugin_Enabled if rollout_p > 0 \
            else CacheAttnPlugin_Disabled
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

    @staticmethod
    def _conf_from_logits(logits):
        # per-position confidence = max softmax probability (their s_i)
        return F.softmax(logits.float(), dim=-1).max(dim=-1).values
    # end

    @torch.no_grad()
    def generate(self, model, tokenizer, config_diffusion, *args, **kwargs):
        assert config_diffusion.num_blocks == 1, \
            'd2cache runners are blockless full-canvas by design: set num_blocks=1'

        size_block = config_diffusion.size_block
        step_per_block = config_diffusion.step_per_block
        id_mask = config_diffusion.id_mask
        sorter = config_diffusion.klass_sorter()
        collector = config_diffusion.klass_collector()

        k_cand = config_diffusion.d2c_k if config_diffusion.d2c_k is not None else 32
        sigma = config_diffusion.d2c_sigma if config_diffusion.d2c_sigma is not None else 10.0
        rollout_p = config_diffusion.d2c_rollout_p if config_diffusion.d2c_rollout_p is not None else 0.1
        conf_mode = config_diffusion.d2c_conf_mode or 'live'
        inflate_w = config_diffusion.d2c_inflate_w if config_diffusion.d2c_inflate_w is not None else 0

        words_stop = list(kwargs['until'])
        len_prompt = kwargs['len_prompt']    # PADDED prompt length
        x = kwargs['ids_input']
        mask_attention = kwargs['attention_mask']    # (B, T), 0 at left pads
        plugin_cache_attn = kwargs['plugin_cache_attn']

        len_full = len_prompt + size_block
        assert x.shape[1] == len_full
        B = x.shape[0]
        device = x.device
        mask_real = mask_attention.bool()    # (B, len_full)

        idx_canvas = torch.arange(len_full, dtype=torch.long, device=device)
        idx_gen = idx_canvas[len_prompt:]
        idx_gen_2d = idx_gen.unsqueeze(0).expand(B, -1)
        shape_target = (B, len_full, -1)

        '''prefill: one full-canvas forward -> KV everywhere + initial conf table'''
        if rollout_p > 0:
            # per-sample queried rows for the rollout: real positions only
            # (pads are forwarded but must stay identity in the rollout)
            type(plugin_cache_attn).set_row_mask(mask_real)
        # end
        logits = model(x, idx_current=idx_canvas, shape_target=shape_target,
                       attention_mask=mask_attention).logits
        logits_gen = logits[:, idx_gen]

        snapshot = SimpleLogitsSnapshot(x, x, id_mask)
        snapshot.update_x0_(idx_gen_2d, logits_gen)
        snapshot.transform_logits(collector, logits_gen, idx_transform=idx_gen_2d)
        conf_table = self._conf_from_logits(logits_gen)    # (B, G) their s_i table

        quota_helper = BlockDiffusionQuotaHelper(x[:, len_prompt:] == id_mask, step_per_block)
        idx_refresh_2d = None    # (B, u) just-unmasked, per sample

        for step in range(step_per_block):
            mask_still = (x[:, len_prompt:] == id_mask)    # (B, G)
            if not bool(mask_still.any()):
                break
            # end

            '''candidate selection: conf_table x certainty_density, top-k masked
            (per sample; equal quotas keep every sample's masked count -- and
            hence k_eff -- identical, so the gather stays rectangular)'''
            density = certainty_density(~mask_still, sigma)    # (B, G)
            score = conf_table * density
            score = score.masked_fill(~mask_still, torch.finfo(score.dtype).min)
            k_eff = min(k_cand, int(mask_still.sum(dim=1).min()))
            idx_cand_local = score.topk(k_eff, dim=1).indices    # (B, k_eff)
            idx_cand_2d = idx_cand_local + len_prompt

            '''extras: just-unmasked resync + rollout nucleus + inflation'''
            mask_q = torch.zeros(B, len_full, dtype=torch.bool, device=device)
            mask_q.scatter_(1, idx_cand_2d, True)
            if idx_refresh_2d is not None:
                mask_q.scatter_(1, idx_refresh_2d, True)
            # end

            if rollout_p > 0:
                importance = plugin_cache_attn.get_global_importance()    # (B, T)
                if importance is not None:
                    # pads carry no information; exclude them from the nucleus
                    mask_q |= nucleus_select(importance.float(), rollout_p, min_k=1,
                                             mask=(~mask_q) & mask_real)
                # end
            # end
            mask_q = inflate_selection(mask_q, inflate_w)
            mask_q &= mask_real    # inflation must never pull pad rows in

            '''ONE forward over the union of all samples' query rows; the
            rollout only counts each sample's OWN rows as queried'''
            idx_union = mask_q.any(dim=0).nonzero(as_tuple=True)[0]
            if rollout_p > 0:
                type(plugin_cache_attn).set_row_mask(mask_q)
            # end
            CachePastKVPlugin_Enabled.set_row_mask(mask_q)
            logits_union = model(x[:, idx_union], idx_current=idx_union,
                                 shape_target=shape_target,
                                 attention_mask=mask_attention).logits    # (B, U, V)
            CachePastKVPlugin_Enabled.set_row_mask(None)

            lut = torch.full((len_full,), -1, dtype=torch.long, device=device)
            lut[idx_union] = torch.arange(idx_union.shape[0], device=device)
            rows = lut[idx_cand_2d]    # (B, k_eff) rows into the union axis
            logits_cand = logits_union.gather(
                1, rows.unsqueeze(-1).expand(-1, -1, logits_union.shape[-1]))

            '''unmask: confidence-argmax confined to each sample's candidate set'''
            snapshot.update_x0_(idx_cand_2d, logits_cand)
            conf_snapshot = snapshot.transform_logits(collector, logits_cand,
                                                      idx_transform=idx_cand_2d)

            mask_no_cand = torch.ones(B, conf_snapshot.shape[-1], dtype=torch.bool, device=device)
            mask_no_cand.scatter_(1, idx_cand_2d, False)
            conf_snapshot = conf_snapshot.masked_fill(mask_no_cand,
                                                      torch.finfo(conf_snapshot.dtype).min)

            idx_sorted_by_conf = sorter.argsort(conf_snapshot, snapshot)
            num_unmask = quota_helper.get_quota(step)
            idx_transform = idx_sorted_by_conf[:, :num_unmask]    # (B, u)

            snapshot.materialize_by_idx_(idx_transform, conf_snapshot)
            snapshot.update_this(1, idx_transform, x0=x)
            idx_refresh_2d = idx_transform

            if conf_mode == 'live':
                # their INTENDED lazy conf table: refresh s_i at queried masked rows
                conf_table.scatter_(1, idx_cand_local, self._conf_from_logits(logits_cand))
            # end
        # end for step

        if rollout_p > 0:
            type(plugin_cache_attn).set_row_mask(None)    # never leak into the next batch
        # end

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
        plugin_cache_attn = config.klass_cache_attn()

        plugin_cache_past_kv.clear(model)
        plugin_cache_attn.clear(model)

        kwargs['plugin_cache_attn'] = plugin_cache_attn

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
