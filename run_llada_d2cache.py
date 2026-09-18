#################################################
# d2Cache reimplementation inside the dllm-meta framework -- LLaDA runner.
# (arXiv 2509.23094; built because the official repo needs a transformers
# window incompatible with our B200 environment.)
#
# Method, faithful to their released design:
#   - ONE full-canvas prefill forward (prompt + all masks): KV for every
#     position + the initial per-position confidence table. NO periodic
#     refresh ever (their design: staleness is unbounded).
#   - per step, query rows = M* U extras:
#       M*     = top-k still-masked positions by  conf_table x certainty_density
#                (k=32, sigma=10 -- their paper values)
#       extras = just-unmasked tokens (KV re-sync) + attention-rollout nucleus
#                (p=0.1 over ALL positions incl. prompt) + gap inflation (w=0
#                in their eval config)
#     Only queried rows are forwarded; keys are the merged full-canvas cache.
#   - the unmask decision is confidence-argmax CONFINED to M* (non-candidates
#     have no fresh logits -- same confinement as their zero-logit scatter).
#   - d2c_conf_mode: 'live' = their INTENDED design (conf_table refreshed at
#     queried masked rows); 'frozen' = their RELEASED code (two bugs make the
#     table immutable after prefill). Default 'live' (stronger baseline).
#
# Runs ONE-BLOCK only (their method arm is blockless full-canvas; num_blocks=1).
# Threads: llada_base (plain prompts) and llada_instruct (use_chat_template /
# use_official_gsm8k_prompt + truncate_at_eos=True via model_args).
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


class RunModel:

    DREAM_SHIFT = False    # run_dream_d2cache overrides

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
        len_prompt = kwargs['len_prompt']
        x = kwargs['ids_input']
        plugin_cache_attn = kwargs['plugin_cache_attn']

        len_full = len_prompt + size_block
        assert x.shape[1] == len_full
        device = x.device

        idx_canvas = torch.arange(len_full, dtype=torch.long, device=device)
        idx_gen = idx_canvas[len_prompt:]
        idx_gen_2d = idx_gen.unsqueeze(0)
        shape_target = (x.shape[0], len_full, -1)

        '''prefill: one full-canvas forward -> KV everywhere + initial conf table'''
        logits = model(x, idx_current=idx_canvas, shape_target=shape_target).logits
        logits_gen = logits[:, idx_gen - 1] if self.DREAM_SHIFT else logits[:, idx_gen]

        snapshot = SimpleLogitsSnapshot(x, x, id_mask)
        snapshot.update_x0_(idx_gen_2d, logits_gen)
        conf_prefill = snapshot.transform_logits(collector, logits_gen, idx_transform=idx_gen_2d)
        conf_table = self._conf_from_logits(logits_gen).squeeze(0)    # (G,) their s_i table

        quota_helper = BlockDiffusionQuotaHelper(x[:, len_prompt:] == id_mask, step_per_block)
        idx_refresh = torch.tensor([], dtype=torch.long, device=device)

        for step in range(step_per_block):
            mask_still = (x[0, len_prompt:] == id_mask)    # (G,)
            if not bool(mask_still.any()):
                break
            # end

            '''candidate selection: conf_table x certainty_density, top-k masked'''
            density = certainty_density((~mask_still).view(1, -1), sigma).squeeze(0)    # (G,)
            score = conf_table * density
            score = score.masked_fill(~mask_still, torch.finfo(score.dtype).min)
            k_eff = min(k_cand, int(mask_still.sum()))
            idx_cand_local = score.topk(k_eff).indices
            idx_cand = idx_cand_local + len_prompt

            '''extras: just-unmasked resync + rollout nucleus + inflation'''
            mask_q = torch.zeros(1, len_full, dtype=torch.bool, device=device)
            mask_q[0, idx_cand] = True
            mask_q[0, idx_refresh] = True

            if rollout_p > 0:
                importance = plugin_cache_attn.get_global_importance()    # (1, T) from last forward
                if importance is not None:
                    mask_q |= nucleus_select(importance.float(), rollout_p, min_k=1, mask=~mask_q)
                # end
            # end
            mask_q = inflate_selection(mask_q, inflate_w)

            mask_extra = mask_q.squeeze(0).clone()
            mask_extra[idx_cand] = False    # candidates go LAST for logits slicing
            idx_extra = mask_extra.nonzero(as_tuple=True)[0]

            if self.DREAM_SHIFT:
                idx_current = torch.cat([idx_extra, idx_cand - 1, idx_cand])
            else:
                idx_current = torch.cat([idx_extra, idx_cand])
            # end

            logits = model(x[:, idx_current], idx_current=idx_current, shape_target=shape_target).logits
            logits_cand = logits[:, -2 * k_eff:-k_eff] if self.DREAM_SHIFT else logits[:, -k_eff:]

            '''unmask: confidence-argmax confined to the candidate set'''
            snapshot.update_x0_(idx_cand.unsqueeze(0), logits_cand)
            conf_snapshot = snapshot.transform_logits(collector, logits_cand, idx_transform=idx_cand.unsqueeze(0))

            mask_no_cand = torch.ones(conf_snapshot.shape[-1], dtype=torch.bool, device=device)
            mask_no_cand[idx_cand] = False
            conf_snapshot = conf_snapshot.masked_fill(mask_no_cand.unsqueeze(0), torch.finfo(conf_snapshot.dtype).min)

            idx_sorted_by_conf = sorter.argsort(conf_snapshot, snapshot)
            num_unmask = quota_helper.get_quota(step)
            idx_transform = idx_sorted_by_conf[:, :num_unmask]

            snapshot.materialize_by_idx_(idx_transform, conf_snapshot)
            snapshot.update_this(1, idx_transform, x0=x)
            idx_refresh = idx_transform.squeeze(0)

            if conf_mode == 'live':
                # their INTENDED lazy conf table: refresh s_i at queried masked rows
                conf_table[idx_cand_local] = self._conf_from_logits(logits_cand).squeeze(0)
            # end
        # end for step

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
        plugin_cache_attn = config.klass_cache_attn()

        plugin_cache_past_kv.clear(model)
        plugin_cache_attn.clear(model)

        kwargs['plugin_cache_attn'] = plugin_cache_attn

        time_start = time.perf_counter()
        sentence_generated, has_done = self.generate(model, tokenizer, config, *args, **kwargs)
        duration_s = time.perf_counter() - time_start

        self.report.add_and_dump(config, kwargs['len_prompt'], has_done, duration_s)

        return sentence_generated, has_done
    # end
# end
