#################################################
# dLLM-Cache reimplementation inside the dllm-meta framework -- LLaDA runner.
# (arXiv 2506.06295; port of the earlier standalone
# test_ppl_yukai_llada_v1_dllm_cache.py into the current runner interface.)
#
# Method, faithful to their design:
#   - the runner forwards the FULL window (prompt + response) EVERY step; the
#     saving happens INSIDE each layer via CacheVOPlugin_Enabled:
#       * fresh V is projected for all rows (cheap), cosine-compared to the
#         cached V, and only the dllmc_v_rate LEAST-similar (= most-changed)
#         RESPONSE rows go through Q/K/attention/FFN ("adaptive partial
#         update", their transfer_ratio); the rest reuse cached layer outputs
#       * prompt rows are frozen between prompt refreshes
#   - every Kr steps (step_refresh_remainder): full RESPONSE recompute
#     (their gen_interval_steps)
#   - every Kp steps (step_refresh_remainder_prompt): full recompute
#     including the prompt (their prompt_interval_steps); step 0 is always
#     a full pass
#   - decode: greedy confidence-argmax, one token per step (the framework's
#     uniform decode rule)
#
# ONE-BLOCK only (their method is interval caching over a fixed window;
# num_blocks=1), batch size 1 (the VO plugin is single-sample).
#
# model_args example (v-rate 0.25, Kp 96, Kr 16):
#   ...,runner=run_llada_dllm_cache,dllmc_v_rate=0.25,
#   step_refresh_remainder=16,step_refresh_remainder_prompt=96,...
#################################################

import time

import torch

from components_llada import SimpleLogitsSnapshot
from tools_llada import (BlockDiffusionQuotaHelper, RunnerReport,
                         collect_ids_stop, truncate_text_at_stop)
from plugins_llada import SaveKVPreviousPlugin_Disabled, CachePastKVPlugin_Enabled,\
                            CacheAttnPlugin_Disabled, CacheVOPlugin_Enabled


class RunModel:

    def __init__(self):
        self.report = RunnerReport()
        self.ids_stop = None
    # end

    def config_plugin_(self, config):
        config.klass_save_kv_previous = SaveKVPreviousPlugin_Disabled
        config.klass_cache_past_kv = CachePastKVPlugin_Enabled
        config.klass_cache_attn = CacheAttnPlugin_Disabled
        config.klass_cache_vo = CacheVOPlugin_Enabled

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
        assert config_diffusion.num_blocks == 1, \
            'the dllm-cache runner is a fixed-window interval cache: set num_blocks=1'

        size_block = config_diffusion.size_block
        step_per_block = config_diffusion.step_per_block
        id_mask = config_diffusion.id_mask
        sorter = config_diffusion.klass_sorter()
        collector = config_diffusion.klass_collector()

        kr = config_diffusion.step_refresh_remainder or 16
        kp = getattr(config_diffusion, 'step_refresh_remainder_prompt', None)

        words_stop = list(kwargs['until'])
        len_prompt = kwargs['len_prompt']
        x = kwargs['ids_input']
        assert x.shape[0] == 1, 'the VO plugin is single-sample: size_batch=1'

        len_full = len_prompt + size_block
        assert x.shape[1] == len_full
        device = x.device

        idx_full = torch.arange(len_full, dtype=torch.long, device=device)
        idx_gen = idx_full[len_prompt:]
        idx_gen_2d = idx_gen.unsqueeze(0)
        shape_target = (x.shape[0], len_full, -1)

        snapshot = SimpleLogitsSnapshot(x, x, id_mask)
        quota_helper = BlockDiffusionQuotaHelper(x[:, len_prompt:] == id_mask, step_per_block)

        for step in range(step_per_block):
            # refresh schedule: step 0 and every Kp -> full (prompt included);
            # every Kr -> full response; otherwise adaptive v-rate update
            if step == 0 or (kp and step % kp == 0):
                CacheVOPlugin_Enabled.set_force_mode('all')
            elif step % kr == 0:
                CacheVOPlugin_Enabled.set_force_mode('response')
            else:
                CacheVOPlugin_Enabled.set_force_mode(None)
            # end

            logits = model(x, idx_current=idx_full, shape_target=shape_target).logits
            logits_gen = logits[:, idx_gen]

            snapshot.update_x0_(idx_gen_2d, logits_gen)
            conf_snapshot = snapshot.transform_logits(collector, logits_gen, idx_transform=idx_gen_2d)

            idx_sorted_by_conf = sorter.argsort(conf_snapshot, snapshot)
            num_unmask = quota_helper.get_quota(step)
            idx_transform = idx_sorted_by_conf[:, :num_unmask]

            snapshot.materialize_by_idx_(idx_transform, conf_snapshot)
            snapshot.update_this(1, idx_transform, x0=x)
        # end for step

        CacheVOPlugin_Enabled.set_force_mode(None)    # never leak into the next sample

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
        # per-sample plugin setup: window geometry + adaptive budget
        v_rate = config.dllmc_v_rate if getattr(config, 'dllmc_v_rate', None) is not None else 0.25
        config.klass_cache_vo\
            .set_prompt_length(kwargs['len_prompt'])\
            .set_response_length(config.size_block)\
            .set_update_budget_p(v_rate)

        plugin_cache_vo = config.klass_cache_vo()
        plugin_cache_past_kv = config.klass_cache_past_kv()

        plugin_cache_vo.clear_layer_past_and_output(model)
        plugin_cache_past_kv.clear(model)

        time_start = time.perf_counter()
        sentence_generated, has_done = self.generate(model, tokenizer, config, *args, **kwargs)
        duration_s = time.perf_counter() - time_start

        self.report.add_and_dump(config, kwargs['len_prompt'], has_done, duration_s)

        return sentence_generated, has_done
    # end
# end
