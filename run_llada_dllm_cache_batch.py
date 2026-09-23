#################################################
# BATCHED dLLM-Cache runner (size_batch > 1) -- the in-framework dLLM-Cache
# reimplementation (run_llada_dllm_cache) over a left-padded batch.
#
# Batching design:
#   - left-padded prompts (batch collater), response region column-aligned;
#     pad keys excluded via attention_mask -> additive bias
#   - the adaptive V-similarity selection happens PER LAYER PER SAMPLE inside
#     CacheVOPlugin_Batch_Enabled: each sample picks its own budget rows, one
#     forward computes the UNION, and the layer-output merge scatters each
#     sample's OWN rows only (per-sample method semantics preserved; the KV
#     cache does get union-fresh K/V -- documented deviation, exact at B=1)
#   - refresh clocks are shared (same steps for every sample), so Kp/Kr
#     ticks batch perfectly; only the adaptive steps pay the union cost --
#     which is the measurement this runner exists for: per-sample budgets of
#     v_rate x response (e.g. 64 rows at 0.25 x 256) are predicted to
#     saturate the union by B~4
#
# model_args example:
#   ...,runner=run_llada_dllm_cache_batch,size_batch=4,dllmc_v_rate=0.25,
#   step_refresh_remainder=16,step_refresh_remainder_prompt=96,...
#################################################

import time

import torch

from components_llada import SimpleLogitsSnapshot
from tools_llada import (BlockDiffusionQuotaHelper, RunnerReport,
                         collect_ids_stop, truncate_text_at_stop)
from plugins_llada import SaveKVPreviousPlugin_Disabled, CachePastKVPlugin_Enabled,\
                            CacheAttnPlugin_Disabled, CacheVOPlugin_Batch_Enabled


class RunModel:

    def __init__(self):
        self.report = RunnerReport()
        self.ids_stop = None
    # end

    def config_plugin_(self, config):
        config.klass_save_kv_previous = SaveKVPreviousPlugin_Disabled
        config.klass_cache_past_kv = CachePastKVPlugin_Enabled
        config.klass_cache_attn = CacheAttnPlugin_Disabled
        config.klass_cache_vo = CacheVOPlugin_Batch_Enabled

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
        len_prompt = kwargs['len_prompt']    # PADDED prompt length
        x = kwargs['ids_input']
        mask_attention = kwargs.get('attention_mask')
        if mask_attention is None:    # bs-1 collater path: no pads exist
            mask_attention = torch.ones_like(x)
        # end

        len_full = len_prompt + size_block
        assert x.shape[1] == len_full
        B = x.shape[0]
        device = x.device

        idx_full = torch.arange(len_full, dtype=torch.long, device=device)
        idx_gen = idx_full[len_prompt:]
        idx_gen_2d = idx_gen.unsqueeze(0).expand(B, -1)
        shape_target = (B, len_full, -1)

        snapshot = SimpleLogitsSnapshot(x, x, id_mask)
        quota_helper = BlockDiffusionQuotaHelper(x[:, len_prompt:] == id_mask, step_per_block)

        for step in range(step_per_block):
            if step == 0 or (kp and step % kp == 0):
                CacheVOPlugin_Batch_Enabled.set_force_mode('all')
            elif step % kr == 0:
                CacheVOPlugin_Batch_Enabled.set_force_mode('response')
            else:
                CacheVOPlugin_Batch_Enabled.set_force_mode(None)
            # end

            logits = model(x, idx_current=idx_full, shape_target=shape_target,
                           attention_mask=mask_attention).logits
            logits_gen = logits[:, idx_gen]

            snapshot.update_x0_(idx_gen_2d, logits_gen)
            conf_snapshot = snapshot.transform_logits(collector, logits_gen, idx_transform=idx_gen_2d)

            idx_sorted_by_conf = sorter.argsort(conf_snapshot, snapshot)
            num_unmask = quota_helper.get_quota(step)
            idx_transform = idx_sorted_by_conf[:, :num_unmask]    # (B, u) per sample

            snapshot.materialize_by_idx_(idx_transform, conf_snapshot)
            snapshot.update_this(1, idx_transform, x0=x)
        # end for step

        CacheVOPlugin_Batch_Enabled.set_force_mode(None)
        CacheVOPlugin_Batch_Enabled._OWN_MASK = None    # tidy per-sample state

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
