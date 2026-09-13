#################################################
# Shared scaffolding for the four router (cached-MLP) runners:
#   run_llada_semi_mlp      (llada-base,      growing window, one-block)
#   run_dream_semi_mlp      (dream-base,      growing window, one-block, dream shift)
#   run_llada_instruct_mlp  (llada-instruct,  FULL-CANVAS block diffusion, eos cut)
#   run_dream_instruct_mlp  (dream-instruct,  growing window, one-block, dream shift, eos cut)
#
# Everything thread-INVARIANT lives here: plugin wiring (KV + attn caches on),
# router-bundle / legacy-MLP loading, selector setup, per-sample cache clearing,
# and the timing report. Each thread file subclasses RunModelMLPBase and
# implements generate() only, so the per-thread decoding loop stays readable
# and free of other threads' machinery.
#################################################

import os

if os.environ.get("JINYU_DEBUG", False):
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # put this at the very top of your script
# end

import time

from tools_llada import RunnerReport
from plugins_llada import SaveKVPreviousPlugin_Disabled, CachePastKVPlugin_Enabled,\
                            CacheAttnPlugin_Enabled, CacheVOPlugin_Disabled

from future_idx_selector import FutureIDXSelector, FutureIdxSelectorModelLoader

from router_deploy import load_router_bundle

from tools_debug import jprint

from constants_llada import DTYPE_EVAL, NAME_MLP


class RunModelMLPBase:

    def __init__(self):
        self.mlp = None
        self.router_bundle = None    # (router, spec) when config.path_router is set
        self.report = RunnerReport()
        self.ids_stop = None    # instruct threads fill lazily via collect_ids_stop
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
        raise NotImplementedError('thread runner must implement generate()')
    # end

    def run_one(self, model, tokenizer, config, *args, **kwargs):

        config.klass_cache_attn.set_size_block(config.size_block)
        config.klass_cache_attn.set_len_prompt(kwargs['len_prompt'])

        path_router = getattr(config, 'path_router', None)
        if path_router:
            if self.router_bundle is None:
                router, spec_router = load_router_bundle(path_router, device=config.device)
                self.router_bundle = (router, spec_router)
                jprint(f'loaded router bundle {path_router}: {spec_router["features"]} norm={spec_router["normalization"]}')
            # end
        elif self.mlp is None:
            # legacy scalar-MLP fallback; NOTE it was trained on LLaDA attention
            # statistics -- retrain per thread before trusting benchmark numbers
            loader_mlp = FutureIdxSelectorModelLoader(1, config.device)
            self.mlp = loader_mlp.load(NAME_MLP).to(DTYPE_EVAL)
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
        kwargs['router_bundle'] = self.router_bundle

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
