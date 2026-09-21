from dataclasses import dataclass
from typing import Optional

from tools_llada import ConfKSorter, ConfCollectorInterface, BlockDiffusionQuotaHelper
from plugins_llada import InspectorPlugin


@dataclass
class DiffusionConfig:
    id_model: str
    len_prompt: int
    len_target: int
    num_blocks: int
    num_unmask_per_step: int
    id_mask: int
    size_batch: int
    device: str
    klass_sorter: ConfKSorter
    klass_collector: ConfCollectorInterface
    klass_save_kv_previous: InspectorPlugin
    klass_cache_past_kv: InspectorPlugin
    klass_cache_attn: InspectorPlugin
    klass_cache_vo: InspectorPlugin
    
    size_block: Optional[int] = None
    step_per_block: Optional[int] = None

    def __post_init__(self):
        self.size_block= int(self.len_target / self.num_blocks)
        self.step_per_block=int(self.size_block / self.num_unmask_per_step)
    # end
# end


@dataclass
class DiffusionConfig_Eval:
    id_model: str
    len_target: int
    num_blocks: int
    num_unmask_per_step: int
    id_mask: int
    size_batch: int
    device: str
    klass_sorter: ConfKSorter
    klass_collector: ConfCollectorInterface

    use_chat_template: Optional[bool] = None    # set True for instruct/SFT checkpoints
    use_official_gsm8k_prompt: Optional[bool] = None    # rebuild the OpenCompass 4-shot CoT
                                                        # multiturn prompt (run with --num_fewshot 0;
                                                        # implies chat template; gsm8k only)
    truncate_at_eos: Optional[bool] = None      # DEPRECATED no-op, kept so old command lines
                                                # still parse; the instruct runners
                                                # (run_llada_instruct[_mlp]/run_dream_instruct[_mlp])
                                                # hard-enable EOS truncation instead

    '''d2cache reimplementation (run_llada_d2cache / run_dream_d2cache)'''
    d2c_k: Optional[int] = None            # masked candidates per step (paper: 32)
    d2c_sigma: Optional[float] = None      # certainty-density gaussian sigma (paper: 10.0)
    d2c_rollout_p: Optional[float] = None  # attention-rollout nucleus threshold (paper: 0.1; 0 disables)
    d2c_conf_mode: Optional[str] = None    # 'live' (their intended design) | 'frozen'
                                           # (their RELEASED code: conf never updated after prefill)
    d2c_inflate_w: Optional[int] = None    # gap inflation window (their eval default: 0)
    '''d2cache'''

    '''with mlp'''
    step_refresh_remainder: Optional[int] = None
    step_refresh_remainder_prompt: Optional[int] = None    # v2 runner: independent PROMPT
                                                           # re-forward interval (their Kp
                                                           # analog); None/0 = prompt KV is
                                                           # never refreshed after init.
                                                           # step_refresh_remainder stays the
                                                           # GENERATION-area interval (Kr)
    step_refresh_remainder_surfix: Optional[int] = None    # instruct runner: re-forward the
                                                           # SUFFIX (future, still-masked
                                                           # blocks after the current one, KV
                                                           # only) every this many steps.
                                                           # None/0 = suffix KV stays as of
                                                           # the initial canvas forward until
                                                           # its block becomes current
                                                           # (the historical behavior)
    h: Optional[int] = None
    select_only_in_h: Optional[bool] = None
    path_router: Optional[str] = None    # router bundle (.pt with .json sidecar); None -> legacy scalar MLP
    path_report: Optional[str] = None    # per-sample runner report (json)
    '''with mlp'''

    klass_save_kv_previous: Optional[InspectorPlugin] = None
    klass_cache_past_kv: Optional[InspectorPlugin] = None
    klass_cache_attn: Optional[InspectorPlugin] = None
    klass_cache_vo: Optional[InspectorPlugin] = None
    
    size_block: Optional[int] = None
    step_per_block: Optional[int] = None

    def __post_init__(self):
        self.size_block= int(self.len_target / self.num_blocks)
        self.step_per_block=int(self.size_block / self.num_unmask_per_step)
    # end
# end






@dataclass
class KVAggregateConfig:
    stamp: str
    type_aggregate: str
    len_prompt: str
    len_target: str
    num_blocks: int
    folder_output: Optional[str] = None
    type_fn: Optional[str] = None
# end


'''
config = DiffusionConfig(
    id_model='GSAI-ML/LLaDA-8B-Base',
    len_prompt=128,
    len_target=256,
    num_blocks=4,
    num_unmask_per_step=1,
    id_mask=126336,
    size_batch=1,
    device='cuda:0',
    klass_sorter=TopKSorter,
    klass_collector=TruthCollector,
    klass_save_kv_previous=SaveKVPreviousPlugin_Disabled,
    klass_cache_past_kv=CachePastKVPlugin_Enabled
)

config.size_block= int(config.len_target / config.num_blocks)
config.step_per_block=int(config.size_block / config.num_unmask_per_step)


config_aggregate = KVAggregateConfig(
    stamp='0326',
    type_aggregate='step',
    len_prompt=config.len_prompt,
    len_target=config.len_target,
    num_blocks=config.num_blocks,
    type_fn='p'
)
config_aggregate.folder_output=f'sims_kv_{config_aggregate.stamp}'

'''