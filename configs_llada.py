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

    '''with mlp'''
    step_refresh_remainder: Optional[int] = None
    h: Optional[int] = None
    select_only_in_h: Optional[bool] = None
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