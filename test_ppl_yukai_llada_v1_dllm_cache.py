import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # put this at the very top of your script


from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn.functional as F

from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from tools_llada import TopKSorter, TruthCollector, MaxCollector
from plugins_llada import\
    SaveKVPreviousPlugin_Disabled, SaveKVPreviousPlugin_Enabled,\
    CachePastKVPlugin_Disabled, CachePastKVPlugin_Enabled,\
    CacheAttnPlugin_Disabled, CacheAttnPlugin_Enabled,\
    CacheVOPlugin_Enabled, CacheVOPlugin_Disabled

from datasets import load_dataset, Dataset

from components_llada import SimpleLogitsSnapshot
from tools_llada import ConfKSorter, ConfCollectorInterface, BlockDiffusionQuotaHelper
from plugins_llada import InspectorPlugin

from dataset_llada import LIST_DATASET
from datapreprocess_llada import parse_lines_with_index, merge_subdocs, PATTEN_REG_WIKI
from dataprocess_llada import Tokenizer_wiki_simple, Collater_wiki_simple

from modeling_llada_yukai_v4 import LLaDAModelLM

from tools_debug import jprint

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


config = DiffusionConfig(
    id_model='GSAI-ML/LLaDA-8B-Base',
    len_prompt=128,
    len_target=256,
    num_blocks=1,
    num_unmask_per_step=1,
    id_mask=126336,
    size_batch=1,
    device='cuda:0',
    klass_sorter=TopKSorter,
    klass_collector=TruthCollector,
    klass_save_kv_previous=SaveKVPreviousPlugin_Disabled,
    klass_cache_past_kv=CachePastKVPlugin_Enabled,
    klass_cache_attn=CacheAttnPlugin_Disabled,
    klass_cache_vo=CacheVOPlugin_Enabled
)

config.size_block= int(config.len_target / config.num_blocks)
config.step_per_block=int(config.size_block / config.num_unmask_per_step)



'''load dataset first'''
name_dataset = LIST_DATASET[1]
ds = load_dataset(*name_dataset, split='all')
docs, _ = parse_lines_with_index(PATTEN_REG_WIKI, ds['text'])
docs = docs['subdocs']

samples = []
for doc in docs:
    lines_1 = doc['texts']
    paragraph_1 = ' '.join(lines_1)
    lines_remain, titles = merge_subdocs(doc['subdocs'])
    paragraph_remain = ' '.join(lines_remain)
    prefix = paragraph_1
    target = paragraph_remain
    samples.append({'text': paragraph_1 + ' ' + paragraph_remain})
# end

ds_origin = Dataset.from_list(samples[:100])


'''initialize constant hyper-parameters'''

'''load tokenizer'''
tokenizer = AutoTokenizer.from_pretrained(
    config.id_model,
    trust_remote_code=True
)

if tokenizer.padding_side != 'left':
    tokenizer.padding_side = 'left'
# end
assert tokenizer.pad_token_id != 126336


'''load model'''
model_kwargs = {}
model = LLaDAModelLM.from_pretrained(
    config.id_model,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    **model_kwargs
)

model = model.eval().to(config.device)


'''start to handle dataset based on hyper-parameter'''
len_max = config.len_prompt + config.len_target
ds = ds_origin\
    .filter(lambda x: x["text"] is not None and len(x["text"].strip()) > 0)\
    .map(Tokenizer_wiki_simple(tokenizer, len_max), remove_columns=ds_origin.column_names)\
    .filter(lambda x: x["length"] >= len_max)\
    .sort("length")
# end

'''prepare dataloader'''
loader = DataLoader(
    ds,
    batch_size=config.size_batch,
    shuffle=False,                 # keep sorted order
    collate_fn=Collater_wiki_simple(len_max, config.len_prompt, config.len_target, config.id_mask),
    drop_last=False
)

@ torch.no_grad()
def run_model_semi(model, x, y, config_diffusion, *args, **kwargs):

    '''declare required variables'''
    num_blocks = config_diffusion.num_blocks
    step_per_block = config_diffusion.step_per_block
    size_block = config_diffusion.size_block
    id_mask = config_diffusion.id_mask
    len_prompt = config_diffusion.len_prompt
    sorter = config_diffusion.klass_sorter()
    collector = config_diffusion.klass_collector()
    
    p_finalized = torch.zeros(x.shape, dtype=torch.float64, device=x.device)
    position_start = 0

    for id_block in range(num_blocks):
        position_end = position_start + len_prompt + (id_block+1) * size_block
        mask_mask_blk = x[:,position_start:position_end] == id_mask
        shape_target = (x.shape[0], position_end, -1)
        
        idx_denoising = torch.arange(position_start, position_end, dtype=torch.long).to(x.device)
        quota_helper = BlockDiffusionQuotaHelper(mask_mask_blk, size_block)

        for step in range(step_per_block):
            x_denoising,  y_denoising= x[:, idx_denoising], y[:, idx_denoising]
            logits = model(x_denoising, idx_current=idx_denoising, shape_target=shape_target).logits
            snapshot = SimpleLogitsSnapshot(logits, x_denoising, y_denoising, id_mask)
            conf_snapshot = snapshot.transform_logits(collector)

            idx_sorted_by_conf = sorter.argsort(conf_snapshot, snapshot)
            num_unmask = quota_helper.get_quota(step)
            idx_transform = idx_sorted_by_conf[:, :num_unmask]

            snapshot.materialize_by_idx_(idx_transform, conf_snapshot)
            snapshot.update_this(1, idx_transform, y=x).update_this(1, idx_transform, p_finalized=p_finalized)
        # end for step
    # end for block

    return p_finalized[:, len_prompt:]
# end

import json
from tqdm import tqdm
from tools_llada import PPLCalculator

calculator_ppl = PPLCalculator()
model\
    .fill_plugin(config.klass_cache_past_kv)\
    .fill_plugin(config.klass_save_kv_previous)\
    .fill_plugin(config.klass_cache_attn)\
    .fill_plugin(config.klass_cache_vo)

config.klass_cache_vo\
    .set_prompt_length(config.len_prompt)\
    .set_response_length(config.len_target)\
    .set_update_budget_p(0.25)

instance_cache_vo = config.klass_cache_vo()
'''start the evaluation process'''
for id_batch, batch in enumerate(tqdm(loader)):

    instance_cache_vo.clear_layer_past_and_output(model)

    conf = run_model_semi(
        model,
        batch['ids_prompt_masked_full'].to(config.device),
        batch['ids_target_masked_full'].to(config.device),
        config
    )

    print(calculator_ppl.cal(conf))
# end for


