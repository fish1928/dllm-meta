import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # put this at the very top of your script

import json

from datasets import load_dataset, Dataset
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn.functional as F

from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from tools_llada import TopKSorter, TruthCollector, MaxCollector
from plugins_llada_06 import \
    SaveKVPreviousPlugin_Disabled, SaveKVPreviousPlugin_Enabled,\
    CachePastKVPlugin_Disabled, CachePastKVPlugin_Enabled,\
    CacheAttnPlugin_Disabled, CacheAttnPlugin_Enabled,\
    CacheVOPlugin_Disabled, CacheVOPlugin_Enabled

from configs_llada import DiffusionConfig

from components_llada import SimpleLogitsSnapshot
from tools_llada import ConfKSorter, ConfCollectorInterface, BlockDiffusionQuotaHelper
from plugins_llada_06 import InspectorPlugin

from dataset_llada import LIST_DATASET
from datapreprocess_llada import parse_lines_with_index, merge_subdocs, PATTEN_REG_WIKI
from dataprocess_llada import Collater_sample

from modeling_llada_yukai_06 import LLaDAModelLM

from tools_debug import jprint


config = DiffusionConfig(
    id_model='GSAI-ML/LLaDA-8B-Base',
    len_prompt=-1,
    len_target=64*4,
    num_blocks=4,
    num_unmask_per_step=1,
    id_mask=126336,
    size_batch=1,
    device='cuda:0',
    klass_sorter=TopKSorter,
    klass_collector=MaxCollector,
    klass_save_kv_previous=SaveKVPreviousPlugin_Disabled,
    klass_cache_past_kv=CachePastKVPlugin_Disabled,
    klass_cache_attn=CacheAttnPlugin_Enabled,
    klass_cache_vo=CacheVOPlugin_Disabled
)

import json
import os
from datasets import load_dataset, Dataset
'''load dataset first'''



folder_samples = 'samples_ifeval'
samples = []
for filename_sample in [f for f in os.listdir(folder_samples) if f[0] != '.']:
    path_sample = os.path.join(folder_samples, filename_sample)
    with open(path_sample, 'r') as file:
        sample = json.load(file)
        samples.append(sample)
    # end
# end

ds_origin = Dataset.from_list(samples[:])


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



'''prepare dataloader'''
loader = DataLoader(
    ds_origin,
    batch_size=config.size_batch,
    shuffle=False,                 # keep sorted order
    collate_fn=Collater_sample(config.id_mask),
    drop_last=False
)

class IndexedElementList:
    def __init__(self, idx_start, idx_end, name=None):
        self.idx_start = idx_start
        self.idx_end = idx_end
        self.indexed_elements = [None] * (idx_end - idx_start)
        self.name = name
    # end

    def add(self, idx_relative, element):
        self.indexed_elements[idx_relative - self.idx_start] = element
    # end

    def get(self, idx_relative):
        return self.indexed_elements[idx_relative - self.idx_start]
    # end

    def has_empty(self):
        for indexed_element in self.indexed_elements:
            if indexed_element is None:
                return True
            # end
        # end

        return False
    # end

    def stack_and_save(self, path_to_save):
        stacked_elements = torch.stack(self.indexed_elements)
        torch.save(stacked_elements, os.path.join(path_to_save, f'{self.name}_{self.idx_start}_{self.idx_end}.pt'))
    # end
# end


class Stats:
    _STATS = ('margin', 'conf', 'attn', 'unmask')

    def __init__(self, idx_start, idx_end, margin_idx_a=0, margin_idx_b=1):
        for name in self._STATS:
            setattr(self, name, IndexedElementList(idx_start, idx_end, name=name))
        # end

        self.margin.idx_a = margin_idx_a
        self.margin.idx_b = margin_idx_b
    # end

    def stack_and_save_all(self, path_to_save):
        for name_stat in self._STATS:
            statlist = getattr(self, name_stat)
            if not statlist.has_empty():
                statlist.stack_and_save(path_to_save)
            # end
        # end
    # end
# end


class RunModelAndCollectStats:
    @ torch.no_grad()
    def run_model_semi(self, model, x, y, config_diffusion, *args, **kwargs):

        '''declare required variables'''
        num_blocks = config_diffusion.num_blocks
        step_per_block = config_diffusion.step_per_block
        size_block = config_diffusion.size_block
        id_mask = config_diffusion.id_mask
        sorter = config_diffusion.klass_sorter()
        collector = config_diffusion.klass_collector()

        plugin_cache_attn = kwargs['plugin_cache_attn']
        id_batch =kwargs['id_batch']
        len_prompt = kwargs['len_prompt']
        
        p_finalized = torch.zeros(x.shape, dtype=torch.float64, device=x.device)
        position_start = 0

        for id_block in range(num_blocks):
            position_end = position_start + len_prompt + (id_block+1) * size_block
            mask_mask_blk = x[:,position_start:position_end] == id_mask

            idx_denoising = torch.arange(position_start, position_end, dtype=torch.long).to(x.device)
            quota_helper = BlockDiffusionQuotaHelper(mask_mask_blk, size_block)
            stats = Stats(position_end-size_block, position_end)

            for step in range(step_per_block):
                step_global = id_block * step_per_block + step + len_prompt
                slice_pos_current = slice(position_end-size_block, position_end)

                x_denoising,  y_denoising= x[:, idx_denoising], y[:, idx_denoising]
                logits = model(x_denoising, idx_current=idx_denoising).logits

                snapshot = SimpleLogitsSnapshot(logits, x_denoising, y_denoising, id_mask)

                margin_p = snapshot.get_margin_p(stats.margin.idx_a, stats.margin.idx_b)[slice_pos_current]
                stats.margin.add(step_global, margin_p)

                conf_snapshot = snapshot.transform_logits(collector)    #(B,L)
                stats.conf.add(step_global, conf_snapshot.squeeze(0)[slice_pos_current])

                idx_sorted_by_conf = sorter.argsort(conf_snapshot, snapshot)
                num_unmask = quota_helper.get_quota(step)
                idx_transform = idx_sorted_by_conf[:, :num_unmask]
                snapshot.materialize_by_idx_(idx_transform, conf_snapshot)

                attn_all = plugin_cache_attn.collect_attn_from_all_blocks(model) #(Ls, Blk, K), Blk=64, Ls=32
                attn = attn_all[:, idx_transform.squeeze(0), slice_pos_current] #(Ls, 1, K)

                stats.attn.add(step_global, attn)
                stats.unmask.add(step_global, idx_transform.squeeze(0))

                snapshot.update_this(1, idx_transform, y=x).update_this(1, idx_transform, p_finalized=p_finalized)

            # end for step
            folder_stats = f'stats_ifeval_256/{id_batch}/'
            os.makedirs(folder_stats,exist_ok=True)
            stats.stack_and_save_all(folder_stats)
            with open(os.path.join(folder_stats, '.pos_root'), 'w+') as file:
                file.write(str(len_prompt))
            # end
        # end for block

        return p_finalized[:, len_prompt:]
    # end

    def run(self):
        import json
        from tqdm import tqdm
        from tools_llada import PPLCalculator, RefreshIdxHelper

        model\
            .fill_plugin(config.klass_cache_past_kv)\
            .fill_plugin(config.klass_save_kv_previous)\
            .fill_plugin(config.klass_cache_attn)\
            .fill_plugin(config.klass_cache_vo)

        plugin_cache_attn = config.klass_cache_attn()


        '''start the evaluation process'''
        for id_batch, sample in enumerate(tqdm(loader)):
            plugin_cache_attn.clear(model)

            conf = RunModelAndCollectStats().run_model_semi(
                model,
                sample['ids_prompt_masked_full'].to(config.device),
                sample['ids_target_masked_full'].to(config.device),
                config,
                plugin_cache_attn=plugin_cache_attn,
                id_batch=id_batch,
                len_prompt=sample['len_prompt']
            )

            # if id_batch >=5:
            #     break
            # # end
        # end for
    # end
# end


import json
from tqdm import tqdm
from tools_llada import PPLCalculator, RefreshIdxHelper

model\
    .fill_plugin(config.klass_cache_past_kv)\
    .fill_plugin(config.klass_save_kv_previous)\
    .fill_plugin(config.klass_cache_attn)\
    .fill_plugin(config.klass_cache_vo)

plugin_cache_attn = config.klass_cache_attn()


'''start the evaluation process'''
for id_sample, sample in enumerate(tqdm(loader)):
    plugin_cache_attn.clear(model)

    conf = RunModelAndCollectStats().run_model_semi(
        model,
        sample['ids_prompt_masked_full'].to(config.device),
        sample['ids_target_masked_full'].to(config.device),
        config,
        plugin_cache_attn=plugin_cache_attn,
        id_batch=id_sample,
        len_prompt=sample['len_prompt']
    )

    # if id_batch >=5:
    #     break
    # # end
# end for