#################################################
# Oracle collection for LLaDA (full denoising, no cache).
#
# Input:  a benchmark mockup CSV from save_benchmark_mockup.py
# Output: per-sample folders under <folder_output>/<id_row>/ with
#           margin_<s>_<e>.pt   (T, size_block)          fp32, -inf at non-masked
#           conf_<s>_<e>.pt     (T, size_block)           fp32, -inf at non-masked
#           attn_<s>_<e>.pt     (T, num_layers, 1, size_block)  attention rows of the
#                                unmasked token, all layers, block-local K
#           unmask_<s>_<e>.pt   (T, 1)                    GLOBAL unmask position per step
#           .pos_root           len_prompt (text)
#           generated.json      generated text + has_done (oracle quality check)
#         consumable by train_mlp.py (same layout as the legacy stats folders).
#
# NOTES:
#   - GENERATION-mode oracle: x is materialized with the model's own predictions
#     (x0), matching deployment. The legacy script teacher-forced with ground
#     truth y; the gold answer is still available in the CSV 'doc' column if a
#     truth-forced variant is wanted later.
#   - num_unmask_per_step is asserted to 1: the oracle records one position per step.
#
# Usage:
#   python run_collect_metrics_llada.py --path_mockup benchmark_mockup/mockup_gsm8k_5shot_p10.csv \
#       --folder_output stats_oracle_gsm8k --len_target 256 --num_blocks 4
#################################################

import argparse
import json
import os

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from tqdm import tqdm

from components_llada import SimpleLogitsSnapshot, Stats
from tools_llada import TopKSorter, MaxCollector, BlockDiffusionQuotaHelper
from plugins_llada import SaveKVPreviousPlugin_Disabled,\
                            CachePastKVPlugin_Disabled,\
                            CacheAttnPlugin_Enabled,\
                            CacheVOPlugin_Disabled
from modeling_llada_yukai_06 import LLaDAModelLM
from save_benchmark_mockup import load_benchmark_mockup
from constants_llada import DTYPE_EVAL, ID_MASK
from tools_debug import jprint


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_mockup', type=str, required=True)
    parser.add_argument('--folder_output', type=str, required=True)
    parser.add_argument('--id_model', type=str, default='GSAI-ML/LLaDA-8B-Base')
    parser.add_argument('--len_target', type=int, default=256)
    parser.add_argument('--num_blocks', type=int, default=4)
    parser.add_argument('--id_mask', type=int, default=ID_MASK)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--limit', type=int, default=None, help='cap on mockup rows')
    parser.add_argument('--seed', type=int, default=233)
    return parser.parse_args()
# end


class CollectMetricsRunner:

    def __init__(self, model, tokenizer, plugin_cache_attn, args):
        self.model = model
        self.tokenizer = tokenizer
        self.plugin_cache_attn = plugin_cache_attn
        self.args = args

        self.size_block = args.len_target // args.num_blocks
        self.step_per_block = self.size_block    # num_unmask_per_step == 1 by design
        self.sorter = TopKSorter()
        self.collector = MaxCollector()
    # end

    @torch.no_grad()
    def collect_one(self, x, len_prompt, folder_stats):
        args = self.args
        id_mask = args.id_mask
        size_block = self.size_block

        neg_inf = torch.finfo(torch.float32).min
        position_start = 0

        for id_block in range(args.num_blocks):
            position_end = position_start + len_prompt + (id_block + 1) * size_block
            block_start = position_end - size_block
            mask_mask_blk = x[:, position_start:position_end] == id_mask

            idx_denoising = torch.arange(position_start, position_end, dtype=torch.long, device=x.device)
            idx_block = torch.arange(block_start, position_end, dtype=torch.long, device=x.device)
            idx_block_2d = idx_block.unsqueeze(0)
            quota_helper = BlockDiffusionQuotaHelper(mask_mask_blk, self.step_per_block)
            shape_target = (x.shape[0], position_end, -1)

            stats = Stats(block_start, position_end)

            for step in range(self.step_per_block):
                x_denoising, y_denoising = x[:, idx_denoising], x[:, idx_denoising]
                logits = self.model(x_denoising, idx_current=idx_denoising, shape_target=shape_target).logits

                # metrics on the current block only (window starts at 0 -> global == row positions)
                logits_blk = logits[:, idx_block].float()
                mask_blk = (x[:, block_start:position_end] == id_mask).squeeze(0)

                # margin = p(top1) - p(top2), fresh, pre-decision
                p_blk = F.softmax(logits_blk, dim=-1)
                top2 = p_blk.topk(2, dim=-1).values.squeeze(0)    # (size_block, 2)
                margin_blk = torch.where(mask_blk, top2[:, 0] - top2[:, 1], torch.tensor(neg_inf, device=x.device))
                stats.margin.add(block_start + step, margin_blk.cpu())

                # confidence via the standard snapshot path
                snapshot = SimpleLogitsSnapshot(x_denoising, y_denoising, id_mask)
                snapshot.update_x0_(idx_block_2d, logits_blk)
                conf_snapshot = snapshot.transform_logits(self.collector, logits_blk, idx_transform=idx_block_2d)
                stats.conf.add(block_start + step, conf_snapshot.squeeze(0)[block_start:position_end].cpu())

                # unmask decision (model's own prediction, matching deployment)
                idx_sorted_by_conf = self.sorter.argsort(conf_snapshot, snapshot)
                num_unmask = quota_helper.get_quota(step)
                idx_transform = idx_sorted_by_conf[:, :num_unmask]
                snapshot.materialize_by_idx_(idx_transform, conf_snapshot)

                # attention rows of the just-unmasked token: all layers, block-local coordinates
                attn_all = self.plugin_cache_attn.collect_attn_from_all_blocks(self.model)   # (num_layers, size_block, size_block)
                idx_local = idx_transform.squeeze(0) - block_start
                assert bool((idx_local >= 0).all() and (idx_local < size_block).all()),\
                    f'unmask position outside current block: {idx_transform.tolist()}'
                stats.attn.add(block_start + step, attn_all[:, idx_local, :].cpu())

                stats.unmask.add(block_start + step, idx_transform.squeeze(0).cpu())

                snapshot.update_this(1, idx_transform, x0=x)
            # end for step

            os.makedirs(folder_stats, exist_ok=True)
            stats.stack_and_save_all(folder_stats)
        # end for block

        with open(os.path.join(folder_stats, '.pos_root'), 'w+') as file:
            file.write(str(len_prompt))
        # end

        return position_end
    # end

    def run(self, rows):
        args = self.args

        for id_row, row in enumerate(tqdm(rows)):
            ids_prompt = self.tokenizer(row['prompt'], add_special_tokens=False)['input_ids']
            len_prompt = len(ids_prompt)

            x = torch.tensor(ids_prompt + [args.id_mask] * args.len_target, dtype=torch.long).view(1, -1)
            x = x.to(args.device)

            # attn plugin block arithmetic depends on per-sample prompt length
            CacheAttnPlugin_Enabled.set_len_prompt(len_prompt).set_size_block(self.size_block)
            self.plugin_cache_attn.clear(self.model)

            folder_stats = os.path.join(args.folder_output, str(id_row))
            position_end = self.collect_one(x, len_prompt, folder_stats)

            text_generated = self.tokenizer.batch_decode(x[:, len_prompt:position_end], skip_special_tokens=False)[0]
            has_done = any(word_stop in text_generated for word_stop in row['until'])
            with open(os.path.join(folder_stats, 'generated.json'), 'w') as file:
                json.dump({
                    'id_request': row['id_request'],
                    'doc_id': row['doc_id'],
                    'has_done': has_done,
                    'text_generated': text_generated,
                }, file)
            # end
        # end for
    # end
# end


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    assert args.len_target % args.num_blocks == 0

    rows = load_benchmark_mockup(args.path_mockup)
    if args.limit is not None:
        rows = rows[:args.limit]
    # end
    jprint(f'collecting oracle for {len(rows)} prompts from {args.path_mockup}')

    tokenizer = AutoTokenizer.from_pretrained(args.id_model, trust_remote_code=True)
    if tokenizer.padding_side != 'left':
        tokenizer.padding_side = 'left'
    # end
    assert tokenizer.pad_token_id != args.id_mask

    model = LLaDAModelLM.from_pretrained(
        args.id_model,
        trust_remote_code=True,
        torch_dtype=DTYPE_EVAL,
    ).eval().to(args.device)

    model\
        .fill_plugin(CachePastKVPlugin_Disabled)\
        .fill_plugin(SaveKVPreviousPlugin_Disabled)\
        .fill_plugin(CacheAttnPlugin_Enabled)\
        .fill_plugin(CacheVOPlugin_Disabled)

    plugin_cache_attn = CacheAttnPlugin_Enabled()

    runner = CollectMetricsRunner(model, tokenizer, plugin_cache_attn, args)
    runner.run(rows)
# end


if __name__ == '__main__':
    main()
# end
