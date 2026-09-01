#################################################
# Oracle collection for LLaDA (full denoising, no cache).
#
# Input:  a benchmark mockup CSV from save_benchmark_mockup.py
# Output: per-sample folders under <folder_output>/<id_row>/ with, per block
#         (row index of every tensor = step within the block):
#           margin_<s>_<e>.pt   (T, size_block)           fp32, p(top1)-p(top2), -inf at non-masked
#           conf_<s>_<e>.pt     (T, size_block)           fp32, p(top1), -inf at non-masked
#           entropy_<s>_<e>.pt  (T, size_block)           fp32, full-vocab entropy, -inf at non-masked
#           attn_<s>_<e>.pt     (T, num_layers, 1, size_block)  attention rows of the
#                                unmasked token, all layers, head-averaged, block-local K
#           unmask_<s>_<e>.pt   (T, 1)                    GLOBAL unmask position per step
#           token_<s>_<e>.pt    (T, 1)                    token id written at the unmask position
#           x0_<s>_<e>.pt       (T, size_block)           long, argmax token id per candidate per step
#           .pos_root           len_prompt (text)
#           generated.json      text, has_done, result: pass/fail/unknown
#         consumable by train_mlp.py (superset of the legacy stats layout).
#
# NOTES:
#   - GENERATION-mode oracle (one-pass): x is materialized with the model's own
#     predictions, matching deployment; metrics are recorded on the same run.
#   - unmask + token together checkpoint the full trajectory, so a future replay
#     can re-verify or collect new metrics on exactly these trajectories.
#   - 'result' is computed by a per-task checker (gsm8k: final-number match after
#     stop-word truncation); tasks without a checker get 'unknown'. Both passed
#     and failed samples are kept -- filter at training time.
#   - num_unmask_per_step is asserted to 1: the oracle records one position per step.
#
# Usage:
#   python run_collect_metrics_llada.py --path_mockup benchmark_mockup/mockup_gsm8k_5shot_p10.csv \
#       --folder_output stats_oracle_gsm8k --len_target 256 --num_blocks 4
#################################################

import argparse
import json
import os
import re

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
    parser.add_argument('--use_chat_template', action='store_true',
                        help='wrap prompts with the chat template (instruct/SFT checkpoints)')
    return parser.parse_args()
# end


def check_result_gsm8k(text_checked, doc):
    match_gold = re.search(r'####\s*(-?[0-9\.,]+)', str(doc.get('answer', '')))
    if match_gold is None:
        return 'unknown'
    # end
    gold = match_gold.group(1).replace(',', '').rstrip('.')

    nums = re.findall(r'-?[0-9][0-9,]*\.?[0-9]*', text_checked.replace('$', ''))
    if not nums:
        return 'fail'
    # end
    pred = nums[-1].replace(',', '').rstrip('.')

    return 'pass' if pred == gold else 'fail'
# end


def check_result_ifeval(text_checked, doc):
    # reuse lm_eval's own rule checkers; doc carries key / prompt /
    # instruction_id_list / kwargs through the mockup CSV round-trip
    try:
        from lm_eval.tasks.ifeval.utils import process_results
        scores = process_results(doc, [text_checked])
    except Exception as error:
        jprint(f'ifeval checker unavailable or failed: {error}')
        return 'unknown'
    # end

    # sample-level flag = strict prompt-level accuracy (all instructions followed);
    # the full score dict is kept as detail for instruction-level ablations
    result = 'pass' if scores.get('prompt_level_strict_acc') else 'fail'
    return result, scores
# end


# a checker returns 'pass'/'fail'/'unknown', optionally as (result, detail_dict)
MAP_TASK_CHECKER = {
    'gsm8k': check_result_gsm8k,
    'ifeval': check_result_ifeval,
}


NAMES_STATS = ('margin', 'conf', 'entropy', 'attn', 'unmask', 'token', 'x0')


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

            stats = Stats(block_start, position_end, names=NAMES_STATS)

            for step in range(self.step_per_block):
                x_denoising, y_denoising = x[:, idx_denoising], x[:, idx_denoising]
                logits = self.model(x_denoising, idx_current=idx_denoising, shape_target=shape_target).logits

                # metrics on the current block only (window starts at 0 -> global == row positions)
                logits_blk = logits[:, idx_block].float()
                mask_blk = (x[:, block_start:position_end] == id_mask).squeeze(0)
                sentinel = torch.tensor(neg_inf, device=x.device)

                logp_blk = F.log_softmax(logits_blk, dim=-1)
                p_blk = logp_blk.exp()

                # margin = p(top1) - p(top2), fresh, pre-decision
                top2 = p_blk.topk(2, dim=-1).values.squeeze(0)    # (size_block, 2)
                margin_blk = torch.where(mask_blk, top2[:, 0] - top2[:, 1], sentinel)
                stats.margin.add(block_start + step, margin_blk.cpu())

                # full-vocab predictive entropy per candidate
                entropy_blk = -(p_blk * logp_blk).sum(dim=-1).squeeze(0)    # (size_block,)
                entropy_blk = torch.where(mask_blk, entropy_blk, sentinel)
                stats.entropy.add(block_start + step, entropy_blk.cpu())

                # argmax token per candidate (for offline argmax-stability features)
                stats.x0.add(block_start + step, logits_blk.argmax(dim=-1).squeeze(0).cpu())

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

                # token id written at the unmask position (unmask + token = full trajectory)
                token_written = torch.gather(snapshot.x0, 1, idx_transform).squeeze(0)
                stats.token.add(block_start + step, token_written.cpu())

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
            text_prompt = row['prompt']
            if args.use_chat_template:
                # same convention as Preprocessor_Until: the whole benchmark
                # context becomes one user turn; the template string carries its
                # own special tokens, so add_special_tokens stays False below
                text_prompt = self.tokenizer.apply_chat_template(
                    [{'role': 'user', 'content': text_prompt}],
                    add_generation_prompt=True,
                    tokenize=False,
                )
            # end

            ids_prompt = self.tokenizer(text_prompt, add_special_tokens=False)['input_ids']
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

            # benchmark check on the cleaned, stop-word-truncated text; cut at the
            # first EOS in the RAW ids first, so stray tokens in the EOS tail
            # cannot corrupt last-number answer extraction (instruct models
            # EOS-fill the block tail)
            ids_generated = x[0, len_prompt:position_end]
            id_eos = self.tokenizer.eos_token_id
            if id_eos is not None:
                hits_eos = (ids_generated == id_eos).nonzero()
                if hits_eos.numel() > 0:
                    ids_generated = ids_generated[:hits_eos[0, 0]]
                # end
            # end
            text_checked = self.tokenizer.decode(ids_generated, skip_special_tokens=True)
            for word_stop in row['until']:
                if word_stop in text_checked:
                    text_checked = text_checked.split(word_stop)[0]
                # end
            # end
            checker = MAP_TASK_CHECKER.get(row['task_name'])
            result, result_detail = 'unknown', None
            if checker is not None:
                result = checker(text_checked, row['doc'])
                if isinstance(result, tuple):
                    result, result_detail = result
                # end
            # end

            record = {
                'id_request': row['id_request'],
                'doc_id': row['doc_id'],
                'has_done': has_done,
                'result': result,
                'text_generated': text_generated,
            }
            if result_detail is not None:
                record['result_detail'] = result_detail
            # end

            with open(os.path.join(folder_stats, 'generated.json'), 'w') as file:
                json.dump(record, file)
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
