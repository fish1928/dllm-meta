#################################################
# Shared scaffolding for the four oracle-trajectory collectors (stage 2):
#   run_collect_metrics_llada_base      (growing window, one-block default)
#   run_collect_metrics_llada_instruct  (FULL-CANVAS block diffusion, chat prompt)
#   run_collect_metrics_dream_base      (growing window, dream shift)
#   run_collect_metrics_dream_instruct  (growing window, dream shift, chat prompt)
#
# Everything thread-INVARIANT lives here: CLI builder (len_target/num_blocks as
# common parameters), model/tokenizer/plugin setup, the per-sample run loop
# (prompt prep via Preprocessor_Until for bit-parity with the eval harness,
# EOS/stop truncation via tools_llada helpers, benchmark result checking), and
# the checker registry for the extended benchmark set. Each thread file
# subclasses OracleCollectorBase and implements collect_one() only -- the
# decoding loop matching its baseline runner.
#
# Output layout per sample (consumed by router training) is unchanged from
# run_collect_metrics_llada.py; generated.json additionally records task_name
# (merged mockup CSVs mix subtasks, e.g. bbh).
#
# Result checkers ('result': pass/fail/unknown; filter at training time):
#   gsm8k          last-number match after truncation
#   ifeval         lm_eval rule checkers (strict prompt-level)
#   minerva_math*  lm_eval minerva utils (sympy equivalence) when importable
#   bbh*           "the answer is X" extraction vs doc target
#   mbpp/humaneval 'unknown' (real check = code execution; eval-stage concern)
#   truthfulqa_gen 'unknown' (bleu/rouge vs references; no binary pass/fail)
#   followbench    'unknown' (LLM-as-judge; eval-stage concern)
#################################################

import argparse
import json
import os
import re

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from tqdm import tqdm

from components_llada import Stats
from tools_llada import TopKSorter, MaxCollector, collect_ids_stop, truncate_text_at_stop
from plugins_llada import SaveKVPreviousPlugin_Disabled,\
                            CachePastKVPlugin_Disabled,\
                            CacheAttnPlugin_Enabled,\
                            CacheVOPlugin_Disabled
from dataprocess_llada import Preprocessor_Until
from save_benchmark_mockup import load_benchmark_mockup
from constants_llada import DTYPE_EVAL
from tools_debug import jprint


NAMES_STATS = ('margin', 'conf', 'entropy', 'attn', 'unmask', 'token', 'x0')


def build_parser(id_model, id_mask, len_target=256, num_blocks=1):
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_mockup', type=str, required=True)
    parser.add_argument('--folder_output', type=str, required=True)
    parser.add_argument('--id_model', type=str, default=id_model)
    parser.add_argument('--len_target', type=int, default=len_target)
    parser.add_argument('--num_blocks', type=int, default=num_blocks)
    parser.add_argument('--id_mask', type=int, default=id_mask)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--limit', type=int, default=None, help='cap on mockup rows')
    parser.add_argument('--filter_task', type=str, default=None,
                        help='keep only this subtask from a merged mockup CSV (e.g. one bbh subtask)')
    parser.add_argument('--seed', type=int, default=233)
    return parser
# end


'''---------------- result checkers ----------------'''


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


def check_result_minerva_math(text_checked, doc):
    # lm_eval's minerva pipeline: last-boxed extraction + sympy equivalence
    try:
        from lm_eval.tasks.minerva_math.utils import process_results
        scores = process_results(doc, [text_checked])
    except Exception as error:
        jprint(f'minerva_math checker unavailable or failed: {error}')
        return 'unknown'
    # end

    return ('pass' if scores.get('exact_match') else 'fail'), scores
# end


def check_result_bbh(text_checked, doc):
    # bbh cot convention: the model states "the answer is X."; targets are short
    # literals like "(A)", "True", "valid", "6"
    target = doc.get('target')
    if target is None:
        return 'unknown'
    # end

    match_pred = re.search(r'the answer is\s*(.+?)(?:\.|$)', text_checked, re.IGNORECASE)
    if match_pred is None:
        return 'fail'
    # end

    def normalize(text):
        return text.strip().strip('.').strip('()').strip().casefold()
    # end

    return 'pass' if normalize(match_pred.group(1)) == normalize(str(target)) else 'fail'
# end


# exact task name -> checker; group subtasks resolve by prefix below
MAP_TASK_CHECKER = {
    'gsm8k': check_result_gsm8k,
    'ifeval': check_result_ifeval,
}

PREFIXES_TASK_CHECKER = (
    ('minerva_math', check_result_minerva_math),
    ('bbh', check_result_bbh),
)


def resolve_checker(task_name):
    if task_name in MAP_TASK_CHECKER:
        return MAP_TASK_CHECKER[task_name]
    # end
    for prefix, checker in PREFIXES_TASK_CHECKER:
        if task_name.startswith(prefix):
            return checker
        # end
    # end
    return None    # mbpp / humaneval / truthfulqa_gen / followbench -> 'unknown'
# end


'''---------------- collector base ----------------'''


class OracleCollectorBase:

    use_chat_template = False    # instruct threads override to True

    def __init__(self, args, klass_model):
        self.args = args

        assert args.len_target % args.num_blocks == 0
        self.size_block = args.len_target // args.num_blocks
        self.step_per_block = self.size_block    # num_unmask_per_step == 1 by design
        self.sorter = TopKSorter()
        self.collector = MaxCollector()

        self.tokenizer = AutoTokenizer.from_pretrained(args.id_model, trust_remote_code=True)
        if self.tokenizer.padding_side != 'left':
            self.tokenizer.padding_side = 'left'
        # end
        assert self.tokenizer.pad_token_id != args.id_mask

        self.model = klass_model.from_pretrained(
            args.id_model,
            trust_remote_code=True,
            torch_dtype=DTYPE_EVAL,
        ).eval().to(args.device)

        self.model\
            .fill_plugin(CachePastKVPlugin_Disabled)\
            .fill_plugin(SaveKVPreviousPlugin_Disabled)\
            .fill_plugin(CacheAttnPlugin_Enabled)\
            .fill_plugin(CacheVOPlugin_Disabled)

        self.plugin_cache_attn = CacheAttnPlugin_Enabled()

        # bit-parity with the eval harness prompt path (run_benchmark_main)
        self.preprocessor = Preprocessor_Until(
            self.tokenizer,
            use_chat_template=self.use_chat_template,
            use_official_gsm8k_prompt=bool(getattr(args, 'use_official_gsm8k_prompt', False)),
        )

        self.ids_stop = collect_ids_stop(self.tokenizer)
    # end

    def collect_one(self, x, len_prompt, folder_stats):
        raise NotImplementedError('thread collector must implement collect_one()')
    # end

    def run(self, rows):
        args = self.args

        for id_row, row in enumerate(tqdm(rows)):
            processed = self.preprocessor({'prompt': row['prompt'], 'until': row['until']})
            ids_prompt = processed['ids_prompt']
            len_prompt = len(ids_prompt)

            x = torch.tensor(ids_prompt + [args.id_mask] * args.len_target, dtype=torch.long).view(1, -1)
            x = x.to(args.device)

            # attn plugin block arithmetic depends on per-sample prompt length
            CacheAttnPlugin_Enabled.set_len_prompt(len_prompt).set_size_block(self.size_block)
            self.plugin_cache_attn.clear(self.model)

            folder_stats = os.path.join(args.folder_output, str(id_row))
            position_end = self.collect_one(x, len_prompt, folder_stats)

            text_generated = self.tokenizer.batch_decode(x[:, len_prompt:position_end], skip_special_tokens=False)[0]

            # benchmark check on the cleaned text: ids-level cut at the first
            # stop id (eos + chat terminators; instruct models EOS-fill the
            # tail), then stop-word truncation
            text_checked, has_done = truncate_text_at_stop(
                self.tokenizer, x[0, len_prompt:position_end], self.ids_stop, row['until'])

            checker = resolve_checker(row['task_name'])
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
                'task_name': row['task_name'],
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


def main_collect(klass_collector, klass_model, parser):
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    rows = load_benchmark_mockup(args.path_mockup, filter_task=args.filter_task)
    if args.limit is not None:
        rows = rows[:args.limit]
    # end
    jprint(f'collecting oracle for {len(rows)} prompts from {args.path_mockup} '
           f'(len_target={args.len_target}, num_blocks={args.num_blocks})')

    collector = klass_collector(args, klass_model)
    collector.run(rows)
# end
