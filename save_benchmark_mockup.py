#################################################
# Benchmark mockup collector.
#
# Registers a fake lm_eval model ("mockup") whose generate_until does NOT run
# any model: lm_eval builds the exact benchmark requests (few-shot context,
# templates, stop strings), and we dump a seeded-free deterministic TAIL subset
# of them to CSV. That CSV is the input pool for oracle generation
# (full-denoising runs with the four baseline runners).
#
# Multi-task aware: one lm_eval invocation delivers ALL tasks' requests in a
# single generate_until call, and group tasks (bbh ~27 subtasks, minerva_math
# 7 subtasks) arrive under their subtask names. Requests are grouped by task
# and each task gets its own tail cut + CSV + meta sidecar. Pass merge=<name>
# to write one combined CSV instead (per-task tails still respected; rows keep
# their task_name column) -- recommended for bbh.
#
# Supported benchmarks (the extended set):
#   task            fewshot  notes
#   gsm8k             5      base threads; instruct threads re-prompt at runtime
#   minerva_math      4      group -> 7 subtask CSVs
#   bbh               3      group -> ~27 subtasks; use merge=bbh
#   mbpp              3      code: HF_ALLOW_CODE_EVAL gate at task load (set below)
#   humaneval         0      code: same gate
#   truthfulqa_gen    0
#   ifeval            0
#   followbench       0      NOT in stock lm_eval; needs a custom task yaml
#                            (--include_path); the collector itself is generic
#                            and handles it once the task resolves
#
# Usage (see run_save_mockups.bash for the full driver):
#   python save_benchmark_mockup.py --tasks gsm8k --model mockup --num_fewshot 5 \
#       --model_args percent=0.1,folder_output=benchmark_mockup,tag=5shot \
#       --predict_only --output_path benchmark_mockup/lm_eval_logs
#   (--predict_only skips metric computation on the empty outputs; without it,
#    code benchmarks additionally need --confirm_run_unsafe_code to score the
#    empty strings, and every printed metric is meaningless either way)
#
# Output: <folder_output>/mockup_<task>_<tag>_p<percent>.csv with columns
#   id_request, task_name, doc_id, prompt, until (json), doc (json)
# plus a .meta.json sidecar per CSV recording percent/counts per task.
#
# NOTES:
#   - the LAST <percent> of each task's DOCUMENTS are kept: lm_eval's --limit N
#     evaluates the FIRST N documents per task, so any later benchmark run with
#     limit <= limit_safe_max (in the meta) never touches the oracle subset.
#   - the tail is cut in doc_id order per task (not raw request order), so it
#     is stable even if request order ever changes.
#   - the 'doc' column keeps the raw document (incl. gold answer) for
#     truth-conditioned oracle modes (TruthCollector / the y problem).
#################################################

import csv
import json
import os

from lm_eval.__main__ import cli_evaluate
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model

from tools_debug import jprint


FIELDNAMES_MOCKUP = ['id_request', 'task_name', 'doc_id', 'prompt', 'until', 'doc']


@register_model("mockup")
class MockupCollectorLM(LM):

    def __init__(self, batch_size=1, percent=0.1, folder_output='benchmark_mockup', tag='', merge='', *args, **kwargs):
        super().__init__()

        self.percent = float(percent)
        self.folder_output = folder_output
        self.tag = tag
        self.merge = merge    # non-empty -> single combined CSV named by this

        assert 0.0 < self.percent <= 1.0, f'percent must be in (0, 1], got {self.percent}'
    # end

    def _build_path_output(self, name):
        name_parts = ['mockup', name]
        if self.tag:
            name_parts.append(self.tag)
        # end
        name_parts.append(f'p{int(self.percent * 100)}')

        return os.path.join(self.folder_output, '_'.join(name_parts) + '.csv')
    # end

    def _collect_task(self, task_name, requests_task):
        # tail cut in DOC units, doc_id order: --limit N takes the first N docs,
        # so docs beyond limit_safe_max are reserved for the oracle
        ids_doc = sorted({request_eval.doc_id for request_eval in requests_task})
        n_total = len(ids_doc)
        n_keep = max(1, int(n_total * self.percent))
        ids_keep = set(ids_doc[n_total - n_keep:])

        rows = []
        for id_request, request_eval in enumerate(requests_task):
            if request_eval.doc_id not in ids_keep:
                continue
            # end

            kwargs_gen = request_eval.args[1] if len(request_eval.args) > 1 else {}
            rows.append({
                'id_request': id_request,
                'task_name': task_name,
                'doc_id': request_eval.doc_id,
                'prompt': request_eval.args[0],
                'until': json.dumps(kwargs_gen.get('until', [])),
                'doc': json.dumps(getattr(request_eval, 'doc', {}), default=str),
            })
        # end

        meta = {
            'task_name': task_name,
            'mode': 'tail',
            'percent': self.percent,
            'n_total_docs': n_total,
            'n_keep_docs': n_keep,
            'n_rows': len(rows),
            'limit_safe_max': n_total - n_keep,    # benchmark runs with --limit <= this never touch the oracle subset
            'ids_doc_keep': sorted(ids_keep),
        }
        return rows, meta
    # end

    def _write_csv(self, path_output, rows, metas):
        os.makedirs(os.path.dirname(path_output) or '.', exist_ok=True)

        with open(path_output, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=FIELDNAMES_MOCKUP, quoting=csv.QUOTE_ALL)
            writer.writeheader()
            writer.writerows(rows)
        # end

        with open(path_output + '.meta.json', 'w') as file:
            json.dump({'tasks': metas}, file, indent=2)
        # end

        n_docs = sum(meta['n_keep_docs'] for meta in metas)
        n_total = sum(meta['n_total_docs'] for meta in metas)
        jprint(f'saved {n_docs}/{n_total} docs ({len(rows)} rows, {len(metas)} task(s)) to {path_output}')
    # end

    def generate_until(self, requests_eval):
        # one call carries ALL tasks' requests (group tasks arrive per subtask)
        tasks = {}
        for request_eval in requests_eval:
            name = getattr(request_eval, 'task_name', 'unknown')
            tasks.setdefault(name, []).append(request_eval)
        # end

        if self.merge:
            rows_all, metas_all = [], []
            for task_name in sorted(tasks):
                rows, meta = self._collect_task(task_name, tasks[task_name])
                rows_all.extend(rows)
                metas_all.append(meta)
            # end
            self._write_csv(self._build_path_output(self.merge), rows_all, metas_all)
        else:
            for task_name in sorted(tasks):
                rows, meta = self._collect_task(task_name, tasks[task_name])
                self._write_csv(self._build_path_output(task_name), rows, [meta])
            # end
        # end

        # lm_eval expects one output per request; empty strings keep it moving
        # (run with --predict_only; any printed metrics are meaningless)
        return [''] * len(requests_eval)
    # end

    def loglikelihood(self, requests):
        raise NotImplementedError('mockup collector only supports generate_until tasks')
    # end

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError('mockup collector only supports generate_until tasks')
    # end
# end


def load_benchmark_mockup(path_csv, filter_task=None):
    '''read a mockup CSV back into a list of dicts; until/doc decoded from json.
       filter_task selects one subtask's rows out of a merged CSV (e.g. bbh).'''
    rows = []
    with open(path_csv, 'r', newline='') as file:
        for row in csv.DictReader(file):
            if filter_task is not None and row['task_name'] != filter_task:
                continue
            # end
            row['until'] = json.loads(row['until'])
            row['doc'] = json.loads(row['doc'])
            row['id_request'] = int(row['id_request'])
            row['doc_id'] = int(row['doc_id'])
            rows.append(row)
        # end
    # end
    return rows
# end


if __name__ == "__main__":
    # mbpp/humaneval refuse to LOAD without this env (code_eval metric gate);
    # the mockup never executes model code, so setting it here is safe
    os.environ.setdefault('HF_ALLOW_CODE_EVAL', '1')
    cli_evaluate()
# end
