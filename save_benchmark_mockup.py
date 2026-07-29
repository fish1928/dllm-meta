#################################################
# Benchmark mockup collector.
#
# Registers a fake lm_eval model ("mockup") whose generate_until does NOT run
# any model: lm_eval builds the exact benchmark requests (few-shot context,
# templates, stop strings), and we dump a seeded random subset of them to CSV.
# That CSV is the input pool for oracle generation (full-denoising runs).
#
# Usage:
#   python save_benchmark_mockup.py --tasks gsm8k --model mockup --num_fewshot 5 \
#       --model_args percent=0.1,folder_output=benchmark_mockup,tag=5shot
#
# Output: <folder_output>/mockup_<task>_<tag>_p<percent>.csv with columns
#   id_request, task_name, doc_id, prompt, until (json), doc (json)
# plus a .meta.json sidecar recording percent/counts.
#
# NOTES:
#   - the LAST <percent> of the requests are kept: lm_eval's --limit N evaluates
#     the FIRST N documents, so any later benchmark run with
#     limit <= (1 - percent) * dataset size never touches the oracle subset.
#   - lm_eval still computes metrics afterwards on our empty outputs; ignore them.
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


@register_model("mockup")
class MockupCollectorLM(LM):

    def __init__(self, batch_size=1, percent=0.1, folder_output='benchmark_mockup', tag='', path_output=None, *args, **kwargs):
        super().__init__()

        self.percent = float(percent)
        self.folder_output = folder_output
        self.tag = tag
        self.path_output = path_output

        assert 0.0 < self.percent <= 1.0, f'percent must be in (0, 1], got {self.percent}'
    # end

    def _build_path_output(self, task_name):
        if self.path_output is not None:
            return self.path_output
        # end

        name_parts = ['mockup', task_name]
        if self.tag:
            name_parts.append(self.tag)
        # end
        name_parts.append(f'p{int(self.percent * 100)}')

        return os.path.join(self.folder_output, '_'.join(name_parts) + '.csv')
    # end

    def generate_until(self, requests_eval):
        n_total = len(requests_eval)
        n_keep = max(1, int(n_total * self.percent))

        # keep the TAIL: --limit N evaluates the first N docs, so the tail stays
        # out of any later benchmark run with limit <= n_total - n_keep
        idxs_keep = list(range(n_total - n_keep, n_total))

        rows = []
        for idx_request in idxs_keep:
            request_eval = requests_eval[idx_request]
            prompt = request_eval.args[0]
            kwargs_gen = request_eval.args[1] if len(request_eval.args) > 1 else {}

            rows.append({
                'id_request': idx_request,
                'task_name': getattr(request_eval, 'task_name', ''),
                'doc_id': getattr(request_eval, 'doc_id', ''),
                'prompt': prompt,
                'until': json.dumps(kwargs_gen.get('until', [])),
                'doc': json.dumps(getattr(request_eval, 'doc', {}), default=str),
            })
        # end

        task_name = rows[0]['task_name'] if rows else 'unknown'
        path_output = self._build_path_output(task_name)
        os.makedirs(os.path.dirname(path_output) or '.', exist_ok=True)

        with open(path_output, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()), quoting=csv.QUOTE_ALL)
            writer.writeheader()
            writer.writerows(rows)
        # end

        with open(path_output + '.meta.json', 'w') as file:
            json.dump({
                'task_name': task_name,
                'mode': 'tail',
                'percent': self.percent,
                'n_total': n_total,
                'n_keep': n_keep,
                'limit_safe_max': n_total - n_keep,    # benchmark runs with --limit <= this never touch the oracle subset
                'ids_request': idxs_keep,
            }, file)
        # end

        jprint(f'saved {n_keep}/{n_total} requests to {path_output}')

        # lm_eval expects one output per request; empty strings keep it moving
        # (the printed metrics are meaningless for this run)
        return [''] * n_total
    # end

    def loglikelihood(self, requests):
        raise NotImplementedError('mockup collector only supports generate_until tasks')
    # end

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError('mockup collector only supports generate_until tasks')
    # end
# end


def load_benchmark_mockup(path_csv):
    '''read a mockup CSV back into a list of dicts; until/doc are decoded from json'''
    rows = []
    with open(path_csv, 'r', newline='') as file:
        for row in csv.DictReader(file):
            row['until'] = json.loads(row['until'])
            row['doc'] = json.loads(row['doc'])
            row['id_request'] = int(row['id_request'])
            rows.append(row)
        # end
    # end
    return rows
# end


if __name__ == "__main__":
    cli_evaluate()
# end
