#################################################
# EXPERIMENT 2 (diagram c) -- teacher-trajectory DOSE-RESPONSE on gsm8k /
# llada_base: give each cached method the first n% of the full-denoising
# teacher's unmask trajectory (positions + tokens, committed in teacher
# order), let it continue with its own mechanism, and measure correctness.
#
#   x-axis: n in {0,10,...,90}% of the teacher trajectory injected
#   y-axis: accuracy over TEACHER-CORRECT docs only (per your design call);
#           per-doc s_i = mean correctness over the n-grid gives the 64-ish
#           continuous per-sample points for the fidelity-performance figure.
#   fast-dllm runs too but is reported separately (block-structured decoder).
#
# Interpretation: this is an INTERVENTION, not a correlation -- fidelity to
# the teacher is set by construction, each doc is its own control across n,
# so prompt difficulty cannot confound the curve.
#
# Prompts come from the benchmark MOCKUP CSV (exact lm_eval-built 5-shot
# contexts; save_benchmark_mockup.py). Scoring is applied offline to the
# decoded continuation: 'flex' = last-number match (lm_eval
# flexible-extract analog, primary), 'strict' = '#### N' (strict-match
# analog). lm_eval itself cannot inject canvases, hence the offline scorer.
#
# BUDGET (single GPU): teacher ~45 min; each method ~35-70 min
# (10 n-values x ~44 teacher-correct docs; model loaded ONCE per phase);
# all four methods + teacher ~ 4.5-5 GPU-hours.
#
# Usage (phases in order; every phase is resume-safe):
#   python run_experiment_2_dose_response.py teacher  --path_mockup benchmark_mockup/mockup_gsm8k_5shot_p10.csv --device cuda:0
#   python run_experiment_2_dose_response.py continue --method ours      --device cuda:0
#   python run_experiment_2_dose_response.py continue --method dllmcache --device cuda:0
#   python run_experiment_2_dose_response.py continue --method d2cache   --device cuda:0
#   python run_experiment_2_dose_response.py continue --method fastdllm  --device cuda:0
#   python run_experiment_2_dose_response.py report
#   python run_experiment_2_dose_response.py export     # fig_ab_<method>.npz for diagrams (a)/(b)
# Overrides: --override step_refresh_remainder=32 (repeatable), --grid 0,25,50,75
#################################################

import argparse
import csv
import importlib
import json
import os
import re
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from configs_llada import DiffusionConfig_Eval
from constants_llada import DTYPE_EVAL
from components_llada import SimpleLogitsSnapshot
from tools_llada import TopKSorter, MaxCollector
from dataprocess_llada import Preprocessor_Until
from save_benchmark_mockup import load_benchmark_mockup
from collect_metrics_common import check_result_gsm8k
from plugins_llada import SaveKVPreviousPlugin_Disabled, CachePastKVPlugin_Disabled,\
                            CacheAttnPlugin_Disabled, CacheVOPlugin_Disabled

ID_MODEL = 'GSAI-ML/LLaDA-8B-Base'
ID_MASK = 126336
LEN_GEN = 256
N_DOCS = 64
FOLDER_DEFAULT = 'results_experiment_2_dose'
GRID_DEFAULT = ','.join(str(n) for n in range(0, 91, 2))    # 0,2,...,90: 46 points
                                                            # (supersets the old
                                                            # 10%-grid, so resume
                                                            # reuses those rows)
TOPK_TEACHER = 32    # teacher top-K dump per position, for diagram (b)'s
                     # p(student token); ~1 MB at 64x256x32 float16
P_FLOOR = 1e-4       # p_teacher when the student token is outside the top-K
BLOCK_FIG = 32       # rank-normalization window for diagrams (a)/(b)

def load_methods_yaml(path):
    '''{name: (runner, num_blocks, args)} from the yaml; see
    experiment_2_methods.yaml for the matched-TFLOPs design + rationale'''
    import yaml
    spec = yaml.safe_load(open(path))
    return {name: (entry['runner'], int(entry['num_blocks']), dict(entry.get('args') or {}))
            for name, entry in spec['methods'].items()}
# end


# fallback when no yaml is present; the yaml (matched-TFLOPs design) wins
METHODS = {
    'ours': ('run_llada_semi_mlp_v2', 1,
             {'step_refresh_remainder': 16, 'step_refresh_remainder_prompt': 64,
              'select_only_in_h': True, 'h': 8,
              'path_router': 'routers_e2e/llada_base__cm_clean.pt'}),
    'dllmcache': ('run_llada_dllm_cache', 1,
                  {'dllmc_v_rate': 0.25, 'step_refresh_remainder': 8,
                   'step_refresh_remainder_prompt': 64}),
    'd2cache': ('run_llada_d2cache', 1,
                {'d2c_k': 32, 'd2c_sigma': 10.0, 'd2c_rollout_p': 0.1,
                 'd2c_conf_mode': 'live'}),
    'fastdllm': ('run_llada_fastdllm', 8, {}),
}


'''----------------------------- shared pieces -----------------------------'''

def score_strict(text, doc):
    match_gold = re.search(r'####\s*(-?[0-9\.,]+)', str(doc.get('answer', '')))
    match_pred = re.search(r'####\s*(-?[0-9\.,]+)', text)
    if match_gold is None or match_pred is None:
        return 0
    norm = lambda s: s.replace(',', '').rstrip('.')
    return int(norm(match_pred.group(1)) == norm(match_gold.group(1)))
# end


def score_flex(text, doc):
    return int(check_result_gsm8k(text, doc) == 'pass')
# end


def cut_at_stops(text, until):
    for word in until:
        if word in text:
            text = text.split(word)[0]
    return text
# end


def load_model(device):
    from modeling_llada_yukai_06 import LLaDAModelLM
    tokenizer = AutoTokenizer.from_pretrained(ID_MODEL, trust_remote_code=True)
    tokenizer.padding_side = 'left'
    model = LLaDAModelLM.from_pretrained(
        ID_MODEL, trust_remote_code=True, torch_dtype=DTYPE_EVAL).eval().to(device)
    return model, tokenizer
# end


def load_docs(folder):
    meta = json.load(open(os.path.join(folder, 'teacher_meta.json')))
    trace = np.load(os.path.join(folder, 'teacher_trace.npz'))
    return meta, trace
# end


'''----------------------------- teacher phase -----------------------------'''

@torch.no_grad()
def phase_teacher(config_cli):
    folder = config_cli.folder
    os.makedirs(folder, exist_ok=True)
    rows = load_benchmark_mockup(config_cli.path_mockup)[:config_cli.n_docs]

    model, tokenizer = load_model(config_cli.device)
    for klass in (CachePastKVPlugin_Disabled, SaveKVPreviousPlugin_Disabled,
                  CacheAttnPlugin_Disabled, CacheVOPlugin_Disabled):
        model.fill_plugin(klass)
    # end
    preprocessor = Preprocessor_Until(tokenizer)    # plain prompts (llada_base)

    steps_all, toks_all, confs_all = [], [], []
    topk_p_all, topk_id_all = [], []
    meta = {'docs': []}
    for id_doc, row in enumerate(rows):
        time_start = time.perf_counter()
        prep = preprocessor({'prompt': row['prompt'], 'until': row['until']})
        ids_prompt = prep['ids_prompt']
        len_prompt = len(ids_prompt)
        len_full = len_prompt + LEN_GEN

        x = torch.tensor(ids_prompt + [ID_MASK] * LEN_GEN,
                         dtype=torch.long, device=config_cli.device).view(1, -1)
        idx_full = torch.arange(len_full, dtype=torch.long, device=config_cli.device)
        shape_target = (1, len_full, -1)

        step_of = np.zeros(LEN_GEN, dtype=np.int64)
        tok_of = np.zeros(LEN_GEN, dtype=np.int64)
        conf_of = np.zeros(LEN_GEN, dtype=np.float32)
        topk_p_of = np.zeros((LEN_GEN, TOPK_TEACHER), dtype=np.float16)
        topk_id_of = np.zeros((LEN_GEN, TOPK_TEACHER), dtype=np.int32)

        # full denoising, greedy confidence-argmax, one token per step --
        # run_llada_semi semantics (float64 softmax like transform_logits)
        for step in range(LEN_GEN):
            logits = model(x, idx_current=idx_full, shape_target=shape_target).logits
            p = F.softmax(logits[0, len_prompt:].to(torch.float64), dim=-1)
            conf, x0 = p.max(dim=-1)
            mask_still = x[0, len_prompt:] == ID_MASK
            conf = conf.masked_fill(~mask_still, -1.0)
            pos = int(conf.argmax())
            x[0, len_prompt + pos] = x0[pos]
            step_of[pos], tok_of[pos], conf_of[pos] = step, int(x0[pos]), float(conf[pos])
            # teacher's commit-time distribution at this position (top-K):
            # lets diagram (b) look up teacher p(student token) after the fact
            p_top, id_top = p[pos].topk(TOPK_TEACHER)
            topk_p_of[pos] = p_top.cpu().numpy().astype(np.float16)
            topk_id_of[pos] = id_top.cpu().numpy().astype(np.int32)
        # end

        text = tokenizer.decode(x[0, len_prompt:], skip_special_tokens=True)
        text = cut_at_stops(text, row['until'])
        correct_flex, correct_strict = score_flex(text, row['doc']), score_strict(text, row['doc'])

        steps_all.append(step_of); toks_all.append(tok_of); confs_all.append(conf_of)
        topk_p_all.append(topk_p_of); topk_id_all.append(topk_id_of)
        meta['docs'].append({'id_doc': id_doc, 'doc_id': row['doc_id'],
                             'ids_prompt': ids_prompt, 'until': row['until'],
                             'doc': row['doc'], 'text': text,
                             'correct_flex': correct_flex, 'correct_strict': correct_strict})
        print(f'[teacher] doc {id_doc}: flex={correct_flex} strict={correct_strict} '
              f'P={len_prompt} {time.perf_counter()-time_start:.0f}s', flush=True)
    # end

    np.savez(os.path.join(folder, 'teacher_trace.npz'),
             teacher_step=np.stack(steps_all), teacher_tok=np.stack(toks_all),
             teacher_conf=np.stack(confs_all),
             teacher_topk_p=np.stack(topk_p_all), teacher_topk_id=np.stack(topk_id_all))
    json.dump(meta, open(os.path.join(folder, 'teacher_meta.json'), 'w'))
    n_ok = sum(d['correct_flex'] for d in meta['docs'])
    print(f'[teacher] done: {n_ok}/{len(rows)} correct (flex) -> continuations use these')
# end


'''----------------------------- continue phase -----------------------------'''

def parse_override(pairs):
    out = {}
    for pair in pairs or []:
        key, value = pair.split('=', 1)
        try:
            out[key] = json.loads(value)
        except json.JSONDecodeError:
            out[key] = value
    return out
# end


@torch.no_grad()
def phase_continue(config_cli):
    folder = config_cli.folder
    meta, trace = load_docs(folder)
    name_runner, num_blocks, args_method = METHODS[config_cli.method]
    args_method = dict(args_method, **parse_override(config_cli.override))
    grid = [int(v) for v in config_cli.grid.split(',')]

    config = DiffusionConfig_Eval(
        id_model=ID_MODEL, len_target=LEN_GEN, num_blocks=num_blocks,
        num_unmask_per_step=1, id_mask=ID_MASK, size_batch=1,
        device=config_cli.device, klass_sorter=TopKSorter, klass_collector=MaxCollector,
        **args_method)

    module = importlib.import_module(name_runner)
    runner = module.RunModel()
    model, tokenizer = load_model(config_cli.device)
    runner.config_plugin_(config)
    runner.register_plugin_(model, config)

    path_csv = os.path.join(folder, f'continuations_{config_cli.method}.csv')
    done = set()
    if os.path.exists(path_csv):
        with open(path_csv) as file:
            done = {(int(row['id_doc']), int(row['n'])) for row in csv.DictReader(file)}
    else:
        with open(path_csv, 'w', newline='') as file:
            csv.writer(file).writerow(
                ['method', 'id_doc', 'n', 'k_injected', 'correct_flex', 'correct_strict',
                 'has_done', 'seconds', 'setting'])
    # end
    setting = json.dumps(args_method, sort_keys=True)

    # student free-run traces (n=0) for diagrams (a)/(b): captured via the
    # SimpleLogitsSnapshot.TRACE hook -- same runs, zero extra GPU time
    path_trace = os.path.join(folder, f'student_trace_{config_cli.method}.npz')
    traces = {}
    if os.path.exists(path_trace):
        saved = np.load(path_trace)
        for pos_row, (step_row, tok_row) in enumerate(zip(saved['student_step'], saved['student_tok'])):
            traces[int(saved['ids_doc'][pos_row])] = (step_row, tok_row)
    # end

    def dump_traces():
        ids_doc = sorted(traces)
        np.savez(path_trace, ids_doc=np.array(ids_doc),
                 student_step=np.stack([traces[i][0] for i in ids_doc]),
                 student_tok=np.stack([traces[i][1] for i in ids_doc]))
    # end

    docs_teacher_ok = [d for d in meta['docs'] if d['correct_flex']]
    print(f'[{config_cli.method}] {len(docs_teacher_ok)} teacher-correct docs x {len(grid)} n-values '
          f'({len(done)} rows already done, {len(traces)} traces on disk)')

    for doc in docs_teacher_ok:
        id_doc = doc['id_doc']
        ids_prompt = doc['ids_prompt']
        len_prompt = len(ids_prompt)
        order = np.argsort(trace['teacher_step'][id_doc])    # positions in teacher unmask order

        for n in grid:
            need_trace = (n == 0 and id_doc not in traces)
            if (id_doc, n) in done and not need_trace:
                continue
            k = round(n / 100 * LEN_GEN)
            x = torch.tensor(ids_prompt + [ID_MASK] * LEN_GEN,
                             dtype=torch.long, device=config_cli.device).view(1, -1)
            pos_inject = order[:k]
            x[0, len_prompt + torch.as_tensor(pos_inject, device=config_cli.device)] = \
                torch.as_tensor(trace['teacher_tok'][id_doc][pos_inject],
                                dtype=torch.long, device=config_cli.device)

            if n == 0:
                SimpleLogitsSnapshot.TRACE = []
            time_start = time.perf_counter()
            text, has_done = runner.run_one(
                model, tokenizer, config,
                ids_input=x, len_prompt=len_prompt,
                until=list(doc['until']), text_prompt='')
            seconds = time.perf_counter() - time_start

            if n == 0:
                commits, SimpleLogitsSnapshot.TRACE = SimpleLogitsSnapshot.TRACE, None
                step_row = np.full(LEN_GEN, -1, dtype=np.int64)
                tok_row = np.zeros(LEN_GEN, dtype=np.int64)
                for id_step, (idx_commit, toks_commit) in enumerate(commits):
                    for pos, tok in zip(np.atleast_1d(idx_commit), np.atleast_1d(toks_commit)):
                        pos_gen = int(pos) - len_prompt
                        if 0 <= pos_gen < LEN_GEN and step_row[pos_gen] < 0:
                            step_row[pos_gen], tok_row[pos_gen] = id_step, int(tok)
                    # end
                # end
                traces[id_doc] = (step_row, tok_row)
                dump_traces()    # crash-safe: rewritten after every traced doc
            # end

            if (id_doc, n) in done:
                continue    # only the trace was missing; csv row already exists
            correct_flex, correct_strict = score_flex(text, doc['doc']), score_strict(text, doc['doc'])
            with open(path_csv, 'a', newline='') as file:
                csv.writer(file).writerow(
                    [config_cli.method, id_doc, n, k, correct_flex, correct_strict,
                     int(bool(has_done)), round(seconds, 2), setting])
            print(f'[{config_cli.method}] doc {id_doc} n={n}%: flex={correct_flex} '
                  f'{seconds:.0f}s', flush=True)
        # end
    # end
    print(f'[{config_cli.method}] complete -> {path_csv} + {path_trace}')
# end


'''----------------------------- report phase -----------------------------'''

def phase_report(config_cli):
    folder = config_cli.folder
    rows = []
    for method in METHODS:
        path_csv = os.path.join(folder, f'continuations_{method}.csv')
        if os.path.exists(path_csv):
            with open(path_csv) as file:
                rows.extend(csv.DictReader(file))
    if not rows:
        print('no continuation CSVs found yet')
        return

    print('\n===== dose-response: accuracy (flex) vs teacher-prefix %, teacher-correct docs =====')
    grid = sorted({int(row['n']) for row in rows})
    header = '  method     ' + ''.join(f'{n:>6d}%' for n in grid)
    print(header)
    for method in METHODS:
        rows_m = [row for row in rows if row['method'] == method]
        if not rows_m:
            continue
        cells = []
        for n in grid:
            vals = [int(row['correct_flex']) for row in rows_m if int(row['n']) == n]
            cells.append(f'{np.mean(vals):>6.3f}' if vals else '     -')
        note = '  (separate: block decoder)' if method == 'fastdllm' else ''
        print(f'  {method:10s} ' + ''.join(cells) + note)
    # end

    # per-sample s_i = mean correctness over the grid -> the 64-point figure
    path_out = os.path.join(folder, 'per_sample_dose.csv')
    with open(path_out, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['model', 'sample_id', 'fidelity', 'score'])
        # here 'fidelity' is the injected-trajectory dose response summary:
        # s_i = fraction of the n-grid at which the continuation is correct
        for method in METHODS:
            rows_m = [row for row in rows if row['method'] == method]
            for id_doc in sorted({int(row['id_doc']) for row in rows_m}):
                vals = [int(row['correct_flex']) for row in rows_m if int(row['id_doc']) == id_doc]
                writer.writerow([method, id_doc, round(np.mean(vals), 4), round(np.mean(vals), 4)])
    print(f'\nper-sample s_i -> {path_out}')
    print('plot: dose-response curves from the table above; s_i distributions from the csv')
# end


'''----------------------------- export phase -----------------------------'''

def phase_export(config_cli):
    # per method: join teacher + student free-run traces into the input format
    # of the (a)/(b) trajectory-fidelity figure: arrays [M, 256] over generated
    # positions + scalars block_length / steps
    folder = config_cli.folder
    meta, trace = load_docs(folder)
    topk_p = trace['teacher_topk_p'].astype(np.float32)
    topk_id = trace['teacher_topk_id']

    for method in METHODS:
        path_trace = os.path.join(folder, f'student_trace_{method}.npz')
        if not os.path.exists(path_trace):
            print(f'  (no student trace yet: {method})')
            continue
        student = np.load(path_trace)
        ids_doc = student['ids_doc']

        t_step = trace['teacher_step'][ids_doc]
        t_tok = trace['teacher_tok'][ids_doc]
        t_conf = trace['teacher_conf'][ids_doc]
        s_step = student['student_step']
        s_tok = student['student_tok']

        # p_teacher: teacher's commit-time prob of the STUDENT's token,
        # looked up in the teacher's top-K dump; floor when outside top-K
        p_teacher = np.full(s_tok.shape, P_FLOOR, dtype=np.float32)
        for row, id_doc in enumerate(ids_doc):
            hit = topk_id[id_doc] == s_tok[row][:, None]        # (256, K)
            any_hit = hit.any(axis=1)
            vals = topk_p[id_doc][np.arange(LEN_GEN), hit.argmax(axis=1)]
            p_teacher[row][any_hit] = vals[any_hit]
        # end

        path_out = os.path.join(folder, f'fig_ab_{method}.npz')
        np.savez(path_out, teacher_step=t_step, student_step=s_step,
                 teacher_tok=t_tok, student_tok=s_tok, teacher_conf=t_conf,
                 p_teacher=p_teacher, block_length=BLOCK_FIG, steps=LEN_GEN)
        n_match = float((s_tok == t_tok).mean())
        print(f'  {method}: {len(ids_doc)} docs -> {path_out} (token match {n_match:.3f})')
    # end
# end


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('teacher', 'continue', 'report', 'export'))
    parser.add_argument('--folder', default=FOLDER_DEFAULT)
    parser.add_argument('--path_mockup', help='gsm8k mockup CSV (teacher phase)')
    parser.add_argument('--n_docs', type=int, default=N_DOCS)
    parser.add_argument('--method')    # validated after the yaml loads
    parser.add_argument('--grid', default=GRID_DEFAULT)
    parser.add_argument('--override', action='append',
                        help='method arg override, key=value (repeatable)')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--methods_yaml', default='experiment_2_methods.yaml',
                        help='per-method hyperparameter spec; loaded when the '
                             'file exists (pass a nonexistent name to use the '
                             'built-in defaults)')
    config_cli = parser.parse_args()

    if os.path.exists(config_cli.methods_yaml):
        METHODS.clear()
        METHODS.update(load_methods_yaml(config_cli.methods_yaml))
        print(f'[methods] loaded {list(METHODS)} from {config_cli.methods_yaml}')
    # end

    if config_cli.phase == 'teacher':
        assert config_cli.path_mockup, 'teacher phase needs --path_mockup'
        phase_teacher(config_cli)
    elif config_cli.phase == 'continue':
        assert config_cli.method in METHODS, \
            f'continue phase needs --method (one of {list(METHODS)})'
        phase_continue(config_cli)
    elif config_cli.phase == 'export':
        phase_export(config_cli)
    else:
        phase_report(config_cli)
    # end
# end


if __name__ == '__main__':
    main()
