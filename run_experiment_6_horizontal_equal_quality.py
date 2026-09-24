#################################################
# EXPERIMENT 1 -- horizontal comparison on gsm8k / llada_base:
# equal-quality operating points vs TFLOPs, then batch scaling.
#
# Pipeline (stages):
#   sweep         hyperparameter grids for dense / ours / dllm-cache /
#                 fast-dllm / d2cache, 64 gsm8k docs, bs=1, GPU-pool queue
#   report        parse scores + runner reports, compute analytic TFLOPs/doc,
#                 pick the EQUAL-QUALITY setting per method: the cheapest
#                 setting whose score >= dense_score - tol (ours: explicitly
#                 the MINIMAL-TFLOPs qualifying setting); writes chosen.json
#   batch         run the chosen setting per method at size_batch 1,2,4,8,16
#                 with the *_batch runners
#   batch_report  batch table: score, s/doc, speedup vs bs=1, tokens/s
#
# TFLOPs accounting (documented approximations):
#   analytic forward FLOPs from the method's row schedule; per-row per-layer
#   cost = QKVO (8 d^2) + gated FFN (6 d ff) + attention (4 T_ctx d); LM head
#   2 d V per logits row; dllm-cache non-selected rows pay the V-projection
#   (2 d^2); d2cache's rollout extras are estimated at their expected size
#   (p * T). Prompt length is the MEASURED per-run average from the runner
#   report. Validated anchors: ours Kr=24/no-Kp ~ 80 TF/doc (matches the v2
#   runner header); dense ~ 18 TF/step at 5-shot gsm8k geometry (matches
#   dLLM-Cache paper's 16.12 TF/token baseline scale).
#
# Usage:
#   python run_experiment_horizontal.py sweep --gpus 0,1,2,3,4,5,6,7
#   python run_experiment_horizontal.py report [--tol 0.03]
#   python run_experiment_horizontal.py batch --gpus 0,1,2,3
#   python run_experiment_horizontal.py batch_report
#   ... sweep --dry     (print commands only)
# Resume-safe: a job with a runner report AND lm_eval results is skipped.
#################################################

import argparse
import glob
import itertools
import json
import os
import subprocess
import threading

'''task + model constants (llada_base / gsm8k only, by design of this thread)'''
ID_MODEL = 'GSAI-ML/LLaDA-8B-Base'
ID_MASK = 126336
TASK = 'gsm8k'
NSHOT = 5
LIMIT = 64
LEN_GEN = 256          # len_target
STEPS = 256            # num_unmask_per_step=1
ROUTER = 'routers_e2e/llada_base__cm_clean.pt'
H_OURS = 8

D = 4096               # LLaDA-8B dims for the analytic FLOPs model
N_LAYERS = 32
FF = 12288
VOCAB = 126464

FOLDER_DEFAULT = 'results_experiment_horizontal'
PORT_BASE = 16000

BATCH_SIZES = (1, 2, 4, 8, 16)
RUNNER_BATCH = {
    'ours': 'run_llada_semi_mlp_v2_batch',
    'dllmcache': 'run_llada_dllm_cache_batch',
    'fastdllm': 'run_llada_fastdllm_batch',
    'd2cache': 'run_llada_d2cache_batch',
}


'''----------------------------- job grids -----------------------------'''

def jobs_sweep():
    jobs = []    # (tag, method, runner, num_blocks, args_extra: dict)

    jobs.append(('dense__full', 'dense', 'run_llada_semi', 1, {}))

    for kr, kp in itertools.product((8, 16, 32, 48), (0, 64, 96)):
        args = {'step_refresh_remainder': kr, 'select_only_in_h': True,
                'h': H_OURS, 'path_router': ROUTER}
        if kp:
            args['step_refresh_remainder_prompt'] = kp
        jobs.append((f'ours__kr{kr}_kp{kp}', 'ours', 'run_llada_semi_mlp_v2', 1, args))

    for v, kr, kp in itertools.product((0.25, 0.5), (8, 16, 32), (64, 100)):
        args = {'dllmc_v_rate': v, 'step_refresh_remainder': kr,
                'step_refresh_remainder_prompt': kp}
        jobs.append((f'dllmcache__v{v}_kr{kr}_kp{kp}', 'dllmcache', 'run_llada_dllm_cache', 1, args))

    for nb in (2, 4, 8, 16, 32):
        jobs.append((f'fastdllm__nb{nb}', 'fastdllm', 'run_llada_fastdllm', nb, {}))

    for k, p in itertools.product((16, 32, 64), (0.05, 0.1)):
        args = {'d2c_k': k, 'd2c_sigma': 10.0, 'd2c_rollout_p': p, 'd2c_conf_mode': 'live'}
        jobs.append((f'd2cache__k{k}_p{p}', 'd2cache', 'run_llada_d2cache', 1, args))

    return jobs
# end


def jobs_batch(chosen):
    jobs = []    # (tag, method, runner, num_blocks, args_extra, size_batch)
    for method, entry in chosen.items():
        if method == 'dense':
            continue
        runner = RUNNER_BATCH[method]
        for bs in BATCH_SIZES:
            jobs.append((f'batch__{method}__bs{bs}', method, runner,
                         entry['num_blocks'], dict(entry['args']), bs))
    return jobs
# end


'''----------------------------- execution -----------------------------'''

def build_command(folder, tag, runner, num_blocks, args_extra, size_batch, port):
    path_runner = os.path.join(folder, f'{tag}__runner.json')
    parts = [f'id_model={ID_MODEL}', f'size_batch={size_batch}', f'len_target={LEN_GEN}',
             f'num_blocks={num_blocks}', 'num_unmask_per_step=1', f'id_mask={ID_MASK}',
             f'runner={runner}', f'path_report={path_runner}']
    parts += [f'{key}={value}' for key, value in args_extra.items()]
    model_args = ','.join(parts)

    cmd = ['accelerate', 'launch', '--num_processes=1', f'--main_process_port={port}',
           'run_benchmark_main.py', '--tasks', TASK, '--limit', str(LIMIT),
           '--model', 'test', '--batch_size', str(size_batch),
           '--num_fewshot', str(NSHOT), '--device', 'cuda',
           '--output_path', os.path.join(folder, tag),
           '--model_args', model_args]
    return cmd, path_runner
# end


def job_is_done(folder, tag):
    path_runner = os.path.join(folder, f'{tag}__runner.json')
    results = glob.glob(os.path.join(folder, tag, '**', 'results_*.json'), recursive=True)
    return os.path.exists(path_runner) and bool(results)
# end


def run_jobs(jobs, folder, gpus, dry):
    os.makedirs(os.path.join(folder, 'logs'), exist_ok=True)
    lock = threading.Lock()
    queue = list(enumerate(jobs))

    def worker(gpu):
        while True:
            with lock:
                if not queue:
                    return
                idx, job = queue.pop(0)
            # end
            tag, _method, runner, num_blocks, args_extra, size_batch = job
            if job_is_done(folder, tag):
                print(f'[gpu{gpu}] SKIP {tag}: already complete')
                continue
            cmd, _ = build_command(folder, tag, runner, num_blocks, args_extra,
                                   size_batch, PORT_BASE + idx)
            if dry:
                print(f'[gpu{gpu}] DRY {tag}:')
                print('  CUDA_VISIBLE_DEVICES=%s %s' % (gpu, ' '.join(cmd)))
                continue
            print(f'[gpu{gpu}] START {tag}')
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
            path_log = os.path.join(folder, 'logs', f'{tag}.log')
            with open(path_log, 'w') as file_log:
                status = subprocess.run(cmd, env=env, stdout=file_log,
                                        stderr=subprocess.STDOUT).returncode
            verdict = 'DONE' if status == 0 and job_is_done(folder, tag) else f'FAILED({status})'
            print(f'[gpu{gpu}] {verdict} {tag}')
        # end while
    # end

    threads = [threading.Thread(target=worker, args=(gpu,)) for gpu in gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
# end


'''----------------------------- parsing -----------------------------'''

METRIC_PREFERENCE = ('exact_match,strict-match', 'exact_match,flexible-extract')

def read_score(folder, tag):
    paths = glob.glob(os.path.join(folder, tag, '**', 'results_*.json'), recursive=True)
    if not paths:
        return None
    results = json.load(open(max(paths, key=os.path.getmtime)))['results'].get(TASK, {})
    for key in METRIC_PREFERENCE:    # zero-fallback, like build_bench_html
        value = results.get(key)
        if value:
            return value
    for key in METRIC_PREFERENCE:
        if results.get(key) is not None:
            return results[key]
    return None
# end


def read_report(folder, tag):
    path = os.path.join(folder, f'{tag}__runner.json')
    if not os.path.exists(path):
        return None
    report = json.load(open(path))
    rows = report.get('rows', [])
    if not rows:
        return None
    return {
        'len_prompt_avg': sum(row['len_prompt'] for row in rows) / len(rows),
        'duration_total_s': report['duration_total_s'],
        'duration_per_doc_s': report['duration_total_s'] / len(rows),
        'num_samples': len(rows),
    }
# end


'''----------------------------- analytic TFLOPs -----------------------------'''

def _fl_rows(rows, t_ctx):
    # full transformer-layer cost for `rows` query rows against context t_ctx
    per_row = 8 * D * D + 6 * D * FF + 4 * t_ctx * D
    return rows * N_LAYERS * per_row
# end

def _fl_head(rows):
    return rows * 2 * D * VOCAB
# end

def _count_ticks(steps, every, include_zero):
    if not every:
        return 1 if include_zero else 0
    ticks = [s for s in range(steps) if s % every == 0]
    return len(ticks) if include_zero else len(ticks) - 1
# end


def flops_doc(method, args, num_blocks, len_prompt):
    P, G, S = len_prompt, LEN_GEN, STEPS
    T = P + G

    if method == 'dense':
        return S * (_fl_rows(T, T) + _fl_head(T))

    if method == 'ours':
        kr = args['step_refresh_remainder']
        kp = args.get('step_refresh_remainder_prompt', 0)
        h = args.get('h', H_OURS)
        n_kr = _count_ticks(S, kr, include_zero=True)
        n_kp = _count_ticks(S, kp, include_zero=False) if kp else 0
        n_adapt = S - n_kr
        total = _fl_rows(P, P)                                    # initial prompt forward (skip_logits)
        total += n_kr * (_fl_rows(G + 1, T) + _fl_head(G + 1))    # gen-area refresh + logits
        total += n_adapt * (_fl_rows(h + 1, T) + _fl_head(h + 1)) # router-selected rows
        total += n_kp * _fl_rows(P, T)                            # prompt KV refresh, no head
        return total

    if method == 'dllmcache':
        v = args['dllmc_v_rate']
        kr = args['step_refresh_remainder']
        kp = args.get('step_refresh_remainder_prompt', 0)
        n_all = _count_ticks(S, kp, include_zero=True) if kp else 1
        n_resp = max(0, _count_ticks(S, kr, include_zero=True) - n_all)
        n_adapt = S - n_all - n_resp
        rows_sel = max(1, int(v * G))
        cost_vrow = 2 * D * D                                     # V-projection for cached rows
        extra_sel = 6 * D * D + 6 * D * FF + 4 * T * D            # full row minus its vrow
        total = n_all * (_fl_rows(T, T) + _fl_head(T))
        total += n_resp * (_fl_rows(G, T) + P * N_LAYERS * cost_vrow + _fl_head(T))
        total += n_adapt * (N_LAYERS * (T * cost_vrow + rows_sel * extra_sel) + _fl_head(T))
        return total

    if method == 'fastdllm':
        size_bk = G // num_blocks
        per_block = _fl_rows(T, T) + _fl_head(T) \
                    + (size_bk - 1) * (_fl_rows(size_bk, T) + _fl_head(size_bk))
        return num_blocks * per_block

    if method == 'd2cache':
        k = args['d2c_k']
        p = args['d2c_rollout_p']
        rows = k + 1 + p * T                                      # cands + resync + E[nucleus]
        return _fl_rows(T, T) + _fl_head(T) + S * (_fl_rows(rows, T) + _fl_head(rows))

    raise ValueError(method)
# end


'''----------------------------- stages -----------------------------'''

def stage_report(folder, tol):
    rows = []
    for tag, method, _runner, num_blocks, args in jobs_sweep():
        score = read_score(folder, tag)
        report = read_report(folder, tag)
        if score is None or report is None:
            print(f'  (incomplete: {tag})')
            continue
        tflops = flops_doc(method, args, num_blocks, report['len_prompt_avg']) / 1e12
        rows.append({'tag': tag, 'method': method, 'num_blocks': num_blocks, 'args': args,
                     'score': score, 'tflops_doc': round(tflops, 1),
                     's_doc': round(report['duration_per_doc_s'], 2),
                     'len_prompt_avg': round(report['len_prompt_avg'], 1)})
    if not rows:
        print('no completed sweep runs found; run the sweep stage first')
        return

    dense_rows = [row for row in rows if row['method'] == 'dense']
    anchor = dense_rows[0]['score'] if dense_rows else max(row['score'] for row in rows)
    floor = anchor - tol
    print(f'\n===== sweep ({folder}) -- dense anchor {anchor:.4f}, quality floor {floor:.4f} =====')

    chosen = {}
    for method in ('dense', 'ours', 'dllmcache', 'fastdllm', 'd2cache'):
        rows_method = sorted([row for row in rows if row['method'] == method],
                             key=lambda row: row['tflops_doc'])
        if not rows_method:
            continue
        print(f'\n--- {method} ---')
        qualifying = [row for row in rows_method if row['score'] >= floor]
        pick = qualifying[0] if qualifying else max(rows_method, key=lambda row: row['score'])
        for row in rows_method:
            mark = ' <== CHOSEN (min TFLOPs at equal quality)' if row is pick else ''
            ok = ' ' if row['score'] >= floor else 'x'
            print(f'  [{ok}] {row["tag"]:32s} score={row["score"]:.4f} '
                  f'TFLOPs/doc={row["tflops_doc"]:>8.1f} s/doc={row["s_doc"]:>7.2f}{mark}')
        if not qualifying:
            print(f'  !! no setting reached the floor; falling back to best score ({pick["score"]:.4f})')
        chosen[method] = pick
    # end

    path_chosen = os.path.join(folder, 'chosen.json')
    json.dump(chosen, open(path_chosen, 'w'), indent=2)
    print(f'\nchosen settings -> {path_chosen}')
# end


def stage_batch_report(folder):
    chosen = json.load(open(os.path.join(folder, 'chosen.json')))
    print(f'\n===== batch scaling ({folder}) =====')
    print(f'  {"method":10s} {"bs":>3s} {"score":>7s} {"s/doc":>8s} {"speedup":>8s} {"tok/s":>8s}')
    for method in ('ours', 'dllmcache', 'fastdllm', 'd2cache'):
        if method not in chosen:
            continue
        base_s_doc = None
        for bs in BATCH_SIZES:
            tag = f'batch__{method}__bs{bs}'
            score = read_score(folder, tag)
            report = read_report(folder, tag)
            if score is None or report is None:
                print(f'  {method:10s} {bs:>3d} (incomplete)')
                continue
            s_doc = report['duration_per_doc_s']
            if bs == 1:
                base_s_doc = s_doc
            speedup = (base_s_doc / s_doc) if base_s_doc else float('nan')
            tok_s = LEN_GEN / s_doc
            print(f'  {method:10s} {bs:>3d} {score:>7.4f} {s_doc:>8.2f} {speedup:>7.2f}x {tok_s:>8.1f}')
        print()
# end


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('sweep', 'report', 'batch', 'batch_report'))
    parser.add_argument('--folder', default=FOLDER_DEFAULT)
    parser.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    parser.add_argument('--tol', type=float, default=0.03,
                        help='equal-quality band: score >= dense - tol')
    parser.add_argument('--dry', action='store_true')
    config = parser.parse_args()

    gpus = [gpu.strip() for gpu in config.gpus.split(',') if gpu.strip()]
    os.makedirs(config.folder, exist_ok=True)

    if config.stage == 'sweep':
        jobs = [job + (1,) for job in jobs_sweep()]    # size_batch=1
        run_jobs(jobs, config.folder, gpus, config.dry)
        print('\nsweep finished; next: python run_experiment_horizontal.py report')
    elif config.stage == 'report':
        stage_report(config.folder, config.tol)
    elif config.stage == 'batch':
        chosen = json.load(open(os.path.join(config.folder, 'chosen.json')))
        jobs = jobs_batch(chosen)
        run_jobs(jobs, config.folder, gpus, config.dry)
        print('\nbatch finished; next: python run_experiment_horizontal.py batch_report')
    elif config.stage == 'batch_report':
        stage_batch_report(config.folder)
    # end
# end


if __name__ == '__main__':
    main()
