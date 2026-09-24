#################################################
# EXPERIMENT 6b -- horizontal comparison, EQUAL-TFLOPs axis (the transpose of
# run_experiment_6_horizontal_equal_quality): fix compute budgets from very
# limited to generous, match every method's hyperparameters to each budget
# with the analytic FLOPs model, run 64 gsm8k docs on llada_base, and read
# the score-vs-TFLOPs curve per method.
#
#   budgets (TF/doc): 60, 120, 250, 500, 1000, 2000   (dense reference ~4600)
#   matching: per method, the candidate setting whose predicted TFLOPs/doc is
#   closest to the budget in log-space, accepted within a 1.35x factor; a
#   method that cannot reach a tier simply has no point there (that IS a
#   finding: the reachable-compute ranges differ, e.g. dllm-cache's floor is
#   ~650 TF/doc -- full-window V-proj + LM head -- while ours reaches ~55).
#
# Stages:
#   plan     no GPU: print the tier -> setting assignment, predicted TFLOPs,
#            and the wall-clock / GPU-hour estimate
#   sweep    run the assigned jobs (+ the dense reference), 64 docs, bs=1
#   report   score vs TFLOPs table per method (TFLOPs recomputed with each
#            run's MEASURED average prompt length) + csv for the plot
#
# Usage:
#   python run_experiment_6_horizontal_equal_TFLOPS.py plan
#   python run_experiment_6_horizontal_equal_TFLOPS.py sweep --gpus 0 [--only ours]
#   python run_experiment_6_horizontal_equal_TFLOPS.py report
#################################################

import argparse
import itertools
import json
import math
import os

import run_experiment_6_horizontal_equal_quality as EQ

BUDGETS_TF = (60, 120, 250, 500, 1000, 2000)
TOL_FACTOR = 1.35          # accept a match within this factor of the budget
P_NOMINAL = 900            # gsm8k 5-shot avg prompt length (planning only;
                           # the report recomputes with measured P)
FOLDER_DEFAULT = 'results_experiment_6_equal_tflops'

# wall-clock model, calibrated on measured runs (dense ~37.5 s/doc at 4617
# TF/doc -> ~123 TF/s effective; small jobs floor at ~6 s/doc of overhead)
def estimate_s_doc(tflops_doc):
    return 6.0 + tflops_doc / 123.0
# end


'''--------------------- candidate settings per method ---------------------'''

def candidates(method):
    # (suffix, num_blocks, args) -- wider grids than the equal-quality sweep,
    # so every budget tier has a nearby setting where the method can reach it
    if method == 'ours':
        for kr, kp in itertools.product((1, 2, 4, 8, 16, 32, 64, 128, 256), (0, 64, 96)):
            args = {'step_refresh_remainder': kr, 'select_only_in_h': True,
                    'h': EQ.H_OURS, 'path_router': EQ.ROUTER}
            if kp:
                args['step_refresh_remainder_prompt'] = kp
            yield f'kr{kr}_kp{kp}', 1, args
    elif method == 'dllmcache':
        for v, kr, kp in itertools.product((0.05, 0.1, 0.25, 0.5, 0.75, 1.0),
                                           (4, 8, 16, 32, 64), (32, 64, 100)):
            yield f'v{v}_kr{kr}_kp{kp}', 1, {'dllmc_v_rate': v,
                                             'step_refresh_remainder': kr,
                                             'step_refresh_remainder_prompt': kp}
    elif method == 'fastdllm':
        for nb in (1, 2, 4, 8, 16, 32, 64):
            yield f'nb{nb}', nb, {}
    elif method == 'd2cache':
        for k, p in itertools.product((8, 16, 32, 64, 128), (0.0, 0.05, 0.1, 0.2)):
            yield f'k{k}_p{p}', 1, {'d2c_k': k, 'd2c_sigma': 10.0,
                                    'd2c_rollout_p': p, 'd2c_conf_mode': 'live'}
    # end
# end


def assign_tiers():
    '''-> {method: {budget: (tag, num_blocks, args, tflops_pred)}}'''
    assignment = {}
    for method in ('ours', 'dllmcache', 'fastdllm', 'd2cache'):
        rows = []
        for suffix, num_blocks, args in candidates(method):
            tflops = EQ.flops_doc(method, args, num_blocks, P_NOMINAL) / 1e12
            rows.append((suffix, num_blocks, args, tflops))
        # end
        assignment[method] = {}
        used = set()
        for budget in BUDGETS_TF:
            best = min(rows, key=lambda row: abs(math.log(row[3] / budget)))
            ratio = max(best[3] / budget, budget / best[3])
            if ratio > TOL_FACTOR or best[0] in used:
                continue    # unreachable tier, or already covered by a nearer tier
            used.add(best[0])
            tag = f'{method}__T{budget}__{best[0]}'
            assignment[method][budget] = (tag, best[1], best[2], best[3])
        # end
    # end
    return assignment
# end


def jobs_from_assignment(assignment):
    jobs = [('dense__full', 'dense', 'run_llada_semi', 1, {}, 1)]    # anchor
    runner_of = {'ours': 'run_llada_semi_mlp_v2', 'dllmcache': 'run_llada_dllm_cache',
                 'fastdllm': 'run_llada_fastdllm', 'd2cache': 'run_llada_d2cache'}
    for method, tiers in assignment.items():
        for budget, (tag, num_blocks, args, _tflops) in sorted(tiers.items()):
            jobs.append((tag, method, runner_of[method], num_blocks, args, 1))
    return jobs
# end


'''----------------------------- stages -----------------------------'''

def stage_plan():
    assignment = assign_tiers()
    total_min = estimate_s_doc(EQ.flops_doc('dense', {}, 1, P_NOMINAL) / 1e12) * EQ.LIMIT / 60
    print(f'===== plan: matched settings per TFLOPs tier (P~{P_NOMINAL}) =====')
    print(f'  {"method":10s} {"budget":>7s} {"setting":26s} {"pred TF/doc":>12s} {"est min":>8s}')
    print(f'  {"dense":10s} {"--":>7s} {"(reference)":26s} '
          f'{EQ.flops_doc("dense", {}, 1, P_NOMINAL)/1e12:>12.0f} '
          f'{estimate_s_doc(EQ.flops_doc("dense", {}, 1, P_NOMINAL)/1e12)*EQ.LIMIT/60:>8.0f}')
    for method, tiers in assignment.items():
        for budget in BUDGETS_TF:
            if budget not in tiers:
                print(f'  {method:10s} {budget:>7d} {"-- unreachable --":26s}')
                continue
            tag, _nb, _args, tflops = tiers[budget]
            minutes = estimate_s_doc(tflops) * EQ.LIMIT / 60
            total_min += minutes
            print(f'  {method:10s} {budget:>7d} {tag.split("__")[-1]:26s} {tflops:>12.0f} {minutes:>8.0f}')
        # end
    # end
    n_jobs = 1 + sum(len(tiers) for tiers in assignment.values())
    print(f'\n  TOTAL: {n_jobs} jobs, ~{total_min/60:.1f} GPU-hours (single GPU, sequential)')
# end


def stage_report(folder):
    assignment = assign_tiers()
    print(f'\n===== score vs TFLOPs ({folder}) =====')
    path_csv = os.path.join(folder, 'curve.csv')
    rows_csv = ['method,budget_tf,tag,tflops_doc,score,s_doc']

    score_dense = EQ.read_score(folder, 'dense__full')
    report_dense = EQ.read_report(folder, 'dense__full')
    if score_dense is not None and report_dense is not None:
        tflops = EQ.flops_doc('dense', {}, 1, report_dense['len_prompt_avg']) / 1e12
        print(f'  dense reference: score={score_dense:.4f} TFLOPs/doc={tflops:.0f}')
        rows_csv.append(f'dense,,dense__full,{tflops:.1f},{score_dense},{report_dense["duration_per_doc_s"]:.2f}')
    # end

    for method, tiers in assignment.items():
        print(f'\n--- {method} ---')
        for budget in BUDGETS_TF:
            if budget not in tiers:
                continue
            tag, num_blocks, args, _pred = tiers[budget]
            score = EQ.read_score(folder, tag)
            report = EQ.read_report(folder, tag)
            if score is None or report is None:
                print(f'  T{budget:<5d} {tag.split("__")[-1]:26s} (incomplete)')
                continue
            tflops = EQ.flops_doc(method, args, num_blocks, report['len_prompt_avg']) / 1e12
            print(f'  T{budget:<5d} {tag.split("__")[-1]:26s} score={score:.4f} '
                  f'TFLOPs/doc={tflops:>7.0f} s/doc={report["duration_per_doc_s"]:>6.2f}')
            rows_csv.append(f'{method},{budget},{tag},{tflops:.1f},{score},{report["duration_per_doc_s"]:.2f}')
        # end
    # end

    with open(path_csv, 'w') as file:
        file.write('\n'.join(rows_csv) + '\n')
    print(f'\ncurve data -> {path_csv}')
# end


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('plan', 'sweep', 'report'))
    parser.add_argument('--folder', default=FOLDER_DEFAULT)
    parser.add_argument('--gpus', default='0')
    parser.add_argument('--only', default='',
                        help='run only jobs whose tag contains this substring')
    parser.add_argument('--dry', action='store_true')
    config = parser.parse_args()

    if config.stage == 'plan':
        stage_plan()
        return
    # end

    os.makedirs(config.folder, exist_ok=True)
    gpus = [gpu.strip() for gpu in config.gpus.split(',') if gpu.strip()]

    if config.stage == 'sweep':
        jobs = [job for job in jobs_from_assignment(assign_tiers()) if config.only in job[0]]
        EQ.run_jobs(jobs, config.folder, gpus, config.dry)
        print('\nsweep finished; next: python run_experiment_6_horizontal_equal_TFLOPS.py report')
    elif config.stage == 'report':
        stage_report(config.folder)
    # end
# end


if __name__ == '__main__':
    main()
