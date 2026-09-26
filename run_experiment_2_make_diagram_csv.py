#################################################
# Build the two CSVs for the exp-2 trajectory figure (exp_2_trajectory_test.py)
# from results_experiment_2_dose.
#
#   exp2_trajectory_points.csv   long format per (model, sample, position):
#                                teacher/student step + token, teacher_conf,
#                                p_teacher, block_length=32, total_steps=256
#   exp2_fidelity_scores.csv     per (model, sample): fidelity = per-sample
#                                within-block(32) rank correlation of the
#                                FREE-RUN (n=0) unmask order vs the teacher;
#                                score = n=0 correctness (flex); s_dose =
#                                dose-response fraction (extra column, ignored
#                                by the plot but kept for reference)
#
# --max_position N keeps only canvas positions < N (uniform across samples,
# so the plot loader's fixed-grid requirement holds). gsm8k answers occupy
# roughly the first ~100 positions; the tail is post-answer free continuation
# whose tokens/order diverge across runs and dilute the statistics. 96 (three
# 32-blocks) is the recommended answer-region cut; default keeps all 256.
#
# Usage:
#   python run_experiment_2_make_diagram_csv.py [--folder results_experiment_2_dose]
#       [--out_dir .] [--max_position 256]
#################################################

import argparse
import csv
import os

import numpy as np

METHODS = ('ours', 'fastdllm', 'd2cache', 'dllmcache')
MODEL_ORDER = {'fastdllm': 1, 'd2cache': 2, 'dllmcache': 3, 'ours': 4}
BLOCK = 32
LEN_GEN = 256


def within_block_rank(step, conf):
    n, length = step.shape
    tie = -conf
    rank = np.empty_like(step, dtype=float)
    for start in range(0, length, BLOCK):
        blk = slice(start, start + BLOCK)
        order = np.lexsort((tie[:, blk], step[:, blk]), axis=1)
        rank[:, blk] = np.argsort(order, axis=1) / (BLOCK - 1)
    return rank
# end


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', default='results_experiment_2_dose')
    parser.add_argument('--out_dir', default='.')
    parser.add_argument('--max_position', type=int, default=LEN_GEN)
    config = parser.parse_args()

    keep = config.max_position
    assert keep % BLOCK == 0, 'max_position must be a multiple of the 32 window'

    rows_traj, rows_score = [], []
    n_clipped = 0
    for method in METHODS:
        fig = np.load(os.path.join(config.folder, f'fig_ab_{method}.npz'))
        ids_doc = np.load(os.path.join(config.folder, f'student_trace_{method}.npz'))['ids_doc']

        # n=0 correctness per doc from the continuation csv
        n0_correct = {}
        with open(os.path.join(config.folder, f'continuations_{method}.csv')) as file:
            for row in csv.DictReader(file):
                if int(row['n']) == 0:
                    n0_correct[int(row['id_doc'])] = int(row['correct_flex'])
        # end

        # dose fraction per doc (reference column)
        s_dose = {}
        with open(os.path.join(config.folder, f'continuations_{method}.csv')) as file:
            acc = {}
            for row in csv.DictReader(file):
                acc.setdefault(int(row['id_doc']), []).append(int(row['correct_flex']))
            s_dose = {k: sum(v) / len(v) for k, v in acc.items()}
        # end

        t_step = fig['teacher_step'][:, :keep]
        s_step = fig['student_step'][:, :keep]
        t_conf = fig['teacher_conf'][:, :keep]
        rank_t = within_block_rank(t_step, t_conf)
        rank_s = within_block_rank(s_step, t_conf)

        for row_index, id_doc in enumerate(ids_doc):
            corr = float(np.corrcoef(rank_t[row_index], rank_s[row_index])[0, 1])
            if corr < 0:
                n_clipped += 1
            fidelity = min(1.0, max(0.0, corr))
            rows_score.append({
                'model': method, 'model_order': MODEL_ORDER[method],
                'sample_id': f'D{int(id_doc):03d}',
                'fidelity': round(fidelity, 6),
                'score': n0_correct.get(int(id_doc), 0),
                's_dose': round(s_dose.get(int(id_doc), 0.0), 4),
                'teacher_correct': 1,    # continuations ran teacher-correct docs only
            })
            for pos in range(keep):
                rows_traj.append({
                    'model': method, 'sample_id': f'D{int(id_doc):03d}', 'position': pos,
                    'teacher_step': int(fig['teacher_step'][row_index, pos]),
                    'student_step': int(fig['student_step'][row_index, pos]),
                    'teacher_token': int(fig['teacher_tok'][row_index, pos]),
                    'student_token': int(fig['student_tok'][row_index, pos]),
                    'teacher_conf': round(float(fig['teacher_conf'][row_index, pos]), 6),
                    'p_teacher': round(float(fig['p_teacher'][row_index, pos]), 6),
                    'block_length': BLOCK, 'total_steps': LEN_GEN,
                })
            # end
        # end
    # end

    '''dose-response curves: per (method, n) accuracy over teacher-correct docs'''
    import yaml
    preds = {name: entry.get('tflops_pred') for name, entry
             in yaml.safe_load(open('experiment_2_methods.yaml'))['methods'].items()}
    rows_dose = []
    for method in METHODS:
        acc = {}
        with open(os.path.join(config.folder, f'continuations_{method}.csv')) as file:
            for row in csv.DictReader(file):
                acc.setdefault(int(row['n']), []).append(int(row['correct_flex']))
        for n in sorted(acc):
            rows_dose.append({'model': method, 'n': n,
                              'accuracy': round(sum(acc[n]) / len(acc[n]), 4),
                              'n_docs': len(acc[n]),
                              'tflops_doc': preds.get(method, '')})
    # end

    for name, rows in (('exp2_trajectory_points.csv', rows_traj),
                       ('exp2_fidelity_scores.csv', rows_score),
                       ('exp2_dose_curves.csv', rows_dose)):
        path = os.path.join(config.out_dir, name)
        with open(path, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f'{path}: {len(rows)} rows')
    if n_clipped:
        print(f'note: {n_clipped} negative per-sample correlations clipped to 0')
# end


if __name__ == '__main__':
    main()
