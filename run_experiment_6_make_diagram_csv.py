#################################################
# Build the two CSVs for the exp-6 paper diagram (exp_6_horizontal_diagram.py)
# from a pulled results_experiment_horizontal folder.
#
#   exp6_sweep_points.csv   panel (a): one row per completed sweep setting
#                           (all methods + the dense anchor), score x100,
#                           analytic TFLOPs/doc, measured s/doc
#   exp6_batch_points.csv   panel (b): one row per (method, batch size),
#                           TFLOPs/doc held at the chosen setting's value
#                           (per-doc compute is batch-invariant by design;
#                           the wall-clock column carries the scaling)
#
# Usage:
#   python run_experiment_6_make_diagram_csv.py [--folder results_experiment_horizontal] [--out_dir .]
#################################################

import argparse
import csv
import json
import os

import run_experiment_6_horizontal_equal_quality as EQ

DISPLAY = {'dense': 'full denoising', 'ours': 'ours', 'dllmcache': 'dllm-cache',
           'fastdllm': 'fast-dllm', 'd2cache': 'd2cache'}
FIELDS = ['model', 'setting', 'budget_index', 'compute_tflops', 'wall_clock_sec',
          'performance_avg', 'batch_samples']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', default='results_experiment_horizontal')
    parser.add_argument('--out_dir', default='.')
    config = parser.parse_args()

    '''panel (a): sweep points'''
    rows_sweep = []
    for tag, method, _runner, num_blocks, args in EQ.jobs_sweep():
        score = EQ.read_score(config.folder, tag)
        report = EQ.read_report(config.folder, tag)
        if score is None or report is None:
            continue
        tflops = EQ.flops_doc(method, args, num_blocks, report['len_prompt_avg']) / 1e12
        rows_sweep.append({
            'model': DISPLAY[method], 'setting': tag.split('__', 1)[1],
            'compute_tflops': round(tflops, 1),
            'wall_clock_sec': round(report['duration_per_doc_s'], 2),
            'performance_avg': round(100 * score, 2), 'batch_samples': 1,
        })
    # end
    # budget_index: rank by TFLOPs within each model (the polyline order)
    for model in {row['model'] for row in rows_sweep}:
        rows_model = sorted((row for row in rows_sweep if row['model'] == model),
                            key=lambda row: row['compute_tflops'])
        for index, row in enumerate(rows_model, start=1):
            row['budget_index'] = index
    rows_sweep.sort(key=lambda row: (row['model'], row['budget_index']))

    '''panel (b): batch points, at each method's chosen setting'''
    chosen = json.load(open(os.path.join(config.folder, 'chosen.json')))
    rows_batch = []
    for method in ('ours', 'dllmcache', 'fastdllm', 'd2cache'):
        entry = chosen.get(method)
        if entry is None:
            continue
        for index, bs in enumerate(EQ.BATCH_SIZES, start=1):
            tag = f'batch__{method}__bs{bs}'
            score = EQ.read_score(config.folder, tag)
            report = EQ.read_report(config.folder, tag)
            if score is None or report is None:
                continue
            rows_batch.append({
                'model': DISPLAY[method], 'setting': f'bs{bs}', 'budget_index': index,
                'compute_tflops': entry['tflops_doc'],    # per-doc compute, bs-invariant
                'wall_clock_sec': round(report['duration_per_doc_s'], 2),
                'performance_avg': round(100 * score, 2), 'batch_samples': bs,
            })
        # end
    # end

    for name, rows in (('exp6_sweep_points.csv', rows_sweep),
                       ('exp6_batch_points.csv', rows_batch)):
        path = os.path.join(config.out_dir, name)
        with open(path, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        print(f'{path}: {len(rows)} rows')
    # end
# end


if __name__ == '__main__':
    main()
