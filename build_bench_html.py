"""Build a single static HTML summary from an e2e benchmark results folder.

Works on any folder produced by the run_bench_*.bash sweeps (each run leaves
<router>__<task>__runner.json next to an lm_eval --output_path directory named
<router>__<task>/): discovers runs from the runner reports, pairs each with
the newest lm_eval results_*.json under its output dir, and renders:

  - a summary matrix: routers x tasks, one headline metric per cell, best
    per task highlighted (headline picked by METRIC_PREFERENCE, e.g. gsm8k
    exact_match,strict-match / bbh exact_match,get-answer)
  - a detail table per run: every lm_eval metric (+- stderr), sample count,
    done rate, total / per-sample wall clock from the runner report

Usage:
  python build_bench_html.py                                # results_bench_confmargin
  python build_bench_html.py --results results_bench_router --out router.html

Stdlib only; open the output in any browser. Column headers sort.
"""

import argparse
import datetime
import glob
import html
import json
import os

METRIC_PREFERENCE = [
    'exact_match,strict-match',
    'exact_match,get-answer',
    'exact_match,flexible-extract',
    'exact_match,none',
    'acc,none',
    'acc_norm,none',
    'pass@1',
    'pass_at_1,none',
    'bleu_acc,none',
]

CSS = """
body { font-family: -apple-system, 'Segoe UI', Roboto, sans-serif; margin: 24px auto;
       max-width: 1500px; padding: 0 16px; background: #fafafa; color: #1a1a1a; }
h1 { font-size: 22px; } h2 { font-size: 17px; margin-top: 34px; }
.meta { color: #666; font-size: 13px; }
table { border-collapse: collapse; width: 100%; font-size: 13px; background: #fff;
        box-shadow: 0 1px 3px rgba(0,0,0,.08); }
th, td { border: 1px solid #e3e3e3; padding: 5px 8px; text-align: left; white-space: nowrap; }
th { background: #f0f0f3; cursor: pointer; user-select: none; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
td.best { background: #e6f4e6; font-weight: 600; }
td.miss { color: #bbb; }
.metricname { color: #888; font-size: 11px; }
.wrap { overflow-x: auto; }
"""

JS = """
document.querySelectorAll('th').forEach(function (th) {
  th.addEventListener('click', function () {
    var table = th.closest('table'), tbody = table.tBodies[0];
    var idx = Array.prototype.indexOf.call(th.parentNode.children, th);
    var dir = th.dataset.dir === 'asc' ? -1 : 1; th.dataset.dir = dir === 1 ? 'asc' : 'desc';
    Array.prototype.slice.call(tbody.rows).sort(function (a, b) {
      var x = a.cells[idx].dataset.v || a.cells[idx].textContent;
      var y = b.cells[idx].dataset.v || b.cells[idx].textContent;
      var nx = parseFloat(x), ny = parseFloat(y);
      if (!isNaN(nx) && !isNaN(ny)) return (nx - ny) * dir;
      return x.localeCompare(y) * dir;
    }).forEach(function (row) { tbody.appendChild(row); });
  });
});
"""


def esc(value):
    return html.escape('' if value is None else str(value))
# end


def discover_runs(folder_results):
    """[(router, task, path_runner_json)] from <router>__<task>__runner.json files."""
    runs = []
    for path in sorted(glob.glob(os.path.join(folder_results, '*__runner.json'))):
        tag = os.path.basename(path)[:-len('__runner.json')]
        router, _, task = tag.rpartition('__')
        runs.append((router, task, path))
    return runs
# end


def load_lm_eval_results(folder_results, router, task):
    """Newest results_*.json under the run's lm_eval output dir; None if absent."""
    tag = f'{router}__{task}'
    candidates = glob.glob(os.path.join(folder_results, tag, '**', 'results_*.json'),
                           recursive=True)
    candidates += glob.glob(os.path.join(folder_results, tag + '_*', '**', 'results_*.json'),
                            recursive=True)
    if not candidates:
        return None
    path = max(candidates, key=os.path.getmtime)
    with open(path, 'r', encoding='utf-8') as file:
        return json.load(file)
# end


def metrics_for_task(payload, task):
    """The task's (or task group's) metric dict {metric_key: value}; numeric only.
    Falls back to the unweighted mean over the group's subtasks when lm_eval
    wrote no group aggregate."""
    if not payload:
        return {}
    results = payload.get('results', {})
    entry = results.get(task)
    if entry:
        return {k: v for k, v in entry.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)}
    # group without aggregate row: average the subtasks
    names_sub = payload.get('group_subtasks', {}).get(task) \
        or [name for name in results if name.startswith(task + '_')]
    if not names_sub:
        return {}
    accum = {}
    for name in names_sub:
        for k, v in results.get(name, {}).items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                accum.setdefault(k, []).append(v)
    return {k: sum(vs) / len(vs) for k, vs in accum.items()}
# end


def headline_metric(metrics):
    """(key, value) for the summary matrix cell."""
    for key in METRIC_PREFERENCE:
        if key in metrics:
            return key, metrics[key]
    for key, value in metrics.items():
        if not key.endswith('_stderr') and ',' in key or key in ('acc', 'exact_match'):
            if '_stderr' not in key:
                return key, value
    for key, value in metrics.items():
        if '_stderr' not in key:
            return key, value
    return None, None
# end


def load_runner(path):
    with open(path, 'r', encoding='utf-8') as file:
        payload = json.load(file)
    rows = payload.get('rows', [])
    n = len(rows) or payload.get('num_samples', 0)
    total_s = payload.get('duration_total_s', 0.0)
    done = sum(1 for r in rows if r.get('has_done'))
    return {
        'num_samples': n,
        'duration_total_s': total_s,
        's_per_sample': (total_s / n) if n else None,
        'done_rate': (done / n) if rows else None,
    }
# end


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', default='results_bench_confmargin')
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    runs = discover_runs(args.results)
    assert runs, f'no *__runner.json under {args.results}'

    routers = sorted({r for r, _, _ in runs})
    tasks = sorted({t for _, t, _ in runs})

    data = {}    # (router, task) -> {'metrics':..., 'headline':(k,v), 'runner':...}
    for router, task, path_runner in runs:
        payload = load_lm_eval_results(args.results, router, task)
        metrics = metrics_for_task(payload, task)
        data[(router, task)] = {
            'metrics': metrics,
            'headline': headline_metric(metrics),
            'runner': load_runner(path_runner),
            'has_lm_eval': payload is not None,
        }
    # end

    # ---- summary matrix ----
    best = {}    # task -> best headline value
    for task in tasks:
        values = [data[(r, task)]['headline'][1] for r in routers
                  if (r, task) in data and data[(r, task)]['headline'][1] is not None]
        best[task] = max(values) if values else None
    # end

    rows_matrix = []
    for router in routers:
        cells = [f'<td>{esc(router)}</td>']
        for task in tasks:
            run = data.get((router, task))
            if not run or run['headline'][1] is None:
                note = 'no lm_eval json' if run and not run['has_lm_eval'] else 'missing'
                cells.append(f'<td class="miss" data-v="-1">{note}</td>')
                continue
            key, value = run['headline']
            hot = ' best' if best[task] is not None and value == best[task] else ''
            cells.append(f'<td class="num{hot}" data-v="{value}">{value:.4f}'
                         f'<br><span class="metricname">{esc(key)}</span></td>')
        # end
        rows_matrix.append('<tr>' + ''.join(cells) + '</tr>')
    # end
    head_matrix = '<th>router</th>' + ''.join(f'<th>{esc(t)}</th>' for t in tasks)

    # ---- detail table ----
    rows_detail = []
    for (router, task) in sorted(data):
        run = data[(router, task)]
        runner = run['runner']
        parts_metrics = []
        for key in sorted(run['metrics']):
            if key.endswith('_stderr') or ('_stderr,' in key):
                continue
            value = run['metrics'][key]
            key_err = key.replace(',', '_stderr,', 1)
            err = run['metrics'].get(key_err)
            text = f'{key}={value:.4f}'
            if isinstance(err, (int, float)):
                text += f' &plusmn;{err:.4f}'
            parts_metrics.append(text)
        # end
        s_ps = runner['s_per_sample']
        done = runner['done_rate']
        rows_detail.append(
            '<tr>'
            f'<td>{esc(router)}</td><td>{esc(task)}</td>'
            f'<td class="num" data-v="{runner["num_samples"]}">{runner["num_samples"]}</td>'
            f'<td class="num" data-v="{done if done is not None else -1}">'
            f'{f"{done:.2f}" if done is not None else "--"}</td>'
            f'<td class="num" data-v="{runner["duration_total_s"]}">{runner["duration_total_s"]:.0f}</td>'
            f'<td class="num" data-v="{s_ps if s_ps is not None else -1}">'
            f'{f"{s_ps:.2f}" if s_ps is not None else "--"}</td>'
            f'<td style="white-space: normal">{" &middot; ".join(parts_metrics) or "--"}</td>'
            '</tr>')
    # end

    parts = [
        f'<style>{CSS}</style>',
        f'<h1>e2e benchmark summary -- {esc(os.path.basename(os.path.abspath(args.results)))}</h1>',
        f'<p class="meta">results: {esc(args.results)} &middot; {len(runs)} runs &middot; '
        f'generated {datetime.datetime.now():%Y-%m-%d %H:%M}</p>',
        '<h2>Summary (headline metric per task; best per column highlighted)</h2>',
        f'<div class="wrap"><table><thead><tr>{head_matrix}</tr></thead>'
        f'<tbody>{"".join(rows_matrix)}</tbody></table></div>',
        '<h2>Per-run detail</h2>',
        '<div class="wrap"><table><thead><tr><th>router</th><th>task</th><th>n</th>'
        '<th>done rate</th><th>total s</th><th>s/sample</th><th>all metrics</th></tr></thead>'
        f'<tbody>{"".join(rows_detail)}</tbody></table></div>',
        f'<script>{JS}</script>',
    ]

    path_out = args.out or f'bench_summary_{os.path.basename(os.path.abspath(args.results))}.html'
    with open(path_out, 'w', encoding='utf-8') as file:
        file.write('<!doctype html><meta charset="utf-8">'
                   f'<title>{esc(os.path.basename(args.results))}</title>' + ''.join(parts))
    print(f'wrote {path_out} ({os.path.getsize(path_out)} bytes)')
# end


if __name__ == '__main__':
    main()
# end
