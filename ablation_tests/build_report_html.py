"""Build a single static HTML page from a fast-ablation report JSON.

Reads the stage-keyed report written by ablation_test_fast.py (every run's
full config + metrics) plus, when present, the routers_final summary (winner
chain, staleness tracks, final bundles), and renders one reviewable page:
stage-A..E tables with per-dataset recalls, eligibility and winner markers,
the four staleness-track leaderboards, and the final bundle scores.

Usage:
  python ablation_tests/build_report_html.py --thread llada_base
  python ablation_tests/build_report_html.py --report my_report.json --out my.html

Stdlib only; open the output file in any browser. Column headers sort.
"""

import argparse
import datetime
import html
import json
import os
import re

PATTERN_SUMM = re.compile(r'^\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*\(n=(\d+)\)\s*$')

STAGES = [
    ('fast_feat', 'Stage A -- features'),
    ('fast_norm', 'Stage B -- normalization'),
    ('fast_loss', 'Stage C -- loss'),
    ('fast_arch', 'Stage D -- architecture'),
    ('fast_h',    'Stage E -- horizon'),
]

TRACK_LABELS = {
    'fresh':  'with conf/margin (fresh -- offline ceiling, leaks at deployment)',
    'aged':   'with aged conf/margin (random-age augmentation)',
    'policy': 'with policy conf/margin (deterministic refresh-clock aging)',
    'clean':  'no conf/margin',
}

NORMALIZATIONS_DEPLOYABLE = ('rank', 'softmax_attn')


def track_of(features):
    if any(f in ('conf', 'margin') for f in features):
        return 'fresh'
    if any(f in ('conf_aged', 'margin_aged') for f in features):
        return 'aged'
    if any(f in ('conf_policy', 'margin_policy') for f in features):
        return 'policy'
    return 'clean'
# end


def parse_summ(value):
    if isinstance(value, str):
        match = PATTERN_SUMM.fullmatch(value)
        if match:
            return float(match.group(1)), int(match.group(2))
    return None
# end


def metric_value(group, key):
    parsed = parse_summ((group or {}).get(key))
    return parsed[0] if parsed else None
# end


def prefixed_metric(group, prefix):
    """First 'ndgc@*'/'pr_auc@*'-style key of the group: (key, value)."""
    for key in sorted(group or {}):
        if key.startswith(prefix):
            parsed = parse_summ(group[key])
            if parsed:
                return key, parsed[0]
    return None, None
# end


def split_tag(name):
    """'norm-rank@full_aged+softmax_attn' -> ('norm-rank', 'full_aged+softmax_attn')"""
    base, _, tag = name.partition('@')
    return base, tag
# end


def ineligible_reason(stage, record):
    config = record.get('config', {})
    features = config.get('features', [])
    if stage == 'fast_feat' and any(f in ('conf', 'margin') for f in features):
        return 'fresh conf/margin (leak)'
    if stage == 'fast_norm' and config.get('normalization') not in NORMALIZATIONS_DEPLOYABLE:
        return 'normalization not deployable'
    if stage == 'fast_arch' and str(config.get('router', '')).startswith('mockup'):
        return 'mockup floor'
    return None
# end


def esc(value):
    return html.escape('' if value is None else str(value))
# end


def fmt(value):
    return '--' if value is None else f'{value:.3f}'
# end


CSS = """
body { font-family: -apple-system, 'Segoe UI', Roboto, sans-serif; margin: 24px auto;
       max-width: 1500px; padding: 0 16px; background: #fafafa; color: #1a1a1a; }
h1 { font-size: 22px; } h2 { font-size: 17px; margin-top: 34px; }
.meta { color: #666; font-size: 13px; }
table { border-collapse: collapse; width: 100%; font-size: 13px; background: #fff;
        box-shadow: 0 1px 3px rgba(0,0,0,.08); }
th, td { border: 1px solid #e3e3e3; padding: 5px 8px; text-align: left; white-space: nowrap; }
th { background: #f0f0f3; cursor: pointer; user-select: none; position: sticky; top: 0; }
tr.winner td { background: #e6f4e6; font-weight: 600; }
tr.runnerup td { background: #f3f9f3; }
tr.ineligible td { color: #999; }
tr.error td { background: #fbecec; color: #a33; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
.best { background: #fff3cf !important; }
.tag { color: #888; font-size: 11px; }
.feat { font-size: 12px; color: #444; white-space: normal; max-width: 300px; }
details { margin: 6px 0 14px; } summary { cursor: pointer; color: #555; font-size: 13px; }
.badge { display: inline-block; padding: 1px 7px; border-radius: 9px; font-size: 11px;
         background: #e8e8ee; margin-left: 6px; }
.tracks td:first-child { white-space: normal; max-width: 420px; }
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


def render_stage(stage, title, records, datasets_all, winner_name):
    rows = []
    # rank rows for winner/runner-up highlighting among eligible finished runs
    scored = [(r, metric_value(r.get('metrics', {}).get('all'), 'recall@5')) for r in records]
    eligible_sorted = sorted(
        [(r, v) for r, v in scored if v is not None
         and not ineligible_reason(stage, r) and not r.get('error')],
        key=lambda p: p[1], reverse=True)
    name_first = eligible_sorted[0][0]['name'] if eligible_sorted else None
    name_second = eligible_sorted[1][0]['name'] if len(eligible_sorted) > 1 else None
    best_r5 = max((v for _, v in scored if v is not None), default=None)

    for record, r5 in scored:
        config = record.get('config', {})
        metrics = record.get('metrics') or {}
        group_all = metrics.get('all', {})
        base, tag = split_tag(record['name'])
        reason = ineligible_reason(stage, record)
        classes = []
        if record.get('error'):
            classes.append('error')
        elif reason:
            classes.append('ineligible')
        if record['name'] == (winner_name or name_first):
            classes.append('winner')
        elif record['name'] == name_second:
            classes.append('runnerup')

        key_ndgc, v_ndgc = prefixed_metric(group_all, 'ndgc@')
        key_auc, v_auc = prefixed_metric(group_all, 'pr_auc@')

        cells = [
            f'<td>{esc(base)}' + (f' <span class="tag">@{esc(tag)}</span>' if tag else '') + '</td>',
            f'<td class="feat">{esc(", ".join(config.get("features", [])))}</td>',
            f'<td>{esc(config.get("normalization"))}</td>',
            f'<td>{esc(config.get("loss"))}</td>',
            f'<td>{esc(config.get("router"))}</td>',
            f'<td class="num">{esc(config.get("h"))}</td>',
            f'<td class="num">{esc(config.get("num_epochs"))}</td>',
        ]
        for key in ('recall@3', 'recall@5', 'recall@10'):
            value = metric_value(group_all, key)
            hot = ' best' if key == 'recall@5' and value is not None and value == best_r5 else ''
            cells.append(f'<td class="num{hot}" data-v="{value if value is not None else -1}">{fmt(value)}</td>')
        cells.append(f'<td class="num" data-v="{v_ndgc if v_ndgc is not None else -1}" '
                     f'title="{esc(key_ndgc)}">{fmt(v_ndgc)}</td>')
        cells.append(f'<td class="num" data-v="{v_auc if v_auc is not None else -1}" '
                     f'title="{esc(key_auc)}">{fmt(v_auc)}</td>')
        for task in datasets_all:
            value = metric_value(metrics.get(f'ds_{task}'), 'recall@5')
            cells.append(f'<td class="num" data-v="{value if value is not None else -1}">{fmt(value)}</td>')
        if record.get('error'):
            status = 'ERROR'
        elif reason:
            status = f'ineligible: {reason}'
        elif record['name'] == (winner_name or name_first):
            status = 'WINNER'
        elif record['name'] == name_second:
            status = 'runner-up'
        else:
            status = ''
        cells.append(f'<td>{esc(status)}</td>')
        rows.append(f'<tr class="{" ".join(classes)}">{"".join(cells)}</tr>')
    # end

    heads = (['run', 'features', 'norm', 'loss', 'router', 'h', 'ep',
              'recall@3', 'recall@5', 'recall@10', 'ndgc', 'pr_auc']
             + [f'r@5 {t}' for t in datasets_all] + ['status'])
    head_html = ''.join(f'<th>{esc(h)}</th>' for h in heads)
    return (f'<h2 id="{stage}">{esc(title)} <span class="badge">{len(records)} runs</span></h2>'
            f'<div class="wrap"><table><thead><tr>{head_html}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')
# end


def render_tracks(records_feat, tracks_saved):
    rows = []
    scored = {r['name']: metric_value(r.get('metrics', {}).get('all'), 'recall@5')
              for r in records_feat}
    features_of = {r['name']: r.get('config', {}).get('features', []) for r in records_feat}
    for track in ('clean', 'policy', 'aged', 'fresh'):
        ranked = sorted(((n, v) for n, v in scored.items()
                         if track_of(features_of.get(n, [])) == track and v is not None),
                        key=lambda p: p[1], reverse=True)
        for rank, (name, value) in enumerate(ranked[:2], 1):
            label = TRACK_LABELS[track] if rank == 1 else ''
            marker = 'winner' if rank == 1 else 'runnerup'
            rows.append(f'<tr class="{marker}"><td>{esc(label)}</td><td>{esc(name)}</td>'
                        f'<td class="feat">{esc(", ".join(features_of.get(name, [])))}</td>'
                        f'<td class="num">{fmt(value)}</td><td>{"winner" if rank == 1 else "runner-up"}</td></tr>')
        if not ranked:
            rows.append(f'<tr><td>{esc(TRACK_LABELS[track])}</td><td colspan="4">no finished runs</td></tr>')
    # end
    return ('<h2 id="tracks">Staleness tracks (winner / runner-up per track)</h2>'
            '<div class="wrap"><table class="tracks"><thead><tr><th>track</th><th>anchor</th>'
            '<th>features</th><th>recall@5</th><th></th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')
# end


def render_bundles(summary):
    bundles = (summary or {}).get('bundles', {})
    if not bundles:
        return '<h2>Final bundles</h2><p class="meta">not trained yet (stage F pending)</p>'
    tasks = sorted({t for b in bundles.values() for t in b.get('recall_eval_no_ifeval', {})})
    head = ('<tr><th>bundle</th><th>features</th>'
            + ''.join(f'<th>r@h {esc(t)}</th>' for t in tasks) + '<th>path</th></tr>')
    rows = []
    for name, bundle in bundles.items():
        cells = [f'<td>{esc(name)}</td>',
                 f'<td class="feat">{esc(", ".join(bundle.get("features", [])))}</td>']
        for task in tasks:
            parsed = parse_summ(bundle.get('recall_eval_no_ifeval', {}).get(task))
            cells.append(f'<td class="num">{fmt(parsed[0] if parsed else None)}</td>')
        cells.append(f'<td class="meta">{esc(bundle.get("path"))}</td>')
        rows.append(f'<tr>{"".join(cells)}</tr>')
    return ('<h2 id="final">Final bundles (holdout recall on eval group)</h2>'
            f'<div class="wrap"><table><thead>{head}</thead><tbody>{"".join(rows)}</tbody></table></div>')
# end


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--thread', default=os.environ.get('THREAD', 'llada_base'))
    parser.add_argument('--report', default=None)
    parser.add_argument('--summary', default=None)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    path_report = args.report
    if path_report is None:
        path_report = f'ablation_test_report_fast_{args.thread}.json'
        if not os.path.exists(path_report) and os.path.exists('ablation_test_report_fast.json'):
            path_report = 'ablation_test_report_fast.json'
    with open(path_report, 'r', encoding='utf-8') as file:
        report = json.load(file)

    path_summary = args.summary or os.path.join('routers_final', f'{args.thread}__ablation_summary.json')
    summary = None
    if os.path.exists(path_summary):
        with open(path_summary, 'r', encoding='utf-8') as file:
            summary = json.load(file)

    datasets_all = sorted({task for records in report.values() if isinstance(records, list)
                           for r in records for task in r.get('config', {}).get('datasets', [])})

    parts = [f'<style>{CSS}</style>',
             f'<h1>Fast ablation -- {esc(args.thread)}</h1>',
             f'<p class="meta">report: {esc(path_report)} &middot; summary: '
             f'{esc(path_summary if summary else "none")} &middot; generated '
             f'{datetime.datetime.now():%Y-%m-%d %H:%M}</p>',
             '<p class="meta">jump to: '
             + ' &middot; '.join(f'<a href="#{s}">{t.split(" -- ")[0]}</a>' for s, t in STAGES)
             + ' &middot; <a href="#tracks">tracks</a> &middot; <a href="#final">final</a></p>']

    if summary:
        parts.append('<h2>Winner chain</h2><details open><summary>winner recipe + stage winners'
                     '</summary><pre>'
                     + esc(json.dumps({k: summary.get(k) for k in ('winner', 'stage_winners')
                                       if k in summary}, indent=2))
                     + '</pre></details>')

    records_feat = report.get('fast_feat', [])
    parts.append(render_tracks(records_feat, (summary or {}).get('tracks')))

    for stage, title in STAGES:
        records = report.get(stage, [])
        if not records:
            parts.append(f'<h2 id="{stage}">{esc(title)}</h2><p class="meta">no runs yet</p>')
            continue
        winner_name = None    # highlight computed from the data; summary only names stage picks
        parts.append(render_stage(stage, title, records, datasets_all, winner_name))
    # end

    # extra stages (side experiments like age_channel) render generically
    names_fixed = {stage for stage, _ in STAGES}
    for stage, records in report.items():
        if stage in names_fixed or not isinstance(records, list) or not records:
            continue
        parts.append(render_stage(stage, f'Extra -- {stage}', records, datasets_all, None))
    # end

    parts.append(render_bundles(summary))
    parts.append(f'<script>{JS}</script>')

    path_out = args.out or f'ablation_report_{args.thread}.html'
    with open(path_out, 'w', encoding='utf-8') as file:
        file.write('<!doctype html><meta charset="utf-8">'
                   f'<title>fast ablation {html.escape(args.thread)}</title>' + ''.join(parts))
    print(f'wrote {path_out} ({os.path.getsize(path_out)} bytes)')
# end


if __name__ == '__main__':
    main()
# end
