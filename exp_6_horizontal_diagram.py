from __future__ import annotations

import argparse
import csv
import html
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = ROOT / "exp6_sweep_points.csv"
DEFAULT_BATCH_PATH = ROOT / "exp6_batch_points.csv"

FIGURE_DIR = ROOT / "figures"
PDF_DIR = ROOT / "output" / "pdf"

PERFORMANCE_SVG = FIGURE_DIR / "scheme-c-combined-resource.svg"
BATCH_SVG = FIGURE_DIR / "scheme-d-combined-batch.svg"
PAIR_SVG = FIGURE_DIR / "model-resource-two-panel.svg"

PERFORMANCE_PDF = PDF_DIR / "scheme-c-combined-resource.pdf"
BATCH_PDF = PDF_DIR / "scheme-d-combined-batch.pdf"
PAIR_PDF = PDF_DIR / "model-resource-two-panel.pdf"

# Native size is the intended paper inclusion size. Two panels plus a 12 pt
# gutter make a 7.17 in wide, double-column figure.
PANEL_W = 252.0
PANEL_H = 188.0
PANEL_GAP = 12.0
PAIR_W = PANEL_W * 2 + PANEL_GAP

COLORS = {
    "ours": "#287C78",
    "fast-dllm": "#B98A2E",
    "d2cache": "#D96C4F",
    "dllm-cache": "#8073AC",
    "full denoising": "#6F6A61",
    "grid": "#DDD8CE",
    "axis": "#6F6A61",
    "text": "#242424",
    "muted": "#625E56",
    "paper": "#FFFFFF",
}

MODEL_ORDER = ("ours", "fast-dllm", "d2cache", "dllm-cache", "full denoising")
REQUIRED_COLUMNS = {
    "model",
    "setting",
    "budget_index",
    "compute_tflops",
    "wall_clock_sec",
    "performance_avg",
    "batch_samples",
}


@dataclass(frozen=True)
class ChartSpec:
    y_key: str
    y_label: str
    aria_label: str
    wall_only: bool = False    # batch panel: per-doc TFLOPs are batch-invariant,
                               # so only wall clock is plotted (bottom axis)
    compute_only: bool = False    # performance panel: bs=1 wall clock is
                                  # overhead-dominated (panel b tells that story),
                                  # so only TFLOPs is plotted, as a fitted
                                  # log-linear curve with raw-setting markers


PERFORMANCE_SPEC = ChartSpec(
    y_key="performance_avg",
    y_label="Performance (Average score)",
    aria_label="TFLOPs versus average performance",
    compute_only=True,
)

BATCH_SPEC = ChartSpec(
    y_key="batch_samples",
    y_label="Batch (samples per batch)",
    aria_label="Wall clock versus samples per batch",
    wall_only=True,
)


def make_demo_data(seed: int = 23) -> list[dict[str, object]]:
    """Create deterministic placeholders only; replace them with measured data."""
    rng = random.Random(seed)
    budgets = [12, 18, 24, 32, 42, 54, 68, 84, 102, 122, 146, 174, 206, 242]
    batch_sizes = [8, 16, 24, 32, 48, 64, 80, 96, 128, 160, 192, 224, 256, 320]
    rows: list[dict[str, object]] = []
    for model in ["other model", "our model"]:
        for index, (budget, batch) in enumerate(zip(budgets, batch_sizes), start=1):
            if model == "our model":
                compute = budget * rng.uniform(0.91, 1.01)
                seconds = budget * rng.uniform(0.99, 1.10) + rng.uniform(-1.2, 1.2)
                score = 58 + 30 * (1 - math.exp(-budget / 58)) + rng.uniform(-0.8, 0.8)
            else:
                compute = budget * rng.uniform(1.00, 1.13)
                seconds = budget * rng.uniform(1.16, 1.31) + rng.uniform(-1.5, 1.5)
                score = 55 + 24 * (1 - math.exp(-budget / 68)) + rng.uniform(-0.9, 0.9)
            rows.append(
                {
                    "model": model,
                    "setting": f"S{index}",
                    "budget_index": index,
                    "compute_tflops": round(compute, 2),
                    "wall_clock_sec": round(seconds, 2),
                    "performance_avg": round(score, 2),
                    "batch_samples": batch,
                }
            )
    return rows


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        columns = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
        rows: list[dict[str, object]] = list(reader)

    if not rows:
        raise ValueError(f"{path} has no data rows")

    for line_number, row in enumerate(rows, start=2):
        if row["model"] not in MODEL_ORDER:
            raise ValueError(
                f"{path}:{line_number}: model must be one of {', '.join(MODEL_ORDER)}"
            )
        for key in [
            "budget_index",
            "compute_tflops",
            "wall_clock_sec",
            "performance_avg",
            "batch_samples",
        ]:
            try:
                float(row[key])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: {key} must be numeric") from exc

    present_models = {str(row["model"]) for row in rows}
    if "ours" not in present_models:
        raise ValueError(f"{path} must contain the 'ours' model rows")
    return rows


def fit_log_linear(points: Sequence[tuple[float, float]]):
    """Least-squares y = a + b*ln(x); None when under-determined."""
    if len(points) < 3:
        return None
    log_x = [math.log(x) for x, _y in points]
    ys = [y for _x, y in points]
    mean_x = sum(log_x) / len(log_x)
    mean_y = sum(ys) / len(ys)
    var_x = sum((v - mean_x) ** 2 for v in log_x)
    if var_x < 1e-12:
        return None
    slope = sum((vx - mean_x) * (vy - mean_y) for vx, vy in zip(log_x, ys)) / var_x
    intercept = mean_y - slope * mean_x
    return lambda x: intercept + slope * math.log(x)


def sample_fit(points: Sequence[tuple[float, float]], y_domain: tuple[float, float]):
    """Fitted curve sampled log-spaced over the model's own x-range, clamped
    to the y domain; None -> caller falls back to the raw polyline."""
    fit = fit_log_linear(points)
    if fit is None:
        return None
    lo = min(x for x, _y in points)
    hi = max(x for x, _y in points)
    samples = []
    for index in range(41):
        x = lo * (hi / lo) ** (index / 40) if lo > 0 else lo + (hi - lo) * index / 40
        y = min(max(fit(x), y_domain[0]), y_domain[1])
        samples.append((x, y))
    return samples


def grouped_rows(rows: Sequence[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    groups = {model: [] for model in MODEL_ORDER}
    for row in rows:
        groups[str(row["model"])].append(row)
    for group in groups.values():
        group.sort(key=lambda row: float(row["budget_index"]))
    return groups


def scale(value: float, domain: tuple[float, float], range_: tuple[float, float]) -> float:
    lo, hi = domain
    start, end = range_
    if hi == lo:
        return (start + end) / 2
    return start + (value - lo) / (hi - lo) * (end - start)


def nice_step(raw_step: float, multipliers: Sequence[float]) -> float:
    if raw_step <= 0:
        return 1.0
    exponent = 10 ** math.floor(math.log10(raw_step))
    fraction = raw_step / exponent
    for multiplier in multipliers:
        if fraction <= multiplier:
            return multiplier * exponent
    return 10 * exponent


def zero_axis(max_value: float, target_intervals: int = 4) -> tuple[tuple[float, float], list[float]]:
    step = nice_step(max_value / target_intervals, (1, 2, 5, 10))
    upper = max(step, math.ceil(max_value / step) * step)
    count = int(round(upper / step))
    return (0.0, upper), [index * step for index in range(count + 1)]


def y_axis(rows: Sequence[dict[str, object]], key: str) -> tuple[tuple[float, float], list[float]]:
    values = [float(row[key]) for row in rows]
    lo, hi = min(values), max(values)
    if key == "batch_samples":
        step = nice_step(hi / 4, (1, 2, 2.5, 4, 5, 8, 10))
        upper = max(step, math.ceil(hi / step) * step)
        count = int(round(upper / step))
        return (0.0, upper), [index * step for index in range(count + 1)]

    span = hi - lo or 1.0
    padded_lo = lo - span * 0.04
    padded_hi = hi + span * 0.04
    step = 5.0 if span <= 40 else nice_step(span / 5, (1, 2, 5, 10))
    domain_lo = math.floor(padded_lo / step) * step
    domain_hi = math.ceil(padded_hi / step) * step
    label_step = step * 2 if (domain_hi - domain_lo) / step > 5 else step
    ticks: list[float] = []
    value = domain_lo
    while value <= domain_hi + label_step * 0.01:
        ticks.append(value)
        value += label_step
    if not math.isclose(ticks[-1], domain_hi):
        ticks.append(domain_hi)
    return (domain_lo, domain_hi), ticks


def number_label(value: float) -> str:
    if math.isclose(value, round(value), abs_tol=1e-8):
        return str(int(round(value)))
    if abs(value) >= 10:
        return f"{value:.1f}".rstrip("0").rstrip(".")
    return f"{value:.2f}".rstrip("0").rstrip(".")


def chart_axes(rows: Sequence[dict[str, object]], spec: ChartSpec) -> dict[str, object]:
    # the dense anchor is drawn as a horizontal reference line, so its huge
    # TFLOPs value must not stretch the x domains
    rows_x = [row for row in rows if row["model"] != "full denoising"] or list(rows)
    compute_max = max(float(row["compute_tflops"]) for row in rows_x)
    wall_max = max(float(row["wall_clock_sec"]) for row in rows_x)
    compute_domain, compute_ticks = zero_axis(compute_max)
    wall_domain, wall_ticks = zero_axis(wall_max)
    y_domain, y_ticks = y_axis(rows, spec.y_key)
    return {
        "compute_domain": compute_domain,
        "compute_ticks": compute_ticks,
        "wall_domain": wall_domain,
        "wall_ticks": wall_ticks,
        "y_domain": y_domain,
        "y_ticks": y_ticks,
    }


def svg_text(
    x: float,
    y: float,
    label: str,
    size: float,
    anchor: str = "start",
    weight: int = 400,
) -> str:
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-family="Helvetica, Arial, sans-serif" '
        f'font-size="{size:.2f}" font-weight="{weight}" text-anchor="{anchor}" '
        f'fill="{COLORS["text"]}">{html.escape(label)}</text>'
    )


def svg_chart_body(
    rows: Sequence[dict[str, object]],
    spec: ChartSpec,
    x_offset: float = 0.0,
    panel_label: str | None = None,
) -> str:
    axes = chart_axes(rows, spec)
    groups = grouped_rows(rows)
    left, top, width, height = x_offset + 42.0, 31.0, 204.0, 122.0
    x_range = (left, left + width)
    y_range = (top + height, top)
    parts: list[str] = []

    if panel_label:
        parts.append(svg_text(x_offset + 3, 10, panel_label, 8.5, weight=700))

    parts.append(
        f'<rect x="{left:.2f}" y="{top:.2f}" width="{width:.2f}" height="{height:.2f}" '
        f'fill="white" stroke="{COLORS["axis"]}" stroke-width="0.75"/>'
    )

    for tick in axes["y_ticks"]:
        y = scale(float(tick), axes["y_domain"], y_range)
        parts.append(
            f'<line x1="{left:.2f}" x2="{left + width:.2f}" y1="{y:.2f}" y2="{y:.2f}" '
            f'stroke="{COLORS["grid"]}" stroke-width="0.55"/>'
        )
        parts.append(svg_text(left - 5, y + 2.4, number_label(float(tick)), 7.2, "end"))

    bottom_key = "wall_clock_sec" if spec.wall_only else "compute_tflops"
    bottom_domain = axes["wall_domain"] if spec.wall_only else axes["compute_domain"]
    bottom_ticks = axes["wall_ticks"] if spec.wall_only else axes["compute_ticks"]
    for tick in bottom_ticks:
        x = scale(float(tick), bottom_domain, x_range)
        parts.append(
            f'<line x1="{x:.2f}" x2="{x:.2f}" y1="{top:.2f}" y2="{top + height:.2f}" '
            f'stroke="{COLORS["grid"]}" stroke-width="0.50"/>'
        )
        parts.append(svg_text(x, top + height + 11.5, number_label(float(tick)), 7.2, "middle"))

    single_metric = spec.wall_only or spec.compute_only
    if not single_metric:
        for tick in axes["wall_ticks"]:
            x = scale(float(tick), axes["wall_domain"], x_range)
            parts.append(
                f'<line x1="{x:.2f}" x2="{x:.2f}" y1="{top:.2f}" y2="{top - 3.2:.2f}" '
                f'stroke="{COLORS["axis"]}" stroke-width="0.65"/>'
            )
            parts.append(svg_text(x, top - 7.2, number_label(float(tick)), 7.2, "middle"))

    if spec.wall_only:
        metrics = [("wall_clock_sec", axes["wall_domain"], "", True)]
    elif spec.compute_only:
        metrics = [("compute_tflops", axes["compute_domain"], "", True)]
    else:
        metrics = [
            ("compute_tflops", axes["compute_domain"], "", True),
            ("wall_clock_sec", axes["wall_domain"], ' stroke-dasharray="4.5 3"', False),
        ]
    # dense anchor: a horizontal quality line, not an x-positioned point
    if groups["full denoising"] and spec.y_key == "performance_avg":
        score = float(groups["full denoising"][0][spec.y_key])
        y = scale(score, axes["y_domain"], y_range)
        parts.append(
            f'<line x1="{left:.2f}" x2="{left + width:.2f}" y1="{y:.2f}" y2="{y:.2f}" '
            f'stroke="{COLORS["full denoising"]}" stroke-width="1.0" stroke-dasharray="2.5 2.5"/>'
        )
        parts.append(svg_text(left + 3, y - 2.5, "full denoising", 6.2))

    for model in reversed(MODEL_ORDER):
        if not groups[model] or model == "full denoising":
            continue
        for metric_key, x_domain, dash, filled in metrics:
            points = [
                (
                    scale(float(row[metric_key]), x_domain, x_range),
                    scale(float(row[spec.y_key]), axes["y_domain"], y_range),
                    row,
                )
                for row in groups[model]
            ]
            samples = None
            if spec.compute_only:
                samples = sample_fit(
                    [(float(row[metric_key]), float(row[spec.y_key])) for row in groups[model]],
                    axes["y_domain"],
                )
            if samples is not None:
                path = " ".join(
                    ("M" if index == 0 else "L")
                    + f"{scale(x, x_domain, x_range):.2f},{scale(y, axes['y_domain'], y_range):.2f}"
                    for index, (x, y) in enumerate(samples)
                )
            else:
                path = " ".join(
                    ("M" if index == 0 else "L") + f"{x:.2f},{y:.2f}"
                    for index, (x, y, _row) in enumerate(points)
                )
            parts.append(
                f'<path d="{path}" fill="none" stroke="{COLORS[model]}" '
                f'stroke-width="1.35" stroke-linejoin="round" stroke-linecap="round"{dash}/>'
            )
            for x, y, row in points:
                fill = COLORS[model] if filled else "white"
                parts.append(
                    f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2.55" fill="{fill}" '
                    f'stroke="{COLORS[model]}" stroke-width="0.85">'
                    f'<title>{html.escape(model)} {html.escape(str(row["setting"]))}: '
                    f'{metric_key}={float(row[metric_key]):.2f}, {spec.y_key}={float(row[spec.y_key]):.2f}</title>'
                    "</circle>"
                )

    entries = [model for model in MODEL_ORDER if groups[model] and model != "full denoising"]
    legend_w = 68.0
    legend_h = 9.5 * len(entries) + (15.0 if not single_metric else 6.0)
    legend_x = x_offset + 42.0 + 204.0 - legend_w - 3.0
    legend_y = 31.0 + 122.0 - legend_h - 3.0
    parts.append(
        f'<rect x="{legend_x:.2f}" y="{legend_y:.2f}" width="{legend_w:.2f}" height="{legend_h:.2f}" '
        f'fill="white" fill-opacity="0.92" stroke="{COLORS["grid"]}" stroke-width="0.55"/>'
    )
    for index, model in enumerate(entries):
        cy = legend_y + 7.5 + 9.5 * index
        parts.append(f'<circle cx="{legend_x + 7:.2f}" cy="{cy:.2f}" r="2.45" fill="{COLORS[model]}"/>')
        parts.append(svg_text(legend_x + 13, cy + 2.3, model, 6.6))
    if not single_metric:
        y_metric = legend_y + 7.5 + 9.5 * len(entries)
        parts.extend(
            [
                f'<line x1="{legend_x + 4:.2f}" x2="{legend_x + 13:.2f}" y1="{y_metric:.2f}" y2="{y_metric:.2f}" stroke="{COLORS["text"]}" stroke-width="1.2"/>',
                svg_text(legend_x + 15, y_metric + 2.3, "TFLOPs", 5.8),
                f'<line x1="{legend_x + 38:.2f}" x2="{legend_x + 47:.2f}" y1="{y_metric:.2f}" y2="{y_metric:.2f}" stroke="{COLORS["text"]}" stroke-width="1.2" stroke-dasharray="4 2.5"/>',
                svg_text(legend_x + 49, y_metric + 2.3, "Wall clock", 5.8),
            ]
        )

    label_bottom = "Wall clock (sec / doc)" if spec.wall_only else "TFLOPs"
    parts.extend(
        [
            svg_text(left + width / 2, 184, label_bottom, 9.0, "middle", 700),
        ]
        + ([] if single_metric else [svg_text(left + width / 2, 9, "Wall clock (sec)", 9.0, "middle", 700)])
        + [
            (
                f'<text x="{x_offset + 10:.2f}" y="{top + height / 2:.2f}" '
                f'font-family="Helvetica, Arial, sans-serif" font-size="9.0" font-weight="700" '
                f'text-anchor="middle" fill="{COLORS["text"]}" '
                f'transform="rotate(-90 {x_offset + 10:.2f} {top + height / 2:.2f})">'
                f'{html.escape(spec.y_label)}</text>'
            ),
        ]
    )
    return "\n".join(parts)


def write_svgs(
    rows_perf: Sequence[dict[str, object]],
    rows_batch: Sequence[dict[str, object]],
) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for path, spec, rows in [
        (PERFORMANCE_SVG, PERFORMANCE_SPEC, rows_perf),
        (BATCH_SVG, BATCH_SPEC, rows_batch),
    ]:
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{PANEL_W:.0f}" height="{PANEL_H:.0f}" '
            f'viewBox="0 0 {PANEL_W:.0f} {PANEL_H:.0f}" role="img" '
            f'aria-label="{html.escape(spec.aria_label)}">\n'
            f'<rect width="{PANEL_W:.0f}" height="{PANEL_H:.0f}" fill="white"/>\n'
            f'{svg_chart_body(rows, spec)}\n</svg>\n'
        )
        path.write_text(svg, encoding="utf-8")

    pair = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{PAIR_W:.0f}" height="{PANEL_H:.0f}" '
        f'viewBox="0 0 {PAIR_W:.0f} {PANEL_H:.0f}" role="img" '
        f'aria-label="Two-panel model resource comparison">\n'
        f'<rect width="{PAIR_W:.0f}" height="{PANEL_H:.0f}" fill="white"/>\n'
        f'{svg_chart_body(rows_perf, PERFORMANCE_SPEC, panel_label="(a)")}\n'
        f'{svg_chart_body(rows_batch, BATCH_SPEC, PANEL_W + PANEL_GAP, "(b)")}\n'
        "</svg>\n"
    )
    PAIR_SVG.write_text(pair, encoding="utf-8")


def draw_pdf_text(
    canvas,
    page_height: float,
    x: float,
    y: float,
    label: str,
    size: float,
    anchor: str = "start",
    bold: bool = False,
) -> None:
    from reportlab.lib.colors import HexColor

    font = "Helvetica-Bold" if bold else "Helvetica"
    canvas.setFont(font, size)
    canvas.setFillColor(HexColor(COLORS["text"]))
    draw_y = page_height - y
    if anchor == "middle":
        canvas.drawCentredString(x, draw_y, label)
    elif anchor == "end":
        canvas.drawRightString(x, draw_y, label)
    else:
        canvas.drawString(x, draw_y, label)


def draw_pdf_chart(
    canvas,
    rows: Sequence[dict[str, object]],
    spec: ChartSpec,
    x_offset: float = 0.0,
    panel_label: str | None = None,
) -> None:
    from reportlab.lib.colors import HexColor, white

    axes = chart_axes(rows, spec)
    groups = grouped_rows(rows)
    left, top, width, height = x_offset + 42.0, 31.0, 204.0, 122.0
    x_range = (left, left + width)
    y_range = (top + height, top)
    py = lambda y: PANEL_H - y

    if panel_label:
        draw_pdf_text(canvas, PANEL_H, x_offset + 3, 10, panel_label, 8.5, bold=True)

    canvas.setStrokeColor(HexColor(COLORS["axis"]))
    canvas.setLineWidth(0.75)
    canvas.rect(left, py(top + height), width, height, stroke=1, fill=0)

    for tick in axes["y_ticks"]:
        y = scale(float(tick), axes["y_domain"], y_range)
        canvas.setStrokeColor(HexColor(COLORS["grid"]))
        canvas.setLineWidth(0.55)
        canvas.line(left, py(y), left + width, py(y))
        draw_pdf_text(canvas, PANEL_H, left - 5, y + 2.4, number_label(float(tick)), 7.2, "end")

    bottom_domain = axes["wall_domain"] if spec.wall_only else axes["compute_domain"]
    bottom_ticks = axes["wall_ticks"] if spec.wall_only else axes["compute_ticks"]
    for tick in bottom_ticks:
        x = scale(float(tick), bottom_domain, x_range)
        canvas.setStrokeColor(HexColor(COLORS["grid"]))
        canvas.setLineWidth(0.50)
        canvas.line(x, py(top), x, py(top + height))
        draw_pdf_text(canvas, PANEL_H, x, top + height + 11.5, number_label(float(tick)), 7.2, "middle")

    single_metric = spec.wall_only or spec.compute_only
    if not single_metric:
        for tick in axes["wall_ticks"]:
            x = scale(float(tick), axes["wall_domain"], x_range)
            canvas.setStrokeColor(HexColor(COLORS["axis"]))
            canvas.setLineWidth(0.65)
            canvas.line(x, py(top), x, py(top - 3.2))
            draw_pdf_text(canvas, PANEL_H, x, top - 7.2, number_label(float(tick)), 7.2, "middle")

    if spec.wall_only:
        metrics = [("wall_clock_sec", axes["wall_domain"], False, True)]
    elif spec.compute_only:
        metrics = [("compute_tflops", axes["compute_domain"], False, True)]
    else:
        metrics = [
            ("compute_tflops", axes["compute_domain"], False, True),
            ("wall_clock_sec", axes["wall_domain"], True, False),
        ]
    if groups["full denoising"] and spec.y_key == "performance_avg":
        score = float(groups["full denoising"][0][spec.y_key])
        y = scale(score, axes["y_domain"], y_range)
        canvas.setStrokeColor(HexColor(COLORS["full denoising"]))
        canvas.setLineWidth(1.0)
        canvas.setDash(2.5, 2.5)
        canvas.line(left, py(y), left + width, py(y))
        canvas.setDash()
        draw_pdf_text(canvas, PANEL_H, left + 3, y - 2.5, "full denoising", 6.2)

    for model in reversed(MODEL_ORDER):
        if not groups[model] or model == "full denoising":
            continue
        for metric_key, x_domain, dashed, filled in metrics:
            points = [
                (
                    scale(float(row[metric_key]), x_domain, x_range),
                    scale(float(row[spec.y_key]), axes["y_domain"], y_range),
                )
                for row in groups[model]
            ]
            canvas.setStrokeColor(HexColor(COLORS[model]))
            canvas.setLineWidth(1.35)
            canvas.setDash(4.5, 3) if dashed else canvas.setDash()
            samples = None
            if spec.compute_only:
                samples = sample_fit(
                    [(float(row[metric_key]), float(row[spec.y_key])) for row in groups[model]],
                    axes["y_domain"],
                )
            path = canvas.beginPath()
            path_points = (
                [(scale(x, x_domain, x_range), scale(y, axes["y_domain"], y_range))
                 for x, y in samples]
                if samples is not None else points
            )
            for index, (x, y) in enumerate(path_points):
                if index == 0:
                    path.moveTo(x, py(y))
                else:
                    path.lineTo(x, py(y))
            canvas.drawPath(path, stroke=1, fill=0)
            canvas.setDash()
            for x, y in points:
                canvas.setFillColor(HexColor(COLORS[model]) if filled else white)
                canvas.setStrokeColor(HexColor(COLORS[model]))
                canvas.setLineWidth(0.85)
                canvas.circle(x, py(y), 2.55, stroke=1, fill=1)

    entries = [model for model in MODEL_ORDER if groups[model] and model != "full denoising"]
    legend_w = 68.0
    legend_h = 9.5 * len(entries) + (15.0 if not single_metric else 6.0)
    legend_x = x_offset + 42.0 + 204.0 - legend_w - 3.0
    legend_y = 31.0 + 122.0 - legend_h - 3.0
    canvas.setFillColor(white)
    canvas.setStrokeColor(HexColor(COLORS["grid"]))
    canvas.setLineWidth(0.55)
    canvas.rect(legend_x, py(legend_y + legend_h), legend_w, legend_h, stroke=1, fill=1)

    for index, model in enumerate(entries):
        cy = legend_y + 7.5 + 9.5 * index
        canvas.setFillColor(HexColor(COLORS[model]))
        canvas.setStrokeColor(HexColor(COLORS[model]))
        canvas.circle(legend_x + 7, py(cy), 2.45, stroke=1, fill=1)
        draw_pdf_text(canvas, PANEL_H, legend_x + 13, cy + 2.3, model, 6.6)
    if not single_metric:
        y_metric = legend_y + 7.5 + 9.5 * len(entries)
        canvas.setStrokeColor(HexColor(COLORS["text"]))
        canvas.setLineWidth(1.2)
        canvas.line(legend_x + 4, py(y_metric), legend_x + 13, py(y_metric))
        draw_pdf_text(canvas, PANEL_H, legend_x + 15, y_metric + 2.3, "TFLOPs", 5.8)
        canvas.setDash(4, 2.5)
        canvas.line(legend_x + 38, py(y_metric), legend_x + 47, py(y_metric))
        canvas.setDash()
        draw_pdf_text(canvas, PANEL_H, legend_x + 49, y_metric + 2.3, "Wall clock", 5.8)

    label_bottom = "Wall clock (sec / doc)" if spec.wall_only else "TFLOPs"
    draw_pdf_text(canvas, PANEL_H, left + width / 2, 184, label_bottom, 9.0, "middle", True)
    if not single_metric:
        draw_pdf_text(canvas, PANEL_H, left + width / 2, 9, "Wall clock (sec)", 9.0, "middle", True)

    canvas.saveState()
    canvas.translate(x_offset + 10, py(top + height / 2))
    canvas.rotate(90)
    canvas.setFillColor(HexColor(COLORS["text"]))
    canvas.setFont("Helvetica-Bold", 9.0)
    canvas.drawCentredString(0, 0, spec.y_label)
    canvas.restoreState()


def write_pdf(path: Path, rows: Sequence[dict[str, object]], spec: ChartSpec) -> None:
    from reportlab.lib.colors import white
    from reportlab.pdfgen import canvas

    path.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(path), pagesize=(PANEL_W, PANEL_H))
    pdf.setFillColor(white)
    pdf.rect(0, 0, PANEL_W, PANEL_H, stroke=0, fill=1)
    draw_pdf_chart(pdf, rows, spec)
    pdf.showPage()
    pdf.save()


def write_pair_pdf(
    path: Path,
    rows_perf: Sequence[dict[str, object]],
    rows_batch: Sequence[dict[str, object]],
) -> None:
    from reportlab.lib.colors import white
    from reportlab.pdfgen import canvas

    path.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(path), pagesize=(PAIR_W, PANEL_H))
    pdf.setFillColor(white)
    pdf.rect(0, 0, PAIR_W, PANEL_H, stroke=0, fill=1)
    draw_pdf_chart(pdf, rows_perf, PERFORMANCE_SPEC, panel_label="(a)")
    draw_pdf_chart(pdf, rows_batch, BATCH_SPEC, PANEL_W + PANEL_GAP, "(b)")
    pdf.showPage()
    pdf.save()


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate compact dual-X-axis paper figures from one CSV file."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="panel (a) CSV: per-setting sweep points (exp6_sweep_points.csv)",
    )
    parser.add_argument(
        "--data-batch",
        type=Path,
        default=DEFAULT_BATCH_PATH,
        help="panel (b) CSV: per-batch-size points (exp6_batch_points.csv)",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    rows_perf = read_csv(args.data.expanduser().resolve())
    rows_batch = read_csv(args.data_batch.expanduser().resolve())

    write_svgs(rows_perf, rows_batch)
    try:
        write_pdf(PERFORMANCE_PDF, rows_perf, PERFORMANCE_SPEC)
        write_pdf(BATCH_PDF, rows_batch, BATCH_SPEC)
        write_pair_pdf(PAIR_PDF, rows_perf, rows_batch)
        print(f"Two-panel PDF: {PAIR_PDF}")
    except ImportError:
        print("reportlab not installed: SVGs written, PDFs skipped")

    print(f"Panel (a) data: {args.data}")
    print(f"Panel (b) data: {args.data_batch}")
    print(f"Two-panel SVG: {PAIR_SVG}")


if __name__ == "__main__":
    main()
