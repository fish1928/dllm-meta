from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde, spearmanr


ROOT = Path(__file__).resolve().parent
DEFAULT_TRAJECTORY_CSV = ROOT / "exp2_trajectory_points.csv"
DEFAULT_SCORE_CSV = ROOT / "exp2_fidelity_scores.csv"
DEFAULT_DOSE_CSV = ROOT / "exp2_dose_curves.csv"
DOSE_FIT = "isotonic"    # 'isotonic' (monotone PAVA fit) | 'moving' (window-5 mean)
DEFAULT_PDF = ROOT / "output" / "pdf" / "diagram3_three_panel.pdf"
DEFAULT_PREVIEW = ROOT / "output" / "figures" / "diagram3_three_panel.png"
DEFAULT_APPENDIX_PDF = ROOT / "output" / "pdf" / "diagram3_token_divergence_appendix.pdf"
DEFAULT_APPENDIX_PREVIEW = ROOT / "output" / "figures" / "diagram3_token_divergence_appendix.png"

BASELINE = "fastdllm"
STUDENT = "ours"

# palette matched to the exp-6 horizontal diagram for cross-figure consistency
COLORS = {
    "ink": "#202020",
    "muted": "#5F5F5F",
    "grid": "#DDDDE0",
    "ours": "#287C78",
    "fastdllm": "#B98A2E",
    "d2cache": "#D96C4F",
    "dllmcache": "#8073AC",
    "random order": "#737373",
}
FALLBACK_COLORS = ("#1F5A99", "#A51C30", "#3D7A3A", "#8A542F")
PROB_CMAP = LinearSegmentedColormap.from_list(
    "full_denoising_probability", ["#F3E7A6", "#E79A57", "#B53A42", "#5D2138"]
)

DISPLAY_LABELS = {
    "ours": "Ours",
    "fastdllm": "Fast-dLLM (cache)",
    "d2cache": "d2Cache",
    "dllmcache": "dLLM-Cache",
    "random order": "Random order",
}

TRAJECTORY_COLUMNS = {
    "model",
    "sample_id",
    "position",
    "teacher_step",
    "student_step",
    "teacher_token",
    "student_token",
    "block_length",
    "total_steps",
}
SCORE_COLUMNS = {"model", "sample_id", "fidelity", "score"}


def _read_rows(path: Path, required: set[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path} contains no data rows")
    return rows


def load_trajectory_csv(path: Path) -> dict[str, dict[str, object]]:
    rows = _read_rows(path, TRAJECTORY_COLUMNS)
    grouped: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for line, row in enumerate(rows, start=2):
        try:
            position = int(row["position"])
            int(row["teacher_step"])
            int(row["student_step"])
            int(row["teacher_token"])
            int(row["student_token"])
            block_length = int(row["block_length"])
            total_steps = int(row["total_steps"])
        except ValueError as exc:
            raise ValueError(f"{path}:{line}: trajectory integer fields must be integers") from exc
        if position < 0 or block_length < 2 or total_steps < 2:
            raise ValueError(f"{path}:{line}: invalid position, block_length, or total_steps")
        grouped[row["model"]][row["sample_id"]].append(row)

    packed: dict[str, dict[str, object]] = {}
    for model, samples in grouped.items():
        sample_ids = sorted(samples)
        expected_positions: list[int] | None = None
        matrices: dict[str, list[list[float]]] = {
            "teacher_step": [],
            "student_step": [],
            "teacher_token": [],
            "student_token": [],
            "teacher_conf": [],
            "p_teacher": [],
        }
        block_lengths: set[int] = set()
        total_steps_values: set[int] = set()
        has_conf = True
        has_probability = True

        for sample_id in sample_ids:
            sample_rows = sorted(samples[sample_id], key=lambda row: int(row["position"]))
            positions = [int(row["position"]) for row in sample_rows]
            if len(set(positions)) != len(positions):
                raise ValueError(f"{path}: duplicate position for {model}/{sample_id}")
            if expected_positions is None:
                expected_positions = positions
            elif positions != expected_positions:
                raise ValueError(f"{path}: position grid differs for {model}/{sample_id}")

            for key in ("teacher_step", "student_step", "teacher_token", "student_token"):
                matrices[key].append([int(row[key]) for row in sample_rows])

            conf_values = [row.get("teacher_conf", "").strip() for row in sample_rows]
            probability_values = [row.get("p_teacher", "").strip() for row in sample_rows]
            has_conf = has_conf and all(conf_values)
            has_probability = has_probability and all(probability_values)
            matrices["teacher_conf"].append([float(value) if value else np.nan for value in conf_values])
            matrices["p_teacher"].append(
                [float(value) if value else np.nan for value in probability_values]
            )
            block_lengths.update(int(row["block_length"]) for row in sample_rows)
            total_steps_values.update(int(row["total_steps"]) for row in sample_rows)

        if len(block_lengths) != 1 or len(total_steps_values) != 1:
            raise ValueError(f"{path}: block_length and total_steps must be constant for {model}")
        block_length = block_lengths.pop()
        length = len(expected_positions or [])
        if length % block_length:
            raise ValueError(f"{path}: sequence length {length} is not divisible by block_length")

        packed[model] = {
            "sample_ids": sample_ids,
            "teacher_step": np.asarray(matrices["teacher_step"], dtype=int),
            "student_step": np.asarray(matrices["student_step"], dtype=int),
            "teacher_token": np.asarray(matrices["teacher_token"], dtype=int),
            "student_token": np.asarray(matrices["student_token"], dtype=int),
            "teacher_conf": (
                np.asarray(matrices["teacher_conf"], dtype=float) if has_conf else None
            ),
            "p_teacher": (
                np.asarray(matrices["p_teacher"], dtype=float) if has_probability else None
            ),
            "block_length": block_length,
            "total_steps": total_steps_values.pop(),
        }
    return packed


def load_score_csv(path: Path) -> tuple[dict[str, dict[str, np.ndarray]], list[str]]:
    rows = _read_rows(path, SCORE_COLUMNS)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    order_values: dict[str, int] = {}
    for line, row in enumerate(rows, start=2):
        try:
            fidelity = float(row["fidelity"])
            score = float(row["score"])
        except ValueError as exc:
            raise ValueError(f"{path}:{line}: fidelity and score must be numeric") from exc
        if not 0 <= fidelity <= 1 or not 0 <= score <= 1:
            raise ValueError(f"{path}:{line}: fidelity and score must be in [0, 1]")
        if row.get("teacher_correct", "").strip() not in ("", "0", "1"):
            raise ValueError(f"{path}:{line}: teacher_correct must be blank, 0, or 1")
        grouped[row["model"]].append(row)
        if row.get("model_order", "").strip():
            order_values[row["model"]] = int(row["model_order"])

    order = sorted(grouped, key=lambda model: (order_values.get(model, 10_000), model))
    packed: dict[str, dict[str, np.ndarray]] = {}
    for model in order:
        model_rows = grouped[model]
        teacher_values = [row.get("teacher_correct", "").strip() for row in model_rows]
        packed[model] = {
            "sample_id": np.asarray([row["sample_id"] for row in model_rows]),
            "fidelity": np.asarray([float(row["fidelity"]) for row in model_rows]),
            "score": np.asarray([float(row["score"]) for row in model_rows]),
            "teacher_correct": np.asarray(
                [value != "0" for value in teacher_values], dtype=bool
            ),
            "has_teacher_correct": np.asarray([all(teacher_values)], dtype=bool),
        }
    return packed, order


def load_dose_csv(path: Path) -> dict[str, dict[str, np.ndarray]]:
    rows = _read_rows(path, {"model", "n", "accuracy"})
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["model"]].append(row)
    packed: dict[str, dict[str, np.ndarray]] = {}
    for model, model_rows in grouped.items():
        model_rows.sort(key=lambda row: int(row["n"]))
        packed[model] = {
            "n": np.asarray([int(row["n"]) for row in model_rows]),
            "accuracy": np.asarray([float(row["accuracy"]) for row in model_rows]),
            "tflops": float(model_rows[0].get("tflops_doc") or 0) or None,
        }
    return packed
# end


def smooth_curve(values: np.ndarray, window: int = 5) -> np.ndarray:
    kernel = np.ones(window) / window
    padded = np.pad(values, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")
# end


def isotonic_curve(values: np.ndarray) -> np.ndarray:
    """Pool-adjacent-violators: least-squares monotone non-decreasing fit.
    Principled for the dose-response (more injected trajectory cannot hurt in
    expectation); applied to EVERY method identically, no points removed."""
    levels = [[float(value), 1] for value in values]
    merged: list[list[float]] = []
    for level in levels:
        merged.append(level)
        while len(merged) > 1 and merged[-2][0] > merged[-1][0]:
            b = merged.pop()
            a = merged.pop()
            weight = a[1] + b[1]
            merged.append([(a[0] * a[1] + b[0] * b[1]) / weight, weight])
    out: list[float] = []
    for value, weight in merged:
        out.extend([value] * int(weight))
    return np.asarray(out)
# end


def plot_dose_panel(fig: plt.Figure, spec, curves: dict, order: list[str]) -> plt.Axes:
    """Accuracy vs teacher-prefix rate, x REVERSED from the convergence point
    n* (all smoothed curves >= 0.995) down to prefill 0: moving rightward
    removes the teacher trajectory and the methods separate."""
    sub = spec.subgridspec(2, 1, height_ratios=(1.83, 4), hspace=0.08)
    legend_ax = fig.add_subplot(sub[0, 0])
    main = fig.add_subplot(sub[1, 0])

    fit = isotonic_curve if DOSE_FIT == "isotonic" else smooth_curve
    smoothed = {model: fit(curves[model]["accuracy"]) for model in order}
    grid_n = curves[order[0]]["n"]
    n_star = None
    for index in range(len(grid_n)):
        if all(smoothed[model][index] >= 0.995 for model in order):
            n_star = int(grid_n[index])
            break
    if n_star is None:
        n_star = int(grid_n[-1])

    for model in order:
        color = COLORS.get(model, FALLBACK_COLORS[0])
        style = ORDER_PANEL_STYLES.get(model, "--")
        label = DISPLAY_LABELS.get(model, model)
        tflops = curves[model]["tflops"]
        if tflops:
            label = f"{label} ({tflops:.0f} TF/doc)"
        n = curves[model]["n"]
        keep = n <= n_star
        main.plot(n[keep], curves[model]["accuracy"][keep], "o", ms=1.8,
                  color=color, alpha=0.3, rasterized=True)
        main.plot(n[keep], smoothed[model][keep], color=color, lw=1.55,
                  ls=style, label=label, solid_capstyle="round")
    main.set_xlim(n_star + 1.5, -1.5)    # REVERSED: teacher help decreases rightward
    main.set_ylim(0.74, 1.008)
    main.axvline(n_star, color=COLORS["muted"], lw=0.8, ls=":")
    main.text(n_star - 0.8, 0.752, "all methods\nrescued",
              fontsize=5.4, color=COLORS["muted"], ha="left", va="bottom")
    main.set_xticks(np.arange(0, n_star + 1, 12)[::-1])
    main.set_xlabel("Teacher-prefix rate (%)  →  less trajectory injected")
    main.set_ylabel("Accuracy (teacher-correct docs)")
    handles, labels = main.get_legend_handles_labels()
    legend_ax.legend(handles, labels, loc="center", frameon=False, fontsize=5.4,
                     handlelength=1.7, labelspacing=0.25, columnspacing=0.8, ncol=2)
    legend_ax.set_axis_off()
    axis_style(main, grid=True)
    return main
# end


def within_block_rank(step: np.ndarray, block_length: int, confidence: np.ndarray | None) -> np.ndarray:
    n_samples, length = step.shape
    position_tie = np.broadcast_to(np.arange(length), (n_samples, length))
    tie = -confidence if confidence is not None else position_tie
    rank = np.empty_like(step, dtype=float)
    for start in range(0, length, block_length):
        block = slice(start, start + block_length)
        order = np.lexsort((tie[:, block], step[:, block]), axis=1)
        rank[:, block] = np.argsort(order, axis=1) / (block_length - 1)
    return rank


def derive_trajectory(data: dict[str, object]) -> dict[str, np.ndarray | float | None]:
    block_length = int(data["block_length"])
    teacher_rank = within_block_rank(
        np.asarray(data["teacher_step"]), block_length, data["teacher_conf"]
    )
    student_rank = within_block_rank(
        np.asarray(data["student_step"]), block_length, data["teacher_conf"]
    )
    token_match = np.asarray(data["student_token"]) == np.asarray(data["teacher_token"])
    correlation = float(np.corrcoef(teacher_rank.ravel(), student_rank.ravel())[0, 1])
    probability = data["p_teacher"]
    return {
        "teacher_rank": teacher_rank.ravel(),
        "student_rank": student_rank.ravel(),
        "token_match": token_match.ravel(),
        "p_teacher": np.asarray(probability).ravel() if probability is not None else None,
        "correlation": correlation,
    }


def binned_mean(x: np.ndarray, y: np.ndarray, bins: int = 10) -> tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(0, 1, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    membership = np.clip(np.digitize(x, edges) - 1, 0, bins - 1)
    mean = np.array([y[membership == index].mean() if np.any(membership == index) else np.nan
                     for index in range(bins)])
    return centers, mean


def safe_kde(values: np.ndarray, grid: np.ndarray, bandwidth: float | str | None = None) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return np.zeros_like(grid)
    if np.std(values) < 1e-8:
        values = values + np.random.default_rng(17).normal(0, 0.005, len(values))
    return gaussian_kde(values, bw_method=bandwidth)(grid)


def density_contour(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    color: str,
    linestyle: str,
    seed: int,
    levels: tuple[float, ...] = (0.5, 0.9),
    fill: bool = True,
) -> None:
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(x), min(2600, len(x)), replace=False)
    jitter = rng.normal(0, 0.017, (2, len(indices)))
    kernel = gaussian_kde(
        np.vstack([x[indices] + jitter[0], y[indices] + jitter[1]]), bw_method=0.22
    )
    grid = np.linspace(-0.05, 1.05, 90)
    xx, yy = np.meshgrid(grid, grid)
    density = kernel(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    sorted_density = np.sort(density.ravel())[::-1]
    mass = np.cumsum(sorted_density) / sorted_density.sum()
    thresholds = sorted(sorted_density[np.searchsorted(mass, level)] for level in levels)
    ax.contour(
        xx,
        yy,
        density,
        levels=thresholds,
        colors=color,
        linewidths=(0.95, 1.35)[-len(thresholds):],
        linestyles=linestyle,
    )
    if fill:
        ax.contourf(
            xx,
            yy,
            density,
            levels=[thresholds[-1], density.max()],
            colors=[color],
            alpha=0.16,
        )


def sigmoid(x: np.ndarray, intercept: float, slope: float) -> np.ndarray:
    z = np.clip(intercept + slope * x, -40, 40)
    return 1 / (1 + np.exp(-z))


def fit_curve(x: np.ndarray, y: np.ndarray):
    try:
        parameters, _ = curve_fit(sigmoid, x, y, p0=(0, 1), maxfev=5000)
        return lambda values: sigmoid(values, *parameters)
    except (RuntimeError, ValueError):
        coefficients = np.polyfit(x, y, 1)
        return lambda values: np.clip(np.polyval(coefficients, values), 0, 1)


def regression_r2(y: np.ndarray, predictors: np.ndarray) -> float:
    design = np.column_stack([np.ones(len(y)), predictors])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual_variance = np.var(y - design @ coefficients)
    total_variance = np.var(y)
    return float(1 - residual_variance / total_variance) if total_variance else 0.0


def axis_style(ax: plt.Axes, grid: bool = False) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(direction="out", length=2.5, width=0.65, pad=1.5)
    if grid:
        ax.grid(True, color=COLORS["grid"], linewidth=0.45, alpha=0.7)
        ax.set_axisbelow(True)


ORDER_PANEL_STYLES = {    # named styles: valid for both ax.contour and ax.plot
    "fastdllm": "dashed",
    "d2cache": "dashdot",
    "dllmcache": "dotted",
    "ours": "solid",
}


def plot_order_panel(
    fig: plt.Figure,
    spec,
    models: dict[str, dict],
    order: list[str],
) -> plt.Axes:
    sub = spec.subgridspec(
        3,
        2,
        width_ratios=(4.2, 1),
        height_ratios=(1.05, 0.78, 4),
        hspace=0.08,
        wspace=0.05,
    )
    legend_ax = fig.add_subplot(sub[0, :])
    top = fig.add_subplot(sub[1, 0])
    main = fig.add_subplot(sub[2, 0], sharex=top)
    right = fig.add_subplot(sub[2, 1])

    for seed_offset, model in enumerate(order):
        data = models[model]
        color = COLORS.get(model, FALLBACK_COLORS[seed_offset % len(FALLBACK_COLORS)])
        style = ORDER_PANEL_STYLES.get(model, "--")
        is_ours = model == "ours"
        density_contour(
            main,
            np.asarray(data["teacher_rank"]),
            np.asarray(data["student_rank"]),
            color,
            style,
            2 + seed_offset,
            levels=(0.8,),
            fill=is_ours,
        )
    main.plot(
        [0, 1], [0, 1], color=COLORS["ink"], lw=1.05, ls=(0, (4, 3)),
        label="Perfect agreement ($y=x$)",
    )
    for seed_offset, model in enumerate(order):
        data = models[model]
        color = COLORS.get(model, FALLBACK_COLORS[seed_offset % len(FALLBACK_COLORS)])
        style = ORDER_PANEL_STYLES.get(model, "--")
        label = DISPLAY_LABELS.get(model, model)
        main.plot(
            [], [], color=color, lw=1.55 if model == "ours" else 1.35, ls=style,
            label=rf"{label}  $\rho_{{\mathrm{{order}}}}$={data['correlation']:.2f}",
        )
    main.set(
        xlim=(-0.03, 1.03),
        ylim=(-0.03, 1.03),
        xlabel="Normalized full-denoising rank",
        ylabel="Normalized cached-decoding rank",
    )
    main.set_xticks((0, 0.5, 1)); main.set_yticks((0, 0.5, 1))
    handles, labels = main.get_legend_handles_labels()
    legend_ax.legend(
        handles,
        labels,
        loc="center",
        frameon=False,
        fontsize=5.8,
        handlelength=1.7,
        labelspacing=0.25,
        columnspacing=0.8,
        ncol=2,
    )
    legend_ax.set_axis_off()
    main.text(
        0.98,
        0.035,
        "KDE contours: 80% probability mass (all methods)\n"
        "Shading: ours' 80% highest-density region",
        transform=main.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.4,
        color=COLORS["muted"],
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "white",
            "edgecolor": "#C8C8C8",
            "linewidth": 0.4,
            "alpha": 0.88,
        },
    )
    axis_style(main)

    for seed_offset, model in enumerate(order):
        data = models[model]
        color = COLORS.get(model, FALLBACK_COLORS[seed_offset % len(FALLBACK_COLORS)])
        centers, values = binned_mean(
            np.asarray(data["teacher_rank"]), np.asarray(data["token_match"], dtype=float), bins=10
        )
        top.plot(centers, values, color=color, lw=1.3,
                 ls=ORDER_PANEL_STYLES.get(model, "--"))
    top.set_ylim(0, 1.02); top.set_yticks((0, 1))
    top.set_ylabel("Token-match\nrate", labelpad=1)
    top.set_title(
        "Token-match diagnostic (not part of trajectory agreement)",
        loc="left",
        fontsize=6.0,
        color=COLORS["muted"],
        pad=1.5,
    )
    top.set_facecolor("#F6F6F6")
    top.tick_params(labelbottom=False)
    axis_style(top)

    residual_grid = np.linspace(-0.6, 0.6, 160)
    for seed_offset, model in enumerate(order):
        data = models[model]
        color = COLORS.get(model, FALLBACK_COLORS[seed_offset % len(FALLBACK_COLORS)])
        residual = np.asarray(data["student_rank"]) - np.asarray(data["teacher_rank"])
        density = safe_kde(residual, residual_grid, bandwidth=0.25)
        if model == "ours":
            right.fill_betweenx(residual_grid, 0, density, color=color, alpha=0.14)
        right.plot(density, residual_grid, color=color, lw=1.1,
                   ls=ORDER_PANEL_STYLES.get(model, "--"))
    right.axhline(0, color=COLORS["ink"], lw=0.75, ls=":")
    right.set_ylim(-0.6, 0.6); right.set_xticks([]); right.set_yticks((-0.5, 0, 0.5))
    right.yaxis.tick_right(); right.yaxis.set_label_position("right")
    right.set_title("Rank-error\nKDE", fontsize=5.8, pad=1)
    right.spines[["top", "left", "bottom"]].set_visible(False)
    right.tick_params(length=2, pad=1)
    return main


def plot_score_panel(
    fig: plt.Figure, spec, scores: dict, order: list[str], seed: int,
    trim_outliers: bool = False,
) -> plt.Axes:
    if trim_outliers:
        # per-model IQR rule on fidelity: keep Q1-1.5*IQR <= x <= Q3+1.5*IQR
        trimmed_counts = {}
        for model in order:
            data = scores[model]
            x = data["fidelity"]
            q1, q3 = np.percentile(x, (25, 75))
            iqr = q3 - q1
            keep = (x >= q1 - 1.5 * iqr) & (x <= q3 + 1.5 * iqr)
            trimmed_counts[model] = int((~keep).sum())
            scores[model] = {key: value[keep] if isinstance(value, np.ndarray) and len(value) == len(keep) else value
                             for key, value in data.items()}
        print("outliers removed (IQR rule):",
              {DISPLAY_LABELS.get(m, m): c for m, c in trimmed_counts.items()})
    sub = spec.subgridspec(2, 1, height_ratios=(1.83, 4), hspace=0.08)
    legend_ax = fig.add_subplot(sub[0, 0])
    main = fig.add_subplot(sub[1, 0])
    rng = np.random.default_rng(seed)
    grid = np.linspace(0, 1, 180)
    pooled_x: list[np.ndarray] = []
    pooled_y: list[np.ndarray] = []
    all_teacher_flags = True

    for index, model in enumerate(order):
        data = scores[model]
        x = data["fidelity"]
        y = data["score"]
        color = COLORS.get(model, FALLBACK_COLORS[index % len(FALLBACK_COLORS)])
        display_label = DISPLAY_LABELS.get(model, model)
        pooled_x.append(x); pooled_y.append(y)
        unique_scores = np.unique(y)
        step = np.diff(unique_scores).min() if len(unique_scores) > 1 else 1
        jittered_y = y + rng.uniform(-0.22, 0.22, len(y)) * step if len(unique_scores) <= 12 else y
        teacher_ok = data["teacher_correct"]
        has_teacher = bool(data["has_teacher_correct"][0])
        all_teacher_flags = all_teacher_flags and has_teacher
        hollow = ~teacher_ok if has_teacher else np.zeros(len(y), dtype=bool)
        main.scatter(x[~hollow], jittered_y[~hollow], s=4.2, color=color, alpha=0.24,
                     linewidths=0, rasterized=True)
        main.scatter(x[hollow], jittered_y[hollow], s=4.5, facecolors="none", edgecolors=color,
                     alpha=0.55, linewidths=0.45, rasterized=True)

        low, high = np.percentile(x, (2, 98))
        curve_x = grid[(grid >= low) & (grid <= high)]
        main.plot(
            curve_x,
            fit_curve(x, y)(curve_x),
            color=color,
            lw=1.45,
            solid_capstyle="round",
        )
        mean_x, mean_y = x.mean(), y.mean()
        err_x = 1.96 * x.std(ddof=1) / np.sqrt(len(x))
        err_y = 1.96 * y.std(ddof=1) / np.sqrt(len(y))
        rho = spearmanr(x, y).statistic
        main.errorbar(mean_x, mean_y, xerr=err_x, yerr=err_y, fmt="o", ms=4.6,
                      color=color, markeredgecolor="white", markeredgewidth=0.65,
                      ecolor=color, elinewidth=0.95, capsize=1.7, zorder=6,
                      label=rf"{display_label}  $\rho_{{\mathrm{{score}}}}$={rho:.2f}")
    pooled_x_array = np.concatenate(pooled_x)
    pooled_y_array = np.concatenate(pooled_y)
    main.plot(grid, fit_curve(pooled_x_array, pooled_y_array)(grid), color=COLORS["ink"],
              lw=1.05, ls=(0, (4, 3)), label="All models pooled")

    model_index = {model: index for index, model in enumerate(order)}
    one_hot = np.zeros((len(pooled_y_array), max(0, len(order) - 1)))
    offset = 0
    for model in order:
        count = len(scores[model]["score"])
        if model_index[model] > 0:
            one_hot[offset:offset + count, model_index[model] - 1] = 1
        offset += count
    r_model = regression_r2(pooled_y_array, one_hot) if one_hot.shape[1] else 0.0
    r_agreement = regression_r2(pooled_y_array, pooled_x_array[:, None])
    r_joint = regression_r2(pooled_y_array, np.column_stack([pooled_x_array, one_hot]))
    main.text(
        0.98, 0.03,
        "Explained variance ($R^2$)\n"
        f"Model identity only: {r_model:.2f}\n"
        f"Trajectory agreement only: {r_agreement:.2f}\n"
        f"Model + agreement: {r_joint:.2f}",
        transform=main.transAxes, ha="right", va="bottom", fontsize=5.2, color=COLORS["muted"],
        bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "#C8C8C8", "linewidth": 0.4},
    )
    main.set(xlim=(0, 1), ylim=(-0.06, 1.06), xlabel="Trajectory agreement", ylabel="Task score")
    main.set_xticks((0, 0.5, 1)); main.set_yticks((0, 0.5, 1))
    if all_teacher_flags:
        main.scatter([], [], s=7, facecolors="none", edgecolors=COLORS["muted"],
                     linewidths=0.55, label="Full-denoising incorrect")
    handles, labels = main.get_legend_handles_labels()
    legend_ax.legend(
        handles,
        labels,
        loc="center",
        frameon=False,
        fontsize=5.0,
        handlelength=1.25,
        handletextpad=0.35,
        labelspacing=0.18,
        columnspacing=0.65,
        ncol=2,
    )
    legend_ax.set_axis_off()
    axis_style(main, grid=True)
    return main


def make_appendix_figure(
    student: dict,
    output_pdf: Path,
    preview_png: Path | None,
    seed: int,
) -> None:
    figure = plt.figure(figsize=(3.55, 3.20), facecolor="white")
    grid_spec = figure.add_gridspec(
        1,
        2,
        width_ratios=(1, 0.055),
        left=0.17,
        right=0.91,
        bottom=0.16,
        top=0.86,
        wspace=0.10,
    )
    axis = figure.add_subplot(grid_spec[0, 0])
    color_axis = figure.add_subplot(grid_spec[0, 1])

    rng = np.random.default_rng(seed)
    full_rank = np.asarray(student["teacher_rank"])
    cached_rank = np.asarray(student["student_rank"])
    token_match = np.asarray(student["token_match"], dtype=bool)
    count = min(1800, len(full_rank))
    indices = rng.choice(len(full_rank), count, replace=False)
    jitter = rng.normal(0, 0.010, (2, count))
    probability = student["p_teacher"]
    colors = (
        np.asarray(probability)[indices]
        if probability is not None
        else token_match[indices].astype(float)
    )
    matched = token_match[indices]
    scatter = axis.scatter(
        full_rank[indices][matched] + jitter[0][matched],
        cached_rank[indices][matched] + jitter[1][matched],
        c=colors[matched],
        cmap=PROB_CMAP,
        vmin=0,
        vmax=1,
        s=4.0,
        alpha=0.32,
        linewidths=0,
        rasterized=True,
    )
    axis.scatter(
        full_rank[indices][~matched] + jitter[0][~matched],
        cached_rank[indices][~matched] + jitter[1][~matched],
        c=colors[~matched],
        cmap=PROB_CMAP,
        vmin=0,
        vmax=1,
        s=12,
        alpha=0.88,
        marker="x",
        linewidths=0.75,
        rasterized=True,
        label="Token mismatch",
    )
    axis.plot(
        [0, 1],
        [0, 1],
        color=COLORS["ink"],
        lw=1.05,
        ls=(0, (4, 3)),
        label="Perfect agreement ($y=x$)",
    )
    axis.set(
        xlim=(-0.04, 1.04),
        ylim=(-0.04, 1.04),
        xlabel="Normalized full-denoising rank",
        ylabel="Normalized cached-decoding rank",
    )
    axis.set_xticks((0, 0.5, 1)); axis.set_yticks((0, 0.5, 1))
    axis.set_aspect("equal", adjustable="box", anchor="S")
    axis.legend(loc="upper left", frameon=False, fontsize=5.6, handletextpad=0.4)
    axis_style(axis)

    colorbar = figure.colorbar(scatter, cax=color_axis)
    colorbar.set_ticks((0, 0.5, 1))
    color_axis.set_title(r"$p_{\mathrm{full}}$", fontsize=6.0, pad=2)
    colorbar.ax.tick_params(length=2, pad=1)
    figure.text(
        0.5,
        0.93,
        "Token-divergence diagnostic",
        ha="center",
        va="bottom",
        fontsize=plt.rcParams["axes.titlesize"],
        fontweight="bold",
    )

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_pdf,
        format="pdf",
        metadata={
            "Title": "Token-divergence diagnostic",
            "Creator": "diagram3_three_panel.py",
        },
    )
    if preview_png is not None:
        preview_png.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(preview_png, dpi=300)
    plt.close(figure)
    print(f"saved {output_pdf}")
    if preview_png is not None:
        print(f"saved {preview_png}")


def make_figure(
    trajectory_csv: Path,
    score_csv: Path,
    output_pdf: Path,
    preview_png: Path | None,
    baseline_model: str,
    student_model: str,
    seed: int,
    panel_b: str = "dose",
    trim_outliers: bool = False,
) -> dict:
    trajectory = load_trajectory_csv(trajectory_csv)
    missing_models = {baseline_model, student_model} - set(trajectory)
    if missing_models:
        raise ValueError(
            f"{trajectory_csv} is missing trajectory models: {', '.join(sorted(missing_models))}"
        )
    curves = load_dose_csv(DEFAULT_DOSE_CSV)
    student = derive_trajectory(trajectory[student_model])
    order_models = [model for model in ("fastdllm", "d2cache", "dllmcache") if model in trajectory]
    order_models.append(student_model)    # ours drawn last, on top
    derived_all = {model: (student if model == student_model else derive_trajectory(trajectory[model]))
                   for model in order_models}

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7.0,
            "axes.titlesize": 8.4,
            "axes.labelsize": 7.4,
            "xtick.labelsize": 6.4,
            "ytick.labelsize": 6.4,
            "legend.fontsize": 5.8,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure = plt.figure(figsize=(7.16, 3.45), facecolor="white")
    outer = figure.add_gridspec(
        1,
        2,
        # Panel (a) includes a side diagnostic while panel (b) does not.
        # Compensate so the two main plotting areas have equal x-axis lengths.
        width_ratios=(1.27, 1.0),
        left=0.066,
        right=0.993,
        bottom=0.13,
        top=0.90,
        wspace=0.16,
    )
    order_axis = plot_order_panel(figure, outer[0], derived_all, order_models)
    if panel_b == "score":
        scores, score_order = load_score_csv(score_csv)
        score_axis = plot_score_panel(figure, outer[1], scores, score_order,
                                      seed + 1, trim_outliers=trim_outliers)
    else:
        score_axis = plot_dose_panel(figure, outer[1], curves, order_models)
    title_y = 0.935
    for panel, title in enumerate(
        (
            "(a) Agreement with the full-denoising order",
            ("(b) Trajectory agreement and task performance" if panel_b == "score"
             else "(b) Removing the injected teacher trajectory"),
        )
    ):
        panel_position = outer[panel].get_position(figure)
        figure.text(
            (panel_position.x0 + panel_position.x1) / 2,
            title_y,
            title,
            ha="center",
            va="bottom",
            fontsize=plt.rcParams["axes.titlesize"],
            fontweight="bold",
        )
    main_axes = (order_axis, score_axis)
    figure.canvas.draw()

    axis_widths = [axis.get_position().width for axis in main_axes]
    relative_spread = (max(axis_widths) - min(axis_widths)) / np.mean(axis_widths)
    if relative_spread > 0.005:
        formatted = ", ".join(f"{width:.5f}" for width in axis_widths)
        raise RuntimeError(f"Main x-axis widths are not aligned: {formatted}")

    axis_bottoms = [axis.get_position().y0 for axis in main_axes]
    if max(axis_bottoms) - min(axis_bottoms) > 0.002:
        formatted = ", ".join(f"{bottom:.5f}" for bottom in axis_bottoms)
        raise RuntimeError(f"Main x-axis baselines are not aligned: {formatted}")

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_pdf,
        format="pdf",
        metadata={
            "Title": "Full-denoising order agreement and task performance",
            "Creator": "diagram3_three_panel.py",
        },
    )
    if preview_png is not None:
        preview_png.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(preview_png, dpi=300)
    plt.close(figure)
    print(f"saved {output_pdf}")
    if preview_png is not None:
        print(f"saved {preview_png}")
    return student


def make_demo_trajectory(seed: int = 23) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    sample_count, length, block_length, total_steps = 100, 128, 32, 64
    blocks = length // block_length
    steps_per_block = total_steps // blocks
    tokens_per_step = block_length // steps_per_block
    teacher_steps = np.zeros((sample_count, length), dtype=int)
    teacher_confidence = rng.uniform(0.05, 0.99, (sample_count, length))
    for sample in range(sample_count):
        for block in range(blocks):
            position = np.arange(block_length)
            score = position + rng.normal(0, 8, block_length)
            order = np.argsort(np.argsort(score))
            teacher_steps[sample, block * block_length:(block + 1) * block_length] = (
                block * steps_per_block + order // tokens_per_step
            )
    teacher_tokens = rng.integers(0, 32000, (sample_count, length))

    rows: list[dict[str, object]] = []
    settings = ((BASELINE, 6.0, 1.4, 1.2), (STUDENT, 1.55, 0.35, 3.0))
    for model_index, (model, sigma, drift, match_base) in enumerate(settings):
        model_rng = np.random.default_rng(seed + 100 + model_index)
        student_steps = np.zeros_like(teacher_steps)
        for sample in range(sample_count):
            for block in range(blocks):
                block_slice = slice(block * block_length, (block + 1) * block_length)
                teacher_block_steps = teacher_steps[sample, block_slice]
                teacher_order = np.argsort(np.argsort(teacher_block_steps + np.arange(block_length) * 1e-3))
                noise = sigma * (1 + drift * block / blocks)
                student_order = np.argsort(np.argsort(teacher_order + model_rng.normal(0, noise, block_length)))
                student_steps[sample, block_slice] = block * steps_per_block + student_order // tokens_per_step
        delta = np.abs(student_steps - teacher_steps)
        match_probability = 1 / (1 + np.exp(-(match_base - 0.6 * delta)))
        token_match = model_rng.random((sample_count, length)) < match_probability
        student_tokens = np.where(
            token_match,
            teacher_tokens,
            model_rng.integers(0, 32000, (sample_count, length)),
        )
        teacher_probability = np.where(
            token_match,
            model_rng.beta(8, 2, (sample_count, length)),
            model_rng.beta(1, 6, (sample_count, length)),
        )
        for sample in range(sample_count):
            for position in range(length):
                rows.append(
                    {
                        "model": model,
                        "sample_id": f"S{sample + 1:03d}",
                        "position": position,
                        "teacher_step": int(teacher_steps[sample, position]),
                        "student_step": int(student_steps[sample, position]),
                        "teacher_token": int(teacher_tokens[sample, position]),
                        "student_token": int(student_tokens[sample, position]),
                        "teacher_conf": round(float(teacher_confidence[sample, position]), 6),
                        "p_teacher": round(float(teacher_probability[sample, position]), 6),
                        "block_length": block_length,
                        "total_steps": total_steps,
                    }
                )
    return rows


def make_demo_scores(seed: int = 29) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    sample_count, repeats = 260, 8
    difficulty = rng.normal(0, 1, sample_count)
    teacher_correct = rng.random(sample_count) < 1 / (1 + np.exp(-(1.3 - 1.1 * difficulty)))
    models = (
        ("random order", 1, 2.2, 9.0),
        (BASELINE, 2, 5.0, 4.5),
        ("baseline cache", 3, 7.0, 3.8),
        (STUDENT, 4, 12.0, 2.3),
    )
    rows: list[dict[str, object]] = []
    for model, model_order, alpha, beta in models:
        fidelity = np.clip(rng.beta(alpha, beta, sample_count) - 0.05 * difficulty, 0.01, 0.99)
        probability = 1 / (1 + np.exp(-(-3.2 + 5.8 * fidelity - 0.9 * difficulty)))
        probability = np.where(teacher_correct, probability, 0.25 * probability)
        score = rng.binomial(repeats, probability) / repeats
        for sample in range(sample_count):
            rows.append(
                {
                    "model": model,
                    "model_order": model_order,
                    "sample_id": f"S{sample + 1:03d}",
                    "fidelity": round(float(fidelity[sample]), 6),
                    "score": round(float(score[sample]), 6),
                    "teacher_correct": int(teacher_correct[sample]),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the paper-ready two-panel trajectory fidelity figure."
    )
    parser.add_argument("--trajectory-csv", type=Path, default=DEFAULT_TRAJECTORY_CSV)
    parser.add_argument("--score-csv", type=Path, default=DEFAULT_SCORE_CSV)
    parser.add_argument("--out", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--preview", type=Path, default=DEFAULT_PREVIEW)
    parser.add_argument("--appendix-out", type=Path, default=DEFAULT_APPENDIX_PDF)
    parser.add_argument("--appendix-preview", type=Path, default=DEFAULT_APPENDIX_PREVIEW)
    parser.add_argument("--baseline-model", default=BASELINE)
    parser.add_argument("--student-model", default=STUDENT)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--panel-b", choices=("dose", "score"), default="dose",
                        help="panel (b): dose-response curves (default) or the "
                             "fidelity-vs-score scatter")
    parser.add_argument("--trim-outliers", action="store_true",
                        help="score panel only: drop per-model fidelity outliers "
                             "by the IQR rule (Q1-1.5*IQR), counts printed")
    parser.add_argument(
        "--refresh-demo",
        action="store_true",
        help="Rewrite the deterministic demo CSV files before plotting.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.refresh_demo:
        write_csv(args.trajectory_csv, make_demo_trajectory(args.seed))
        write_csv(args.score_csv, make_demo_scores(args.seed + 6))
    student = make_figure(
        args.trajectory_csv,
        args.score_csv,
        args.out,
        args.preview,
        args.baseline_model,
        args.student_model,
        args.seed,
        panel_b=args.panel_b,
        trim_outliers=args.trim_outliers,
    )
    make_appendix_figure(
        student,
        args.appendix_out,
        args.appendix_preview,
        args.seed,
    )
