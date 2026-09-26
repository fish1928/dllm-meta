from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.transforms import blended_transform_factory as blend


ROOT = Path(__file__).resolve().parent
DEFAULT_STAGE1_CSV = ROOT / "data" / "diagram4_stage1.csv"
DEFAULT_STAGE2_CSV = ROOT / "data" / "diagram4_stage2.csv"
DEFAULT_PDF = ROOT / "output" / "pdf" / "diagram4_ablation.pdf"
DEFAULT_PREVIEW = ROOT / "output" / "figures" / "diagram4_ablation.png"

COLORS = {
    "ink": "#242424",
    "muted": "#666666",
    "grid": "#DDDDE0",
    "accent": "#287C78",
    "accent_bg": "#CFE5E1",
    "alternative": "#D9764A",
    "neutral": "#9E9E9E",
    "benchmark": "#BEBEC2",
    "separator": "#E5E5E7",
    # Pale tints of the figure's coral and teal accents distinguish the two stages.
    "stage1_bg": "#FCF4F0",
    "stage2_bg": "#F0F7F6",
}

STAGE1_COLUMNS = {
    "group",
    "group_order",
    "option",
    "option_order",
    "is_adopted",
    "delta",
    "ci_low",
    "ci_high",
}
STAGE2_COLUMNS = {
    "option",
    "option_order",
    "is_adopted",
    "delta",
    "ci_low",
    "ci_high",
    "benchmark_deltas",
}


def paired_bootstrap(
    candidate: np.ndarray,
    adopted: np.ndarray,
    n_boot: int = 10000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Return mean candidate-minus-adopted accuracy and its paired 95% CI in points."""
    candidate = np.asarray(candidate, dtype=float)
    adopted = np.asarray(adopted, dtype=float)
    if candidate.shape != adopted.shape or candidate.size == 0:
        raise ValueError("candidate and adopted must contain the same non-zero number of prompts")
    rng = np.random.default_rng(seed)
    differences = (candidate - adopted) * 100
    samples = rng.integers(0, differences.size, (n_boot, differences.size))
    boot_means = differences[samples].mean(axis=1)
    low, high = np.percentile(boot_means, (2.5, 97.5))
    return float(differences.mean()), float(low), float(high)


def read_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path} contains no data rows")
    return rows


def parse_bool(value: str, path: Path, line: int) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise ValueError(f"{path}:{line}: is_adopted must be 0/1, true/false, or yes/no")


def parse_interval(row: dict[str, str], path: Path, line: int) -> tuple[float, float, float]:
    try:
        delta = float(row["delta"])
        low = float(row["ci_low"])
        high = float(row["ci_high"])
    except ValueError as error:
        raise ValueError(f"{path}:{line}: delta and CI values must be numeric") from error
    if low > high or not low <= delta <= high:
        raise ValueError(f"{path}:{line}: expected ci_low <= delta <= ci_high")
    return delta, low, high


def load_stage1(path: Path) -> list[tuple[str, str, list[tuple[str, float, float, float]]]]:
    rows = read_csv(path, STAGE1_COLUMNS)
    grouped: dict[
        tuple[int, str],
        list[tuple[int, bool, str, float | None, float | None, float | None]],
    ] = defaultdict(list)
    for line, row in enumerate(rows, start=2):
        try:
            group_order = int(row["group_order"])
            option_order = int(row["option_order"])
        except ValueError as error:
            raise ValueError(f"{path}:{line}: group_order and option_order must be integers") from error
        is_adopted = parse_bool(row["is_adopted"], path, line)
        if is_adopted:
            interval = (None, None, None)
        else:
            interval = parse_interval(row, path, line)
        grouped[(group_order, row["group"].strip())].append(
            (option_order, is_adopted, row["option"].strip(), *interval)
        )

    output: list[tuple[str, str, list[tuple[str, float, float, float]]]] = []
    for (_, group), options in sorted(grouped.items()):
        options.sort(key=lambda item: item[0])
        adopted_options = [item for item in options if item[1]]
        if len(adopted_options) != 1:
            raise ValueError(f"{path}: group '{group}' must contain exactly one adopted option")
        adopted_name = adopted_options[0][2]
        alternatives = [
            (name, float(delta), float(low), float(high))
            for _, is_adopted, name, delta, low, high in options
            if not is_adopted
        ]
        if not alternatives:
            raise ValueError(f"{path}: group '{group}' must contain at least one alternative")
        output.append((group, adopted_name, alternatives))
    return output


def load_stage2(path: Path) -> tuple[str, list[dict[str, object]]]:
    rows = read_csv(path, STAGE2_COLUMNS)
    parsed: list[tuple[int, dict[str, object]]] = []
    for line, row in enumerate(rows, start=2):
        try:
            order = int(row["option_order"])
        except ValueError as error:
            raise ValueError(f"{path}:{line}: option_order must be an integer") from error
        is_adopted = parse_bool(row["is_adopted"], path, line)
        if is_adopted:
            delta, low, high = 0.0, 0.0, 0.0
        else:
            delta, low, high = parse_interval(row, path, line)
        values = []
        for value in row["benchmark_deltas"].split(";"):
            if value.strip():
                try:
                    values.append(float(value))
                except ValueError as error:
                    raise ValueError(
                        f"{path}:{line}: benchmark_deltas must be semicolon-separated numbers"
                    ) from error
        if not is_adopted and not values:
            raise ValueError(f"{path}:{line}: each alternative needs at least one benchmark delta")
        parsed.append(
            (
                order,
                {
                    "name": row["option"].strip(),
                    "adopted": is_adopted,
                    "delta": delta,
                    "low": low,
                    "high": high,
                    "benchmarks": values,
                    "annotation": row.get("annotation", "").strip(),
                },
            )
        )

    parsed.sort(key=lambda item: item[0])
    items = [item for _, item in parsed]
    adopted_items = [item for item in items if item["adopted"]]
    if len(adopted_items) != 1:
        raise ValueError(f"{path}: Stage II must contain exactly one adopted option")
    alternatives = [item for item in items if not item["adopted"]]
    if not alternatives:
        raise ValueError(f"{path}: Stage II must contain at least one alternative")
    return str(adopted_items[0]["name"]), alternatives


def compute_ylim_stage1(stage1: list) -> tuple[float, float]:
    values = [0.0]
    for _, _, alternatives in stage1:
        for _, delta, low, high in alternatives:
            values.extend((delta, low, high))
    low, high = min(values), max(values)
    span = max(high - low, 1.0)
    return low - 0.10 * span, high + 0.26 * span


def compute_ylim_stage2(stage2: list[dict[str, object]]) -> tuple[float, float]:
    values = [0.0]
    for item in stage2:
        values.extend((float(item["delta"]), float(item["low"]), float(item["high"])))
        values.extend(float(value) for value in item["benchmarks"])
    low, high = min(values), max(values)
    span = max(high - low, 1.0)
    return low - 0.10 * span, high + 0.26 * span


def draw_interval(ax: plt.Axes, x: float, delta: float, low: float, high: float) -> None:
    significant = high < 0 or low > 0
    color = COLORS["alternative"] if significant else COLORS["neutral"]
    ax.plot([x, x], [low, high], color=color, lw=1.0, zorder=3, solid_capstyle="butt")
    ax.plot(
        x,
        delta,
        "o",
        ms=3.8,
        mfc=color if significant else "white",
        mec=color,
        mew=0.85,
        zorder=4,
    )


def draw_adopted(ax: plt.Axes, x: float, band_half_width: float = 0.42) -> None:
    ax.axvspan(
        x - band_half_width,
        x + band_half_width,
        color=COLORS["accent_bg"],
        lw=0,
        zorder=0,
    )
    ax.plot(x, 0, "D", ms=3.8, color=COLORS["accent"], zorder=5)


def style_axis(
    ax: plt.Axes,
    ticks: list[float],
    labels: list[str],
    adopted_indices: set[int],
) -> None:
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=38, ha="right", rotation_mode="anchor")
    for index, label in enumerate(ax.get_xticklabels()):
        label.set_fontsize(5.6)
        label.set_color(COLORS["accent"] if index in adopted_indices else COLORS["muted"])
        if index in adopted_indices:
            label.set_fontweight("bold")
    ax.axhline(0, color=COLORS["accent"], lw=0.75, zorder=2)
    ax.yaxis.grid(True, color=COLORS["grid"], lw=0.45, alpha=0.75)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(direction="out", length=2.5, width=0.65, pad=1.5)
    ax.tick_params(axis="x", length=0, pad=2)


def make_figure(
    stage1_csv: Path,
    stage2_csv: Path,
    output_pdf: Path,
    preview_png: Path | None,
) -> None:
    stage1 = load_stage1(stage1_csv)
    stage2_adopted, stage2 = load_stage2(stage2_csv)
    # separate y scales: Stage I is measured in offline recall@5 points,
    # Stage II in closed-loop accuracy points
    y_limits1 = compute_ylim_stage1(stage1)
    y_limits = compute_ylim_stage2(stage2)

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 6.2,
            "axes.titlesize": 7.2,
            "axes.labelsize": 6.3,
            "xtick.labelsize": 5.6,
            "ytick.labelsize": 5.6,
            "legend.fontsize": 5.2,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    figure = plt.figure(figsize=(7.16, 2.06), facecolor="white")
    outer = figure.add_gridspec(
        1,
        2,
        width_ratios=(2.35, 1.0),
        left=0.075,
        right=0.982,
        bottom=0.215,
        top=0.82,
        wspace=0.30,
    )

    panel_bottom, panel_top = 0.0, 0.99
    for panel, color in enumerate((COLORS["stage1_bg"], COLORS["stage2_bg"])):
        position = outer[panel].get_position(figure)
        horizontal_pad = 0.006
        figure.add_artist(
            Rectangle(
                (position.x0 - horizontal_pad, panel_bottom),
                position.width + 2 * horizontal_pad,
                panel_top - panel_bottom,
                transform=figure.transFigure,
                facecolor=color,
                edgecolor="none",
                zorder=-10,
            )
        )

    ax1 = figure.add_subplot(outer[0])
    ax2 = figure.add_subplot(outer[1])
    ax1.set_facecolor(COLORS["stage1_bg"])
    ax2.set_facecolor(COLORS["stage2_bg"])

    transform1 = blend(ax1.transData, ax1.transAxes)
    x = 0.0
    gap = 0.62
    ticks1: list[float] = []
    labels1: list[str] = []
    adopted1: set[int] = set()
    for group_index, (group, adopted_name, alternatives) in enumerate(stage1):
        group_start = x
        draw_adopted(ax1, x)
        ticks1.append(x)
        labels1.append(adopted_name)
        adopted1.add(len(labels1) - 1)
        x += 1
        for name, delta, low, high in alternatives:
            draw_interval(ax1, x, delta, low, high)
            ticks1.append(x)
            labels1.append(name)
            x += 1
        group_center = (group_start + x - 1) / 2
        ax1.text(
            group_center,
            1.025,
            group,
            transform=transform1,
            ha="center",
            va="bottom",
            fontsize=6.2,
            fontweight="bold",
            color=COLORS["ink"],
        )
        if group_index < len(stage1) - 1:
            separator_x = x - 1 + (1 + gap) / 2
            ax1.axvline(separator_x, color=COLORS["separator"], lw=0.6, zorder=1)
        x += gap

    ax1.set_xlim(-0.60, x - gap - 0.40)
    ax1.set_ylim(*y_limits1)
    style_axis(ax1, ticks1, labels1, adopted1)
    ax1.set_ylabel(r"$\Delta$ recall@5 vs. adopted (points)", labelpad=2)

    spacing = 1.55
    ticks2 = [0.0]
    labels2 = [stage2_adopted]
    draw_adopted(ax2, 0.0)
    for index, item in enumerate(stage2, start=1):
        x_position = index * spacing
        benchmark_values = np.asarray(item["benchmarks"], dtype=float)
        jitter = np.linspace(-0.30, 0.30, benchmark_values.size)
        ax2.plot(
            x_position + jitter,
            benchmark_values,
            "o",
            ms=2.1,
            color=COLORS["benchmark"],
            zorder=2,
            mew=0,
        )
        draw_interval(
            ax2,
            x_position,
            float(item["delta"]),
            float(item["low"]),
            float(item["high"]),
        )
        ticks2.append(x_position)
        labels2.append(str(item["name"]))

        annotation = str(item["annotation"])
        if annotation:
            bracket_y = y_limits[1] - 0.17 * (y_limits[1] - y_limits[0])
            bracket_height = 0.06 * (y_limits[1] - y_limits[0])
            ax2.plot(
                [0, 0, x_position, x_position],
                [bracket_y - bracket_height, bracket_y, bracket_y, bracket_y - bracket_height],
                color=COLORS["accent"],
                lw=0.65,
            )
            ax2.text(
                x_position / 2,
                bracket_y + 0.02 * (y_limits[1] - y_limits[0]),
                annotation,
                ha="center",
                va="bottom",
                fontsize=5.4,
                color=COLORS["accent"],
                style="italic",
            )

    ax2.set_xlim(-0.68, len(stage2) * spacing + 0.62)
    ax2.set_ylim(*y_limits)
    style_axis(ax2, ticks2, labels2, {0})
    ax2.set_ylabel(r"$\Delta$ accuracy vs. adopted (points)", labelpad=2)

    legend_handles = [
        Line2D([], [], ls="", marker="D", ms=3.6, color=COLORS["accent"], label="Adopted"),
        Line2D(
            [], [], ls="", marker="o", ms=3.6,
            mfc=COLORS["alternative"], mec=COLORS["alternative"],
            label="95% CI excludes zero",
        ),
        Line2D(
            [], [], ls="", marker="o", ms=3.6,
            mfc="white", mec=COLORS["neutral"],
            label="95% CI includes zero",
        ),
        Line2D(
            [], [], ls="", marker="o", ms=2.4,
            color=COLORS["benchmark"], label="Per benchmark",
        ),
    ]
    ax2.legend(
        handles=legend_handles,
        loc="lower left",
        fontsize=5.0,
        frameon=False,
        handletextpad=0.25,
        labelspacing=0.28,
        borderaxespad=0.25,
    )

    title_y = 0.895
    for panel, title in enumerate(
        ("(a) Stage I: Router design (offline recall@5)",
         "(b) Stage II: Feature aging (e2e)")
    ):
        position = outer[panel].get_position(figure)
        figure.text(
            (position.x0 + position.x1) / 2,
            title_y,
            title,
            ha="center",
            va="bottom",
            fontsize=plt.rcParams["axes.titlesize"],
            fontweight="bold",
            color=COLORS["ink"],
        )

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "Title": "TRACEQUERY ablation study",
        "Creator": Path(__file__).name,
    }
    figure.savefig(output_pdf, metadata=metadata)
    if preview_png is not None:
        preview_png.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(preview_png, dpi=300)
    plt.close(figure)
    print(f"saved {output_pdf}")
    if preview_png is not None:
        print(f"saved {preview_png}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the paper-ready two-panel TRACEQUERY ablation figure."
    )
    parser.add_argument("--stage1-csv", type=Path, default=DEFAULT_STAGE1_CSV)
    parser.add_argument("--stage2-csv", type=Path, default=DEFAULT_STAGE2_CSV)
    parser.add_argument("--out", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--preview", type=Path, default=DEFAULT_PREVIEW)
    args = parser.parse_args()
    make_figure(args.stage1_csv, args.stage2_csv, args.out, args.preview)


if __name__ == "__main__":
    main()
