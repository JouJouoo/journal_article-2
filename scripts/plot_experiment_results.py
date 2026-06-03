from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from figure_style import (
    CONSTRAINT_COLORS,
    apply_publication_style,
    display_variant,
    format_label,
    save_publication_figure,
    style_axes,
    variant_color,
    variant_sort_key,
)


METRIC_PANELS = [
    ("eval_feasible_rate", "Feasible rate", "higher"),
    ("eval_system_cost", "System cost", "lower"),
    ("eval_grid_carbon_emission", "Grid carbon emission", "lower"),
    ("eval_net_carbon_allowance_need", "Net allowance need", "lower"),
    ("eval_renewable_consumption_rate", "Renewable consumption rate", "higher"),
    ("eval_settlement_success_rate", "Settlement success rate", "higher"),
    ("eval_max_violation", "Max violation", "lower"),
]

CONSTRAINT_METRICS = [
    ("eval_voltage_violation", "Voltage"),
    ("eval_line_violation", "Line"),
    ("eval_soc_violation", "SOC"),
    ("eval_trade_violation", "Trade"),
]


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _group_rows(rows: list[dict], keys: tuple[str, ...]) -> dict[tuple, list[dict]]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key, "") for key in keys)].append(row)
    return dict(grouped)


def _mean_std(rows: list[dict], metric: str) -> tuple[float, float]:
    values = [float(row[metric]) for row in rows if isinstance(row.get(metric), (int, float))]
    if not values:
        return 0.0, 0.0
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def _metrics_path(row: dict) -> Path:
    return Path(row["output_dir"]) / f"{row['variant']}_metrics.json"


def _grid_shape(item_count: int, max_cols: int = 3) -> tuple[int, int]:
    cols = min(max_cols, max(item_count, 1))
    rows = int(math.ceil(item_count / cols))
    return rows, cols


def _group_sort_key(key: tuple) -> tuple:
    if len(key) == 1:
        return (variant_sort_key(str(key[0])), str(key[0]))
    return (str(key[0]), variant_sort_key(str(key[-1])), str(key[-1]))


def _group_variant(key: tuple) -> str:
    return str(key[-1])


def _group_name(key: tuple) -> str:
    if len(key) == 1:
        return display_variant(str(key[0]))
    return f"{format_label(str(key[0]))}\n{display_variant(str(key[1]))}"


def _save(fig, output_dir: Path, name: str) -> list[Path]:
    return save_publication_figure(fig, output_dir / name)


def _plot_learning_curves(rows: list[dict], output_dir: Path) -> list[Path]:
    paths = []
    by_label = _group_rows(rows, ("label",))
    for (label,), label_rows in by_label.items():
        by_variant = _group_rows(label_rows, ("variant",))
        fig, ax = plt.subplots(figsize=(9, 5.2))
        for (variant,), variant_rows in sorted(by_variant.items()):
            curves = []
            for row in variant_rows:
                metrics_file = _metrics_path(row)
                if not metrics_file.exists():
                    continue
                metrics = _read_json(metrics_file)
                curve = [float(item.get("mean_reward", 0.0)) for item in metrics]
                if curve:
                    curves.append(curve)
            if not curves:
                continue
            length = min(len(curve) for curve in curves)
            arr = np.asarray([curve[:length] for curve in curves], dtype=float)
            xs = np.arange(length)
            mean = arr.mean(axis=0)
            std = arr.std(axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros(length)
            color = variant_color(variant)
            ax.plot(xs, mean, label=display_variant(variant), color=color, linewidth=1.3)
            ax.fill_between(xs, mean - std, mean + std, color=color, alpha=0.12, linewidth=0)
        ax.set_title(f"Training mean reward - {format_label(label) if label else 'default'}")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Mean reward")
        style_axes(ax)
        ax.legend(frameon=False, ncol=3)
        fig.tight_layout()
        saved = _save(fig, output_dir, f"learning_curves_{label or 'default'}")
        plt.close(fig)
        paths.extend(saved)
    return paths


def _plot_eval_panels(rows: list[dict], output_dir: Path) -> list[Path]:
    labels = sorted({row.get("label", "") for row in rows})
    single_label = len(labels) <= 1
    group_keys = ("variant",) if single_label else ("label", "variant")
    grouped = _group_rows(rows, group_keys)
    ordered_keys = sorted(grouped, key=_group_sort_key)
    names = [_group_name(key) for key in ordered_keys]
    x = np.arange(len(names))

    nrows, ncols = _grid_shape(len(METRIC_PANELS), max_cols=3)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 3.2 * nrows))
    flat_axes = np.asarray(axes).reshape(-1)
    colors = [variant_color(_group_variant(key)) for key in ordered_keys]
    for ax, (metric, title, direction) in zip(flat_axes, METRIC_PANELS):
        means = []
        stds = []
        for key in ordered_keys:
            mean, std = _mean_std(grouped[key], metric)
            means.append(mean)
            stds.append(std)
        ax.bar(x, means, yerr=stds, color=colors, alpha=0.86, capsize=3)
        ax.set_title(f"{title} ({direction})", loc="left", fontweight="bold")
        ax.set_xticks(x, names, rotation=45, ha="right")
        style_axes(ax)
    for ax in flat_axes[len(METRIC_PANELS) :]:
        ax.remove()
    fig.tight_layout()
    paths = _save(fig, output_dir, "evaluation_metric_panels")
    plt.close(fig)
    return paths


def _plot_constraint_bars(rows: list[dict], output_dir: Path) -> list[Path]:
    grouped = _group_rows(rows, ("variant",))
    variants = [key[0] for key in sorted(grouped, key=lambda key: variant_sort_key(str(key[0])))]
    x = np.arange(len(variants))
    width = 0.18
    fig, ax = plt.subplots(figsize=(10, 5.4))
    for idx, (metric, label) in enumerate(CONSTRAINT_METRICS):
        means = [_mean_std(grouped[(variant,)], metric)[0] for variant in variants]
        ax.bar(
            x + (idx - 1.5) * width,
            means,
            width=width,
            label=label,
            color=CONSTRAINT_COLORS.get(label),
        )
    ax.set_title("Evaluation constraint violation totals", loc="left", fontweight="bold")
    ax.set_xlabel("Variant")
    ax.set_ylabel("Violation total")
    ax.set_xticks(x, [display_variant(variant) for variant in variants], rotation=25, ha="right")
    ax.legend(frameon=False, ncol=4)
    style_axes(ax)
    fig.tight_layout()
    paths = _save(fig, output_dir, "constraint_violations")
    plt.close(fig)
    return paths


def _plot_label_sensitivity(rows: list[dict], output_dir: Path) -> list[Path]:
    labels = sorted({row.get("label", "") for row in rows})
    if len(labels) <= 1:
        return []
    grouped = _group_rows(rows, ("label", "variant"))
    variants = sorted({row.get("variant", "") for row in rows}, key=lambda value: variant_sort_key(str(value)))
    x = np.arange(len(labels))
    width = min(0.8 / max(len(variants), 1), 0.25)
    fig, axes = plt.subplots(2, 1, figsize=(max(10, len(labels) * 1.25), 8.0), sharex=True)
    for ax, metric, ylabel in [
        (axes[0], "eval_lccoins", "LCCoins"),
        (axes[1], "eval_grid_carbon_emission", "Grid carbon emission"),
    ]:
        for idx, variant in enumerate(variants):
            means = []
            for label in labels:
                means.append(_mean_std(grouped.get((label, variant), []), metric)[0])
            ax.bar(
                x + (idx - (len(variants) - 1) / 2) * width,
                means,
                width=width,
                label=display_variant(variant),
                color=variant_color(variant),
            )
        ax.set_ylabel(ylabel)
        style_axes(ax)
    axes[0].set_title("Setting sensitivity", loc="left", fontweight="bold")
    axes[-1].set_xticks(x, [format_label(str(label)) for label in labels], rotation=35, ha="right")
    axes[0].legend(frameon=False, ncol=max(1, len(variants)))
    fig.tight_layout()
    paths = _save(fig, output_dir, "setting_sensitivity")
    plt.close(fig)
    return paths


def _plot_runtime(rows: list[dict], output_dir: Path) -> list[Path]:
    if not any(isinstance(row.get("total_seconds"), (int, float)) for row in rows):
        return []
    labels = sorted({row.get("label", "") for row in rows})
    single_label = len(labels) <= 1
    group_keys = ("variant",) if single_label else ("label", "variant")
    grouped = _group_rows(rows, group_keys)
    ordered_keys = sorted(grouped, key=_group_sort_key)
    names = [_group_name(key) for key in ordered_keys]
    means = [_mean_std(grouped[key], "total_seconds")[0] for key in ordered_keys]
    stds = [_mean_std(grouped[key], "total_seconds")[1] for key in ordered_keys]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(9, len(names) * 0.6), 5.0))
    ax.bar(x, means, yerr=stds, color=[variant_color(_group_variant(key)) for key in ordered_keys], alpha=0.85, capsize=3)
    ax.set_title("Training plus evaluation wall time", loc="left", fontweight="bold")
    ax.set_ylabel("Seconds")
    ax.set_xticks(x, names, rotation=45, ha="right")
    style_axes(ax)
    fig.tight_layout()
    paths = _save(fig, output_dir, "runtime_seconds")
    plt.close(fig)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create figures from TECSF experiment summary.json files."
    )
    parser.add_argument("summary", help="summary.json generated by an experiment script.")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    summary_path = Path(args.summary)
    payload = _read_json(summary_path)
    rows = payload.get("runs", [])
    if not rows:
        raise SystemExit(f"No runs found in {summary_path}")
    output_dir = Path(args.output_dir) if args.output_dir else summary_path.parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    apply_publication_style()

    paths = []
    paths.extend(_plot_learning_curves(rows, output_dir))
    paths.extend(_plot_eval_panels(rows, output_dir))
    paths.extend(_plot_constraint_bars(rows, output_dir))
    paths.extend(_plot_runtime(rows, output_dir))
    paths.extend(_plot_label_sensitivity(rows, output_dir))
    for path in paths:
        print(f"figure={path}")


if __name__ == "__main__":
    main()
