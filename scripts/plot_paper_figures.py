from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from figure_style import (
    apply_publication_style,
    clean_label,
    display_variant,
    format_label,
    label_value,
    mean_std_ci,
    moving_average,
    save_publication_figure,
    style_axes,
    variant_color,
    variant_sort_key,
)


MAIN_VARIANTS = ["tecsf", "mappo", "no_lccoins", "no_lagrange", "preset_low_carbon"]
LCCOINS_VARIANTS = ["tecsf", "no_feedback", "no_lccoins", "preset_low_carbon"]
NETWORK_VARIANTS = ["tecsf", "heuristic", "no_lagrange"]
BENCHMARK_VARIANTS = ["tecsf", "myopic_opt", "heuristic"]


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _summary_rows(root: Path, suite: str) -> list[dict]:
    path = root / suite / "summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing summary: {path}")
    return list(_read_json(path).get("runs", []))


def _settlement_rows(root: Path) -> list[dict]:
    path = root / "settlement_stress" / "summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing settlement summary: {path}")
    return list(_read_json(path).get("rows", []))


def _available_variants(rows: list[dict], preferred: Iterable[str]) -> list[str]:
    present = {str(row.get("variant", "")) for row in rows}
    variants = [variant for variant in preferred if variant in present]
    extras = sorted(present - set(variants), key=variant_sort_key)
    return variants + extras


def _preferred_variants(rows: list[dict], preferred: Iterable[str]) -> list[str]:
    present = {str(row.get("variant", "")) for row in rows}
    return [variant for variant in preferred if variant in present]


def _metric_values(rows: list[dict], metric: str, variant: str | None = None, label: str | None = None) -> list[float]:
    values = []
    for row in rows:
        if variant is not None and row.get("variant") != variant:
            continue
        if label is not None and row.get("label", "") != label:
            continue
        value = row.get(metric)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def _mean_ci(rows: list[dict], metric: str, variant: str, label: str | None = None) -> tuple[float, float]:
    mean, _, ci = mean_std_ci(_metric_values(rows, metric, variant=variant, label=label))
    return mean, ci


def _variant_metric_series(rows: list[dict], metric: str, variants: list[str]) -> tuple[list[float], list[float]]:
    means = []
    cis = []
    for variant in variants:
        mean, ci = _mean_ci(rows, metric, variant)
        means.append(mean)
        cis.append(ci)
    return means, cis


def _load_curves(rows: list[dict], variant: str, metric: str = "mean_reward", window: int = 50) -> np.ndarray | None:
    curves = []
    for row in rows:
        if row.get("variant") != variant:
            continue
        output_dir = row.get("output_dir")
        if not output_dir:
            continue
        path = Path(output_dir) / f"{variant}_metrics.json"
        if not path.exists():
            continue
        metrics = _read_json(path)
        values = np.asarray([float(item.get(metric, 0.0)) for item in metrics], dtype=float)
        if values.size:
            curves.append(moving_average(values, window))
    if not curves:
        return None
    length = min(curve.size for curve in curves)
    return np.vstack([curve[:length] for curve in curves])


def _add_panel_label(ax, label: str) -> None:
    ax.text(-0.12, 1.08, label, transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")


def _plot_curve_panel(ax, rows: list[dict], variants: list[str]) -> None:
    plotted = False
    for variant in variants:
        arr = _load_curves(rows, variant)
        if arr is None:
            continue
        xs = np.arange(arr.shape[1])
        mean = arr.mean(axis=0)
        ci = 1.959963984540054 * arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0]) if arr.shape[0] > 1 else 0.0
        color = variant_color(variant)
        ax.plot(xs, mean, color=color, linewidth=1.6, label=display_variant(variant))
        ax.fill_between(xs, mean - ci, mean + ci, color=color, alpha=0.13, linewidth=0)
        plotted = True
    if not plotted:
        ax.text(0.5, 0.5, "Training curves unavailable", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean reward")
    ax.set_title("Training convergence")
    if plotted:
        ax.legend(frameon=False, ncol=3, loc="lower right")
    style_axes(ax)


def _plot_dot_panel(ax, rows: list[dict], variants: list[str], metric: str, title: str, xlabel: str) -> None:
    y = np.arange(len(variants))
    means, cis = _variant_metric_series(rows, metric, variants)
    for idx, variant in enumerate(variants):
        ax.errorbar(
            means[idx],
            y[idx],
            xerr=cis[idx],
            fmt="o",
            color=variant_color(variant),
            capsize=2,
            markersize=4,
        )
    ax.set_yticks(y, [display_variant(variant) for variant in variants])
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    style_axes(ax, grid_axis="x")


def _plot_grouped_metric_bars(
    ax,
    rows: list[dict],
    variants: list[str],
    metrics: list[tuple[str, str]],
    show_legend: bool = True,
) -> None:
    x = np.arange(len(metrics))
    width = min(0.8 / max(len(variants), 1), 0.16)
    for idx, variant in enumerate(variants):
        means = [_mean_ci(rows, metric, variant)[0] for metric, _ in metrics]
        ax.bar(
            x + (idx - (len(variants) - 1) / 2) * width,
            means,
            width=width,
            color=variant_color(variant),
            label=display_variant(variant),
        )
    ax.set_xticks(x, [label for _, label in metrics])
    ax.set_title("Economic and carbon outcomes")
    if show_legend:
        ax.legend(frameon=False, ncol=2)
    style_axes(ax)


def _plot_safety_panel(ax, rows: list[dict], variants: list[str]) -> None:
    x = np.arange(len(variants))
    success = [_mean_ci(rows, "eval_settlement_success_rate", variant)[0] for variant in variants]
    violation = [_mean_ci(rows, "eval_max_violation", variant)[0] for variant in variants]
    ax.bar(x - 0.16, success, width=0.32, color=[variant_color(v) for v in variants], alpha=0.82)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Settlement success")
    ax.set_xticks(x, [display_variant(variant) for variant in variants], rotation=30, ha="right")
    ax.set_title("Settlement safety")
    style_axes(ax)
    ax2 = ax.twinx()
    ax2.plot(x + 0.16, violation, color="#111111", marker="D", linewidth=1.2, label="Max violation")
    ax2.set_ylabel("Max violation")
    ax2.spines["top"].set_visible(False)
    ax2.legend(frameon=False, loc="upper right")


def _plot_lccoins_panel(ax, rows: list[dict], variants: list[str]) -> None:
    means, cis = _variant_metric_series(rows, "eval_lccoins", variants)
    x = np.arange(len(variants))
    ax.bar(x, means, yerr=cis, color=[variant_color(variant) for variant in variants], capsize=2)
    ax.set_xticks(x, [display_variant(variant) for variant in variants], rotation=30, ha="right")
    ax.set_ylabel("LCCoins")
    ax.set_title("Low-carbon incentive")
    style_axes(ax)


def plot_fig1_main_comparison(
    root: Path,
    output_dir: Path,
    formats: Iterable[str] = ("pdf", "svg", "png"),
    dpi: int = 600,
) -> list[Path]:
    rows = _summary_rows(root, "formal_multiseed")
    variants = _preferred_variants(rows, MAIN_VARIANTS)
    fig = plt.figure(figsize=(7.2, 7.0), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1.0, 1.0])
    axes = [
        fig.add_subplot(gs[0, :]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[2, 0]),
        fig.add_subplot(gs[2, 1]),
    ]
    _plot_curve_panel(axes[0], rows, variants)
    _plot_dot_panel(axes[1], rows, variants, "eval_mean_reward", "Evaluation reward", "Mean reward")
    _plot_grouped_metric_bars(
        axes[2],
        rows,
        variants,
        [
            ("eval_system_cost", "Cost"),
            ("eval_grid_carbon_emission", "Carbon"),
        ],
        show_legend=False,
    )
    _plot_safety_panel(axes[3], rows, variants)
    _plot_lccoins_panel(axes[4], rows, variants)
    for label, ax in zip("ABCDE", axes):
        _add_panel_label(ax, label)
    return save_publication_figure(fig, output_dir / "fig1_main_comparison", formats=formats, dpi=dpi)


def _setting_grid(rows: list[dict]) -> tuple[list[float], list[float]]:
    lines = sorted({float(label_value(str(row.get("label", "")), "line", 0.0)) for row in rows})
    trades = sorted({float(label_value(str(row.get("label", "")), "trade", 0.0)) for row in rows})
    return lines, trades


def _network_matrix(rows: list[dict], variant: str, metric: str) -> tuple[np.ndarray, list[float], list[float]]:
    lines, trades = _setting_grid(rows)
    matrix = np.full((len(lines), len(trades)), np.nan, dtype=float)
    for i, line in enumerate(lines):
        for j, trade in enumerate(trades):
            values = [
                float(row[metric])
                for row in rows
                if row.get("variant") == variant
                and float(label_value(str(row.get("label", "")), "line", -1.0)) == line
                and float(label_value(str(row.get("label", "")), "trade", -1.0)) == trade
                and isinstance(row.get(metric), (int, float))
            ]
            if values:
                matrix[i, j] = float(np.mean(values))
    return matrix, lines, trades


def _draw_heatmap(ax, matrix: np.ndarray, xlabels: list, ylabels: list, title: str, cmap: str, vmin=None, vmax=None):
    image = ax.imshow(matrix, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(xlabels)), [str(item) for item in xlabels])
    ax.set_yticks(np.arange(len(ylabels)), [str(item) for item in ylabels])
    ax.set_xlabel("Trade scale")
    ax.set_ylabel("Line scale")
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7, color="black")
    return image


def _plot_settlement_matrix(ax, rows: list[dict]) -> None:
    cases = sorted({row["case"] for row in rows})
    checks = [
        ("passed", "Pass"),
        ("rollback_energy_error", "Energy"),
        ("rollback_carbon_error", "Carbon"),
        ("rollback_lccoins_error", "LCCoins"),
    ]
    matrix = np.zeros((len(cases), len(checks)), dtype=float)
    for i, case in enumerate(cases):
        case_rows = [row for row in rows if row["case"] == case]
        matrix[i, 0] = 1.0 if all(bool(row["passed"]) for row in case_rows) else 0.0
        for j, (field, _) in enumerate(checks[1:], start=1):
            matrix[i, j] = 1.0 if max(abs(float(row[field])) for row in case_rows) <= 1e-9 else 0.0
    cmap = ListedColormap(["#D55E00", "#009E73"])
    ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(checks)), [label for _, label in checks], rotation=30, ha="right")
    ax.set_yticks(np.arange(len(cases)), [clean_label(case) for case in cases])
    ax.set_title("Settlement stress matrix")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, "pass" if matrix[i, j] else "fail", ha="center", va="center", fontsize=6)


def plot_fig2_safety_stress(
    root: Path,
    output_dir: Path,
    formats: Iterable[str] = ("pdf", "svg", "png"),
    dpi: int = 600,
) -> list[Path]:
    rows = _summary_rows(root, "network_stress")
    settlement_rows = _settlement_rows(root)
    tecsf_success, lines, trades = _network_matrix(rows, "tecsf", "eval_settlement_success_rate")
    no_lagrange_success, _, _ = _network_matrix(rows, "no_lagrange", "eval_settlement_success_rate")
    tecsf_violation, _, _ = _network_matrix(rows, "tecsf", "eval_max_violation")
    no_lagrange_violation, _, _ = _network_matrix(rows, "no_lagrange", "eval_max_violation")

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.2), constrained_layout=True)
    im0 = _draw_heatmap(axes[0, 0], tecsf_success, trades, lines, "TECSF success rate", "viridis", 0, 1)
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.02)
    im1 = _draw_heatmap(axes[0, 1], no_lagrange_success, trades, lines, "w/o Lagrange success rate", "viridis", 0, 1)
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.02)
    reduction = no_lagrange_violation - tecsf_violation
    im2 = _draw_heatmap(axes[1, 0], reduction, trades, lines, "Violation reduction", "cividis")
    fig.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.02)
    _plot_settlement_matrix(axes[1, 1], settlement_rows)
    for label, ax in zip("ABCD", axes.reshape(-1)):
        _add_panel_label(ax, label)
    return save_publication_figure(fig, output_dir / "fig2_safety_stress", formats=formats, dpi=dpi)


def _kappa_series(rows: list[dict], variant: str, metric: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    kappas = sorted({float(label_value(str(row.get("label", "")), "kappa", 0.0)) for row in rows})
    means = []
    cis = []
    for kappa in kappas:
        values = [
            float(row[metric])
            for row in rows
            if row.get("variant") == variant
            and float(label_value(str(row.get("label", "")), "kappa", -1.0)) == kappa
            and isinstance(row.get(metric), (int, float))
        ]
        mean, _, ci = mean_std_ci(values)
        means.append(mean)
        cis.append(ci)
    return np.asarray(kappas), np.asarray(means), np.asarray(cis)


def _plot_kappa_metric(ax, rows: list[dict], variants: list[str], metric: str, title: str, ylabel: str, annotate=False) -> None:
    for variant in variants:
        x, means, cis = _kappa_series(rows, variant, metric)
        color = variant_color(variant)
        ax.plot(x, means, marker="o", linewidth=1.4, color=color, label=display_variant(variant))
        ax.fill_between(x, means - cis, means + cis, color=color, alpha=0.12, linewidth=0)
    ax.axvline(0.1, color="#666666", linestyle="--", linewidth=0.8)
    if annotate:
        ax.text(0.105, 0.05, "kappa=0.1\nweak stability", transform=ax.get_xaxis_transform(), fontsize=6)
    ax.set_xlabel("kappa")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    style_axes(ax)


def plot_fig3_lccoins_sensitivity(
    root: Path,
    output_dir: Path,
    formats: Iterable[str] = ("pdf", "svg", "png"),
    dpi: int = 600,
) -> list[Path]:
    rows = _summary_rows(root, "lccoins_sensitivity")
    variants = _preferred_variants(rows, LCCOINS_VARIANTS)
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 5.4), constrained_layout=True)
    panels = [
        ("eval_mean_reward", "Reward response", "Mean reward", True),
        ("eval_lccoins", "LCCoins feedback", "LCCoins", False),
        ("eval_grid_carbon_emission", "Grid carbon", "Emission", False),
        ("eval_settlement_success_rate", "Settlement success", "Success rate", False),
        ("eval_max_violation", "Constraint violation", "Max violation", False),
    ]
    for ax, (metric, title, ylabel, annotate) in zip(axes.reshape(-1), panels):
        _plot_kappa_metric(ax, rows, variants, metric, title, ylabel, annotate=annotate)
    axes.reshape(-1)[-1].axis("off")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes.reshape(-1)[-1].legend(handles, labels, frameon=False, loc="center")
    for label, ax in zip("ABCDE", axes.reshape(-1)[:5]):
        _add_panel_label(ax, label)
    return save_publication_figure(fig, output_dir / "fig3_lccoins_sensitivity", formats=formats, dpi=dpi)


def _plot_ieee33_seed_scatter(ax, rows: list[dict]) -> None:
    variants = _available_variants(rows, BENCHMARK_VARIANTS)
    rng = np.random.default_rng(2026)
    for idx, variant in enumerate(variants):
        values = _metric_values(rows, "eval_max_violation", variant=variant)
        jitter = rng.normal(0.0, 0.035, len(values))
        ax.scatter(
            np.full(len(values), idx) + jitter,
            values,
            color=variant_color(variant),
            edgecolor="white",
            linewidth=0.3,
            s=28,
            label=display_variant(variant),
        )
        if values:
            ax.hlines(np.mean(values), idx - 0.22, idx + 0.22, color="#111111", linewidth=1.1)
    ax.set_xticks(np.arange(len(variants)), [display_variant(variant) for variant in variants], rotation=25, ha="right")
    ax.set_ylabel("Max violation")
    ax.set_title("IEEE 33-bus seed-level safety")
    style_axes(ax)


def _plot_ieee69_cost_carbon(ax, rows: list[dict]) -> None:
    variants = _available_variants(rows, BENCHMARK_VARIANTS)
    _plot_grouped_metric_bars(
        ax,
        rows,
        variants,
        [
            ("eval_system_cost", "Cost"),
            ("eval_grid_carbon_emission", "Carbon"),
        ],
    )
    ax.set_title("IEEE 69-bus cost/carbon")


def _plot_runtime_ratio(ax, rows: list[dict]) -> None:
    labels = sorted({str(row.get("label", "")) for row in rows})
    x = np.arange(len(labels))
    tecsf_ratio = []
    for label in labels:
        base = _mean_ci(rows, "total_seconds", "heuristic", label=label)[0]
        tecsf = _mean_ci(rows, "total_seconds", "tecsf", label=label)[0]
        tecsf_ratio.append(tecsf / base if base > 0 else 0.0)
    ax.bar(x, tecsf_ratio, color=variant_color("tecsf"), width=0.6)
    ax.axhline(1.0, color="#111111", linestyle="--", linewidth=0.8)
    ax.set_xticks(x, [format_label(label) for label in labels], rotation=25, ha="right")
    ax.set_ylabel("TECSF / heuristic time")
    ax.set_title("Relative runtime")
    style_axes(ax)


def _plot_scalability_boundary(ax, rows: list[dict]) -> None:
    agents = sorted({int(label_value(str(row.get("label", "")), "agents", 0)) for row in rows})
    nodes = sorted({int(label_value(str(row.get("label", "")), "nodes", 0)) for row in rows})
    matrix = np.full((len(agents), len(nodes)), np.nan, dtype=float)
    for i, agent_count in enumerate(agents):
        for j, node_count in enumerate(nodes):
            values = [
                float(row["eval_settlement_success_rate"])
                for row in rows
                if row.get("variant") == "tecsf"
                and int(label_value(str(row.get("label", "")), "agents", -1)) == agent_count
                and int(label_value(str(row.get("label", "")), "nodes", -1)) == node_count
                and isinstance(row.get("eval_settlement_success_rate"), (int, float))
            ]
            if values:
                matrix[i, j] = float(np.mean(values))
    image = ax.imshow(matrix, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(nodes)), [str(item) for item in nodes])
    ax.set_yticks(np.arange(len(agents)), [str(item) for item in agents])
    ax.set_xlabel("Nodes")
    ax.set_ylabel("Agents")
    ax.set_title("TECSF scalability boundary")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if np.isfinite(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white")
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.02)


def plot_fig4_generalization_scalability(
    root: Path,
    output_dir: Path,
    formats: Iterable[str] = ("pdf", "svg", "png"),
    dpi: int = 600,
) -> list[Path]:
    ieee33_rows = _summary_rows(root, "benchmark_ieee33bw")
    ieee69_rows = _summary_rows(root, "benchmark_ieee69")
    scalability_rows = _summary_rows(root, "scalability")
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8), constrained_layout=True)
    _plot_ieee33_seed_scatter(axes[0, 0], ieee33_rows)
    _plot_ieee69_cost_carbon(axes[0, 1], ieee69_rows)
    _plot_runtime_ratio(axes[1, 0], scalability_rows)
    _plot_scalability_boundary(axes[1, 1], scalability_rows)
    for label, ax in zip("ABCD", axes.reshape(-1)):
        _add_panel_label(ax, label)
    return save_publication_figure(fig, output_dir / "fig4_generalization_scalability", formats=formats, dpi=dpi)


def plot_all(
    root: str | Path,
    output_dir: str | Path | None = None,
    formats: Iterable[str] = ("pdf", "svg", "png"),
    dpi: int = 600,
) -> dict[str, list[Path]]:
    apply_publication_style()
    root = Path(root)
    output_dir = Path(output_dir) if output_dir else root / "paper_figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "fig1_main_comparison": plot_fig1_main_comparison(root, output_dir, formats=formats, dpi=dpi),
        "fig2_safety_stress": plot_fig2_safety_stress(root, output_dir, formats=formats, dpi=dpi),
        "fig3_lccoins_sensitivity": plot_fig3_lccoins_sensitivity(root, output_dir, formats=formats, dpi=dpi),
        "fig4_generalization_scalability": plot_fig4_generalization_scalability(root, output_dir, formats=formats, dpi=dpi),
    }
    plt.close("all")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Create publication-oriented TECSF paper figures.")
    parser.add_argument("root", help="Experiment report root containing suite summary.json files.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--formats", nargs="+", default=["pdf", "svg", "png"])
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()

    outputs = plot_all(args.root, output_dir=args.output_dir, formats=args.formats, dpi=args.dpi)
    for name, paths in outputs.items():
        for path in paths:
            print(f"{name}={path}")


if __name__ == "__main__":
    main()
