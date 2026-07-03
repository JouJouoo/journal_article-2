from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from figure_style import (
    apply_publication_style,
    display_variant,
    save_publication_figure,
    style_axes,
    variant_color,
)


CASES = [
    ("benchmark_ieee33bw", "IEEE 33-bus"),
    ("benchmark_ieee69", "IEEE 69-bus"),
]
VARIANTS = ["tecsf", "myopic_opt", "heuristic"]


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_case_summaries(root: Path) -> list[dict]:
    cases = []
    for suite, display in CASES:
        path = root / suite / "summary.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing summary: {path}")
        payload = _read_json(path)
        by_variant = {row["variant"]: row for row in payload.get("by_variant", [])}
        missing = [variant for variant in VARIANTS if variant not in by_variant]
        if missing:
            raise ValueError(f"{path} missing variants: {', '.join(missing)}")
        cases.append(
            {
                "suite": suite,
                "display": display,
                "benchmark": payload.get("benchmark_case", {}),
                "by_variant": by_variant,
            }
        )
    return cases


def _metric(row: dict, name: str) -> tuple[float, float]:
    return float(row[f"{name}_mean"]), float(row.get(f"{name}_std", 0.0))


def _add_panel_label(ax, label: str) -> None:
    ax.text(-0.12, 1.08, label, transform=ax.transAxes, fontsize=10, fontweight="bold")


def _plot_case_metric_bars(ax, cases: list[dict], metric: str, ylabel: str, title: str) -> None:
    x = np.arange(len(cases), dtype=float)
    width = 0.23
    for idx, variant in enumerate(VARIANTS):
        means = []
        stds = []
        for case in cases:
            mean, std = _metric(case["by_variant"][variant], metric)
            means.append(mean)
            stds.append(std)
        ax.bar(
            x + (idx - 1) * width,
            means,
            width=width,
            yerr=stds,
            color=variant_color(variant),
            label=display_variant(variant),
            capsize=2,
            alpha=0.86,
        )
    ax.set_xticks(x, [case["display"] for case in cases])
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold")
    style_axes(ax)


def _plot_allowance_lccoins(ax, cases: list[dict]) -> None:
    positions = []
    labels = []
    rows = []
    x = 0.0
    for case in cases:
        for variant in VARIANTS:
            positions.append(x)
            labels.append(f"{case['display'].replace('IEEE ', '')}\n{display_variant(variant)}")
            rows.append((case, variant))
            x += 1.0
        x += 0.6

    width = 0.34
    for pos, (case, variant) in zip(positions, rows):
        color = variant_color(variant)
        allowance, allowance_std = _metric(case["by_variant"][variant], "eval_net_carbon_allowance_need")
        lccoins, lccoins_std = _metric(case["by_variant"][variant], "eval_lccoins")
        ax.bar(
            pos - width / 2,
            allowance,
            width=width,
            yerr=allowance_std,
            color=color,
            alpha=0.86,
            capsize=2,
        )
        ax.bar(
            pos + width / 2,
            lccoins,
            width=width,
            yerr=lccoins_std,
            color=color,
            alpha=0.38,
            hatch="///",
            edgecolor=color,
            linewidth=0.7,
            capsize=2,
        )
    ax.set_xticks(positions, labels, rotation=35, ha="right")
    ax.set_ylabel("Amount")
    ax.set_title("Allowance need and LCCoins", loc="left", fontweight="bold")
    ax.legend(
        handles=[
            Patch(facecolor="#777777", alpha=0.86, label="Net allowance need"),
            Patch(facecolor="#777777", alpha=0.38, hatch="///", label="LCCoins"),
        ],
        frameon=False,
        loc="upper left",
    )
    style_axes(ax)


def _plot_feasibility_gate(ax, cases: list[dict]) -> None:
    cell_text = []
    for case in cases:
        tecsf = case["by_variant"]["tecsf"]
        success, _ = _metric(tecsf, "eval_settlement_success_rate")
        violation, _ = _metric(tecsf, "eval_max_violation")
        cell_text.append([f"{success:.3f}", f"{violation:.3f}"])

    ax.axis("off")
    ax.set_title("LC-MAPPO feasibility gate", loc="left", fontweight="bold", pad=8)
    table = ax.table(
        cellText=cell_text,
        rowLabels=[case["display"] for case in cases],
        colLabels=["Settlement success", "Max violation"],
        loc="center",
        cellLoc="center",
        rowLoc="center",
        colLoc="center",
        colWidths=[0.36, 0.30],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.5)
    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.5)
        if row == 0:
            cell.set_facecolor("#EAEAEA")
            cell.set_text_props(weight="bold")
        elif col >= 0:
            cell.set_facecolor("#EAF4EA")


def plot_ieee_standard_cases(
    root: str | Path,
    output_dir: str | Path,
    formats: Iterable[str] = ("pdf", "svg", "png"),
    dpi: int = 600,
) -> list[Path]:
    root = Path(root)
    output_dir = Path(output_dir)
    cases = _load_case_summaries(root)
    apply_publication_style()

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.0), constrained_layout=True)
    _plot_case_metric_bars(
        axes[0, 0],
        cases,
        "eval_system_social_cost",
        "System social cost",
        "Economic outcome",
    )
    _plot_case_metric_bars(
        axes[0, 1],
        cases,
        "eval_grid_carbon_emission",
        "Grid carbon emission",
        "Carbon outcome",
    )
    _plot_allowance_lccoins(axes[1, 0], cases)
    _plot_feasibility_gate(axes[1, 1], cases)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[0, 0].legend(handles, labels, frameon=False, loc="upper left")
    for label, ax in zip("ABCD", axes.reshape(-1)):
        _add_panel_label(ax, label)

    paths = save_publication_figure(
        fig,
        output_dir / "fig7_ieee_standard_cases",
        formats=formats,
        dpi=dpi,
    )
    plt.close(fig)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the IEEE 33/69 paper figure.")
    parser.add_argument("--root", required=True, help="Root directory containing benchmark summaries.")
    parser.add_argument("--output-dir", required=True, help="Directory for paper figure outputs.")
    parser.add_argument("--formats", nargs="+", default=["pdf", "svg", "png"])
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()

    for path in plot_ieee_standard_cases(
        root=args.root,
        output_dir=args.output_dir,
        formats=args.formats,
        dpi=args.dpi,
    ):
        print(f"figure={path}")


if __name__ == "__main__":
    main()
