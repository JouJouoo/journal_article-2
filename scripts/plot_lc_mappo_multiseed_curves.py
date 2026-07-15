#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LC-MAPPO 三种子训练曲线对比图生成脚本。

在种子 7/42/100 的 10000-episode 训练全部完成后运行，生成：
1. 每个种子的独立收敛曲线（奖励/P2P/电网）
2. 三种子叠加对比曲线（8个核心指标，含均值±标准差阴影）
3. 最终性能汇总表（最后1000 episodes统计）

用法:
    python scripts/plot_lc_mappo_multiseed_curves.py
    python scripts/plot_lc_mappo_multiseed_curves.py --base-dir outputs/lc_mappo_10000ep_multiseed_20260705
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from figure_style import (
    OKABE_ITO,
    apply_publication_style,
    moving_average,
    save_publication_figure,
    style_axes,
)


DEFAULT_BASE_DIR = "outputs/lc_mappo_10000ep_multiseed_20260705"
VARIANT = "lc_mappo"
SMOOTH_WINDOW = 100
TAIL_EPISODES = 1000

# 需要绘制对比图的指标
PLOT_METRICS = [
    ("mean_reward", "Mean Reward"),
    ("total_reward", "Total Reward"),
    ("p2p_energy", "P2P Energy (kWh)"),
    ("system_cost", "System Cost"),
    ("grid_carbon_emission", "Grid Carbon Emission"),
    ("carbon_reduction", "Carbon Reduction"),
    ("lccoins", "LCCoins Minted"),
    ("settlement_success_rate", "Settlement Success Rate"),
]

# 最终汇总关心的指标
SUMMARY_METRICS = [
    "mean_reward",
    "total_reward",
    "p2p_energy",
    "system_cost",
    "grid_sell_energy",
    "grid_sell_revenue",
    "grid_carbon_emission",
    "carbon_reduction",
    "lccoins",
    "settlement_success_rate",
    "feasible_rate",
    "max_violation",
    "renewable_consumption_rate",
    "consensus_confirmed_rate",
]


def load_metrics(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_metric(metrics: list[dict], key: str) -> np.ndarray:
    return np.array([m[key] for m in metrics], dtype=float)


def extract_episodes(metrics: list[dict]) -> np.ndarray:
    return np.array([m["episode"] for m in metrics], dtype=float)


def plot_single_seed_curves(metrics, seed, output_dir, window=SMOOTH_WINDOW):
    """为单个种子生成收敛曲线图（3子图：奖励/P2P/电网）。"""
    episodes = extract_episodes(metrics)
    total_rewards = extract_metric(metrics, "total_reward")
    p2p_energy = extract_metric(metrics, "p2p_energy")
    grid_buy = extract_metric(metrics, "grid_buy_cost")

    apply_publication_style()
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    fig.suptitle(
        f"LC-MAPPO Training Convergence - Seed {seed} (10000 Episodes)",
        fontsize=14, fontweight="bold",
    )

    lines = [
        (axes[0], total_rewards, "#0072B2", "Total Reward"),
        (axes[1], p2p_energy, "#009E73", "P2P Energy (kWh)"),
        (axes[2], grid_buy, "#D55E00", "Grid Purchase Cost"),
    ]

    for ax, raw, color, ylabel in lines:
        ax.plot(episodes, raw, alpha=0.22, color=color, linewidth=0.5, label="Raw")
        if len(raw) > window:
            smoothed = moving_average(raw, window)
            ax.plot(
                episodes[window - 1:],
                smoothed[window - 1:],
                color=color, linewidth=1.8,
                label=f"Smoothed (window={window})",
            )
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False)
        style_axes(ax)

    plt.tight_layout()
    paths = save_publication_figure(fig, output_dir / f"convergence_seed{seed}")
    plt.close(fig)

    tail = min(TAIL_EPISODES, len(metrics))
    print(f"  [Seed {seed} last {tail} ep] "
          f"Reward={np.mean(total_rewards[-tail:]):.2f}+/-{np.std(total_rewards[-tail:]):.2f}  "
          f"P2P={np.mean(p2p_energy[-tail:]):.2f}  "
          f"Grid={np.mean(grid_buy[-tail:]):.2f}")
    return paths


def plot_multiseed_comparison(all_metrics, output_dir, window=SMOOTH_WINDOW):
    """生成多种子对比曲线图（每个指标一张图）。"""
    seeds = sorted(all_metrics.keys())
    min_len = min(len(all_metrics[s]) for s in seeds)

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    apply_publication_style()

    for metric_key, metric_label in PLOT_METRICS:
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        all_smoothed = []

        for idx, seed in enumerate(seeds):
            metrics = all_metrics[seed][:min_len]
            episodes = extract_episodes(metrics)
            values = extract_metric(metrics, metric_key)
            smoothed = moving_average(values, window)
            all_smoothed.append(smoothed)

            color = OKABE_ITO[idx % len(OKABE_ITO)]
            ax.plot(episodes, smoothed, color=color, linewidth=1.5,
                    alpha=0.85, label=f"Seed {seed}")

        stacked = np.vstack(all_smoothed)
        mean_curve = stacked.mean(axis=0)
        std_curve = stacked.std(axis=0, ddof=0)
        ax.plot(episodes, mean_curve, color="#111111", linewidth=2.0, label="Mean")
        ax.fill_between(
            episodes, mean_curve - std_curve, mean_curve + std_curve,
            color="#999999", alpha=0.18, linewidth=0, label="Mean +/- std",
        )

        ax.set_title(f"LC-MAPPO {metric_label} - Multi-Seed Comparison")
        ax.set_xlabel("Episode")
        ax.set_ylabel(metric_label)
        style_axes(ax)
        ax.legend(frameon=False, ncol=min(5, len(seeds) + 2), loc="best")
        fig.tight_layout()

        safe_name = metric_key.replace("/", "_")
        save_publication_figure(fig, figures_dir / f"multiseed_{safe_name}")
        plt.close(fig)

    print(f"Multi-seed comparison figures saved to: {figures_dir}")


def generate_summary_table(all_metrics, output_dir, tail=TAIL_EPISODES):
    """生成最终性能指标汇总表。"""
    seeds = sorted(all_metrics.keys())

    seed_stats = {}
    for seed in seeds:
        metrics = all_metrics[seed]
        tail_data = metrics[-min(tail, len(metrics)):]
        stats = {}
        for key in SUMMARY_METRICS:
            vals = [m[key] for m in tail_data if key in m]
            if vals:
                stats[key] = float(np.mean(vals))
        seed_stats[seed] = stats

    aggregated = {}
    for key in SUMMARY_METRICS:
        vals = [seed_stats[s][key] for s in seeds if key in seed_stats[s]]
        if vals:
            aggregated[key] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }

    summary = {
        "variant": VARIANT,
        "episodes": 10000,
        "tail_episodes": tail,
        "seeds": seeds,
        "per_seed": {str(seed): seed_stats[seed] for seed in seeds},
        "aggregated": aggregated,
    }

    summary_path = output_dir / "multiseed_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*110}")
    print(f"LC-MAPPO 10000 Episodes - Final {tail} Episodes Performance Summary")
    print(f"{'='*110}")
    header = f"{'Metric':<35}"
    for s in seeds:
        header += f" {'Seed '+str(s):>15}"
    header += f" {'Mean':>15} {'Std':>15}"
    print(header)
    print("-" * 110)

    for key in SUMMARY_METRICS:
        row = f"{key:<35}"
        for s in seeds:
            val = seed_stats[s].get(key, float("nan"))
            row += f" {val:>15.4f}"
        agg = aggregated.get(key, {})
        row += f" {agg.get('mean', float('nan')):>15.4f} {agg.get('std', float('nan')):>15.4f}"
        print(row)

    print(f"\nSummary saved: {summary_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="LC-MAPPO 3-seed 10000ep comparison plot generator"
    )
    parser.add_argument(
        "--base-dir", default=DEFAULT_BASE_DIR,
        help=f"Base directory containing seed_7/, seed_42/, seed_100/ (default: {DEFAULT_BASE_DIR})",
    )
    parser.add_argument("--window", type=int, default=SMOOTH_WINDOW)
    parser.add_argument("--tail", type=int, default=TAIL_EPISODES)
    args = parser.parse_args()

    base = Path(args.base_dir)

    # 加载所有种子的指标
    all_metrics = {}
    for seed in [7, 42, 100]:
        metrics_path = base / f"seed_{seed}" / f"{VARIANT}_metrics.json"
        if metrics_path.exists():
            all_metrics[seed] = load_metrics(metrics_path)
            print(f"Loaded seed {seed}: {metrics_path} ({len(all_metrics[seed])} episodes)")
        else:
            print(f"WARNING: seed {seed} metrics not found: {metrics_path}")

    if len(all_metrics) < 2:
        print(f"\nERROR: need at least 2 seeds, found {len(all_metrics)}")
        sys.exit(1)

    # 1. 单种子曲线
    print(f"\n{'='*60}")
    print("Phase 1: Single-seed convergence curves")
    print(f"{'='*60}")
    for seed in sorted(all_metrics.keys()):
        plot_single_seed_curves(all_metrics[seed], seed, base / f"seed_{seed}")

    # 2. 多种子对比图
    print(f"\n{'='*60}")
    print("Phase 2: Multi-seed comparison curves")
    print(f"{'='*60}")
    plot_multiseed_comparison(all_metrics, base)

    # 3. 汇总表
    print(f"\n{'='*60}")
    print("Phase 3: Final performance summary")
    print(f"{'='*60}")
    generate_summary_table(all_metrics, base)

    print(f"\n{'='*60}")
    print("All done!")
    print(f"Output: {base}")
    print(f"Figures: {base / 'figures'}")
    print(f"Summary: {base / 'multiseed_summary.json'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
