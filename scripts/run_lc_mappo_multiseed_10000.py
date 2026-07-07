#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LC-MAPPO 三种子 10000 episodes 批量实验脚本。

运行种子 42 和 100 的训练（种子 7 已有结果），
然后生成多种子对比训练曲线和最终性能指标汇总。

用法:
    python scripts/run_lc_mappo_multiseed_10000.py [--skip-train] [--seed7-dir PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from figure_style import (
    OKABE_ITO,
    apply_publication_style,
    display_variant,
    moving_average,
    save_publication_figure,
    style_axes,
)
from tecsf.rl.mappo import train


# ---- 配置 ----
CONFIG_PATH = "configs/default.yaml"
VARIANT = "lc_mappo"
EPISODES = 10000
DEVICE = "cpu"
NEW_SEEDS = [42, 100]
EXISTING_SEED = 7
OUTPUT_BASE = "outputs/lc_mappo_10000ep_multiseed_20260705"
SMOOTH_WINDOW = 100
TAIL_EPISODES = 1000  # 最后 N 个 episode 用于统计

# 需要绘制的核心指标
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
    "grid_carbon_emission",
    "carbon_reduction",
    "lccoins",
    "settlement_success_rate",
    "feasible_rate",
    "max_violation",
    "renewable_consumption_rate",
    "consensus_confirmed_rate",
]


def ensure_output_dir(base: Path, seed: int) -> Path:
    d = base / f"seed_{seed}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_single_seed(seed: int, output_base: Path, device: str = "cpu", episodes: int = 10000) -> Path:
    """运行单个种子的训练，返回 metrics 文件路径。"""
    out_dir = ensure_output_dir(output_base, seed)
    metrics_path = out_dir / f"{VARIANT}_metrics.json"

    if metrics_path.exists():
        print(f"[SKIP] 种子 {seed} 已有结果: {metrics_path}")
        return metrics_path

    print(f"\n{'='*60}")
    print(f"开始训练: seed={seed}, variant={VARIANT}, episodes={episodes}")
    print(f"输出目录: {out_dir}")
    print(f"{'='*60}\n")

    t0 = time.perf_counter()
    result = train(
        config=CONFIG_PATH,
        variant=VARIANT,
        output_dir=str(out_dir),
        episodes=episodes,
        seed=seed,
        device=device,
    )
    elapsed = time.perf_counter() - t0
    hours = elapsed / 3600.0
    print(f"\n种子 {seed} 训练完成，耗时 {elapsed:.0f}s ({hours:.1f}h)")
    print(f"checkpoint: {result.checkpoint_path}")
    print(f"metrics: {result.metrics_path}")
    return Path(result.metrics_path)


def load_metrics(path: Path) -> list[dict]:
    """加载训练指标 JSON。"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_metric(metrics: list[dict], key: str) -> np.ndarray:
    """从指标列表中提取指定 key 的数值数组。"""
    return np.array([m[key] for m in metrics], dtype=float)


def extract_episodes(metrics: list[dict]) -> np.ndarray:
    """提取 episode 编号数组。"""
    return np.array([m["episode"] for m in metrics], dtype=float)


# ---- 绘图函数 ----


def plot_single_seed_curves(
    metrics: list[dict],
    seed: int,
    output_dir: Path,
    window: int = SMOOTH_WINDOW,
):
    """为单个种子生成收敛曲线图。"""
    episodes = extract_episodes(metrics)
    total_rewards = extract_metric(metrics, "total_reward")
    p2p_energy = extract_metric(metrics, "p2p_energy")
    grid_buy = extract_metric(metrics, "grid_buy_cost")

    apply_publication_style()
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    fig.suptitle(
        f"LC-MAPPO Training Convergence — Seed {seed} ({EPISODES} Episodes)",
        fontsize=14,
        fontweight="bold",
    )

    lines = [
        (axes[0], total_rewards, "blue", "Total Reward"),
        (axes[1], p2p_energy, "green", "P2P Energy (kWh)"),
        (axes[2], grid_buy, "red", "Grid Purchase Cost"),
    ]

    for ax, raw, color, ylabel in lines:
        ax.plot(episodes, raw, alpha=0.25, color=color, linewidth=0.5, label="Raw")
        if len(raw) > window:
            smoothed = moving_average(raw, window)
            ax.plot(
                episodes[window - 1 :],
                smoothed[window - 1 :],
                color=color,
                linewidth=1.8,
                label=f"Smoothed (window={window})",
            )
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False)
        style_axes(ax)

    plt.tight_layout()
    paths = save_publication_figure(fig, output_dir / f"convergence_seed{seed}")
    plt.close(fig)

    # 打印统计
    tail = min(TAIL_EPISODES, len(metrics))
    final = total_rewards[-tail:]
    final_p2p = p2p_energy[-tail:]
    final_grid = grid_buy[-tail:]
    print(f"\n  [种子 {seed} 最后 {tail} episodes 统计]")
    print(f"    Total Reward:  {final.mean():.2f} +/- {final.std():.2f}")
    print(f"    P2P Energy:    {final_p2p.mean():.2f} +/- {final_p2p.std():.2f}")
    print(f"    Grid Cost:     {final_grid.mean():.2f} +/- {final_grid.std():.2f}")

    return paths


def plot_multiseed_comparison(
    all_metrics: dict[int, list[dict]],
    output_dir: Path,
    window: int = SMOOTH_WINDOW,
):
    """生成多种子对比曲线图（每个指标一张图）。"""
    seeds = sorted(all_metrics.keys())
    # 确保所有种子的 episode 数一致（取最小值对齐）
    min_len = min(len(all_metrics[s]) for s in seeds)
    print(f"\n多种子对比：{len(seeds)} 个种子，对齐到 {min_len} episodes")

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
            ax.plot(
                episodes,
                smoothed,
                color=color,
                linewidth=1.5,
                alpha=0.85,
                label=f"Seed {seed}",
            )

        # 均值 ± 标准差阴影
        stacked = np.vstack(all_smoothed)
        mean_curve = stacked.mean(axis=0)
        std_curve = stacked.std(axis=0, ddof=0)
        ax.plot(episodes, mean_curve, color="#111111", linewidth=2.0, label="Mean")
        ax.fill_between(
            episodes,
            mean_curve - std_curve,
            mean_curve + std_curve,
            color="#999999",
            alpha=0.18,
            linewidth=0,
            label="Mean ± std",
        )

        ax.set_title(f"LC-MAPPO {metric_label} — Multi-Seed Comparison")
        ax.set_xlabel("Episode")
        ax.set_ylabel(metric_label)
        style_axes(ax)
        ax.legend(frameon=False, ncol=min(5, len(seeds) + 2), loc="best")
        fig.tight_layout()

        safe_name = metric_key.replace("/", "_")
        save_publication_figure(fig, figures_dir / f"multiseed_{safe_name}")
        plt.close(fig)

    print(f"多种子对比图已保存到: {figures_dir}")


def generate_summary_table(
    all_metrics: dict[int, list[dict]],
    output_dir: Path,
    tail: int = TAIL_EPISODES,
) -> dict:
    """生成最终性能指标汇总表（JSON 和 文本表格）。"""
    seeds = sorted(all_metrics.keys())

    # 每个种子取最后 tail 个 episode 的均值
    seed_stats: dict[int, dict[str, float]] = {}
    for seed in seeds:
        metrics = all_metrics[seed]
        tail_data = metrics[-min(tail, len(metrics)) :]
        stats = {}
        for key in SUMMARY_METRICS:
            vals = [m[key] for m in tail_data if key in m]
            if vals:
                stats[key] = float(np.mean(vals))
        seed_stats[seed] = stats

    # 三种子聚合
    aggregated: dict[str, dict[str, float]] = {}
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
        "episodes": EPISODES,
        "tail_episodes": tail,
        "seeds": seeds,
        "per_seed": {
            str(seed): seed_stats[seed] for seed in seeds
        },
        "aggregated": aggregated,
    }

    # 保存 JSON
    summary_path = output_dir / "multiseed_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 打印文本表格
    print(f"\n{'='*100}")
    print(f"LC-MAPPO {EPISODES} Episodes — Final {tail} Episodes Performance Summary")
    print(f"{'='*100}")
    header = f"{'Metric':<35}"
    for s in seeds:
        header += f" {'Seed '+str(s):>15}"
    header += f" {'Mean':>15} {'Std':>15}"
    print(header)
    print("-" * 100)

    for key in SUMMARY_METRICS:
        row = f"{key:<35}"
        for s in seeds:
            val = seed_stats[s].get(key, float("nan"))
            row += f" {val:>15.4f}"
        agg = aggregated.get(key, {})
        row += f" {agg.get('mean', float('nan')):>15.4f} {agg.get('std', float('nan')):>15.4f}"
        print(row)

    print(f"\n汇总已保存: {summary_path}")

    # 保存文本版本
    txt_path = output_dir / "multiseed_summary.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"LC-MAPPO {EPISODES} Episodes — Final {tail} Episodes Performance Summary\n")
        f.write("=" * 100 + "\n")
        f.write(header + "\n")
        f.write("-" * 100 + "\n")
        for key in SUMMARY_METRICS:
            row = f"{key:<35}"
            for s in seeds:
                val = seed_stats[s].get(key, float("nan"))
                row += f" {val:>15.4f}"
            agg = aggregated.get(key, {})
            row += f" {agg.get('mean', float('nan')):>15.4f} {agg.get('std', float('nan')):>15.4f}"
            f.write(row + "\n")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="LC-MAPPO 三种子 10000ep 批量实验"
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="跳过训练，仅用现有数据生成图表",
    )
    parser.add_argument(
        "--seed7-dir",
        default="outputs/lc_mappo_10000episodes_plateau_debug_20260704/lc_mappo",
        help="种子 7 的现有结果目录",
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_BASE,
        help=f"输出根目录 (默认: {OUTPUT_BASE})",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=NEW_SEEDS,
        help="需要运行的种子列表",
    )
    parser.add_argument(
        "--device",
        default=DEVICE,
        help="训练设备",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=EPISODES,
        help="训练 episodes 数",
    )
    args = parser.parse_args()

    device_val = args.device
    episodes_val = args.episodes
    new_seeds_val = args.seeds

    output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    # ---- 阶段 1: 训练 ----
    if not args.skip_train:
        print(f"\n{'#'*60}")
        print(f"# 阶段 1: 多种子训练")
        print(f"# 已有种子: {EXISTING_SEED}")
        print(f"# 新增种子: {new_seeds_val}")
        print(f"# Variant: {VARIANT}, Episodes: {episodes_val}, Device: {device_val}")
        print(f"{'#'*60}")

        t_start = time.perf_counter()
        for seed in new_seeds_val:
            run_single_seed(seed, output_base, device_val, episodes_val)
        total_elapsed = time.perf_counter() - t_start
        print(f"\n所有训练完成，总耗时 {total_elapsed:.0f}s ({total_elapsed/3600:.1f}h)")
    else:
        print("跳过训练阶段 (--skip-train)")

    # ---- 阶段 2: 收集所有种子的指标 ----
    print(f"\n{'#'*60}")
    print(f"# 阶段 2: 收集数据 & 生成图表")
    print(f"{'#'*60}")

    all_metrics: dict[int, list[dict]] = {}

    # 种子 7：从已有目录加载
    seed7_dir = Path(args.seed7_dir)
    seed7_metrics = seed7_dir / f"{VARIANT}_metrics.json"
    if seed7_metrics.exists():
        all_metrics[7] = load_metrics(seed7_metrics)
        print(f"加载种子 7: {seed7_metrics} ({len(all_metrics[7])} episodes)")
    else:
        print(f"警告: 种子 7 指标文件不存在: {seed7_metrics}")

    # 新种子
    for seed in new_seeds_val:
        metrics_path = ensure_output_dir(output_base, seed) / f"{VARIANT}_metrics.json"
        if metrics_path.exists():
            all_metrics[seed] = load_metrics(metrics_path)
            print(f"加载种子 {seed}: {metrics_path} ({len(all_metrics[seed])} episodes)")
        else:
            print(f"警告: 种子 {seed} 指标文件不存在: {metrics_path}")

    if len(all_metrics) < 2:
        print("错误: 至少需要 2 个种子的数据才能对比")
        sys.exit(1)

    # ---- 阶段 3: 生成单种子曲线 ----
    print(f"\n{'#'*60}")
    print(f"# 阶段 3: 生成单种子收敛曲线")
    print(f"{'#'*60}")
    for seed in sorted(all_metrics.keys()):
        plot_single_seed_curves(all_metrics[seed], seed, output_base / f"seed_{seed}")

    # ---- 阶段 4: 生成多种子对比图 ----
    print(f"\n{'#'*60}")
    print(f"# 阶段 4: 生成多种子对比图")
    print(f"{'#'*60}")
    plot_multiseed_comparison(all_metrics, output_base)

    # ---- 阶段 5: 生成最终性能汇总 ----
    print(f"\n{'#'*60}")
    print(f"# 阶段 5: 生成最终性能汇总")
    print(f"{'#'*60}")
    summary = generate_summary_table(all_metrics, output_base)

    print(f"\n{'='*60}")
    print("全部完成！")
    print(f"输出目录: {output_base}")
    print(f"图表: {output_base / 'figures'}")
    print(f"汇总: {output_base / 'multiseed_summary.json'}")
    print(f"{'='*60}")

    return summary


if __name__ == "__main__":
    main()
