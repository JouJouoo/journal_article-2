#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
绘制 IEEE 33 LC-MAPPO 2000 回合训练曲线。
从 summary.json 读取数据，生成多面板训练曲线图。
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from figure_style import (
    apply_publication_style,
    save_publication_figure,
    moving_average,
    style_axes,
)


def load_data(summary_path: str) -> dict:
    with open(summary_path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="绘制 IEEE 33 LC-MAPPO 训练曲线")
    parser.add_argument(
        "--summary",
        default="outputs/ieee33bw_lcmappo_2000ep/summary.json",
        help="summary.json 路径",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/ieee33bw_lcmappo_2000ep",
        help="输出目录",
    )
    parser.add_argument("--window", type=int, default=50, help="平滑窗口大小")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_data(args.summary)
    variant = "tecsf"
    if variant not in data:
        variants = list(data.keys())
        print(f"未找到 variant '{variant}'，可用: {variants}")
        variant = variants[0] if variants else exit(1)

    metrics = data[variant]
    n = len(metrics)
    eps = np.array([m["episode"] for m in metrics])
    print(f"加载 {n} 条记录，variant={variant}")

    # ----- 提取各指标序列 -----
    mean_reward = np.array([m["mean_reward"] for m in metrics])
    total_reward = np.array([m["total_reward"] for m in metrics])
    reward_eco = np.array([m["reward_eco"] for m in metrics])
    reward_coin = np.array([m["reward_coin"] for m in metrics])

    carbon_emission = np.array([m["carbon_emission"] for m in metrics])
    carbon_reduction = np.array([m["carbon_reduction"] for m in metrics])
    carbon_offset = np.array([m["carbon_offset"] for m in metrics])
    lccoins = np.array([m["lccoins"] for m in metrics])

    p2p_energy = np.array([m["p2p_energy"] for m in metrics])
    grid_buy_cost = np.array([m["grid_buy_cost"] for m in metrics])
    renewable_rate = np.array([m["renewable_consumption_rate"] for m in metrics])

    actor_loss = np.array([m.get("actor_loss", 0) for m in metrics])
    critic_loss = np.array([m.get("critic_loss", 0) for m in metrics])
    approx_kl = np.array([m.get("approx_kl", 0) for m in metrics])

    feasible_rate = np.array([m["feasible_rate"] for m in metrics])
    settlement_success = np.array([m["settlement_success_rate"] for m in metrics])
    consensus_confirmed = np.array([m["consensus_confirmed_rate"] for m in metrics])

    # ----- 1. 主训练曲线（总奖励，带平滑） -----
    apply_publication_style()
    fig, axes = plt.subplots(3, 2, figsize=(10, 11))
    fig.subplots_adjust(hspace=0.35, wspace=0.30)

    # (a) Mean reward
    ax = axes[0, 0]
    ax.plot(eps, mean_reward, alpha=0.25, color="#0072B2", linewidth=0.6, label="Raw")
    ax.plot(
        eps, moving_average(mean_reward, args.window),
        color="#0072B2", linewidth=1.6, label=f"Smooth (w={args.window})",
    )
    ax.set_title("Mean Reward", fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean Reward")
    ax.legend(frameon=False)
    style_axes(ax)
    ax.annotate(
        f"{mean_reward[-1]:.4f}",
        xy=(eps[-1], mean_reward[-1]),
        fontsize=7, color="#0072B2", ha="right", va="bottom",
    )

    # (b) Total reward
    ax = axes[0, 1]
    ax.plot(eps, total_reward, alpha=0.25, color="#D55E00", linewidth=0.6, label="Raw")
    ax.plot(
        eps, moving_average(total_reward, args.window),
        color="#D55E00", linewidth=1.6, label=f"Smooth (w={args.window})",
    )
    ax.set_title("Total Reward", fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.legend(frameon=False)
    style_axes(ax)
    ax.annotate(
        f"{total_reward[-1]:.2f}",
        xy=(eps[-1], total_reward[-1]),
        fontsize=7, color="#D55E00", ha="right", va="bottom",
    )

    # (c) Reward decomposition: eco vs coin
    ax = axes[1, 0]
    ax.plot(eps, moving_average(reward_eco, args.window), color="#0072B2", linewidth=1.6, label="Economic")
    ax.plot(eps, moving_average(reward_coin, args.window), color="#009E73", linewidth=1.6, label="LCCoins")
    ax.set_title("Reward Decomposition", fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.legend(frameon=False)
    style_axes(ax)

    # (d) Carbon metrics
    ax = axes[1, 1]
    ax.plot(eps, moving_average(carbon_emission, args.window), color="#D55E00", linewidth=1.4, label="Emission")
    ax.plot(eps, moving_average(carbon_reduction, args.window), color="#009E73", linewidth=1.4, label="Reduction")
    ax.plot(eps, moving_average(carbon_offset, args.window), color="#56B4E9", linewidth=1.4, label="Offset")
    ax.set_title("Carbon Emission / Reduction / Offset", fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("CO₂ (kg)")
    ax.legend(frameon=False)
    style_axes(ax)

    # (e) LCCoins & Renewable Rate
    ax = axes[2, 0]
    ax2 = ax.twinx()
    l1 = ax.plot(
        eps, moving_average(lccoins, args.window),
        color="#CC79A7", linewidth=1.6, label="LCCoins",
    )
    l2 = ax2.plot(
        eps, moving_average(renewable_rate, args.window),
        color="#E69F00", linewidth=1.4, label="Renewable rate",
    )
    ax.set_title("LCCoins & Renewable Consumption Rate", fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("LCCoins")
    ax2.set_ylabel("Rate")
    ax2.set_ylim(0.98, 1.002)
    ax.legend(handles=l1 + l2, frameon=False)
    style_axes(ax)

    # (f) P2P Energy & Grid Cost
    ax = axes[2, 1]
    ax2 = ax.twinx()
    l1 = ax.plot(
        eps, moving_average(p2p_energy, args.window),
        color="#0072B2", linewidth=1.6, label="P2P energy",
    )
    l2 = ax2.plot(
        eps, moving_average(grid_buy_cost, args.window),
        color="#D55E00", linewidth=1.6, label="Grid cost",
    )
    ax.set_title("P2P Energy & Grid Cost", fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("P2P Energy (kWh)")
    ax2.set_ylabel("Grid Cost (¥)")
    ax.legend(handles=l1 + l2, frameon=False)
    style_axes(ax)

    fig.suptitle(
        f"IEEE 33 — LC-MAPPO Training Curves ({n} Episodes)",
        fontsize=11, fontweight="bold", y=0.985,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    paths = save_publication_figure(fig, output_dir / "training_curves_panel")
    plt.close(fig)
    for p in paths:
        print(f"  saved: {p}")

    # ----- 2. 损失和 KL 散度 -----
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    fig.subplots_adjust(wspace=0.30)

    ax = axes[0]
    ax.plot(eps, moving_average(actor_loss, args.window), color="#0072B2", linewidth=1.4)
    ax.set_title("Actor Loss", fontweight="bold")
    ax.set_xlabel("Episode")
    style_axes(ax)

    ax = axes[1]
    ax.plot(eps, moving_average(critic_loss, args.window), color="#D55E00", linewidth=1.4)
    ax.set_title("Critic Loss", fontweight="bold")
    ax.set_xlabel("Episode")
    style_axes(ax)

    ax = axes[2]
    ax.plot(eps, moving_average(approx_kl, args.window), color="#009E73", linewidth=1.4)
    ax.set_title("Approx. KL Divergence", fontweight="bold")
    ax.set_xlabel("Episode")
    style_axes(ax)

    fig.suptitle(
        "Loss & KL Convergence",
        fontsize=10, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    paths = save_publication_figure(fig, output_dir / "training_loss_curves")
    plt.close(fig)
    for p in paths:
        print(f"  saved: {p}")

    # ----- 3. 约束满足率 -----
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.plot(eps, feasible_rate, color="#009E73", linewidth=0.8, alpha=0.4, label="Feasible rate")
    ax.plot(eps, settlement_success, color="#0072B2", linewidth=0.8, alpha=0.4, label="Settlement success")
    ax.plot(eps, consensus_confirmed, color="#CC79A7", linewidth=0.8, alpha=0.4, label="Consensus confirmed")
    # Smooth overlay
    ax.plot(
        eps, moving_average(feasible_rate, args.window),
        color="#009E73", linewidth=1.8, label=f"Feasible (smooth)",
    )
    ax.plot(
        eps, moving_average(settlement_success, args.window),
        color="#0072B2", linewidth=1.8, label=f"Settlement (smooth)",
    )
    ax.plot(
        eps, moving_average(consensus_confirmed, args.window),
        color="#CC79A7", linewidth=1.8, label=f"Consensus (smooth)",
    )
    ax.set_title("Constraint Satisfaction & Settlement Rates", fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Rate")
    ax.set_ylim(0.94, 1.005)
    ax.legend(frameon=False, ncol=3)
    style_axes(ax)
    fig.tight_layout()
    paths = save_publication_figure(fig, output_dir / "training_constraint_rates")
    plt.close(fig)
    for p in paths:
        print(f"  saved: {p}")

    # ----- 打印摘要统计 -----
    window = min(200, n // 2)
    print()
    print("=" * 60)
    print(f"IEEE 33 LC-MAPPO 训练摘要 ({n} episodes)")
    print("=" * 60)
    print(f"{'指标':<30} {'起始':>10} {'末段均值':>10} {'变化':>10}")
    print("-" * 60)
    print(f"{'Mean Reward':<30} {mean_reward[0]:>10.4f} {np.mean(mean_reward[-window:]):>10.4f} {np.mean(mean_reward[-window:]) - mean_reward[0]:>+10.4f}")
    print(f"{'Total Reward':<30} {total_reward[0]:>10.2f} {np.mean(total_reward[-window:]):>10.2f} {np.mean(total_reward[-window:]) - total_reward[0]:>+10.2f}")
    print(f"{'Economic Reward':<30} {reward_eco[0]:>10.2f} {np.mean(reward_eco[-window:]):>10.2f} {np.mean(reward_eco[-window:]) - reward_eco[0]:>+10.2f}")
    print(f"{'LCCoins Utility':<30} {reward_coin[0]:>10.2f} {np.mean(reward_coin[-window:]):>10.2f} {np.mean(reward_coin[-window:]) - reward_coin[0]:>+10.2f}")
    print(f"{'Carbon Emission (kg)':<30} {carbon_emission[0]:>10.2f} {np.mean(carbon_emission[-window:]):>10.2f} {np.mean(carbon_emission[-window:]) - carbon_emission[0]:>+10.2f}")
    print(f"{'Carbon Reduction (kg)':<30} {carbon_reduction[0]:>10.2f} {np.mean(carbon_reduction[-window:]):>10.2f} {np.mean(carbon_reduction[-window:]) - carbon_reduction[0]:>+10.2f}")
    print(f"{'LCCoins':<30} {lccoins[0]:>10.2f} {np.mean(lccoins[-window:]):>10.2f} {np.mean(lccoins[-window:]) - lccoins[0]:>+10.2f}")
    print(f"{'P2P Energy (kWh)':<30} {p2p_energy[0]:>10.4f} {np.mean(p2p_energy[-window:]):>10.4f} {np.mean(p2p_energy[-window:]) - p2p_energy[0]:>+10.4f}")
    print(f"{'Grid Cost (¥)':<30} {grid_buy_cost[0]:>10.2f} {np.mean(grid_buy_cost[-window:]):>10.2f} {np.mean(grid_buy_cost[-window:]) - grid_buy_cost[0]:>+10.2f}")
    print(f"{'Renewable Rate':<30} {renewable_rate[0]:>10.4f} {np.mean(renewable_rate[-window:]):>10.4f} {np.mean(renewable_rate[-window:]) - renewable_rate[0]:>+10.4f}")
    print(f"{'Feasible Rate':<30} {feasible_rate[0]:>10.4f} {np.mean(feasible_rate[-window:]):>10.4f} {np.mean(feasible_rate[-window:]) - feasible_rate[0]:>+10.4f}")
    print(f"{'Settlement Success':<30} {settlement_success[0]:>10.4f} {np.mean(settlement_success[-window:]):>10.4f} {np.mean(settlement_success[-window:]) - settlement_success[0]:>+10.4f}")
    print("-" * 60)


if __name__ == "__main__":
    main()
