#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MAPPO vs LC-MAPPO 奖励差异分解分析图
揭示 total_reward 差异的来源：reward_eco（经济奖励）vs reward_coin（LCCoins效用）
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

LC_MAPPO_SUMMARY = Path(__file__).resolve().parent.parent / 'outputs' / 'lc_mappo_10000episodes_plateau_debug_20260704' / 'summary.json'
MAPPO_SUMMARY = Path(__file__).resolve().parent.parent / 'outputs' / 'experiments' / 'summary.json'
OUTPUT_DIR = Path(__file__).resolve().parent.parent / 'outputs' / 'comparison'

def smooth(data, window=200):
    if len(data) < window:
        return np.array(data)
    cumsum = np.cumsum(np.insert(data, 0, 0))
    return (cumsum[window:] - cumsum[:-window]) / window

def main():
    with open(LC_MAPPO_SUMMARY, 'r') as f:
        lc_data = json.load(f)['lc_mappo']
    with open(MAPPO_SUMMARY, 'r') as f:
        mp_data = json.load(f)['mappo']

    # 提取各奖励分量
    lc_total = np.array([ep['total_reward'] for ep in lc_data])
    mp_total = np.array([ep['total_reward'] for ep in mp_data])
    lc_eco = np.array([ep['reward_eco'] for ep in lc_data])
    mp_eco = np.array([ep['reward_eco'] for ep in mp_data])
    lc_coin = np.array([ep['reward_coin'] for ep in lc_data])
    mp_coin = np.array([ep['reward_coin'] for ep in mp_data])

    episodes = np.arange(1, len(lc_total) + 1)
    window = 200

    color_lc = '#E63946'
    color_mp = '#1D3557'
    color_coin = '#2A9D8F'
    color_eco_lc = '#E76F51'
    color_eco_mp = '#457B9D'

    # ========== 图1：奖励分解堆叠图 ==========
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    # 子图1: total_reward（原始对比）
    ax1 = axes[0]
    ax1.plot(episodes, mp_total, alpha=0.12, color=color_mp, linewidth=0.4)
    ax1.plot(episodes, lc_total, alpha=0.12, color=color_lc, linewidth=0.4)
    ax1.plot(episodes[window-1:], smooth(mp_total, window), color=color_mp, linewidth=2.2, label='MAPPO total_reward')
    ax1.plot(episodes[window-1:], smooth(lc_total, window), color=color_lc, linewidth=2.2, label='LC-MAPPO total_reward')
    ax1.set_ylabel('Total Reward', fontsize=13)
    ax1.set_title('(a) Total Reward (eco + coin - penalty)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, loc='lower right')
    ax1.grid(True, alpha=0.25)
    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

    # 子图2: reward_eco（经济奖励，同尺度可比）
    ax2 = axes[1]
    ax2.plot(episodes, mp_eco, alpha=0.12, color=color_eco_mp, linewidth=0.4)
    ax2.plot(episodes, lc_eco, alpha=0.12, color=color_eco_lc, linewidth=0.4)
    ax2.plot(episodes[window-1:], smooth(mp_eco, window), color=color_eco_mp, linewidth=2.2, label='MAPPO reward_eco')
    ax2.plot(episodes[window-1:], smooth(lc_eco, window), color=color_eco_lc, linewidth=2.2, label='LC-MAPPO reward_eco')
    ax2.set_ylabel('Economic Reward', fontsize=13)
    ax2.set_title('(b) Economic Reward (same scale, comparable)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11, loc='lower right')
    ax2.grid(True, alpha=0.25)
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

    # 子图3: reward_coin（LCCoins效用，仅LC-MAPPO有）
    ax3 = axes[2]
    ax3.plot(episodes, lc_coin, alpha=0.12, color=color_coin, linewidth=0.4)
    ax3.plot(episodes[window-1:], smooth(lc_coin, window), color=color_coin, linewidth=2.2, label='LC-MAPPO reward_coin (CRRA utility)')
    ax3.axhline(y=0, color=color_mp, linewidth=2, linestyle='--', alpha=0.6, label='MAPPO reward_coin = 0 (by design)')
    ax3.set_ylabel('LCCoins Utility Reward', fontsize=13)
    ax3.set_xlabel('Training Episode', fontsize=13)
    ax3.set_title('(c) LCCoins Utility Reward (only LC-MAPPO, by design)', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11, loc='lower right')
    ax3.grid(True, alpha=0.25)
    ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

    plt.tight_layout()
    for ext in ['png', 'pdf', 'svg']:
        path = OUTPUT_DIR / f'reward_decomposition_analysis.{ext}'
        fig.savefig(path, dpi=300 if ext == 'png' else None, bbox_inches='tight')
        print(f"saved: {path}")
    plt.close()

    # ========== 图2：最终收敛值柱状分解 ==========
    fig, ax = plt.subplots(figsize=(10, 6))

    lc_final_eco = np.mean(lc_eco[-1000:])
    mp_final_eco = np.mean(mp_eco[-1000:])
    lc_final_coin = np.mean(lc_coin[-1000:])
    mp_final_coin = np.mean(mp_coin[-1000:])
    lc_final_total = np.mean(lc_total[-1000:])
    mp_final_total = np.mean(mp_total[-1000:])
    lc_penalty = lc_final_total - lc_final_eco - lc_final_coin
    mp_penalty = mp_final_total - mp_final_eco - mp_final_coin

    categories = ['MAPPO\n(Baseline)', 'LC-MAPPO\n(Ours)']
    eco_vals = [mp_final_eco, lc_final_eco]
    coin_vals = [mp_final_coin, lc_final_coin]
    penalty_vals = [mp_penalty, lc_penalty]

    x = np.arange(len(categories))
    width = 0.5

    bars1 = ax.bar(x, eco_vals, width, label='reward_eco (Economic)', color='#457B9D', alpha=0.85)
    bars2 = ax.bar(x, coin_vals, width, bottom=eco_vals, label='reward_coin (LCCoins CRRA)', color='#2A9D8F', alpha=0.85)
    bars3 = ax.bar(x, penalty_vals, width,
                   bottom=[e+c for e,c in zip(eco_vals, coin_vals)],
                   label='penalty (violation + action bound)', color='#E63946', alpha=0.6)

    # 标注数值
    for i, (e, c, p) in enumerate(zip(eco_vals, coin_vals, penalty_vals)):
        total = e + c + p
        ax.text(i, total + (3 if total > 0 else -8), f'{total:.1f}',
                ha='center', fontsize=13, fontweight='bold')
        if abs(e) > 2:
            ax.text(i, e/2, f'{e:.1f}', ha='center', va='center', fontsize=10, color='white', fontweight='bold')
        if abs(c) > 2:
            ax.text(i, e + c/2, f'{c:.1f}', ha='center', va='center', fontsize=10, color='white', fontweight='bold')

    ax.set_ylabel('Reward (last 1000 episodes mean)', fontsize=13)
    ax.set_title('Reward Decomposition: MAPPO vs LC-MAPPO\n(total_reward = reward_eco + reward_coin + penalty)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12)
    ax.legend(fontsize=11, loc='upper left')
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.4)
    ax.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    for ext in ['png', 'pdf', 'svg']:
        path = OUTPUT_DIR / f'reward_decomposition_bar.{ext}'
        fig.savefig(path, dpi=300 if ext == 'png' else None, bbox_inches='tight')
        print(f"saved: {path}")
    plt.close()

    # ========== 打印分析摘要 ==========
    print("\n" + "="*80)
    print("奖励差异分解分析")
    print("="*80)
    print(f"\n{'Component':<30} {'LC-MAPPO':<15} {'MAPPO':<15} {'Diff':<15} {'% of gap':<10}")
    print("-"*80)
    gap = lc_final_total - mp_final_total
    for name, lc_v, mp_v in [
        ('reward_eco', lc_final_eco, mp_final_eco),
        ('reward_coin', lc_final_coin, mp_final_coin),
        ('penalty', lc_penalty, mp_penalty),
        ('TOTAL', lc_final_total, mp_final_total),
    ]:
        diff = lc_v - mp_v
        pct = (diff / gap * 100) if gap != 0 else 0
        print(f"{name:<30} {lc_v:<15.2f} {mp_v:<15.2f} {diff:<+15.2f} {pct:<.1f}%")

if __name__ == '__main__':
    main()
