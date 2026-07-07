#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成 MAPPO vs LC-MAPPO 训练曲线对比图
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# 路径
LC_MAPPO_SUMMARY = Path(__file__).resolve().parent.parent / 'outputs' / 'lc_mappo_10000episodes_plateau_debug_20260704' / 'summary.json'
MAPPO_SUMMARY = Path(__file__).resolve().parent.parent / 'outputs' / 'experiments' / 'summary.json'
OUTPUT_DIR = Path(__file__).resolve().parent.parent / 'outputs' / 'comparison'

def smooth(data, window=200):
    if len(data) < window:
        return data
    cumsum = np.cumsum(np.insert(data, 0, 0))
    return (cumsum[window:] - cumsum[:-window]) / window

def main():
    # 加载数据
    with open(LC_MAPPO_SUMMARY, 'r') as f:
        lc_data = json.load(f)['lc_mappo']
    with open(MAPPO_SUMMARY, 'r') as f:
        mp_data = json.load(f)['mappo']

    lc_rewards = np.array([ep['total_reward'] for ep in lc_data])
    mp_rewards = np.array([ep['total_reward'] for ep in mp_data])
    lc_episodes = np.arange(1, len(lc_rewards) + 1)
    mp_episodes = np.arange(1, len(mp_rewards) + 1)

    window = 200
    lc_smooth = smooth(lc_rewards, window)
    mp_smooth = smooth(mp_rewards, window)
    lc_smooth_ep = lc_episodes[window - 1:]
    mp_smooth_ep = mp_episodes[window - 1:]

    # 统计
    lc_final = np.mean(lc_rewards[-1000:])
    mp_final = np.mean(mp_rewards[-1000:])
    lc_final_std = np.std(lc_rewards[-1000:])
    mp_final_std = np.std(mp_rewards[-1000:])
    improvement_abs = lc_final - mp_final
    improvement_pct = improvement_abs / abs(mp_final) * 100

    print(f"LC-MAPPO: last1000 mean={lc_final:.2f}, std={lc_final_std:.2f}")
    print(f"MAPPO:    last1000 mean={mp_final:.2f}, std={mp_final_std:.2f}")
    print(f"Improvement: +{improvement_abs:.2f} ({improvement_pct:.1f}%)")

    # ---------- 绘图 ----------
    fig, ax = plt.subplots(figsize=(14, 7))

    # 颜色
    color_lc = '#E63946'   # 红色 - LC-MAPPO
    color_mp = '#1D3557'   # 深蓝 - MAPPO

    # 原始曲线（浅色）
    ax.plot(mp_episodes, mp_rewards, alpha=0.15, color=color_mp, linewidth=0.4)
    ax.plot(lc_episodes, lc_rewards, alpha=0.15, color=color_lc, linewidth=0.4)

    # 平滑曲线
    ax.plot(mp_smooth_ep, mp_smooth, color=color_mp, linewidth=2.2,
            label=f'MAPPO (Baseline)')
    ax.plot(lc_smooth_ep, lc_smooth, color=color_lc, linewidth=2.2,
            label=f'LC-MAPPO (Ours)')

    # 收敛区域填充（最后1000 episodes均值）
    ax.axhline(y=mp_final, color=color_mp, linestyle='--', alpha=0.5, linewidth=1.2)
    ax.axhline(y=lc_final, color=color_lc, linestyle='--', alpha=0.5, linewidth=1.2)

    # 标注性能提升
    mid_x = len(lc_rewards) * 0.72
    y_top = lc_final
    y_bot = mp_final
    # 箭头从 MAPPO 均值指向 LC-MAPPO 均值
    ax.annotate('', xy=(mid_x, y_top), xytext=(mid_x, y_bot),
                arrowprops=dict(arrowstyle='<->', color='#2A9D8F', lw=2.0))
    ax.text(mid_x + 150, (y_top + y_bot) / 2,
            f'+{improvement_abs:.1f}\n(+{improvement_pct:.1f}%)',
            fontsize=13, fontweight='bold', color='#2A9D8F',
            va='center', ha='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#2A9D8F', alpha=0.9))

    # 标注收敛值
    ax.text(lc_episodes[-1] + 80, lc_final,
            f'{lc_final:.1f}', fontsize=11, color=color_lc,
            va='center', fontweight='bold')
    ax.text(mp_episodes[-1] + 80, mp_final,
            f'{mp_final:.1f}', fontsize=11, color=color_mp,
            va='center', fontweight='bold')

    ax.set_xlabel('Training Episode', fontsize=14)
    ax.set_ylabel('Total Reward', fontsize=14)
    ax.set_title('Training Convergence: LC-MAPPO vs MAPPO (10000 Episodes)',
                 fontsize=16, fontweight='bold', pad=15)
    ax.legend(fontsize=13, loc='lower right', framealpha=0.9)
    ax.grid(True, alpha=0.25)
    ax.set_xlim(0, max(len(lc_rewards), len(mp_rewards)) + 500)
    ax.tick_params(labelsize=12)

    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ['png', 'pdf', 'svg']:
        path = OUTPUT_DIR / f'mappo_vs_lcmappo_comparison.{ext}'
        fig.savefig(path, dpi=300 if ext == 'png' else None, bbox_inches='tight')
        print(f"saved: {path}")
    plt.close()

    # ---------- 第二张图：带子图（奖励 + 局部放大） ----------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [3, 1]})

    # 左图：完整训练曲线
    ax1.plot(mp_episodes, mp_rewards, alpha=0.15, color=color_mp, linewidth=0.4)
    ax1.plot(lc_episodes, lc_rewards, alpha=0.15, color=color_lc, linewidth=0.4)
    ax1.plot(mp_smooth_ep, mp_smooth, color=color_mp, linewidth=2.2, label='MAPPO (Baseline)')
    ax1.plot(lc_smooth_ep, lc_smooth, color=color_lc, linewidth=2.2, label='LC-MAPPO (Ours)')
    ax1.axhline(y=mp_final, color=color_mp, linestyle='--', alpha=0.4, linewidth=1)
    ax1.axhline(y=lc_final, color=color_lc, linestyle='--', alpha=0.4, linewidth=1)
    ax1.set_xlabel('Training Episode', fontsize=13)
    ax1.set_ylabel('Total Reward', fontsize=13)
    ax1.set_title('Training Convergence Curves', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=12, loc='lower right')
    ax1.grid(True, alpha=0.25)

    # 右图：最后2000 episodes 收敛箱线对比
    lc_box = lc_rewards[-2000:]
    mp_box = mp_rewards[-2000:]
    bp = ax2.boxplot([mp_box, lc_box], labels=['MAPPO', 'LC-MAPPO'],
                     patch_artist=True, widths=0.5,
                     showmeans=True,
                     meanprops=dict(marker='D', markerfacecolor='white',
                                    markeredgecolor='black', markersize=7))
    bp['boxes'][0].set_facecolor(color_mp)
    bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_facecolor(color_lc)
    bp['boxes'][1].set_alpha(0.7)
    ax2.set_ylabel('Total Reward (last 2000 episodes)', fontsize=13)
    ax2.set_title('Convergence Distribution', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.25, axis='y')

    # 标注提升
    ax2.annotate(f'+{improvement_abs:.1f}\n(+{improvement_pct:.1f}%)',
                 xy=(2, lc_final), xytext=(2.35, (lc_final + mp_final) / 2),
                 fontsize=12, fontweight='bold', color='#2A9D8F',
                 va='center', ha='left',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                           edgecolor='#2A9D8F', alpha=0.9))
    ax2.annotate('', xy=(2.2, lc_final), xytext=(2.2, mp_final),
                 arrowprops=dict(arrowstyle='<->', color='#2A9D8F', lw=1.8))

    fig.suptitle('LC-MAPPO vs MAPPO: Training Performance Comparison (10000 Episodes)',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    for ext in ['png', 'pdf', 'svg']:
        path = OUTPUT_DIR / f'mappo_vs_lcmappo_comparison_detail.{ext}'
        fig.savefig(path, dpi=300 if ext == 'png' else None, bbox_inches='tight')
        print(f"saved: {path}")
    plt.close()

    print("\n=== Summary ===")
    print(f"LC-MAPPO final 1000ep: {lc_final:.2f} ± {lc_final_std:.2f}")
    print(f"MAPPO    final 1000ep: {mp_final:.2f} ± {mp_final_std:.2f}")
    print(f"Improvement: +{improvement_abs:.2f} ({improvement_pct:+.1f}%)")

if __name__ == '__main__':
    main()
