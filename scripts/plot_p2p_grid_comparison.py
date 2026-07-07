#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MAPPO vs LC-MAPPO: P2P 交易量 & 电网交易量 训练过程对比图
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

    # 提取字段
    lc_p2p = np.array([ep['p2p_energy'] for ep in lc_data])
    mp_p2p = np.array([ep['p2p_energy'] for ep in mp_data])
    lc_grid = np.array([ep['grid_buy_cost'] for ep in lc_data])
    mp_grid = np.array([ep['grid_buy_cost'] for ep in mp_data])
    lc_renewable = np.array([ep['renewable_consumption_rate'] for ep in lc_data])
    mp_renewable = np.array([ep['renewable_consumption_rate'] for ep in mp_data])
    lc_carbon = np.array([ep['carbon_emission'] for ep in lc_data])
    mp_carbon = np.array([ep['carbon_emission'] for ep in mp_data])

    episodes = np.arange(1, len(lc_p2p) + 1)
    window = 200
    sm_ep = episodes[window - 1:]

    color_lc = '#E63946'
    color_mp = '#1D3557'

    # ========== 主图：2x2 对比 ==========
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # (a) P2P 交易量
    ax = axes[0, 0]
    ax.plot(episodes, mp_p2p, alpha=0.10, color=color_mp, linewidth=0.4)
    ax.plot(episodes, lc_p2p, alpha=0.10, color=color_lc, linewidth=0.4)
    ax.plot(sm_ep, smooth(mp_p2p, window), color=color_mp, linewidth=2.4, label='MAPPO (Baseline)')
    ax.plot(sm_ep, smooth(lc_p2p, window), color=color_lc, linewidth=2.4, label='LC-MAPPO (Ours)')
    ax.set_xlabel('Training Episode', fontsize=12)
    ax.set_ylabel('P2P Energy (kWh)', fontsize=12)
    ax.set_title('(a) P2P Trading Volume', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.25)

    # 标注最终值
    lc_final_p2p = np.mean(lc_p2p[-1000:])
    mp_final_p2p = np.mean(mp_p2p[-1000:])
    ax.annotate(f'{lc_final_p2p:.1f}', xy=(episodes[-1], lc_final_p2p),
                xytext=(episodes[-1]+100, lc_final_p2p),
                fontsize=10, color=color_lc, fontweight='bold', va='center')
    ax.annotate(f'{mp_final_p2p:.1f}', xy=(episodes[-1], mp_final_p2p),
                xytext=(episodes[-1]+100, mp_final_p2p),
                fontsize=10, color=color_mp, fontweight='bold', va='center')

    # (b) 电网交易量（购电成本）
    ax = axes[0, 1]
    ax.plot(episodes, mp_grid, alpha=0.10, color=color_mp, linewidth=0.4)
    ax.plot(episodes, lc_grid, alpha=0.10, color=color_lc, linewidth=0.4)
    ax.plot(sm_ep, smooth(mp_grid, window), color=color_mp, linewidth=2.4, label='MAPPO (Baseline)')
    ax.plot(sm_ep, smooth(lc_grid, window), color=color_lc, linewidth=2.4, label='LC-MAPPO (Ours)')
    ax.set_xlabel('Training Episode', fontsize=12)
    ax.set_ylabel('Grid Purchase Cost (¥)', fontsize=12)
    ax.set_title('(b) Grid Trading Volume (Purchase Cost)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.25)

    lc_final_grid = np.mean(lc_grid[-1000:])
    mp_final_grid = np.mean(mp_grid[-1000:])
    ax.annotate(f'{lc_final_grid:.1f}', xy=(episodes[-1], lc_final_grid),
                xytext=(episodes[-1]+100, lc_final_grid),
                fontsize=10, color=color_lc, fontweight='bold', va='center')
    ax.annotate(f'{mp_final_grid:.1f}', xy=(episodes[-1], mp_final_grid),
                xytext=(episodes[-1]+100, mp_final_grid),
                fontsize=10, color=color_mp, fontweight='bold', va='center')

    # (c) 可再生能源消纳率
    ax = axes[1, 0]
    ax.plot(episodes, mp_renewable * 100, alpha=0.10, color=color_mp, linewidth=0.4)
    ax.plot(episodes, lc_renewable * 100, alpha=0.10, color=color_lc, linewidth=0.4)
    ax.plot(sm_ep, smooth(mp_renewable * 100, window), color=color_mp, linewidth=2.4, label='MAPPO (Baseline)')
    ax.plot(sm_ep, smooth(lc_renewable * 100, window), color=color_lc, linewidth=2.4, label='LC-MAPPO (Ours)')
    ax.set_xlabel('Training Episode', fontsize=12)
    ax.set_ylabel('Renewable Consumption Rate (%)', fontsize=12)
    ax.set_title('(c) Renewable Energy Consumption Rate', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.25)
    ax.set_ylim(80, 100)

    # (d) 碳排放
    ax = axes[1, 1]
    ax.plot(episodes, mp_carbon, alpha=0.10, color=color_mp, linewidth=0.4)
    ax.plot(episodes, lc_carbon, alpha=0.10, color=color_lc, linewidth=0.4)
    ax.plot(sm_ep, smooth(mp_carbon, window), color=color_mp, linewidth=2.4, label='MAPPO (Baseline)')
    ax.plot(sm_ep, smooth(lc_carbon, window), color=color_lc, linewidth=2.4, label='LC-MAPPO (Ours)')
    ax.set_xlabel('Training Episode', fontsize=12)
    ax.set_ylabel('Carbon Emission (kgCO₂)', fontsize=12)
    ax.set_title('(d) Carbon Emission', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.25)

    fig.suptitle('Training Process Comparison: LC-MAPPO vs MAPPO (10000 Episodes)',
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()

    for ext in ['png', 'pdf', 'svg']:
        path = OUTPUT_DIR / f'p2p_grid_comparison_4panel.{ext}'
        fig.savefig(path, dpi=300 if ext == 'png' else None, bbox_inches='tight')
        print(f"saved: {path}")
    plt.close()

    # ========== 专项图：P2P + Grid 并排（用户要求的主图） ==========
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 左图: P2P 交易量
    ax1.fill_between(sm_ep, smooth(mp_p2p, window), alpha=0.12, color=color_mp)
    ax1.fill_between(sm_ep, smooth(lc_p2p, window), alpha=0.12, color=color_lc)
    ax1.plot(episodes, mp_p2p, alpha=0.12, color=color_mp, linewidth=0.4)
    ax1.plot(episodes, lc_p2p, alpha=0.12, color=color_lc, linewidth=0.4)
    ax1.plot(sm_ep, smooth(mp_p2p, window), color=color_mp, linewidth=2.4, label='MAPPO (Baseline)')
    ax1.plot(sm_ep, smooth(lc_p2p, window), color=color_lc, linewidth=2.4, label='LC-MAPPO (Ours)')
    ax1.axhline(y=mp_final_p2p, color=color_mp, linestyle='--', alpha=0.4, linewidth=1)
    ax1.axhline(y=lc_final_p2p, color=color_lc, linestyle='--', alpha=0.4, linewidth=1)
    # 提升标注
    p2p_improve = (lc_final_p2p - mp_final_p2p) / mp_final_p2p * 100 if mp_final_p2p != 0 else 0
    mid_x1 = len(episodes) * 0.75
    ax1.annotate('', xy=(mid_x1, lc_final_p2p), xytext=(mid_x1, mp_final_p2p),
                 arrowprops=dict(arrowstyle='<->', color='#2A9D8F', lw=1.8))
    ax1.text(mid_x1 + 120, (lc_final_p2p + mp_final_p2p) / 2,
             f'+{lc_final_p2p - mp_final_p2p:.1f} kWh\n(+{p2p_improve:.0f}%)',
             fontsize=11, fontweight='bold', color='#2A9D8F', va='center',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#2A9D8F', alpha=0.9))
    ax1.set_xlabel('Training Episode', fontsize=13)
    ax1.set_ylabel('P2P Energy (kWh)', fontsize=13)
    ax1.set_title('(a) P2P Trading Volume Convergence', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=12, loc='center right', framealpha=0.9)
    ax1.grid(True, alpha=0.25)
    ax1.set_xlim(0, len(episodes) + 600)

    # 右图: 电网交易量
    ax2.fill_between(sm_ep, smooth(mp_grid, window), alpha=0.12, color=color_mp)
    ax2.fill_between(sm_ep, smooth(lc_grid, window), alpha=0.12, color=color_lc)
    ax2.plot(episodes, mp_grid, alpha=0.12, color=color_mp, linewidth=0.4)
    ax2.plot(episodes, lc_grid, alpha=0.12, color=color_lc, linewidth=0.4)
    ax2.plot(sm_ep, smooth(mp_grid, window), color=color_mp, linewidth=2.4, label='MAPPO (Baseline)')
    ax2.plot(sm_ep, smooth(lc_grid, window), color=color_lc, linewidth=2.4, label='LC-MAPPO (Ours)')
    ax2.axhline(y=mp_final_grid, color=color_mp, linestyle='--', alpha=0.4, linewidth=1)
    ax2.axhline(y=lc_final_grid, color=color_lc, linestyle='--', alpha=0.4, linewidth=1)
    # 降低标注
    grid_reduce = (mp_final_grid - lc_final_grid) / mp_final_grid * 100 if mp_final_grid != 0 else 0
    mid_x2 = len(episodes) * 0.75
    ax2.annotate('', xy=(mid_x2, lc_final_grid), xytext=(mid_x2, mp_final_grid),
                 arrowprops=dict(arrowstyle='<->', color='#2A9D8F', lw=1.8))
    ax2.text(mid_x2 + 120, (lc_final_grid + mp_final_grid) / 2,
             f'−{mp_final_grid - lc_final_grid:.1f} ¥\n(−{grid_reduce:.1f}%)',
             fontsize=11, fontweight='bold', color='#2A9D8F', va='center',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#2A9D8F', alpha=0.9))
    ax2.set_xlabel('Training Episode', fontsize=13)
    ax2.set_ylabel('Grid Purchase Cost (¥)', fontsize=13)
    ax2.set_title('(b) Grid Trading Volume Convergence', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=12, loc='center right', framealpha=0.9)
    ax2.grid(True, alpha=0.25)
    ax2.set_xlim(0, len(episodes) + 600)

    fig.suptitle('P2P & Grid Trading Volume: LC-MAPPO vs MAPPO (10000 Episodes)',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()

    for ext in ['png', 'pdf', 'svg']:
        path = OUTPUT_DIR / f'p2p_grid_comparison.{ext}'
        fig.savefig(path, dpi=300 if ext == 'png' else None, bbox_inches='tight')
        print(f"saved: {path}")
    plt.close()

    # ========== 打印统计 ==========
    print("\n" + "=" * 70)
    print("训练过程对比统计（最后 1000 episodes 均值）")
    print("=" * 70)
    print(f"{'Metric':<30} {'LC-MAPPO':<15} {'MAPPO':<15} {'Diff':<15}")
    print("-" * 70)
    for name, lc_v, mp_v, unit in [
        ('P2P Energy', lc_final_p2p, mp_final_p2p, 'kWh'),
        ('Grid Purchase Cost', lc_final_grid, mp_final_grid, '¥'),
        ('Renewable Rate', np.mean(lc_renewable[-1000:])*100, np.mean(mp_renewable[-1000:])*100, '%'),
        ('Carbon Emission', np.mean(lc_carbon[-1000:]), np.mean(mp_carbon[-1000:]), 'kgCO₂'),
    ]:
        diff = lc_v - mp_v
        print(f"{name:<30} {lc_v:<15.2f} {mp_v:<15.2f} {diff:<+15.2f}")

if __name__ == '__main__':
    main()
