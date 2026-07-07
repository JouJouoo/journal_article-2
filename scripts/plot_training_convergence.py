#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从训练记录中生成收敛曲线图
包括：奖励函数、P2P交易量、电网交易量
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def load_training_data(summary_path):
    """加载训练数据"""
    with open(summary_path, 'r') as f:
        data = json.load(f)
    
    # 数据在 'lc_mappo' 键下，是一个列表
    episodes = data['lc_mappo']
    return episodes


def smooth_data(data, window=100):
    """使用移动平均平滑数据"""
    return np.convolve(data, np.ones(window)/window, mode='valid')


def plot_convergence_curves(episodes, output_dir, window=100):
    """
    生成三个收敛曲线图
    1. 奖励函数收敛图
    2. P2P交易量收敛图
    3. 电网交易量收敛图
    """
    # 提取数据
    episode_nums = [ep['episode'] for ep in episodes]
    total_rewards = [ep['total_reward'] for ep in episodes]
    p2p_energy = [ep['p2p_energy'] for ep in episodes]
    grid_buy_cost = [ep['grid_buy_cost'] for ep in episodes]  # 作为电网交易量代理
    
    # 创建图形
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    fig.suptitle('Training Convergence Curves (10000 Episodes)', fontsize=16, fontweight='bold')
    
    # 1. 奖励函数收敛图
    ax1 = axes[0]
    ax1.plot(episode_nums, total_rewards, alpha=0.3, color='blue', linewidth=0.5, label='Raw')
    
    # 平滑曲线
    if len(total_rewards) > window:
        smoothed_rewards = smooth_data(total_rewards, window)
        smoothed_episodes = episode_nums[window-1:]
        ax1.plot(smoothed_episodes, smoothed_rewards, color='blue', linewidth=2, label=f'Smoothed (window={window})')
    
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Total Reward', fontsize=12)
    ax1.set_title('Reward Function Convergence', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. P2P交易量收敛图
    ax2 = axes[1]
    ax2.plot(episode_nums, p2p_energy, alpha=0.3, color='green', linewidth=0.5, label='Raw')
    
    # 平滑曲线
    if len(p2p_energy) > window:
        smoothed_p2p = smooth_data(p2p_energy, window)
        ax2.plot(smoothed_episodes, smoothed_p2p, color='green', linewidth=2, label=f'Smoothed (window={window})')
    
    ax2.set_xlabel('Episode', fontsize=12)
    ax2.set_ylabel('P2P Energy (kWh)', fontsize=12)
    ax2.set_title('P2P Trading Volume Convergence', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. 电网交易量收敛图（使用grid_buy_cost作为代理）
    ax3 = axes[2]
    ax3.plot(episode_nums, grid_buy_cost, alpha=0.3, color='red', linewidth=0.5, label='Raw')
    
    # 平滑曲线
    if len(grid_buy_cost) > window:
        smoothed_grid = smooth_data(grid_buy_cost, window)
        ax3.plot(smoothed_episodes, smoothed_grid, color='red', linewidth=2, label=f'Smoothed (window={window})')
    
    ax3.set_xlabel('Episode', fontsize=12)
    ax3.set_ylabel('Grid Purchase Cost (¥)', fontsize=12)
    ax3.set_title('Grid Trading Volume Convergence (Proxy: Purchase Cost)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图形
    output_path_png = Path(output_dir) / 'training_convergence_curves.png'
    output_path_pdf = Path(output_dir) / 'training_convergence_curves.pdf'
    output_path_svg = Path(output_dir) / 'training_convergence_curves.svg'
    
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    plt.savefig(output_path_pdf, bbox_inches='tight')
    plt.savefig(output_path_svg, bbox_inches='tight')
    
    print(f"✓ 图表已保存到:")
    print(f"  - {output_path_png}")
    print(f"  - {output_path_pdf}")
    print(f"  - {output_path_svg}")
    
    # 显示统计信息
    print(f"\n=== 最终1000 episodes统计 ===")
    final_1000_rewards = total_rewards[-1000:]
    final_1000_p2p = p2p_energy[-1000:]
    final_1000_grid = grid_buy_cost[-1000:]
    
    print(f"奖励函数: 均值={np.mean(final_1000_rewards):.2f}, 标准差={np.std(final_1000_rewards):.2f}")
    print(f"P2P交易量: 均值={np.mean(final_1000_p2p):.2f}, 标准差={np.std(final_1000_p2p):.2f}")
    print(f"电网交易成本: 均值={np.mean(final_1000_grid):.2f}, 标准差={np.std(final_1000_grid):.2f}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='从训练记录生成收敛曲线图')
    parser.add_argument('--summary', type=str, required=True,
                        help='summary.json文件路径')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='输出目录（默认与summary.json同目录）')
    parser.add_argument('--window', type=int, default=100,
                        help='平滑窗口大小（默认100）')
    
    args = parser.parse_args()
    
    # 确定输出目录
    summary_path = Path(args.summary)
    if args.output_dir is None:
        output_dir = summary_path.parent
    else:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"正在加载训练数据: {summary_path}")
    episodes = load_training_data(summary_path)
    print(f"✓ 已加载 {len(episodes)} 个episodes的数据")
    
    print(f"\n正在生成收敛曲线图...")
    plot_convergence_curves(episodes, output_dir, window=args.window)
    
    print("\n✓ 完成！")


if __name__ == '__main__':
    main()
