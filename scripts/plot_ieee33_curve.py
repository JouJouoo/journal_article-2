"""Plot IEEE 33 LC-MAPPO training curves (200 episodes)."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

DATA = "C:/Users/zrway/Desktop/期刊论文-2/outputs/ieee_benchmark_lcmappo_20260706/benchmark_ieee33bw/tecsf/seed_7/tecsf_metrics.json"
OUT_DIR = "C:/Users/zrway/Desktop/期刊论文-2/outputs/ieee_benchmark_lcmappo_20260706"

with open(DATA) as f:
    metrics = json.load(f)

episodes = np.arange(1, len(metrics) + 1)

def smooth(y, window=10):
    y = np.array(y, dtype=float)
    s = np.convolve(y, np.ones(window)/window, mode='valid')
    x = np.arange(window//2, len(y) - window//2 + 1)
    if len(s) != len(x):
        s = s[:len(x)]
    return x, s

# ---- Figure 1: Main training curve (total_reward) ----
fig, ax = plt.subplots(figsize=(10, 5.5))

total_reward = [e['total_reward'] for e in metrics]
ax.plot(episodes, total_reward, alpha=0.3, color='#E74C3C', linewidth=0.8, label='Raw')
sx, sy = smooth(total_reward, 20)
ax.plot(sx, sy, color='#E74C3C', linewidth=2.2, label='Smoothed (window=20)')

ax.axhline(y=57.5, color='gray', linestyle='--', alpha=0.4, linewidth=1)
ax.text(195, 58.2, f'Final: {total_reward[-1]:.1f}', fontsize=10, ha='right', color='#E74C3C', fontweight='bold')
ax.text(5, 23.5, f'Start: {total_reward[0]:.1f}', fontsize=10, ha='left', color='gray')

ax.set_xlabel('Training Episode', fontsize=12)
ax.set_ylabel('Total Reward', fontsize=12)
ax.set_title('IEEE 33 — LC-MAPPO Training Curve (200 Episodes)', fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='lower right')
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "ieee33_training_curve.png"), dpi=150)
fig.savefig(os.path.join(OUT_DIR, "ieee33_training_curve.pdf"))
fig.savefig(os.path.join(OUT_DIR, "ieee33_training_curve.svg"))
plt.close()
print("Saved: ieee33_training_curve.png/pdf/svg")

# ---- Figure 2: Multi-panel decomposition ----
fig, axes = plt.subplots(3, 2, figsize=(14, 12))
fig.suptitle('IEEE 33 — LC-MAPPO Training Decomposition', fontsize=14, fontweight='bold')

panels = [
    (0, 0, 'total_reward', 'Total Reward', '#E74C3C', None),
    (0, 1, 'reward_eco', 'Economic Reward (¥)', '#2980B9', None),
    (1, 0, 'reward_coin', 'LCCoins Utility Reward', '#27AE60', None),
    (1, 1, 'p2p_energy', 'P2P Energy (kWh)', '#8E44AD', 'kW·h'),
    (2, 0, 'grid_buy_cost', 'Grid Purchase Cost (¥)', '#E67E22', None),
    (2, 1, 'carbon_emission', 'Carbon Emission (kgCO₂)', '#2C3E50', 'kg'),
]

for row, col, key, title, color, unit in panels:
    ax = axes[row, col]
    vals = [e[key] for e in metrics]
    ax.plot(episodes, vals, alpha=0.3, color=color, linewidth=0.8)
    sx, sy = smooth(vals, 20)
    ax.plot(sx, sy, color=color, linewidth=2.2)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Episode')
    ax.grid(True, alpha=0.3)
    # Annotate final value
    final_val = np.mean(vals[-10:])
    ax.annotate(f'{final_val:.2f}', xy=(195, final_val), fontsize=9,
                color=color, fontweight='bold', ha='right',
                xytext=(0, 8), textcoords='offset points')

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "ieee33_decomposition.png"), dpi=150)
fig.savefig(os.path.join(OUT_DIR, "ieee33_decomposition.pdf"))
plt.close()
print("Saved: ieee33_decomposition.png/pdf")

print("\n=== IEEE 33 Training Summary ===")
print(f"Episodes: {len(metrics)}")
print(f"Total Reward:  {total_reward[0]:.1f} → {total_reward[-1]:.1f}  (+{total_reward[-1]-total_reward[0]:.1f})")
print(f"Economic:      {metrics[0]['reward_eco']:.1f} → {metrics[-1]['reward_eco']:.1f}")
print(f"LCCoins:       {metrics[0]['reward_coin']:.1f} → {metrics[-1]['reward_coin']:.1f}")
print(f"P2P Energy:    {metrics[0]['p2p_energy']:.3f} → {metrics[-1]['p2p_energy']:.3f} kWh")
print(f"Grid Buy:      {metrics[0]['grid_buy_cost']:.1f} → {metrics[-1]['grid_buy_cost']:.1f} ¥")
print(f"Carbon:        {metrics[0]['carbon_emission']:.1f} → {metrics[-1]['carbon_emission']:.1f} kg")
