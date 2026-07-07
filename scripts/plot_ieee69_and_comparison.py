"""
Plot IEEE 69 training curve and comparison with IEEE 33.
"""
import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = "C:/Users/zrway/Desktop/期刊论文-2/outputs/ieee_benchmark_lcmappo_20260706"
SMOOTH = 10

# ── Load IEEE 69 metrics ───────────────────────────────────────────────
with open(f"{OUT}/benchmark_ieee69/tecsf/seed_7/tecsf_metrics.json") as f:
    d69 = json.load(f)
eps69 = np.arange(1, len(d69)+1)
R69  = {k: np.array([e[k] for e in d69]) for k in d69[0] if isinstance(d69[0][k], (int,float))}

# ── Load IEEE 33 metrics ───────────────────────────────────────────────
with open(f"{OUT}/benchmark_ieee33bw/tecsf/seed_7/tecsf_metrics.json") as f:
    d33 = json.load(f)
eps33 = np.arange(1, len(d33)+1)
R33  = {k: np.array([e[k] for e in d33]) for k in d33[0] if isinstance(d33[0][k], (int,float))}

def smooth(x, w=SMOOTH):
    s = np.convolve(x, np.ones(w)/w, mode='same')
    return s

# ═══════════════════════════════════════════════════════════════════
#  FIG 1 — IEEE 69 training curve (same style as IEEE 33)
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(1, 1, figsize=(9, 5))
ax.plot(eps69, smooth(R69['total_reward']), color='#e74c3c', lw=2, label='Smoothed (w=10)')
ax.plot(eps69, R69['total_reward'], color='#e74c3c', alpha=0.25, lw=0.8)
ax.set_xlabel('Training Episode', fontsize=12)
ax.set_ylabel('Total Reward', fontsize=12)
ax.set_title('IEEE 69 — LC-MAPPO Training Curve (200 Episodes)', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11)
fig.tight_layout()
fig.savefig(f"{OUT}/ieee69_training_curve.png", dpi=200)
fig.savefig(f"{OUT}/ieee69_training_curve.pdf")
fig.savefig(f"{OUT}/ieee69_training_curve.svg")
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════
#  FIG 2 — Reward decomposition for IEEE 69 (6-panel)
# ═══════════════════════════════════════════════════════════════════
decomp_keys = ['total_reward','reward_eco','reward_coin',
               'p2p_energy','grid_buy_cost','carbon_emission']
titles      = ['Total Reward','Economic Reward','LCCoins Utility',
               'P2P Energy (kWh)','Grid Purchase (¥)','Carbon (kgCO₂)']
colors      = ['#e74c3c','#3498db','#2ecc71','#f39c12','#9b59b6','#1abc9c']
ylabels     = ['Total Reward','Reward (¥)','Utility (CRRA)','kWh','¥','kgCO₂']

fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))
axes = axes.flatten()
for i, (key, title, color, ylabel) in enumerate(zip(decomp_keys, titles, colors, ylabels)):
    ax = axes[i]
    if key in R69:
        ax.plot(eps69, smooth(R69[key]), color=color, lw=2)
        ax.plot(eps69, R69[key], color=color, alpha=0.2, lw=0.7)
    ax.set_title(f'({chr(97+i)}) {title}', fontsize=11, fontweight='bold')
    ax.set_xlabel('Episode', fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/ieee69_decomposition.png", dpi=200)
fig.savefig(f"{OUT}/ieee69_decomposition.pdf")
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════
#  FIG 3 — IEEE 33 vs IEEE 69 comparison (total_reward)
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(1, 1, figsize=(10, 5.5))
ax.plot(eps33, smooth(R33['total_reward']), color='#2980b9', lw=2.5, label='IEEE 33 (32 agents)')
ax.plot(eps33, R33['total_reward'], color='#2980b9', alpha=0.2, lw=0.7)
ax.plot(eps69, smooth(R69['total_reward']), color='#e74c3c', lw=2.5, label='IEEE 69 (68 agents)')
ax.plot(eps69, R69['total_reward'], color='#e74c3c', alpha=0.2, lw=0.7)

# Annotate final values
f33 = smooth(R33['total_reward'])[-1]
f69 = smooth(R69['total_reward'])[-1]
ax.annotate(f'IEEE 33 final: {f33:.1f}', xy=(200, f33), xytext=(180, f33+8),
            color='#2980b9', fontsize=10, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#2980b9'))
ax.annotate(f'IEEE 69 final: {f69:.1f}', xy=(200, f69), xytext=(180, f69-6),
            color='#e74c3c', fontsize=10, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#e74c3c'))

ax.set_xlabel('Training Episode', fontsize=12)
ax.set_ylabel('Total Reward', fontsize=12)
ax.set_title('LC-MAPPO: IEEE 33 vs IEEE 69 Training Comparison', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11, loc='lower right')
fig.tight_layout()
fig.savefig(f"{OUT}/ieee33_vs_ieee69_comparison.png", dpi=200)
fig.savefig(f"{OUT}/ieee33_vs_ieee69_comparison.pdf")
fig.savefig(f"{OUT}/ieee33_vs_ieee69_comparison.svg")
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════
#  FIG 4 — 4-panel comparison (total / eco / P2P / carbon)
# ═══════════════════════════════════════════════════════════════════
compare_keys = ['total_reward','reward_eco','p2p_energy','carbon_emission']
ctitles     = ['Total Reward','Economic Reward (¥)','P2P Energy (kWh)','Carbon Emission (kgCO₂)']
cylables     = ['Total Reward','Reward (¥)','kWh','kgCO₂']
ccolors33   = ['#2980b9'] * 4
ccolors69   = ['#e74c3c'] * 4

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()
for i, (key, title, ylabel) in enumerate(zip(compare_keys, ctitles, cylables)):
    ax = axes[i]
    if key in R33:
        ax.plot(eps33, smooth(R33[key]), color=ccolors33[i], lw=2, label='IEEE 33')
        ax.plot(eps33, R33[key], color=ccolors33[i], alpha=0.15, lw=0.6)
    if key in R69:
        ax.plot(eps69, smooth(R69[key]), color=ccolors69[i], lw=2, label='IEEE 69')
        ax.plot(eps69, R69[key], color=ccolors69[i], alpha=0.15, lw=0.6)
    ax.set_title(f'({chr(97+i)}) {title}', fontsize=11, fontweight='bold')
    ax.set_xlabel('Episode', fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(f"{OUT}/ieee33_vs_ieee69_4panel.png", dpi=200)
fig.savefig(f"{OUT}/ieee33_vs_ieee69_4panel.pdf")
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════
#  Print summary statistics
# ═══════════════════════════════════════════════════════════════════
print("=== IEEE 33 vs IEEE 69 Summary (last 20 episodes) ===")
for key in ['total_reward','reward_eco','reward_coin','p2p_energy','grid_buy_cost','carbon_emission']:
    if key in R33 and key in R69:
        m33 = np.mean(R33[key][-20:])
        m69 = np.mean(R69[key][-20:])
        diff = m69 - m33
        pct = diff / abs(m33) * 100 if m33 != 0 else 0
        print(f"  {key:<20} IEEE33={m33:>8.2f}  IEEE69={m69:>8.2f}  diff={diff:>+8.2f} ({pct:>+.1f}%)")
