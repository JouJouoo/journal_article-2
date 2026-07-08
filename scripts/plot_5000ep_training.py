"""
Plot IEEE 33/69 5000-episode training curves.
Data source: stdout logs (reward snapshots every 50 episodes)
+ 200-ep benchmark metrics for detailed convergence curves.
"""
import re
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(r"C:\Users\zrway\Desktop\期刊论文-2\outputs\ieee_benchmark_lcmappo_20260706")
OUT = BASE / "figures_5000ep"
OUT.mkdir(parents=True, exist_ok=True)

# ─── 1. Parse 5000-ep log files ───
def parse_log(path):
    episodes, rewards = [], []
    with open(path) as f:
        for line in f:
            m = re.search(r"episode\s+(\d+)/(\d+)\s+reward=([\d.\-]+)", line)
            if m:
                episodes.append(int(m.group(1)))
                rewards.append(float(m.group(3)))
    return np.array(episodes), np.array(rewards)

eps33, rw33 = parse_log(BASE / "parallel_5000_logs" / "ieee33_5000.log")
eps69, rw69 = parse_log(BASE / "parallel_5000_logs" / "ieee69_5000.log")

print(f"IEEE 33: {len(eps33)} snapshots, ep range {eps33[0]}-{eps33[-1]}, rw range {rw33.min():.4f}-{rw33.max():.4f}")
print(f"IEEE 69: {len(eps69)} snapshots, ep range {eps69[0]}-{eps69[-1]}, rw range {rw69.min():.4f}-{rw69.max():.4f}")

# ─── 2. Load 200-ep benchmark metrics for detailed curves ───
def load_benchmark_metrics(case_dir):
    path = BASE / case_dir / "tecsf" / "seed_7" / "tecsf_metrics.json"
    with open(path) as f:
        data = json.load(f)
    eps = np.array([m["episode"] for m in data])
    total_rw = np.array([m["total_reward"] for m in data])
    mean_rw = np.array([m["mean_reward"] for m in data])
    sys_cost = np.array([m["system_cost"] for m in data])
    carbon = np.array([m["grid_carbon_emission"] for m in data])
    lccoins = np.array([m["lccoins"] for m in data])
    p2p = np.array([m["p2p_energy"] for m in data])
    settlement = np.array([m["settlement_success_rate"] for m in data])
    carbon_red = np.array([m["carbon_reduction"] for m in data])
    actor_loss = np.array([m["actor_loss"] for m in data])
    critic_loss = np.array([m["critic_loss"] for m in data])
    return {
        "eps": eps, "total_reward": total_rw, "mean_reward": mean_rw,
        "system_cost": sys_cost, "carbon": carbon, "lccoins": lccoins,
        "p2p": p2p, "settlement": settlement, "carbon_reduction": carbon_red,
        "actor_loss": actor_loss, "critic_loss": critic_loss,
    }

metrics33 = load_benchmark_metrics("benchmark_ieee33bw")
metrics69 = load_benchmark_metrics("benchmark_ieee69")

print(f"IEEE 33 benchmark: {len(metrics33['eps'])} episodes")
print(f"IEEE 69 benchmark: {len(metrics69['eps'])} episodes")

smooth = lambda x, w=20: np.convolve(x, np.ones(w)/w, mode='valid')

# ─── 3. Publication style ───
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman"],
    "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 9, "figure.dpi": 150,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})

# ─── Figure 1: 5000-episode reward curves ───
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

for ax, eps, rw, name, color in [
    (ax1, eps33, rw33, "IEEE 33-bus", "#0072B2"),
    (ax2, eps69, rw69, "IEEE 69", "#009E73"),
]:
    ax.plot(eps, rw, "o-", color=color, linewidth=1.2, markersize=3, alpha=0.8)
    ax.axhline(y=np.mean(rw), color="red", linestyle="--", alpha=0.4, linewidth=0.8,
               label=f"Mean={np.mean(rw):.4f}")
    ax.set_xlim(0, 5000)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Average Reward")
    ax.set_title(f"{name} Training Reward — LC-MAPPO (5000 ep)", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

fig.tight_layout()
for fmt in ["png", "pdf", "svg"]:
    fig.savefig(OUT / f"fig1_5000ep_reward.{fmt}")
print(f"Fig 1 saved: 5000-ep reward curves")
plt.close(fig)

# ─── Figure 2: Combined reward overlay ───
fig, ax = plt.subplots(figsize=(10, 5.5))
ax.plot(eps33, rw33, "o-", color="#0072B2", linewidth=1.2, markersize=3, alpha=0.8, label="IEEE 33-bus")
ax.plot(eps69, rw69, "s-", color="#009E73", linewidth=1.2, markersize=3, alpha=0.8, label="IEEE 69-bus")
ax.axhline(y=np.mean(rw33), color="#0072B2", linestyle="--", alpha=0.3, linewidth=0.8)
ax.axhline(y=np.mean(rw69), color="#009E73", linestyle="--", alpha=0.3, linewidth=0.8)
ax.set_xlim(0, 5000)
ax.set_xlabel("Episode")
ax.set_ylabel("Average Reward")
ax.set_title("IEEE 33 vs IEEE 69 — LC-MAPPO Training Reward Comparison", fontweight="bold")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
for fmt in ["png", "pdf", "svg"]:
    fig.savefig(OUT / f"fig2_reward_comparison.{fmt}")
print(f"Fig 2 saved: reward comparison")
plt.close(fig)

# ─── Figure 3: Detailed convergence from 200-ep benchmark (IEEE 33) ───
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes_flat = axes.flatten()
titles_33 = [
    ("Total Reward", "blue", metrics33["total_reward"]),
    ("Mean Reward", "teal", metrics33["mean_reward"]),
    ("System Cost", "red", metrics33["system_cost"]),
    ("Grid Carbon Emission", "darkred", metrics33["carbon"]),
    ("Carbon Reduction", "green", metrics33["carbon_reduction"]),
    ("LCCoins Minted", "purple", metrics33["lccoins"]),
]

for ax, (title, color, vals) in zip(axes_flat, titles_33):
    ax.plot(metrics33["eps"], vals, color=color, alpha=0.25, linewidth=0.5)
    if len(vals) > 10:
        s = smooth(vals, 10)
        ax.plot(metrics33["eps"][9:], s, color=color, linewidth=1.8, label="Smoothed")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Episode")
    ax.grid(True, alpha=0.3)

fig.suptitle("IEEE 33-bus — LC-MAPPO Training Convergence (200-ep Benchmark)", fontsize=14, fontweight="bold")
fig.tight_layout()
for fmt in ["png", "pdf", "svg"]:
    fig.savefig(OUT / f"fig3_ieee33_convergence.{fmt}")
print(f"Fig 3 saved: IEEE 33 convergence detail")
plt.close(fig)

# ─── Figure 4: Detailed convergence from 200-ep benchmark (IEEE 69) ───
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes_flat = axes.flatten()
titles_69 = [
    ("Total Reward", "blue", metrics69["total_reward"]),
    ("Mean Reward", "teal", metrics69["mean_reward"]),
    ("System Cost", "red", metrics69["system_cost"]),
    ("Grid Carbon Emission", "darkred", metrics69["carbon"]),
    ("Carbon Reduction", "green", metrics69["carbon_reduction"]),
    ("LCCoins Minted", "purple", metrics69["lccoins"]),
]

for ax, (title, color, vals) in zip(axes_flat, titles_69):
    ax.plot(metrics69["eps"], vals, color=color, alpha=0.25, linewidth=0.5)
    if len(vals) > 10:
        s = smooth(vals, 10)
        ax.plot(metrics69["eps"][9:], s, color=color, linewidth=1.8, label="Smoothed")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Episode")
    ax.grid(True, alpha=0.3)

fig.suptitle("IEEE 69-bus — LC-MAPPO Training Convergence (200-ep Benchmark)", fontsize=14, fontweight="bold")
fig.tight_layout()
for fmt in ["png", "pdf", "svg"]:
    fig.savefig(OUT / f"fig4_ieee69_convergence.{fmt}")
print(f"Fig 4 saved: IEEE 69 convergence detail")
plt.close(fig)

# ─── Figure 5: IEEE 33 vs 69 side-by-side comparison (key metrics) ───
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
compare_metrics = [
    ("Total Reward", "total_reward"),
    ("System Cost", "system_cost"),
    ("Grid Carbon", "carbon"),
    ("LCCoins", "lccoins"),
    ("Carbon Reduction", "carbon_reduction"),
    ("P2P Energy", "p2p"),
]

for ax, (title, key) in zip(axes.flatten(), compare_metrics):
    v33 = metrics33[key]
    v69 = metrics69[key]
    ax.plot(metrics33["eps"], v33, color="#0072B2", alpha=0.3, linewidth=0.5)
    ax.plot(metrics69["eps"], v69, color="#009E73", alpha=0.3, linewidth=0.5)
    if len(v33) > 10:
        s33 = smooth(v33, 10)
        ax.plot(metrics33["eps"][9:], s33, color="#0072B2", linewidth=1.8, label="IEEE 33")
    if len(v69) > 10:
        s69 = smooth(v69, 10)
        ax.plot(metrics69["eps"][9:], s69, color="#009E73", linewidth=1.8, label="IEEE 69")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Episode")
    ax.legend()
    ax.grid(True, alpha=0.3)

fig.suptitle("IEEE 33 vs IEEE 69 — LC-MAPPO Key Metrics Comparison", fontsize=14, fontweight="bold")
fig.tight_layout()
for fmt in ["png", "pdf", "svg"]:
    fig.savefig(OUT / f"fig5_ieee_comparison.{fmt}")
print(f"Fig 5 saved: IEEE comparison")
plt.close(fig)

# ─── Figure 6: Training losses ───
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
for ax, key, ylabel in [(ax1, "actor_loss", "Actor Loss"), (ax2, "critic_loss", "Critic Loss")]:
    v33 = metrics33[key]
    v69 = metrics69[key]
    ax.plot(metrics33["eps"], v33, color="#0072B2", alpha=0.3, linewidth=0.5)
    ax.plot(metrics69["eps"], v69, color="#009E73", alpha=0.3, linewidth=0.5)
    if len(v33) > 5:
        s33 = smooth(v33, 5)
        ax.plot(metrics33["eps"][4:], s33, color="#0072B2", linewidth=1.5, label="IEEE 33")
    if len(v69) > 5:
        s69 = smooth(v69, 5)
        ax.plot(metrics69["eps"][4:], s69, color="#009E73", linewidth=1.5, label="IEEE 69")
    ax.set_title(ylabel, fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)

fig.suptitle("LC-MAPPO Training Losses", fontsize=13, fontweight="bold")
fig.tight_layout()
for fmt in ["png", "pdf", "svg"]:
    fig.savefig(OUT / f"fig6_losses.{fmt}")
print(f"Fig 6 saved: training losses")
plt.close(fig)

# ─── Summary stats ───
print("\n" + "="*60)
print("TRAINING SUMMARY")
print("="*60)

for name, eps, rw, mname, m in [
    ("IEEE 33 (5000ep)", eps33, rw33, "IEEE 33 (200ep)", metrics33),
    ("IEEE 69 (5000ep)", eps69, rw69, "IEEE 69 (200ep)", metrics69),
]:
    print(f"\n{name}:")
    print(f"  Snapshots:          {len(eps)}")
    print(f"  Episode range:      {eps[0]}–{eps[-1]}")
    print(f"  Final reward:        {rw[-1]:.4f}")
    print(f"  Mean reward:         {np.mean(rw):.4f} ± {np.std(rw):.4f}")
    print(f"  Max reward:          {np.max(rw):.4f} (ep {eps[np.argmax(rw)]})")
    print(f"  Min reward:          {np.min(rw):.4f} (ep {eps[np.argmin(rw)]})")

    print(f"\n{mname} (benchmark metrics):")
    tail = 50
    print(f"  Last {tail} ep mean_rw:     {np.mean(m['mean_reward'][-tail:]):.4f}")
    print(f"  Last {tail} ep system_cost: {np.mean(m['system_cost'][-tail:]):.2f}")
    print(f"  Last {tail} ep carbon:      {np.mean(m['carbon'][-tail:]):.2f}")
    print(f"  Last {tail} ep lccoins:     {np.mean(m['lccoins'][-tail:]):.2f}")
    print(f"  Last {tail} ep carbon_red:  {np.mean(m['carbon_reduction'][-tail:]):.2f}")

print(f"\nAll figures saved to: {OUT}")
for f in sorted(OUT.glob("*.png")):
    print(f"  {f.name}")
