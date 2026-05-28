from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]


def _load_metric(path: Path, metric: str) -> tuple[np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    episodes = np.asarray([float(row.get("episode", idx)) for idx, row in enumerate(rows)])
    values = np.asarray([float(row[metric]) for row in rows], dtype=float)
    return episodes, values


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    out = np.empty_like(values, dtype=float)
    cumsum = np.cumsum(np.insert(values, 0, 0.0))
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        out[idx] = (cumsum[idx + 1] - cumsum[start]) / (idx - start + 1)
    return out


def _summarize(seeds: list[str], values: np.ndarray, tail: int) -> dict:
    final_values = values[:, -1]
    tail_values = values[:, -min(tail, values.shape[1]) :].mean(axis=1)
    return {
        "seeds": seeds,
        "episodes": int(values.shape[1]),
        "final_mean_reward_by_seed": {
            seed: float(value) for seed, value in zip(seeds, final_values)
        },
        f"last_{tail}_episode_mean_reward_by_seed": {
            seed: float(value) for seed, value in zip(seeds, tail_values)
        },
        f"last_{tail}_episode_mean_reward_mean": float(tail_values.mean()),
        f"last_{tail}_episode_mean_reward_std": float(tail_values.std(ddof=0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot multi-seed TECSF training curves.")
    parser.add_argument(
        "--run-dir",
        default="outputs/tecsf_chain_1000_multiseed_20260527",
        help="Directory containing seed_*/tecsf_metrics.json.",
    )
    parser.add_argument("--metric", default="mean_reward")
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--tail", type=int, default=100)
    parser.add_argument(
        "--seeds",
        nargs="*",
        default=None,
        help="Optional seed list, e.g. --seeds 100 42 2026.",
    )
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if args.seeds:
        metric_paths = [run_dir / f"seed_{seed}" / "tecsf_metrics.json" for seed in args.seeds]
    else:
        metric_paths = sorted(run_dir.glob("seed_*/tecsf_metrics.json"))
    if not metric_paths:
        raise FileNotFoundError(f"No seed_*/tecsf_metrics.json found under {run_dir}")
    missing = [str(path) for path in metric_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing metrics files: " + ", ".join(missing))

    seeds: list[str] = []
    episodes_ref: np.ndarray | None = None
    curves: list[np.ndarray] = []
    for path in metric_paths:
        seed = path.parent.name.removeprefix("seed_")
        episodes, values = _load_metric(path, args.metric)
        if episodes_ref is None:
            episodes_ref = episodes
        elif len(episodes) != len(episodes_ref):
            raise ValueError("All seed runs must have the same episode count.")
        seeds.append(seed)
        curves.append(values)

    assert episodes_ref is not None
    values = np.vstack(curves)
    smoothed = np.vstack([_moving_average(curve, args.window) for curve in values])
    mean_curve = smoothed.mean(axis=0)
    std_curve = smoothed.std(axis=0, ddof=0)

    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "SimHei"],
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.unicode_minus": False,
        }
    )

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for idx, (seed, curve) in enumerate(zip(seeds, smoothed)):
        ax.plot(
            episodes_ref,
            curve,
            color=OKABE_ITO[idx % len(OKABE_ITO)],
            linewidth=1.35,
            alpha=0.8,
            label=f"Seed {seed}",
        )
    ax.plot(episodes_ref, mean_curve, color="#111111", linewidth=2.2, label="Mean")
    ax.fill_between(
        episodes_ref,
        mean_curve - std_curve,
        mean_curve + std_curve,
        color="#999999",
        alpha=0.22,
        linewidth=0,
        label="Mean ± std",
    )
    ax.set_title("TECSF training reward across three random seeds")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean episode reward")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=3, loc="lower right")
    fig.tight_layout()

    png_path = output_dir / f"{args.metric}_multiseed_curve.png"
    pdf_path = output_dir / f"{args.metric}_multiseed_curve.pdf"
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)

    summary = _summarize(seeds, values, args.tail)
    summary_path = output_dir / "multiseed_training_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")

    print(f"png={png_path}")
    print(f"pdf={pdf_path}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
