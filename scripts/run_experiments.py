from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tecsf.metrics import write_json
from tecsf.rl.mappo import train
from tecsf.variants import VARIANTS


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TECSF baselines and ablations.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--output-dir", default="outputs/experiments")
    parser.add_argument(
        "--variants",
        nargs="*",
        default=["tecsf", "no_chain", "no_lccoins", "no_feedback", "mappo", "no_lagrange", "heuristic"],
        choices=sorted(VARIANTS),
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = {}
    for variant in args.variants:
        result = train(
            config=args.config,
            variant=variant,
            output_dir=output_dir / variant,
            episodes=args.episodes,
        )
        all_metrics[variant] = result.episode_metrics

    write_json(output_dir / "summary.json", all_metrics)
    plt.figure(figsize=(8, 4.5))
    for variant, metrics in all_metrics.items():
        xs = [m["episode"] for m in metrics]
        ys = [m["mean_reward"] for m in metrics]
        plt.plot(xs, ys, label=variant)
    plt.xlabel("Episode")
    plt.ylabel("Mean reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "reward_curves.png", dpi=160)
    print(f"summary={output_dir / 'summary.json'}")
    print(f"figure={output_dir / 'reward_curves.png'}")


if __name__ == "__main__":
    main()
