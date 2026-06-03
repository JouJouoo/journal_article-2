from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from figure_style import (
    apply_publication_style,
    display_variant,
    save_publication_figure,
    style_axes,
    variant_color,
)
from tecsf.metrics import write_json
from tecsf.rl.mappo import train
from tecsf.variants import VARIANTS


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TECSF baselines and ablations.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None, help="auto, cpu, cuda, or cuda:<index>")
    parser.add_argument("--output-dir", default="outputs/experiments")
    parser.add_argument(
        "--variants",
        nargs="*",
        default=[
            "tecsf",
            "no_chain",
            "no_lccoins",
            "no_feedback",
            "mappo",
            "no_lagrange",
            "preset_low_carbon",
            "heuristic",
        ],
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
            seed=args.seed,
            device=args.device,
        )
        all_metrics[variant] = result.episode_metrics

    write_json(output_dir / "summary.json", all_metrics)
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for variant, metrics in all_metrics.items():
        xs = [m["episode"] for m in metrics]
        ys = [m["mean_reward"] for m in metrics]
        ax.plot(xs, ys, label=display_variant(variant), color=variant_color(variant), linewidth=1.4)
    ax.set_title("Training reward curves")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean reward")
    ax.legend(frameon=False, ncol=2)
    style_axes(ax)
    fig.tight_layout()
    figure_paths = save_publication_figure(fig, output_dir / "reward_curves")
    plt.close(fig)
    print(f"summary={output_dir / 'summary.json'}")
    for path in figure_paths:
        print(f"figure={path}")


if __name__ == "__main__":
    main()
