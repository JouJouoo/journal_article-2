from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tecsf.rl.mappo import train


def main() -> None:
    parser = argparse.ArgumentParser(description="训练低碳资产感知 MAPPO 变体.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--variant", default="tecsf")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None, help="auto, cpu, cuda, or cuda:<index>")
    parser.add_argument("--output-dir", default="outputs/train")
    args = parser.parse_args()
    result = train(
        config=args.config,
        variant=args.variant,
        output_dir=args.output_dir,
        episodes=args.episodes,
        seed=args.seed,
        device=args.device,
    )
    print(f"checkpoint={result.checkpoint_path}")
    print(f"metrics={result.metrics_path}")


if __name__ == "__main__":
    main()
