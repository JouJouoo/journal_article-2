from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tecsf.metrics import write_json
from tecsf.rl.mappo import evaluate_policy


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a TECSF checkpoint.")
    parser.add_argument("checkpoint")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--output", default="outputs/evaluation.json")
    args = parser.parse_args()
    result = evaluate_policy(args.checkpoint, config=args.config, episodes=args.episodes)
    write_json(args.output, result)
    print(f"evaluation={args.output}")


if __name__ == "__main__":
    main()
