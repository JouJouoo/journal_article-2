from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tecsf.config import load_config
from tecsf.device import resolve_device
from tecsf.env import EnergyCarbonEnv
from tecsf.metrics import write_json
from tecsf.rl.networks import RecurrentGaussianActor
from tecsf.variants import get_variant


def _load_actor(checkpoint_path: str | Path, env: EnergyCarbonEnv, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    variant = checkpoint.get("variant", "tecsf")
    variant_spec = get_variant(variant)
    actor = RecurrentGaussianActor(
        obs_dim=env.observation_dim,
        action_dim=env.action_dim,
        hidden_dim=env.config.rl.hidden_dim,
        recurrent_dim=env.config.rl.recurrent_dim,
        use_recurrence=variant_spec.use_recurrence,
    ).to(device)
    actor.load_state_dict(checkpoint["actor_state_dict"])
    actor.eval()
    return actor, variant


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a deterministic policy rollout and export simulated TECS-Chain ledgers."
    )
    parser.add_argument("checkpoint", nargs="?", default=None)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None, help="auto, cpu, cuda, or cuda:<index>")
    parser.add_argument("--output-dir", default="outputs/ledger_export")
    parser.add_argument(
        "--fallback-policy",
        choices=["heuristic", "zero"],
        default="heuristic",
        help="Policy to use when no checkpoint is supplied.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device or cfg.rl.device)
    base_seed = cfg.scenario.seed if args.seed is None else args.seed

    summaries = []
    for episode in range(args.episodes):
        env = EnergyCarbonEnv(cfg, variant="tecsf")
        actor = None
        variant = "tecsf"
        if args.checkpoint is not None:
            actor, variant = _load_actor(args.checkpoint, env, device)
        obs, _, _ = env.reset(seed=base_seed + episode)
        hidden = (
            actor.initial_hidden(env.num_agents, device)
            if actor is not None
            else None
        )

        done = False
        rewards = []
        while not done:
            if actor is None:
                if args.fallback_policy == "heuristic":
                    action = env.heuristic_action()
                else:
                    action = np.zeros((env.num_agents, env.action_dim), dtype=np.float32)
            else:
                with torch.no_grad():
                    dist, hidden = actor(
                        torch.as_tensor(obs, dtype=torch.float32, device=device),
                        hidden,
                    )
                    action = torch.clamp(dist.mean, -1.0, 1.0).cpu().numpy()
            result = env.step(action)
            rewards.append(result.reward)
            obs = result.observation
            done = result.terminated or result.truncated

        ledger_path = output_dir / f"ledger_ep{episode}.json"
        env.chain.export_ledger(ledger_path)
        blocks = env.chain.blocks
        records = env.chain.records
        summaries.append(
            {
                "episode": episode,
                "variant": variant,
                "seed": base_seed + episode,
                "ledger": str(ledger_path),
                "blocks": len(blocks),
                "transactions": sum(len(block.transactions) for block in blocks),
                "settled_records": sum(record.settled for record in records),
                "total_records": len(records),
                "total_lccoins": float(env.chain.lccoins_balances.sum()),
                "mean_reward": float(np.asarray(rewards, dtype=np.float32).mean()),
                "head_hash": blocks[-1].block_hash if blocks else env.chain.genesis_hash,
            }
        )

    summary_path = output_dir / "summary.json"
    write_json(summary_path, summaries)
    print(f"summary={summary_path}")
    for item in summaries:
        print(
            "ledger={ledger} blocks={blocks} txs={transactions} "
            "settled={settled_records}/{total_records} head={head_hash}".format(**item)
        )


if __name__ == "__main__":
    main()
