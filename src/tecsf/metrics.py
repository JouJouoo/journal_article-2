from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def summarize_episode(infos: list[dict], rewards: list[np.ndarray]) -> dict[str, float]:
    if not infos:
        return {}
    reward_arr = np.asarray(rewards, dtype=np.float32)
    total_reward = float(reward_arr.sum())
    mean_reward = float(reward_arr.mean())
    total_cost = 0.0
    total_emission = 0.0
    total_lccoins = 0.0
    total_p2p = 0.0
    settled = 0
    rejected = 0
    max_violation = 0.0
    for info in infos:
        total_cost += float(info.get("system_cost", 0.0))
        total_emission += float(info.get("carbon_emission", 0.0))
        total_lccoins += float(info.get("lccoins", 0.0))
        total_p2p += float(info.get("p2p_energy", 0.0))
        settled += int(info.get("settled", False))
        rejected += int(not info.get("settled", False))
        max_violation = max(max_violation, float(info.get("max_violation", 0.0)))
    return {
        "total_reward": total_reward,
        "mean_reward": mean_reward,
        "system_cost": total_cost,
        "carbon_emission": total_emission,
        "lccoins": total_lccoins,
        "p2p_energy": total_p2p,
        "settlement_success_rate": settled / max(len(infos), 1),
        "rejected_records": float(rejected),
        "max_violation": max_violation,
    }


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)
