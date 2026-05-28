from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(
        to_jsonable(payload),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def ensure_2d_actions(actions: np.ndarray, num_agents: int, action_dim: int) -> np.ndarray:
    arr = np.asarray(actions, dtype=np.float32)
    if arr.shape != (num_agents, action_dim):
        raise ValueError(
            f"Expected actions with shape {(num_agents, action_dim)}, got {arr.shape}"
        )
    return arr
