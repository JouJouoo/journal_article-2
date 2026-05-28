from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RolloutBuffer:
    observations: list[np.ndarray] = field(default_factory=list)
    global_states: list[np.ndarray] = field(default_factory=list)
    hidden_states: list[np.ndarray] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    log_probs: list[np.ndarray] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    infos: list[dict] = field(default_factory=list)
    per_agent_rewards: list[np.ndarray] = field(default_factory=list)

    def add(
        self,
        observation: np.ndarray,
        global_state: np.ndarray,
        hidden_state: np.ndarray,
        action: np.ndarray,
        log_prob: np.ndarray,
        reward: np.ndarray,
        done: bool,
        value: float,
        info: dict,
    ) -> None:
        self.observations.append(observation.astype(np.float32))
        self.global_states.append(global_state.astype(np.float32))
        self.hidden_states.append(hidden_state.astype(np.float32))
        self.actions.append(action.astype(np.float32))
        self.log_probs.append(log_prob.astype(np.float32))
        self.rewards.append(float(np.mean(reward)))
        self.per_agent_rewards.append(reward.astype(np.float32))
        self.dones.append(bool(done))
        self.values.append(float(value))
        self.infos.append(info)

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "observations": np.asarray(self.observations, dtype=np.float32),
            "global_states": np.asarray(self.global_states, dtype=np.float32),
            "hidden_states": np.asarray(self.hidden_states, dtype=np.float32),
            "actions": np.asarray(self.actions, dtype=np.float32),
            "log_probs": np.asarray(self.log_probs, dtype=np.float32),
            "rewards": np.asarray(self.rewards, dtype=np.float32),
            "dones": np.asarray(self.dones, dtype=np.float32),
            "values": np.asarray(self.values, dtype=np.float32),
            "per_agent_rewards": np.asarray(self.per_agent_rewards, dtype=np.float32),
        }
