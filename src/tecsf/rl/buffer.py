from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from tecsf.variants import VariantSpec


@dataclass
class RolloutBuffer:
    observations: list[np.ndarray] = field(default_factory=list)
    global_states: list[np.ndarray] = field(default_factory=list)
    hidden_states: list[np.ndarray] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    action_priors: list[np.ndarray] = field(default_factory=list)
    log_probs: list[np.ndarray] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)
    values: list[np.ndarray] = field(default_factory=list)
    infos: list[dict] = field(default_factory=list)
    per_agent_rewards: list[np.ndarray] = field(default_factory=list)
    per_agent_reward_eco: list[np.ndarray] = field(default_factory=list)
    per_agent_reward_coin: list[np.ndarray] = field(default_factory=list)
    asset_states: list[np.ndarray] = field(default_factory=list)
    trade_relations: list[np.ndarray] = field(default_factory=list)
    _use_asset_observation: bool = field(default=False, init=False)

    def set_variant(self, variant: VariantSpec) -> None:
        """设置变体配置，用于正确填充asset_states."""
        self._use_asset_observation = variant.use_asset_observation

    def add(
        self,
        observation: np.ndarray,
        global_state: np.ndarray,
        hidden_state: np.ndarray,
        action: np.ndarray,
        log_prob: np.ndarray,
        reward: np.ndarray,
        done: bool,
        value: np.ndarray,
        info: dict,
        action_prior: np.ndarray | None = None,
    ) -> None:
        n_agents = int(reward.shape[0])
        value_arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if value_arr.size == 1:
            value_arr = np.asarray([value_arr[0], 0.0], dtype=np.float32)
        self.observations.append(observation.astype(np.float32))
        self.global_states.append(global_state.astype(np.float32))
        self.hidden_states.append(hidden_state.astype(np.float32))
        self.actions.append(action.astype(np.float32))
        prior = action if action_prior is None else action_prior
        self.action_priors.append(prior.astype(np.float32))
        self.log_probs.append(log_prob.astype(np.float32))
        self.rewards.append(float(np.mean(reward)))
        self.per_agent_rewards.append(reward.astype(np.float32))
        self.dones.append(bool(done))
        self.values.append(value_arr.astype(np.float32))
        self.infos.append(info)
        self.per_agent_reward_eco.append(
            np.asarray(info.get("agent_reward_eco", reward), dtype=np.float32)
        )
        self.per_agent_reward_coin.append(
            np.asarray(
                info.get("agent_reward_coin", np.zeros(n_agents, dtype=np.float32)),
                dtype=np.float32,
            )
        )
        # 根据变体配置正确填充asset_states
        obs_arr = np.asarray(observation, dtype=np.float32)
        if self._use_asset_observation:
            # 启用资产观测时，从第-1维提取LCCoins余额
            self.asset_states.append(obs_arr[:, -1].astype(np.float32))
        else:
            # 不启用资产观测时，填充全0（无资产信息）
            self.asset_states.append(np.zeros(n_agents, dtype=np.float32))
        self.trade_relations.append(
            np.asarray(
                info.get(
                    "p2p_matrix",
                    np.zeros((n_agents, n_agents), dtype=np.float32),
                ),
                dtype=np.float32,
            )
        )

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "observations": np.asarray(self.observations, dtype=np.float32),
            "global_states": np.asarray(self.global_states, dtype=np.float32),
            "hidden_states": np.asarray(self.hidden_states, dtype=np.float32),
            "actions": np.asarray(self.actions, dtype=np.float32),
            "action_priors": np.asarray(self.action_priors, dtype=np.float32),
            "log_probs": np.asarray(self.log_probs, dtype=np.float32),
            "rewards": np.asarray(self.rewards, dtype=np.float32),
            "dones": np.asarray(self.dones, dtype=np.float32),
            "values": np.asarray(self.values, dtype=np.float32),
            "per_agent_rewards": np.asarray(self.per_agent_rewards, dtype=np.float32),
            "per_agent_reward_eco": np.asarray(
                self.per_agent_reward_eco, dtype=np.float32
            ),
            "per_agent_reward_coin": np.asarray(
                self.per_agent_reward_coin, dtype=np.float32
            ),
            "asset_states": np.asarray(self.asset_states, dtype=np.float32),
            "trade_relations": np.asarray(self.trade_relations, dtype=np.float32),
        }
