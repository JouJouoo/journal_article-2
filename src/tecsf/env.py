from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from tecsf.chain import SettlementRecord, SimulatedTECSChain
from tecsf.config import ExperimentConfig
from tecsf.data import SyntheticScenario, generate_synthetic_scenario
from tecsf.market import (
    ACTION_DIM,
    ActionBatch,
    clear_market,
    normalize_physical_actions,
    scale_raw_actions,
)
from tecsf.variants import VariantSpec, get_variant


OBS_DIM = 20


@dataclass
class StepResult:
    observation: np.ndarray
    global_state: np.ndarray
    reward: np.ndarray
    terminated: bool
    truncated: bool
    info: dict


class EnergyCarbonEnv:
    """Multi-agent TECSF environment with Gymnasium-style spaces."""

    def __init__(
        self,
        config: ExperimentConfig,
        variant: str | VariantSpec = "tecsf",
        scenario: SyntheticScenario | None = None,
    ):
        self.config = config
        self.variant = get_variant(variant) if isinstance(variant, str) else variant
        self.scenario = scenario or generate_synthetic_scenario(config, seed=config.scenario.seed)
        self.num_agents = self.scenario.num_agents
        self.action_dim = ACTION_DIM
        self.observation_dim = OBS_DIM
        self.global_state_dim = self.num_agents * self.observation_dim + self.num_agents**2 + 2 * self.num_agents + self.scenario.num_nodes + 4
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_agents, self.action_dim),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_agents, self.observation_dim),
            dtype=np.float32,
        )
        self.chain = SimulatedTECSChain(config, self.num_agents)
        self.t = 0
        self.soc = np.full(self.num_agents, config.storage.init_soc, dtype=np.float32)
        self.prev_feedback = np.zeros((self.num_agents, 8), dtype=np.float32)
        self.prev_actions = np.zeros((self.num_agents, ACTION_DIM), dtype=np.float32)
        self.last_p2p = np.zeros((self.num_agents, self.num_agents), dtype=np.float32)
        self.last_grid_buy = np.zeros(self.num_agents, dtype=np.float32)
        self.last_grid_sell = np.zeros(self.num_agents, dtype=np.float32)
        self.last_voltages = np.ones(self.scenario.num_nodes, dtype=np.float32)
        self.last_violations = np.zeros(4, dtype=np.float32)
        self.lagrange = np.zeros(4, dtype=np.float32)

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.scenario = generate_synthetic_scenario(self.config, seed=seed)
        self.chain = SimulatedTECSChain(self.config, self.num_agents)
        self.t = 0
        self.soc = np.full(self.num_agents, self.config.storage.init_soc, dtype=np.float32)
        self.prev_feedback.fill(0.0)
        self.prev_actions.fill(0.0)
        self.last_p2p.fill(0.0)
        self.last_grid_buy.fill(0.0)
        self.last_grid_sell.fill(0.0)
        self.last_voltages.fill(self.config.network.voltage_ref)
        self.last_violations.fill(0.0)
        self.lagrange.fill(0.0)
        obs = self._observation()
        return obs, self._global_state(obs), {}

    def step(self, raw_actions: np.ndarray) -> StepResult:
        raw = np.asarray(raw_actions, dtype=np.float32)
        if raw.shape != (self.num_agents, self.action_dim):
            raise ValueError(
                f"Expected action shape {(self.num_agents, self.action_dim)}, got {raw.shape}"
            )
        physical_actions = scale_raw_actions(raw, self.config)
        package = clear_market(self.scenario, self.config, self.t, self.soc, physical_actions)

        if self.variant.use_chain:
            record = self.chain.settle(package, enable_lccoins=self.variant.use_lccoins)
        else:
            record = self.chain.bypass_settlement(
                package, enable_lccoins=self.variant.use_lccoins
            )

        lccoins = self._settled_lccoins(record)
        rewards, info = self._rewards(package, record, lccoins)
        if self.variant.use_lagrange:
            violation_vec = np.asarray(
                [package.violations[k] for k in ("voltage", "line", "soc", "trade")],
                dtype=np.float32,
            )
            self.lagrange = np.maximum(0.0, self.lagrange + 0.05 * violation_vec)
        else:
            violation_vec = np.zeros(4, dtype=np.float32)

        self.soc = package.soc_next.copy()
        self.last_p2p = package.p2p_power.copy()
        self.last_grid_buy = package.grid_buy.copy()
        self.last_grid_sell = package.grid_sell.copy()
        self.last_voltages = package.voltages.copy()
        self.last_violations = violation_vec
        self.prev_actions = normalize_physical_actions(physical_actions, self.config)
        self.prev_feedback = self._feedback(package, lccoins) if self.variant.use_feedback and record.settled else np.zeros_like(self.prev_feedback)

        self.t += 1
        terminated = self.t >= self.scenario.horizon
        obs = self._observation()
        return StepResult(
            observation=obs,
            global_state=self._global_state(obs),
            reward=rewards.astype(np.float32),
            terminated=terminated,
            truncated=False,
            info=info,
        )

    def heuristic_action(self) -> np.ndarray:
        idx = self.scenario.time_index(self.t)
        load = self.scenario.load[:, idx]
        pv = self.scenario.pv[:, idx]
        surplus = np.maximum(pv - load, 0.0)
        deficit = np.maximum(load - pv, 0.0)
        raw = np.zeros((self.num_agents, self.action_dim), dtype=np.float32)
        raw[:, 0] = np.clip(deficit / max(self.config.market.max_buy_power, 1e-8), 0.0, 1.0)
        raw[:, 1] = np.clip(surplus / max(self.config.market.max_sell_power, 1e-8), 0.0, 1.0)
        raw[:, 2] = 0.4
        raw[:, 3] = -0.2
        raw[:, 4] = np.where(surplus > 0.1, 0.2, 0.0)
        raw[:, 5] = np.where(deficit > 0.1, 0.2, 0.0)
        return raw

    def _observation(self) -> np.ndarray:
        idx = self.scenario.time_index(self.t)
        load = self.scenario.load[:, idx]
        pv = self.scenario.pv[:, idx]
        soc_norm = self.soc / max(self.config.storage.capacity, 1e-8)
        buy_price = np.full(self.num_agents, self.scenario.grid_buy_price[idx], dtype=np.float32)
        sell_price = np.full(self.num_agents, self.scenario.grid_sell_price[idx], dtype=np.float32)
        gamma_prev = np.full(
            self.num_agents,
            self.scenario.grid_emission_factor[self.scenario.time_index(self.t - 1)],
            dtype=np.float32,
        )
        parts = [
            load.reshape(-1, 1),
            pv.reshape(-1, 1),
            soc_norm.reshape(-1, 1),
            buy_price.reshape(-1, 1),
            sell_price.reshape(-1, 1),
            gamma_prev.reshape(-1, 1),
            self.prev_feedback,
            self.prev_actions,
        ]
        return np.concatenate(parts, axis=1).astype(np.float32)

    def _global_state(self, obs: np.ndarray) -> np.ndarray:
        return np.concatenate(
            [
                obs.reshape(-1),
                self.last_p2p.reshape(-1),
                self.last_grid_buy,
                self.last_grid_sell,
                self.last_voltages,
                self.last_violations,
            ]
        ).astype(np.float32)

    def _feedback(self, package, lccoins: np.ndarray) -> np.ndarray:
        c = package.carbon
        return np.stack(
            [
                c.e_grid,
                c.e_pv_credit,
                c.c_offset,
                c.a_buy,
                c.c_sell,
                c.e_resp,
                lccoins,
                c.carbon_reduction,
            ],
            axis=1,
        ).astype(np.float32)

    def _settled_lccoins(self, record: SettlementRecord) -> np.ndarray:
        out = np.zeros(self.num_agents, dtype=np.float32)
        if not record.settled or not self.variant.use_lccoins:
            return out
        for agent, amount in record.lccoins.items():
            out[int(agent)] = float(amount)
        return out

    def _rewards(
        self, package, record: SettlementRecord, lccoins: np.ndarray
    ) -> tuple[np.ndarray, dict]:
        idx = self.scenario.time_index(package.epoch)
        dt = self.scenario.delta_t
        p2p_revenue = (package.p2p_power * package.p2p_price).sum(axis=1) * dt
        p2p_cost = (package.p2p_power * package.p2p_price).sum(axis=0) * dt
        grid_revenue = package.grid_sell * self.scenario.grid_sell_price[idx] * dt
        grid_cost = package.grid_buy * self.scenario.grid_buy_price[idx] * dt
        op_cost = self.config.storage.op_cost * (package.charge + package.discharge) * dt
        carbon_cost = (
            self.scenario.carbon_allowance_price[idx] * package.carbon.a_buy
            - self.scenario.low_carbon_sell_price[idx] * package.carbon.c_sell
        )
        lc_reward = self.config.lccoins.kappa * lccoins
        violation_vec = np.asarray(
            [package.violations[k] for k in ("voltage", "line", "soc", "trade")],
            dtype=np.float32,
        )
        lagrange_penalty = float(np.dot(self.lagrange, violation_vec))
        fixed_penalty = self.config.clearing.violation_penalty * float(violation_vec.sum())
        penalty = 0.0 if not self.variant.use_lagrange else lagrange_penalty + fixed_penalty
        rewards = p2p_revenue + grid_revenue - p2p_cost - grid_cost - op_cost - carbon_cost + lc_reward
        rewards = rewards - penalty / max(self.num_agents, 1)
        if not record.settled:
            rewards = rewards - 1.0
        info = {
            "settled": record.settled,
            "record_state": record.state,
            "record_reason": record.reason,
            "system_cost": float(grid_cost.sum() + p2p_cost.sum() + op_cost.sum() + np.maximum(carbon_cost, 0.0).sum()),
            "carbon_emission": float(package.carbon.e_grid.sum()),
            "lccoins": float(lccoins.sum()),
            "p2p_energy": float(package.p2p_power.sum() * dt),
            "max_violation": float(package.max_violation()),
            "violations": package.violations,
        }
        return rewards.astype(np.float32), info
