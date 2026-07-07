from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import gymnasium as gym
import numpy as np

from tecsf.chain import ConsortiumChainLedger, SettlementRecord
from tecsf.config import ExperimentConfig
from tecsf.data import SyntheticScenario, generate_synthetic_scenario
from tecsf.market import (
    ACTION_DIM,
    clear_market,
    p2p_reference_price,
    scale_raw_actions,
)
from tecsf.variants import VariantSpec, get_variant


P2P_REF_PRICE_OBS_INDEX = 3


def compute_obs_dim(variant: VariantSpec) -> int:
    """根据变体配置计算观测空间维度."""
    base_dim = 6  # load, pv, soc, p2p_ref_price, buy_price, sell_price
    if variant.use_asset_observation:
        base_dim += 1  # + lccoins_balance
    return base_dim


@dataclass
class StepResult:
    observation: np.ndarray
    global_state: np.ndarray
    reward: np.ndarray
    terminated: bool
    truncated: bool
    info: dict


class EnergyCarbonEnv:
    """面向 P2P 低碳能源交易的多智能体环境."""

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
        self.observation_dim = compute_obs_dim(self.variant)
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
        self.chain = ConsortiumChainLedger(config, self.num_agents)
        self.t = 0
        self.soc = np.full(self.num_agents, config.storage.init_soc, dtype=np.float32)
        self.last_p2p = np.zeros((self.num_agents, self.num_agents), dtype=np.float32)
        self.last_grid_buy = np.zeros(self.num_agents, dtype=np.float32)
        self.last_grid_sell = np.zeros(self.num_agents, dtype=np.float32)
        self.last_voltages = np.ones(self.scenario.num_nodes, dtype=np.float32)
        self.last_violations = np.zeros(4, dtype=np.float32)
        self.lagrange = np.zeros(4, dtype=np.float32)

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.scenario = generate_synthetic_scenario(self.config, seed=seed)
        self.chain = ConsortiumChainLedger(self.config, self.num_agents)
        self.t = 0
        self.soc = np.full(self.num_agents, self.config.storage.init_soc, dtype=np.float32)
        self.last_p2p.fill(0.0)
        self.last_grid_buy.fill(0.0)
        self.last_grid_sell.fill(0.0)
        self.last_voltages.fill(self.config.network.voltage_ref)
        self.last_violations.fill(0.0)
        if not self.config.clearing.preserve_lagrange_on_reset:
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
        clear_start = perf_counter()
        package = clear_market(self.scenario, self.config, self.t, self.soc, physical_actions)
        clear_seconds = perf_counter() - clear_start

        settlement_start = perf_counter()
        lccoins_balance_before = self.chain.lccoins_balances.copy()
        if self.variant.use_chain:
            record = self.chain.settle(package, enable_lccoins=self.variant.use_lccoins)
        else:
            record = self.chain.bypass_settlement(
                package, enable_lccoins=self.variant.use_lccoins
            )
        lccoins_balance_after = self.chain.lccoins_balances.copy()
        settlement_seconds = perf_counter() - settlement_start

        lccoins = self._settled_lccoins(record)
        reward_start = perf_counter()
        rewards, info = self._rewards(
            package,
            record,
            lccoins,
            lccoins_balance_before,
            lccoins_balance_after,
        )
        info["clear_seconds"] = float(clear_seconds)
        info["settlement_seconds"] = float(settlement_seconds)
        info["reward_seconds"] = float(perf_counter() - reward_start)
        if self.variant.use_lagrange:
            violation_vec = np.asarray(
                [package.violations[k] for k in ("voltage", "line", "soc", "trade")],
                dtype=np.float32,
            )
            self.lagrange = np.maximum(
                0.0,
                self.lagrange + self.config.clearing.lagrange_step_size * violation_vec,
            )
        else:
            violation_vec = np.zeros(4, dtype=np.float32)

        self.soc = package.soc_next.copy()
        self.last_p2p = package.p2p_power.copy()
        self.last_grid_buy = package.grid_buy.copy()
        self.last_grid_sell = package.grid_sell.copy()
        self.last_voltages = package.voltages.copy()
        self.last_violations = violation_vec
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

    def greedy_feasible_action(self) -> np.ndarray:
        idx = self.scenario.time_index(self.t)
        load = self.scenario.load[:, idx]
        pv = self.scenario.pv[:, idx]
        net = load - pv
        s = self.config.storage
        m = self.config.market
        dt = max(self.config.scenario.delta_t, 1e-8)
        max_charge_by_soc = np.maximum(0.0, (s.soc_max - self.soc) / (s.charge_efficiency * dt))
        max_discharge_by_soc = np.maximum(0.0, (self.soc - s.soc_min) * s.discharge_efficiency / dt)
        charge = np.minimum(np.maximum(-net, 0.0), np.minimum(s.max_charge_power, max_charge_by_soc))
        discharge = np.minimum(np.maximum(net, 0.0), np.minimum(s.max_discharge_power, max_discharge_by_soc))
        residual = load + charge - pv - discharge
        buy = np.minimum(np.maximum(residual, 0.0), m.max_buy_power)
        sell = np.minimum(np.maximum(-residual, 0.0), m.max_sell_power)

        raw = np.zeros((self.num_agents, self.action_dim), dtype=np.float32)
        raw[:, 0] = np.clip(buy / max(m.max_buy_power, 1e-8), 0.0, 1.0)
        raw[:, 1] = np.clip(sell / max(m.max_sell_power, 1e-8), 0.0, 1.0)
        raw[:, 2] = 0.8
        raw[:, 3] = -0.8
        raw[:, 4] = np.clip(charge / max(s.max_charge_power, 1e-8), 0.0, 1.0)
        raw[:, 5] = np.clip(discharge / max(s.max_discharge_power, 1e-8), 0.0, 1.0)
        return raw

    def deterministic_action(self) -> np.ndarray:
        return self.heuristic_action()

    def _observation(self) -> np.ndarray:
        idx = self.scenario.time_index(self.t)
        load = self.scenario.load[:, idx]
        pv = self.scenario.pv[:, idx]
        soc_norm = self.soc / max(self.config.storage.capacity, 1e-8)
        p2p_ref_price = np.full(
            self.num_agents,
            self._p2p_reference_price(self.t),
            dtype=np.float32,
        )
        buy_price = np.full(self.num_agents, self.scenario.grid_buy_price[idx], dtype=np.float32)
        sell_price = np.full(self.num_agents, self.scenario.grid_sell_price[idx], dtype=np.float32)
        parts = [
            load.reshape(-1, 1),
            pv.reshape(-1, 1),
            soc_norm.reshape(-1, 1),
            p2p_ref_price.reshape(-1, 1),
            buy_price.reshape(-1, 1),
            sell_price.reshape(-1, 1),
        ]
        if self.variant.use_asset_observation:
            parts.append(self._normalized_lccoins_balance().reshape(-1, 1))
        return np.concatenate(parts, axis=1).astype(np.float32)

    def _p2p_reference_price(self, t: int) -> float:
        return p2p_reference_price(self.scenario, self.config, t)

    def _normalized_lccoins_balance(self) -> np.ndarray:
        if not self.variant.use_asset_observation:
            return np.zeros(self.num_agents, dtype=np.float32)
        scale = max(float(self.config.lccoins.balance_norm), 1e-8)
        return (self.chain.lccoins_balances / scale).astype(np.float32)

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

    def _settled_lccoins(self, record: SettlementRecord) -> np.ndarray:
        out = np.zeros(self.num_agents, dtype=np.float32)
        if not record.settled or not self.variant.use_lccoins:
            return out
        for agent, amount in record.lccoins.items():
            out[int(agent)] = float(amount)
        return out

    def _crra_utility(self, balance: np.ndarray) -> np.ndarray:
        cfg = self.config.lccoins
        shifted = np.maximum(balance.astype(np.float64) + float(cfg.crra_b0), 1e-8)
        rho = float(cfg.crra_rho)
        if abs(rho - 1.0) <= 1e-8:
            utility = np.log(shifted)
        else:
            utility = ((shifted ** (1.0 - rho)) - 1.0) / (1.0 - rho)
        return utility.astype(np.float32)

    def _rewards(
        self,
        package,
        record: SettlementRecord,
        lccoins: np.ndarray,
        lccoins_balance_before: np.ndarray,
        lccoins_balance_after: np.ndarray,
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
        load_shed = package.load_shed * dt
        pv_curtailment = package.pv_curtailment * dt
        emergency_cost = (
            self.config.clearing.load_shed_penalty * load_shed
            + self.config.clearing.pv_curtail_penalty * pv_curtailment
        )
        utility_before = self._crra_utility(lccoins_balance_before)
        utility_stock = self._crra_utility(lccoins_balance_after)
        utility_increment = utility_stock - utility_before
        if self.variant.use_asset_utility and self.variant.use_lccoins:
            agent_reward_coin = (
                float(self.config.lccoins.stock_utility_weight) * utility_stock
                + float(self.config.lccoins.increment_utility_weight) * utility_increment
            )
        else:
            agent_reward_coin = np.zeros(self.num_agents, dtype=np.float32)
        violation_vec = np.asarray(
            [package.violations[k] for k in ("voltage", "line", "soc", "trade")],
            dtype=np.float32,
        )
        lagrange_penalty = float(np.dot(self.lagrange, violation_vec))
        violation_sum = float(violation_vec.sum())
        penalty_multiplier = min(
            float(self.config.clearing.violation_penalty_max_multiplier),
            1.0 + self.config.clearing.adaptive_violation_penalty_gain * violation_sum,
        )
        fixed_penalty = (
            self.config.clearing.violation_penalty
            * penalty_multiplier
            * violation_sum
        )
        action_bound_penalty = (
            float(self.config.clearing.action_saturation_penalty)
            * float(package.action_bound_deviation)
        )
        penalty = action_bound_penalty
        if self.variant.use_lagrange:
            penalty += lagrange_penalty + fixed_penalty
        agent_reward_eco = (
            p2p_revenue
            + grid_revenue
            - p2p_cost
            - grid_cost
            - op_cost
            - carbon_cost
            - emergency_cost
        )
        agent_profit = agent_reward_eco + agent_reward_coin
        rewards = agent_profit.copy()
        rewards = rewards - penalty / max(self.num_agents, 1)
        if not record.settled:
            rewards = rewards - 1.0
        pv_generation = float(self.scenario.pv[:, idx].sum() * dt)
        pv_used = float(package.carbon.q_lc.sum())
        trade_repair_deviation = float(
            np.abs(package.attempted_p2p_power - package.p2p_power).sum() * dt
        )
        storage_repair_deviation = float(
            (
                np.abs(package.attempted_charge - package.charge).sum()
                + np.abs(package.attempted_discharge - package.discharge).sum()
            )
            * dt
        )
        feasible = bool(
            record.settled
            and package.max_violation() <= self.config.clearing.network_tolerance
        )
        system_social_cost = float(
            grid_cost.sum()
            + op_cost.sum()
            + np.maximum(carbon_cost, 0.0).sum()
            + emergency_cost.sum()
        )
        participant_payment_cost = float(system_social_cost + p2p_cost.sum())
        info = {
            "settled": record.settled,
            "feasible": feasible,
            "record_state": record.state,
            "record_reason": record.reason,
            "system_cost": system_social_cost,
            "system_social_cost": system_social_cost,
            "participant_payment_cost": participant_payment_cost,
            "p2p_transfer_payment": float(p2p_cost.sum()),
            "p2p_transfer_revenue": float(p2p_revenue.sum()),
            "emergency_cost": float(emergency_cost.sum()),
            "load_shed": float(load_shed.sum()),
            "pv_curtailment": float(pv_curtailment.sum()),
            "carbon_emission": float(package.carbon.e_grid.sum()),
            "grid_carbon_emission": float(package.carbon.e_grid.sum()),
            "pv_credit": float(package.carbon.e_pv_credit.sum()),
            "net_carbon_allowance_need": float(package.carbon.e_ca_need.sum()),
            "allowance_buy": float(package.carbon.a_buy.sum()),
            "low_carbon_sell": float(package.carbon.c_sell.sum()),
            "carbon_reduction": float(package.carbon.carbon_reduction.sum()),
            "lccoins": float(lccoins.sum()),
            "agent_lccoins": lccoins.astype(float).tolist(),
            "agent_lccoins_balance": lccoins_balance_after.astype(float).tolist(),
            "agent_lccoins_asset_state": (
                lccoins_balance_after / max(float(self.config.lccoins.balance_norm), 1e-8)
            ).astype(float).tolist(),
            "agent_reward_eco": agent_reward_eco.astype(float).tolist(),
            "agent_reward_coin": agent_reward_coin.astype(float).tolist(),
            "agent_utility_stock": utility_stock.astype(float).tolist(),
            "agent_utility_increment": utility_increment.astype(float).tolist(),
            "agent_q_lc": package.carbon.q_lc.astype(float).tolist(),
            "agent_c_offset": package.carbon.c_offset.astype(float).tolist(),
            "agent_c_sell": package.carbon.c_sell.astype(float).tolist(),
            "agent_lccoins_candidate": package.carbon.lccoins_candidate.astype(float).tolist(),
            "agent_low_carbon_contribution": package.carbon.low_carbon_contribution.astype(float).tolist(),
            "agent_profit": agent_profit.astype(float).tolist(),
            "agent_carbon_emission": package.carbon.e_grid.astype(float).tolist(),
            "consensus_confirmed": [
                bool(record.consensus_confirmed.get(agent, False))
                for agent in range(self.num_agents)
            ],
            "consensus_votes": [
                int(record.consensus_votes.get(agent, 0))
                for agent in range(self.num_agents)
            ],
            "consensus_threshold": int(record.consensus_threshold),
            "validator_confirmations": [
                {
                    k: v
                    for k, v in record.validator_confirmations.get(
                        agent, {}
                    ).items()
                }
                for agent in range(self.num_agents)
            ],
            "p2p_reference_price": float(self._p2p_reference_price(package.epoch)),
            "p2p_matrix": package.p2p_power.astype(float).tolist(),
            "p2p_energy": float(package.p2p_power.sum() * dt),
            "attempted_p2p_energy": float(package.attempted_p2p_power.sum() * dt),
            "grid_buy_cost": float(grid_cost.sum()),
            "pv_generation": pv_generation,
            "pv_used": pv_used,
            "trade_repair_deviation": trade_repair_deviation,
            "storage_repair_deviation": storage_repair_deviation,
            "safety_adjustment": float(package.safety_adjustment),
            "action_bound_deviation": float(package.action_bound_deviation),
            "action_bound_penalty": float(action_bound_penalty),
            "repair_iterations": float(package.repair_iterations),
            "penalty_multiplier": float(penalty_multiplier),
            "max_violation": float(package.max_violation()),
            "violations": package.violations,
        }
        return rewards.astype(np.float32), info
