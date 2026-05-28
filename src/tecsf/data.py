from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tecsf.config import ExperimentConfig


@dataclass
class SyntheticScenario:
    num_agents: int
    num_nodes: int
    horizon: int
    delta_t: float
    agent_nodes: np.ndarray
    load: np.ndarray
    pv: np.ndarray
    grid_buy_price: np.ndarray
    grid_sell_price: np.ndarray
    carbon_allowance_price: np.ndarray
    low_carbon_sell_price: np.ndarray
    grid_emission_factor: np.ndarray
    line_from: np.ndarray
    line_to: np.ndarray
    resistance: np.ndarray
    reactance: np.ndarray
    line_capacity: np.ndarray

    def time_index(self, t: int) -> int:
        return int(t % self.horizon)


def _daily_profile(horizon: int, phase: float = 0.0) -> np.ndarray:
    x = np.arange(horizon, dtype=np.float32)
    return 0.5 + 0.5 * np.sin(2.0 * np.pi * (x / max(horizon, 1)) + phase)


def generate_synthetic_scenario(config: ExperimentConfig, seed: int | None = None) -> SyntheticScenario:
    sc = config.scenario
    rng = np.random.default_rng(sc.seed if seed is None else seed)
    horizon = sc.horizon
    num_agents = sc.num_agents
    num_nodes = sc.num_nodes

    agent_nodes = 1 + (np.arange(num_agents) % max(1, num_nodes - 1))

    evening = _daily_profile(horizon, phase=-np.pi / 2.0)
    midday = np.maximum(0.0, np.sin(np.pi * np.arange(horizon) / max(horizon - 1, 1)))
    load_base = rng.uniform(0.7, 1.5, size=(num_agents, 1))
    load_shape = 0.8 + 0.7 * evening.reshape(1, horizon)
    load_noise = rng.normal(0.0, 0.05, size=(num_agents, horizon))
    load = np.maximum(0.15, load_base * load_shape + load_noise).astype(np.float32)

    pv_cap = rng.uniform(0.4, 2.4, size=(num_agents, 1))
    pv_noise = rng.normal(0.0, 0.04, size=(num_agents, horizon))
    pv = np.maximum(0.0, pv_cap * midday.reshape(1, horizon) + pv_noise).astype(np.float32)

    price_shape = 0.85 + 0.25 * evening
    grid_buy_price = (config.market.grid_buy_base * price_shape).astype(np.float32)
    grid_sell_price = (
        config.market.grid_sell_base * (0.95 + 0.1 * midday)
    ).astype(np.float32)
    carbon_allowance_price = np.full(
        horizon, config.market.carbon_allowance_price, dtype=np.float32
    )
    low_carbon_sell_price = np.full(
        horizon, config.market.low_carbon_sell_price, dtype=np.float32
    )
    grid_emission_factor = (
        config.market.grid_emission_base * (1.0 + 0.12 * evening - 0.08 * midday)
    ).astype(np.float32)

    # Radial feeder: parent of node k is floor((k - 1) / 2).
    line_to = np.arange(1, num_nodes, dtype=np.int64)
    line_from = ((line_to - 1) // 2).astype(np.int64)
    line_count = max(0, num_nodes - 1)
    resistance = np.full(line_count, config.network.default_resistance, dtype=np.float32)
    reactance = np.full(line_count, config.network.default_reactance, dtype=np.float32)
    line_capacity = np.full(
        line_count, config.network.default_line_capacity, dtype=np.float32
    )

    return SyntheticScenario(
        num_agents=num_agents,
        num_nodes=num_nodes,
        horizon=horizon,
        delta_t=sc.delta_t,
        agent_nodes=agent_nodes.astype(np.int64),
        load=load,
        pv=pv,
        grid_buy_price=grid_buy_price,
        grid_sell_price=grid_sell_price,
        carbon_allowance_price=carbon_allowance_price,
        low_carbon_sell_price=low_carbon_sell_price,
        grid_emission_factor=grid_emission_factor,
        line_from=line_from,
        line_to=line_to,
        resistance=resistance,
        reactance=reactance,
        line_capacity=line_capacity,
    )
