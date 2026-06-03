from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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


def _load_optional_vector(
    data: np.lib.npyio.NpzFile,
    key: str,
    default: np.ndarray,
    dtype,
) -> np.ndarray:
    if key not in data:
        return default.astype(dtype)
    return np.asarray(data[key], dtype=dtype)


def _require_shape(name: str, value: np.ndarray, expected: tuple[int, ...]) -> None:
    if value.shape != expected:
        raise ValueError(f"profile {name} must have shape {expected}, got {value.shape}")


def _load_profile_scenario(config: ExperimentConfig, path: str | Path) -> SyntheticScenario:
    sc = config.scenario
    profile_path = Path(path)
    if not profile_path.exists():
        raise FileNotFoundError(f"Scenario profile not found: {profile_path}")
    data = np.load(profile_path)
    load = np.asarray(data["load"], dtype=np.float32) * sc.load_scale
    pv = np.asarray(data["pv"], dtype=np.float32) * sc.pv_scale
    if load.ndim != 2 or pv.ndim != 2 or load.shape != pv.shape:
        raise ValueError("profile load and pv arrays must be 2-D with matching shapes")

    num_agents, horizon = load.shape
    num_nodes = int(data["num_nodes"]) if "num_nodes" in data else sc.num_nodes
    if num_nodes < 2:
        raise ValueError("profile num_nodes must be >= 2")
    agent_nodes = _load_optional_vector(
        data,
        "agent_nodes",
        1 + (np.arange(num_agents) % max(1, num_nodes - 1)),
        np.int64,
    )
    _require_shape("agent_nodes", agent_nodes, (num_agents,))
    if np.any(agent_nodes < 0) or np.any(agent_nodes >= num_nodes):
        raise ValueError("profile agent_nodes must reference valid node indices")
    line_to = _load_optional_vector(data, "line_to", np.arange(1, num_nodes), np.int64)
    line_from = _load_optional_vector(data, "line_from", ((line_to - 1) // 2), np.int64)
    if line_from.shape != line_to.shape:
        raise ValueError("profile line_from and line_to arrays must have matching shape")
    if np.any(line_from < 0) or np.any(line_from >= num_nodes):
        raise ValueError("profile line_from must reference valid node indices")
    if np.any(line_to < 0) or np.any(line_to >= num_nodes):
        raise ValueError("profile line_to must reference valid node indices")
    line_count = int(line_to.shape[0])
    resistance = _load_optional_vector(
        data,
        "resistance",
        np.full(line_count, config.network.default_resistance),
        np.float32,
    )
    reactance = _load_optional_vector(
        data,
        "reactance",
        np.full(line_count, config.network.default_reactance),
        np.float32,
    )
    line_capacity = _load_optional_vector(
        data,
        "line_capacity",
        np.full(line_count, config.network.default_line_capacity),
        np.float32,
    )
    for name, value in [
        ("resistance", resistance),
        ("reactance", reactance),
        ("line_capacity", line_capacity),
    ]:
        _require_shape(name, value, (line_count,))
    grid_buy_price = _load_optional_vector(
        data,
        "grid_buy_price",
        np.full(horizon, config.market.grid_buy_base),
        np.float32,
    )
    grid_sell_price = _load_optional_vector(
        data,
        "grid_sell_price",
        np.full(horizon, config.market.grid_sell_base),
        np.float32,
    )
    carbon_allowance_price = _load_optional_vector(
        data,
        "carbon_allowance_price",
        np.full(horizon, config.market.carbon_allowance_price),
        np.float32,
    )
    low_carbon_sell_price = _load_optional_vector(
        data,
        "low_carbon_sell_price",
        np.full(horizon, config.market.low_carbon_sell_price),
        np.float32,
    )
    grid_emission_factor = _load_optional_vector(
        data,
        "grid_emission_factor",
        np.full(horizon, config.market.grid_emission_base),
        np.float32,
    )
    for name, value in [
        ("grid_buy_price", grid_buy_price),
        ("grid_sell_price", grid_sell_price),
        ("carbon_allowance_price", carbon_allowance_price),
        ("low_carbon_sell_price", low_carbon_sell_price),
        ("grid_emission_factor", grid_emission_factor),
    ]:
        _require_shape(name, value, (horizon,))
    return SyntheticScenario(
        num_agents=num_agents,
        num_nodes=num_nodes,
        horizon=horizon,
        delta_t=sc.delta_t,
        agent_nodes=agent_nodes.astype(np.int64),
        load=load.astype(np.float32),
        pv=pv.astype(np.float32),
        grid_buy_price=grid_buy_price.astype(np.float32),
        grid_sell_price=grid_sell_price.astype(np.float32),
        carbon_allowance_price=carbon_allowance_price.astype(np.float32),
        low_carbon_sell_price=low_carbon_sell_price.astype(np.float32),
        grid_emission_factor=grid_emission_factor.astype(np.float32),
        line_from=line_from.astype(np.int64),
        line_to=line_to.astype(np.int64),
        resistance=resistance.astype(np.float32),
        reactance=reactance.astype(np.float32),
        line_capacity=line_capacity.astype(np.float32),
    )


def generate_synthetic_scenario(config: ExperimentConfig, seed: int | None = None) -> SyntheticScenario:
    sc = config.scenario
    if sc.profile_path:
        return _load_profile_scenario(config, sc.profile_path)
    rng = np.random.default_rng(sc.seed if seed is None else seed)
    horizon = sc.horizon
    num_agents = sc.num_agents
    num_nodes = sc.num_nodes

    agent_nodes = 1 + (np.arange(num_agents) % max(1, num_nodes - 1))

    evening = _daily_profile(horizon, phase=-np.pi / 2.0)
    midday = np.maximum(0.0, np.sin(np.pi * np.arange(horizon) / max(horizon - 1, 1)))
    load_base = rng.uniform(0.7, 1.5, size=(num_agents, 1))
    load_shape = 0.8 + 0.7 * evening.reshape(1, horizon)
    load_noise = rng.normal(0.0, 0.05 * sc.load_noise_scale, size=(num_agents, horizon))
    load = np.maximum(0.15, sc.load_scale * (load_base * load_shape + load_noise)).astype(np.float32)

    pv_cap = rng.uniform(0.4, 2.4, size=(num_agents, 1))
    pv_noise = rng.normal(0.0, 0.04 * sc.pv_noise_scale, size=(num_agents, horizon))
    pv = np.maximum(0.0, sc.pv_scale * (pv_cap * midday.reshape(1, horizon) + pv_noise)).astype(np.float32)

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
