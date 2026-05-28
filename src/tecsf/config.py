from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

import yaml


@dataclass
class ScenarioConfig:
    num_agents: int = 8
    num_nodes: int = 5
    horizon: int = 24
    delta_t: float = 1.0
    seed: int = 7


@dataclass
class NetworkConfig:
    voltage_ref: float = 1.0
    voltage_min: float = 0.95
    voltage_max: float = 1.05
    default_resistance: float = 0.003
    default_reactance: float = 0.0
    default_line_capacity: float = 10.0


@dataclass
class MarketConfig:
    p2p_price_min: float = 0.2
    p2p_price_max: float = 1.2
    max_buy_power: float = 3.0
    max_sell_power: float = 3.0
    grid_buy_base: float = 0.82
    grid_sell_base: float = 0.35
    carbon_allowance_price: float = 0.06
    low_carbon_sell_price: float = 0.03
    grid_emission_base: float = 0.58
    pv_credit_beta: float = 1.0


@dataclass
class StorageConfig:
    capacity: float = 5.0
    soc_min: float = 0.5
    soc_max: float = 4.5
    init_soc: float = 2.5
    charge_efficiency: float = 0.95
    discharge_efficiency: float = 0.95
    max_charge_power: float = 1.5
    max_discharge_power: float = 1.5
    op_cost: float = 0.01


@dataclass
class LccoinsConfig:
    alpha_q: float = 1.0
    alpha_offset: float = 0.5
    kappa: float = 0.2
    q_norm: float = 5.0
    offset_norm: float = 3.0


@dataclass
class ClearingConfig:
    max_repair_iters: int = 20
    repair_shrink: float = 0.9
    network_tolerance: float = 1e-4
    violation_penalty: float = 20.0


@dataclass
class RLConfig:
    hidden_dim: int = 64
    recurrent_dim: int = 64
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    gamma: float = 0.98
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    update_epochs: int = 3
    batch_size: int = 128
    episodes: int = 20
    seed: int = 7
    device: str = "cpu"


@dataclass
class ExperimentConfig:
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    market: MarketConfig = field(default_factory=MarketConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    lccoins: LccoinsConfig = field(default_factory=LccoinsConfig)
    clearing: ClearingConfig = field(default_factory=ClearingConfig)
    rl: RLConfig = field(default_factory=RLConfig)


def _merge_dataclass(cls: type, data: dict[str, Any] | None):
    base = cls()
    if not data:
        return base
    type_hints = get_type_hints(cls)
    values = {}
    for f in fields(base):
        raw = data.get(f.name, getattr(base, f.name))
        field_type = type_hints.get(f.name, f.type)
        if is_dataclass(field_type):
            values[f.name] = _merge_dataclass(field_type, raw)
        else:
            values[f.name] = raw
    return cls(**values)


def load_config(path: str | Path = "configs/default.yaml") -> ExperimentConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return _merge_dataclass(ExperimentConfig, data)
