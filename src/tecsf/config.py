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
    profile_path: str = ""
    load_scale: float = 1.0
    pv_scale: float = 1.0
    load_noise_scale: float = 1.0
    pv_noise_scale: float = 1.0


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
    clean_energy_weight: float = 1.0
    carbon_reduction_weight: float = 1.0
    minting_coefficient: float = 1.0
    crra_rho: float = 0.5
    crra_b0: float = 1.0
    stock_utility_weight: float = 0.1
    increment_utility_weight: float = 0.2
    balance_norm: float = 10.0
    validator_count: int = 4
    consensus_threshold: int = 0


@dataclass
class ClearingConfig:
    enable_dynamic_action_bounds: bool = True
    local_trade_margin: float = 1.2
    local_storage_margin: float = 1.0
    action_saturation_penalty: float = 0.02
    max_repair_iters: int = 20
    repair_shrink: float = 0.9
    network_tolerance: float = 1e-4
    violation_penalty: float = 20.0
    enable_safety_shield: bool = True
    safety_storage_fraction: float = 1.0
    enable_emergency_balancing: bool = True
    max_emergency_iters: int = 20
    emergency_balance_shrink: float = 0.7
    load_shed_penalty: float = 5.0
    pv_curtail_penalty: float = 0.5
    lagrange_step_size: float = 0.05
    preserve_lagrange_on_reset: bool = False
    adaptive_violation_penalty_gain: float = 0.5
    violation_penalty_max_multiplier: float = 5.0


@dataclass
class RLConfig:
    hidden_dim: int = 64
    recurrent_dim: int = 64
    actor_lr: float = 5e-5
    critic_lr: float = 1e-3
    gamma: float = 0.98
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    clip_eps_min: float = 0.1
    clip_eps_max: float = 0.3
    entropy_coef: float = 0.0
    log_std_init: float = -3.0
    log_std_min: float = -4.0
    log_std_max: float = -2.5
    action_prior_coef: float = 2.0
    action_prior_decay_episodes: int = 0
    lr_decay_episodes: int = 5000
    min_lr_factor: float = 0.0
    target_kl: float = 0.02
    best_checkpoint_window: int = 500
    early_stop_warmup_episodes: int = 4000
    early_stop_patience: int = 1000
    early_stop_min_delta: float = 5e-4
    restore_best_on_plateau: bool = True
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    update_epochs: int = 1
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
