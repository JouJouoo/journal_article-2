from __future__ import annotations

import numpy as np

from tecsf.carbon import compute_carbon_result
from tecsf.config import ExperimentConfig, NetworkConfig, ScenarioConfig
from tecsf.data import generate_synthetic_scenario
from tecsf.market import ActionBatch, clear_market, double_auction, scale_raw_actions


def test_double_auction_matches_price_compatible_orders():
    actions = ActionBatch(
        q_buy=np.asarray([0.0, 2.0, 1.0], dtype=np.float32),
        q_sell=np.asarray([2.5, 0.0, 0.0], dtype=np.float32),
        price_buy=np.asarray([0.0, 0.8, 0.7], dtype=np.float32),
        price_sell=np.asarray([0.4, 0.0, 0.0], dtype=np.float32),
        charge=np.zeros(3, dtype=np.float32),
        discharge=np.zeros(3, dtype=np.float32),
    )
    power, price = double_auction(actions)
    assert np.isclose(power[0, 1], 2.0)
    assert np.isclose(power[0, 2], 0.5)
    assert np.isclose(price[0, 1], 0.6)
    assert np.isclose(price[0, 2], 0.55)


def test_zero_raw_power_actions_are_no_op():
    config = ExperimentConfig(scenario=ScenarioConfig(num_agents=3, num_nodes=3, horizon=4))
    raw = np.zeros((3, 6), dtype=np.float32)
    actions = scale_raw_actions(raw, config)

    assert np.allclose(actions.q_buy, 0.0)
    assert np.allclose(actions.q_sell, 0.0)
    assert np.allclose(actions.charge, 0.0)
    assert np.allclose(actions.discharge, 0.0)


def test_clearing_outputs_carbon_accounts_and_hash():
    config = ExperimentConfig(scenario=ScenarioConfig(num_agents=3, num_nodes=3, horizon=4))
    scenario = generate_synthetic_scenario(config, seed=123)
    actions = ActionBatch(
        q_buy=np.asarray([1.0, 1.0, 0.0], dtype=np.float32),
        q_sell=np.asarray([0.0, 0.0, 2.0], dtype=np.float32),
        price_buy=np.asarray([0.8, 0.75, 0.2], dtype=np.float32),
        price_sell=np.asarray([0.2, 0.2, 0.45], dtype=np.float32),
        charge=np.zeros(3, dtype=np.float32),
        discharge=np.zeros(3, dtype=np.float32),
    )
    package = clear_market(
        scenario=scenario,
        config=config,
        t=0,
        soc=np.full(3, config.storage.init_soc, dtype=np.float32),
        actions=actions,
    )
    assert package.package_hash
    assert package.attempted_p2p_power.shape == (3, 3)
    assert package.p2p_power.shape == (3, 3)
    assert package.carbon.e_grid.shape == (3,)
    assert np.all(package.carbon.a_buy >= -1e-7)
    assert np.all(package.carbon.c_sell >= -1e-7)
    assert set(package.violations) == {"voltage", "line", "soc", "trade"}


def test_grid_exchange_is_clearing_fallback_not_actor_action():
    config = ExperimentConfig(scenario=ScenarioConfig(num_agents=3, num_nodes=3, horizon=4))
    scenario = generate_synthetic_scenario(config, seed=123)
    raw = np.zeros((3, 6), dtype=np.float32)
    actions = scale_raw_actions(raw, config)

    assert not hasattr(actions, "grid_buy")
    assert not hasattr(actions, "grid_sell")

    package = clear_market(
        scenario=scenario,
        config=config,
        t=0,
        soc=np.full(3, config.storage.init_soc, dtype=np.float32),
        actions=actions,
    )
    p2p_sold = package.p2p_power.sum(axis=1)
    p2p_bought = package.p2p_power.sum(axis=0)
    residual = (
        package.effective_load
        + package.charge
        + p2p_sold
        - package.effective_pv
        - package.discharge
        - p2p_bought
    )

    assert np.allclose(package.grid_buy, np.maximum(residual, 0.0))
    assert np.allclose(package.grid_sell, np.maximum(-residual, 0.0))


def test_lccoins_clean_energy_excludes_external_grid_export():
    config = ExperimentConfig(scenario=ScenarioConfig(num_agents=3, num_nodes=3, horizon=4))
    scenario = generate_synthetic_scenario(config, seed=123)
    load = np.asarray([1.0, 0.5, 0.25], dtype=np.float32)
    pv = np.asarray([5.0, 5.0, 5.0], dtype=np.float32)
    zeros = np.zeros(3, dtype=np.float32)
    grid_sell = np.asarray([4.0, 4.5, 4.75], dtype=np.float32)

    carbon = compute_carbon_result(
        scenario=scenario,
        config=config,
        t=0,
        load=load,
        pv=pv,
        charge=zeros,
        p2p_sell=zeros,
        grid_buy=zeros,
        grid_sell=grid_sell,
    )

    assert np.allclose(carbon.q_lc, load * scenario.delta_t)


def test_default_no_trade_no_storage_scenario_is_feasible():
    config = ExperimentConfig()
    scenario = generate_synthetic_scenario(config, seed=config.scenario.seed)
    zeros = np.zeros(config.scenario.num_agents, dtype=np.float32)
    prices = np.full(
        config.scenario.num_agents, config.market.p2p_price_min, dtype=np.float32
    )
    soc = np.full(
        config.scenario.num_agents, config.storage.init_soc, dtype=np.float32
    )
    actions = ActionBatch(
        q_buy=zeros,
        q_sell=zeros,
        price_buy=prices,
        price_sell=prices,
        charge=zeros,
        discharge=zeros,
    )

    max_violation = 0.0
    for t in range(config.scenario.horizon):
        package = clear_market(scenario, config, t, soc, actions)
        max_violation = max(max_violation, package.max_violation())

    assert max_violation <= config.clearing.network_tolerance


def test_safety_shield_reduces_storage_induced_network_violation():
    config = ExperimentConfig(
        scenario=ScenarioConfig(num_agents=16, num_nodes=9, horizon=24),
        network=NetworkConfig(default_line_capacity=3.0),
    )
    config.clearing.enable_dynamic_action_bounds = False
    scenario = generate_synthetic_scenario(config, seed=7)
    n_agents = config.scenario.num_agents
    soc = np.full(n_agents, config.storage.init_soc, dtype=np.float32)
    actions = ActionBatch(
        q_buy=np.zeros(n_agents, dtype=np.float32),
        q_sell=np.zeros(n_agents, dtype=np.float32),
        price_buy=np.full(n_agents, 0.8, dtype=np.float32),
        price_sell=np.full(n_agents, 0.4, dtype=np.float32),
        charge=np.full(n_agents, config.storage.max_charge_power, dtype=np.float32),
        discharge=np.zeros(n_agents, dtype=np.float32),
    )

    shielded = clear_market(scenario, config, 18, soc, actions)
    config.clearing.enable_safety_shield = False
    unshielded = clear_market(scenario, config, 18, soc, actions)

    assert shielded.safety_adjustment > 0.0
    assert shielded.max_violation() < unshielded.max_violation()


def test_dynamic_action_bounds_prevent_extreme_storage_export():
    config = ExperimentConfig(
        scenario=ScenarioConfig(num_agents=8, num_nodes=5, horizon=24),
        network=NetworkConfig(default_line_capacity=10.0),
    )
    scenario = generate_synthetic_scenario(config, seed=7)
    n_agents = config.scenario.num_agents
    soc = np.full(n_agents, config.storage.init_soc, dtype=np.float32)
    actions = ActionBatch(
        q_buy=np.full(n_agents, config.market.max_buy_power, dtype=np.float32),
        q_sell=np.full(n_agents, config.market.max_sell_power, dtype=np.float32),
        price_buy=np.full(n_agents, 1.2, dtype=np.float32),
        price_sell=np.full(n_agents, 0.2, dtype=np.float32),
        charge=np.zeros(n_agents, dtype=np.float32),
        discharge=np.full(n_agents, config.storage.max_discharge_power, dtype=np.float32),
    )

    package = clear_market(scenario, config, 0, soc, actions)
    local_deficit = np.maximum(scenario.load[:, 0] - scenario.pv[:, 0], 0.0)

    assert package.action_bound_deviation > 0.0
    assert np.all(package.discharge <= local_deficit + 1e-4)
    assert package.max_violation() <= config.clearing.network_tolerance


def test_emergency_balancing_records_load_shed_and_pv_curtailment():
    config = ExperimentConfig(
        scenario=ScenarioConfig(num_agents=16, num_nodes=9, horizon=24),
        network=NetworkConfig(default_line_capacity=0.5),
    )
    config.storage.max_charge_power = 0.0
    config.storage.max_discharge_power = 0.0
    scenario = generate_synthetic_scenario(config, seed=7)
    n_agents = config.scenario.num_agents
    zeros = np.zeros(n_agents, dtype=np.float32)
    prices = np.full(n_agents, 0.4, dtype=np.float32)
    actions = ActionBatch(
        q_buy=zeros,
        q_sell=zeros,
        price_buy=prices,
        price_sell=prices,
        charge=zeros,
        discharge=zeros,
    )

    package = clear_market(
        scenario,
        config,
        18,
        np.full(n_agents, config.storage.init_soc, dtype=np.float32),
        actions,
    )

    assert package.max_violation() <= config.clearing.network_tolerance
    assert float(package.load_shed.sum() + package.pv_curtailment.sum()) > 0.0
