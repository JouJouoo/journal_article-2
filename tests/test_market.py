from __future__ import annotations

import numpy as np

from tecsf.config import ExperimentConfig, ScenarioConfig
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
    assert package.p2p_power.shape == (3, 3)
    assert package.carbon.e_grid.shape == (3,)
    assert np.all(package.carbon.a_buy >= -1e-7)
    assert np.all(package.carbon.c_sell >= -1e-7)
    assert set(package.violations) == {"voltage", "line", "soc", "trade"}


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
