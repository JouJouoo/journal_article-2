from __future__ import annotations

import numpy as np

from tecsf.config import ExperimentConfig, ScenarioConfig
from tecsf.env import (
    ASSET_BALANCE_OBS_INDEX,
    P2P_REF_PRICE_OBS_INDEX,
    EnergyCarbonEnv,
)


def test_env_step_shapes_and_finite_rewards():
    config = ExperimentConfig(scenario=ScenarioConfig(num_agents=4, num_nodes=3, horizon=3))
    env = EnergyCarbonEnv(config, variant="tecsf")
    obs, global_state, _ = env.reset(seed=9)
    assert env.observation_dim == 7
    assert obs.shape == (4, env.observation_dim)
    assert global_state.shape == (env.global_state_dim,)
    assert np.all(obs[:, P2P_REF_PRICE_OBS_INDEX] > 0.0)
    assert np.allclose(obs[:, ASSET_BALANCE_OBS_INDEX], 0.0)
    result = env.step(np.zeros((4, env.action_dim), dtype=np.float32))
    assert result.observation.shape == (4, env.observation_dim)
    assert result.global_state.shape == (env.global_state_dim,)
    assert result.reward.shape == (4,)
    assert np.all(np.isfinite(result.reward))
    assert "settled" in result.info
    assert "net_carbon_allowance_need" in result.info
    assert "trade_repair_deviation" in result.info
    assert "system_social_cost" in result.info
    assert "participant_payment_cost" in result.info
    assert "action_bound_deviation" in result.info
    assert "agent_reward_eco" in result.info
    assert "agent_reward_coin" in result.info
    assert "agent_utility_stock" in result.info
    assert "agent_utility_increment" in result.info
    assert "agent_lccoins_balance" in result.info
    assert "consensus_confirmed" in result.info
    assert "p2p_matrix" in result.info


def test_standard_mappo_does_not_observe_lccoins_asset_state():
    config = ExperimentConfig(scenario=ScenarioConfig(num_agents=4, num_nodes=3, horizon=3))
    env = EnergyCarbonEnv(config, variant="mappo")
    obs, _, _ = env.reset(seed=9)
    result = env.step(np.zeros((4, env.action_dim), dtype=np.float32))
    assert np.allclose(obs[:, ASSET_BALANCE_OBS_INDEX], 0.0)
    assert np.allclose(result.observation[:, ASSET_BALANCE_OBS_INDEX], 0.0)
    assert result.info["lccoins"] == 0.0


def test_heuristic_action_is_valid():
    config = ExperimentConfig(scenario=ScenarioConfig(num_agents=4, num_nodes=3, horizon=3))
    env = EnergyCarbonEnv(config, variant="heuristic")
    env.reset(seed=10)
    action = env.heuristic_action()
    assert action.shape == (4, env.action_dim)
    assert np.max(action) <= 1.0
    assert np.min(action) >= -1.0


def test_greedy_feasible_action_is_valid():
    config = ExperimentConfig(scenario=ScenarioConfig(num_agents=4, num_nodes=3, horizon=6))
    env = EnergyCarbonEnv(config, variant="greedy_feasible")
    env.reset(seed=11)

    action = env.greedy_feasible_action()

    assert action.shape == (4, env.action_dim)
    assert np.max(action) <= 1.0
    assert np.min(action) >= -1.0


def test_myopic_opt_action_is_valid_and_deterministic_variant_uses_it():
    config = ExperimentConfig(scenario=ScenarioConfig(num_agents=4, num_nodes=3, horizon=6))
    env = EnergyCarbonEnv(config, variant="myopic_opt")
    env.reset(seed=13)

    action = env.myopic_opt_action()

    assert action.shape == (4, env.action_dim)
    assert np.max(action) <= 1.0
    assert np.min(action) >= -1.0
    assert np.allclose(action, env.deterministic_action())


def test_preset_low_carbon_variant_uses_internal_low_carbon_reward():
    config = ExperimentConfig(scenario=ScenarioConfig(num_agents=4, num_nodes=3, horizon=3))
    env = EnergyCarbonEnv(config, variant="preset_low_carbon")
    env.reset(seed=12)
    result = env.step(np.zeros((4, env.action_dim), dtype=np.float32))
    assert result.info["record_reason"] == "TECS-Chain bypassed"
    assert result.info["lccoins"] == 0.0


def test_tecsf_coin_reward_uses_crra_stock_and_increment_utility():
    config = ExperimentConfig(scenario=ScenarioConfig(num_agents=4, num_nodes=3, horizon=3))
    env = EnergyCarbonEnv(config, variant="tecsf")
    env.reset(seed=12)

    result = env.step(np.zeros((env.num_agents, env.action_dim), dtype=np.float32))

    expected = (
        config.lccoins.stock_utility_weight
        * np.asarray(result.info["agent_utility_stock"], dtype=np.float32)
        + config.lccoins.increment_utility_weight
        * np.asarray(result.info["agent_utility_increment"], dtype=np.float32)
    )
    assert np.allclose(result.info["agent_reward_coin"], expected)
