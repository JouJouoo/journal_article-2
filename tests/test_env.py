from __future__ import annotations

import numpy as np

from tecsf.config import ExperimentConfig, NetworkConfig, ScenarioConfig
from tecsf.env import EnergyCarbonEnv


def test_env_step_shapes_and_finite_rewards():
    config = ExperimentConfig(scenario=ScenarioConfig(num_agents=4, num_nodes=3, horizon=3))
    env = EnergyCarbonEnv(config, variant="tecsf")
    obs, global_state, _ = env.reset(seed=9)
    assert obs.shape == (4, env.observation_dim)
    assert global_state.shape == (env.global_state_dim,)
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


def test_lccoins_reward_is_clipped_and_adaptive_weight_reacts_to_risk():
    config = ExperimentConfig(
        scenario=ScenarioConfig(num_agents=8, num_nodes=5, horizon=24),
        network=NetworkConfig(default_line_capacity=0.5),
    )
    config.clearing.enable_safety_shield = False
    config.lccoins.kappa = 0.4
    config.lccoins.kappa_max = 0.4
    config.lccoins.reward_clip = 0.01
    env = EnergyCarbonEnv(config, variant="tecsf")
    env.reset(seed=7)
    action = np.ones((env.num_agents, env.action_dim), dtype=np.float32)

    result = env.step(action)

    assert abs(result.info["lccoins_reward_clipped"]) <= env.num_agents * 0.01 + 1e-6
    assert env.lccoins_reward_weight <= config.lccoins.kappa
    assert "feasible" in result.info
