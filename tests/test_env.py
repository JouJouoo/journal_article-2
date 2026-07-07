from __future__ import annotations

import numpy as np

from tecsf.config import ExperimentConfig, ScenarioConfig
from tecsf.env import (
    P2P_REF_PRICE_OBS_INDEX,
    EnergyCarbonEnv,
)


def test_env_step_shapes_and_finite_rewards():
    config = ExperimentConfig(scenario=ScenarioConfig(num_agents=4, num_nodes=3, horizon=3))
    env = EnergyCarbonEnv(config, variant="tecsf")
    obs, global_state, _ = env.reset(seed=9)
    assert env.observation_dim == 7  # tecsf变体有7维观测
    assert obs.shape == (4, env.observation_dim)
    assert global_state.shape == (env.global_state_dim,)
    assert np.all(obs[:, P2P_REF_PRICE_OBS_INDEX] > 0.0)
    # tecsf变体第7维（索引6）是LCCoins余额，初始为0
    if env.observation_dim == 7:
        assert np.allclose(obs[:, -1], 0.0)
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
    # 标准MAPPO只有6维观测（无LCCoins余额）
    assert env.observation_dim == 6
    assert obs.shape == (4, 6)
    assert result.observation.shape == (4, 6)
    assert result.info["lccoins"] == 0.0


def test_greedy_feasible_action_is_valid():
    config = ExperimentConfig(scenario=ScenarioConfig(num_agents=4, num_nodes=3, horizon=6))
    env = EnergyCarbonEnv(config, variant="tecsf")
    env.reset(seed=11)

    action = env.greedy_feasible_action()

    assert action.shape == (4, env.action_dim)
    assert np.max(action) <= 1.0
    assert np.min(action) >= -1.0


def test_no_chain_variant_disables_blockchain_and_asset_state():
    config = ExperimentConfig(scenario=ScenarioConfig(num_agents=4, num_nodes=3, horizon=3))
    env = EnergyCarbonEnv(config, variant="no_chain")
    obs, _, _ = env.reset(seed=14)
    result = env.step(np.zeros((4, env.action_dim), dtype=np.float32))
    # no_chain变体只有6维观测（无LCCoins余额）
    assert env.observation_dim == 6
    assert obs.shape == (4, 6)
    assert result.observation.shape == (4, 6)
    assert result.info["lccoins"] == 0.0


def test_lc_mappo_coin_reward_uses_crra_stock_and_increment_utility():
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
