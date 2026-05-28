from __future__ import annotations

import numpy as np

from tecsf.config import ExperimentConfig, ScenarioConfig
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


def test_heuristic_action_is_valid():
    config = ExperimentConfig(scenario=ScenarioConfig(num_agents=4, num_nodes=3, horizon=3))
    env = EnergyCarbonEnv(config, variant="heuristic")
    env.reset(seed=10)
    action = env.heuristic_action()
    assert action.shape == (4, env.action_dim)
    assert np.max(action) <= 1.0
    assert np.min(action) >= -1.0
