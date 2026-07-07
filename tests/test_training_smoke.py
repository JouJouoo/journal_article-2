from __future__ import annotations

import numpy as np
import torch

from tecsf.config import ExperimentConfig, RLConfig, ScenarioConfig
from tecsf.rl.buffer import RolloutBuffer
from tecsf.rl.mappo import (
    _dual_reward_components,
    _inactive_update_metrics,
    _lr_factor,
    _should_restore_best,
    train,
)
from tecsf.rl.networks import RecurrentGaussianActor


def test_rollout_buffer_uses_pre_action_asset_state():
    from tecsf.variants import get_variant
    buffer = RolloutBuffer()
    # 设置为启用资产观测的变体（tecsf）
    buffer.set_variant(get_variant("tecsf"))
    observation = np.zeros((2, 7), dtype=np.float32)
    observation[:, -1] = np.asarray([0.1, 0.2], dtype=np.float32)
    info = {
        "agent_lccoins_asset_state": [9.0, 9.0],
        "p2p_matrix": [[0.0, 1.0], [0.0, 0.0]],
    }

    buffer.add(
        observation=observation,
        global_state=np.zeros(20, dtype=np.float32),
        hidden_state=np.zeros((2, 4), dtype=np.float32),
        action=np.zeros((2, 6), dtype=np.float32),
        log_prob=np.zeros(2, dtype=np.float32),
        reward=np.ones(2, dtype=np.float32),
        done=False,
        value=np.zeros(2, dtype=np.float32),
        info=info,
    )

    arrays = buffer.arrays()
    assert np.allclose(arrays["asset_states"][0], [0.1, 0.2])


def test_actor_log_std_is_clamped():
    actor = RecurrentGaussianActor(
        obs_dim=4,
        action_dim=2,
        hidden_dim=8,
        recurrent_dim=8,
        log_std_init=1.0,
        log_std_min=-2.0,
        log_std_max=-1.0,
    )
    obs = torch.zeros(3, 4)
    hidden = actor.initial_hidden(3, torch.device("cpu"))

    dist, _ = actor(obs, hidden)

    assert torch.allclose(
        torch.log(dist.stddev), torch.full((3, 2), -1.0)
    )


def test_dual_reward_components_keep_penalties_in_eco_advantage():
    data = {
        "per_agent_reward_eco": np.asarray([[10.0, 8.0]], dtype=np.float32),
        "per_agent_reward_coin": np.asarray([[1.5, 2.0]], dtype=np.float32),
        "per_agent_rewards": np.asarray([[9.0, 7.0]], dtype=np.float32),
    }

    reward_eco, reward_coin = _dual_reward_components(data)

    assert np.allclose(reward_eco, [[7.5, 5.0]])
    assert np.allclose(reward_coin, [[1.5, 2.0]])


def test_lr_factor_decays_to_floor():
    config = ExperimentConfig(
        rl=RLConfig(lr_decay_episodes=10, min_lr_factor=0.2)
    )

    assert _lr_factor(config, 0) == 1.0
    assert np.isclose(_lr_factor(config, 5), 0.6)
    assert np.isclose(_lr_factor(config, 20), 0.2)


def test_should_restore_best_after_plateau_patience():
    config = ExperimentConfig(
        rl=RLConfig(
            early_stop_warmup_episodes=10,
            early_stop_patience=5,
            restore_best_on_plateau=True,
        )
    )

    assert not _should_restore_best(config, 9, 0, False)
    assert not _should_restore_best(config, 14, 10, False)
    assert _should_restore_best(config, 15, 10, False)
    assert not _should_restore_best(config, 15, 10, True)


def test_inactive_update_metrics_has_stable_keys():
    metrics = _inactive_update_metrics()

    assert metrics["policy_update_accepted"] == 0.0
    assert "approx_kl" in metrics


def test_training_smoke(tmp_path):
    config = ExperimentConfig(
        scenario=ScenarioConfig(num_agents=2, num_nodes=3, horizon=3, seed=11),
        rl=RLConfig(
            hidden_dim=16,
            recurrent_dim=16,
            update_epochs=1,
            episodes=1,
            seed=11,
            device="cpu",
        ),
    )
    for variant in ["tecsf", "lc_mappo"]:
        result = train(
            config=config,
            variant=variant,
            output_dir=tmp_path / variant,
            episodes=1,
        )
        checkpoint = torch.load(result.checkpoint_path, map_location="cpu", weights_only=True)
        assert result.episode_metrics
        assert result.checkpoint_path.endswith(".pt")
        assert result.best_checkpoint_path.endswith("_best_checkpoint.pt")
        assert checkpoint["variant"] == variant
        assert "critic_state_dict" in checkpoint
        assert "credit_assignment_state_dict" in checkpoint
        assert "advantage_gate_state_dict" in checkpoint
        assert "clip_gate_state_dict" in checkpoint
        assert "best_score" in checkpoint
