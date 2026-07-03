from __future__ import annotations

import numpy as np
import torch

from tecsf.config import ExperimentConfig, RLConfig, ScenarioConfig
from tecsf.rl.buffer import RolloutBuffer
from tecsf.rl.mappo import train


def test_rollout_buffer_uses_pre_action_asset_state():
    buffer = RolloutBuffer()
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
        assert checkpoint["variant"] == variant
        assert "critic_state_dict" in checkpoint
        assert "credit_assignment_state_dict" in checkpoint
        assert "advantage_gate_state_dict" in checkpoint
        assert "clip_gate_state_dict" in checkpoint
