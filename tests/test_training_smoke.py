from __future__ import annotations

from tecsf.config import ExperimentConfig, RLConfig, ScenarioConfig
from tecsf.rl.mappo import train


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
    result = train(config=config, variant="tecsf", output_dir=tmp_path, episodes=1)
    assert result.episode_metrics
    assert result.checkpoint_path.endswith(".pt")
