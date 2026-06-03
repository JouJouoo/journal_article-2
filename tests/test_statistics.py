from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_stats_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "analyze_experiment_statistics.py"
    spec = importlib.util.spec_from_file_location("analyze_experiment_statistics", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_paired_statistics_marks_direction_and_pairs_by_seed():
    stats = _load_stats_module()
    summary = {
        "runs": [
            {"label": "formal", "variant": "tecsf", "seed": 1, "eval_mean_reward": 1.0},
            {"label": "formal", "variant": "tecsf", "seed": 2, "eval_mean_reward": 1.0},
            {"label": "formal", "variant": "baseline", "seed": 1, "eval_mean_reward": 0.5},
            {"label": "formal", "variant": "baseline", "seed": 2, "eval_mean_reward": 0.4},
        ]
    }

    rows = stats.analyze(summary, baseline="baseline", metrics=["eval_mean_reward"])

    assert len(rows) == 1
    assert rows[0]["n_pairs"] == 2
    assert rows[0]["direction"] == "higher"
    assert rows[0]["mean_diff_variant_minus_baseline"] > 0.0
    assert "p_holm" in rows[0]
