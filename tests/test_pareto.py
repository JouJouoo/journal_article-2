from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_pareto_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "analyze_pareto_front.py"
    spec = importlib.util.spec_from_file_location("analyze_pareto_front", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_pareto_filters_infeasible_runs_before_dominance():
    pareto = _load_pareto_module()
    summary = {
        "runs": [
            {
                "label": "formal",
                "variant": "safe",
                "seed": 1,
                "eval_settlement_success_rate": 1.0,
                "eval_max_violation": 0.0,
                "eval_system_cost": 10.0,
                "eval_grid_carbon_emission": 5.0,
                "eval_mean_reward": 1.0,
                "eval_feasible_rate": 1.0,
            },
            {
                "label": "formal",
                "variant": "unsafe_low_cost",
                "seed": 1,
                "eval_settlement_success_rate": 0.5,
                "eval_max_violation": 2.0,
                "eval_system_cost": 1.0,
                "eval_grid_carbon_emission": 1.0,
                "eval_mean_reward": 2.0,
                "eval_feasible_rate": 0.5,
            },
        ]
    }

    rows = pareto.analyze(summary, pareto.DEFAULT_OBJECTIVES, 0.95, 0.1)

    by_variant = {row["variant"]: row for row in rows}
    assert by_variant["safe"]["passes_feasibility_gate"]
    assert by_variant["safe"]["pareto_efficient"]
    assert not by_variant["unsafe_low_cost"]["passes_feasibility_gate"]
    assert not by_variant["unsafe_low_cost"]["pareto_efficient"]
