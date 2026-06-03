from __future__ import annotations

import numpy as np

from tecsf.benchmark_cases import BENCHMARK_CASES, is_radial
from tecsf.config import ExperimentConfig, ScenarioConfig
from tecsf.data import generate_synthetic_scenario
from scripts.create_benchmark_profile import build_profile


def test_npz_profile_scenario_loader(tmp_path):
    profile = tmp_path / "scenario_profile.npz"
    load = np.ones((3, 4), dtype=np.float32)
    pv = np.full((3, 4), 0.5, dtype=np.float32)
    np.savez(
        profile,
        load=load,
        pv=pv,
        num_nodes=np.asarray(3),
        agent_nodes=np.asarray([1, 1, 2], dtype=np.int64),
        line_from=np.asarray([0, 1], dtype=np.int64),
        line_to=np.asarray([1, 2], dtype=np.int64),
        line_capacity=np.asarray([5.0, 4.0], dtype=np.float32),
    )
    config = ExperimentConfig(
        scenario=ScenarioConfig(profile_path=str(profile), load_scale=2.0, pv_scale=0.5)
    )

    scenario = generate_synthetic_scenario(config)

    assert scenario.num_agents == 3
    assert scenario.num_nodes == 3
    assert scenario.horizon == 4
    assert np.allclose(scenario.load, 2.0)
    assert np.allclose(scenario.pv, 0.25)
    assert np.array_equal(scenario.agent_nodes, np.asarray([1, 1, 2]))


def test_npz_profile_loader_rejects_bad_agent_nodes(tmp_path):
    profile = tmp_path / "bad_profile.npz"
    np.savez(
        profile,
        load=np.ones((2, 3), dtype=np.float32),
        pv=np.zeros((2, 3), dtype=np.float32),
        num_nodes=np.asarray(3),
        agent_nodes=np.asarray([1, 5], dtype=np.int64),
    )
    config = ExperimentConfig(scenario=ScenarioConfig(profile_path=str(profile)))

    try:
        generate_synthetic_scenario(config)
    except ValueError as exc:
        assert "agent_nodes" in str(exc)
    else:
        raise AssertionError("invalid profile should have failed")


def test_benchmark_profile_can_be_loaded(tmp_path):
    profile = tmp_path / "ieee33_profile.npz"
    np.savez(profile, **build_profile(agents=6, horizon=8, seed=3, day_type="cloudy"))
    config = ExperimentConfig(scenario=ScenarioConfig(profile_path=str(profile)))

    scenario = generate_synthetic_scenario(config)

    assert scenario.num_agents == 6
    assert scenario.num_nodes == 33
    assert scenario.horizon == 8
    assert scenario.line_from.shape == scenario.line_to.shape


def test_standard_benchmark_cases_have_expected_totals_and_radial_topology():
    ieee33 = BENCHMARK_CASES["ieee33bw"]
    ieee69 = BENCHMARK_CASES["ieee69"]

    assert ieee33.num_buses == 33
    assert ieee33.num_branches == 32
    assert np.isclose(ieee33.total_load_p_mw, 3.715)
    assert np.isclose(ieee33.total_load_q_mvar, 2.3)
    assert is_radial(ieee33)

    assert ieee69.num_buses == 69
    assert ieee69.num_branches == 68
    assert np.isclose(ieee69.total_load_p_mw, 3.8021, atol=1e-3)
    assert np.isclose(ieee69.total_load_q_mvar, 2.6947, atol=1e-3)
    assert is_radial(ieee69)


def test_standard_benchmark_profile_metadata_and_loader(tmp_path):
    for case_name, expected_nodes in [("ieee33bw", 33), ("ieee69", 69)]:
        profile = tmp_path / f"{case_name}.npz"
        np.savez(
            profile,
            **build_profile(
                agents=None,
                horizon=6,
                seed=4,
                day_type="weekday",
                case=case_name,
            ),
        )
        config = ExperimentConfig(scenario=ScenarioConfig(profile_path=str(profile)))

        scenario = generate_synthetic_scenario(config)
        with np.load(profile) as data:
            assert str(np.asarray(data["case_name"]).item()) == case_name
            assert float(np.asarray(data["benchmark_total_p_mw"]).item()) > 0.0

        assert scenario.num_nodes == expected_nodes
        assert scenario.horizon == 6
        assert scenario.line_from.shape[0] == expected_nodes - 1
