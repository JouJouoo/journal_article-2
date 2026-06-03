from __future__ import annotations

import numpy as np

from tecsf.device import assign_parallel_device, resolve_device
from tecsf.metrics import summarize_episode


def test_summarize_episode_exports_extended_metrics():
    infos = [
        {
            "settled": True,
            "system_cost": 2.0,
            "system_social_cost": 2.0,
            "participant_payment_cost": 2.5,
            "p2p_transfer_payment": 0.5,
            "carbon_emission": 1.0,
            "grid_carbon_emission": 1.0,
            "emergency_cost": 0.9,
            "load_shed": 0.1,
            "pv_curtailment": 0.2,
            "net_carbon_allowance_need": 0.7,
            "allowance_buy": 0.7,
            "low_carbon_sell": 0.1,
            "carbon_reduction": 0.2,
            "lccoins": 0.5,
            "p2p_energy": 1.2,
            "attempted_p2p_energy": 1.5,
            "pv_generation": 2.0,
            "pv_used": 1.0,
            "trade_repair_deviation": 0.3,
            "storage_repair_deviation": 0.4,
            "action_bound_deviation": 0.6,
            "action_bound_penalty": 0.03,
            "safety_adjustment": 0.4,
            "repair_iterations": 2.0,
            "lccoins_reward_raw": 0.2,
            "lccoins_reward_clipped": 0.1,
            "lccoins_reward_weight": 0.2,
            "clear_seconds": 0.01,
            "settlement_seconds": 0.02,
            "reward_seconds": 0.03,
            "feasible": True,
            "record_state": "Settled",
            "record_reason": "",
            "max_violation": 0.0,
            "violations": {"voltage": 0.0, "line": 0.0, "soc": 0.0, "trade": 0.0},
            "agent_profit": [1.0, 2.0],
            "agent_lccoins": [0.2, 0.3],
            "agent_q_lc": [1.0, 2.0],
            "agent_c_offset": [0.1, 0.2],
            "agent_c_sell": [0.05, 0.1],
            "agent_lccoins_candidate": [0.2, 0.3],
            "agent_carbon_emission": [0.6, 0.4],
        },
        {
            "settled": False,
            "system_cost": 0.0,
            "carbon_emission": 0.0,
            "p2p_energy": 0.0,
            "attempted_p2p_energy": 0.0,
            "pv_generation": 0.0,
            "pv_used": 0.0,
            "feasible": False,
            "record_state": "Rejected",
            "record_reason": "constraint violation exceeds tolerance",
            "max_violation": 1.0,
            "violations": {"voltage": 1.0, "line": 0.0, "soc": 0.0, "trade": 0.0},
        }
    ]
    rewards = [
        np.asarray([1.0, 2.0], dtype=np.float32),
        np.asarray([-1.0, -1.0], dtype=np.float32),
    ]
    summary = summarize_episode(infos, rewards)
    assert summary["renewable_consumption_rate"] == 0.5
    assert summary["q_lc"] == 1.0
    assert summary["pv_used"] == 1.0
    assert np.isclose(summary["trade_repair_rate"], 0.2)
    assert summary["feasible_rate"] == 0.5
    assert summary["rejected_records"] == 1.0
    assert summary["constraint_rejection_records"] == 1.0
    assert summary["hash_rejection_records"] == 0.0
    assert summary["execution_revert_records"] == 0.0
    assert summary["storage_repair_deviation"] == 0.4
    assert summary["system_social_cost"] == 2.0
    assert summary["participant_payment_cost"] == 2.5
    assert summary["p2p_transfer_payment"] == 0.5
    assert summary["action_bound_deviation"] == 0.6
    assert summary["action_bound_penalty"] == 0.03
    assert summary["emergency_cost"] == 0.9
    assert summary["load_shed"] == 0.1
    assert summary["pv_curtailment"] == 0.2
    assert summary["safety_adjustment"] == 0.4
    assert summary["repair_iterations"] == 2.0
    assert summary["lccoins_reward_weight_mean"] == 0.1
    assert summary["clear_seconds"] == 0.01
    assert summary["settlement_seconds"] == 0.02
    assert summary["reward_seconds"] == 0.03
    assert summary["agent_profit_mean"] == 1.5
    assert "agent_profit_gini" in summary
    assert summary["agent_q_lc"] == [1.0, 2.0]
    assert summary["agent_c_offset"] == [0.1, 0.2]
    assert summary["agent_c_sell"] == [0.05, 0.1]
    assert summary["agent_lccoins"] == [0.2, 0.3]
    assert summary["lccoins_candidate"] == 0.5
    assert summary["carbon_offset"] == 0.30000000000000004
    assert np.isclose(summary["agent_lccoins_q_lc_corr"], 1.0)
    assert np.isclose(summary["agent_lccoins_c_offset_corr"], 1.0)
    assert np.isclose(summary["agent_lccoins_candidate_corr"], 1.0)


def test_device_helpers_cpu_and_auto_assignment():
    assert str(resolve_device("cpu")) == "cpu"
    assigned = assign_parallel_device("auto", worker_index=0)
    assert assigned == "cpu" or assigned.startswith("cuda:")
