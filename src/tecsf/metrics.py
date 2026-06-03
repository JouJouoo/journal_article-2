from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def gini(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return 0.0
    min_value = float(arr.min())
    if min_value < 0.0:
        arr = arr - min_value
    total = float(arr.sum())
    if total <= 1e-12:
        return 0.0
    sorted_arr = np.sort(arr)
    n = sorted_arr.size
    weights = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * np.sum(weights * sorted_arr)) / (n * total) - (n + 1) / n)


def jain_index(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    denom = arr.size * float(np.sum(arr * arr))
    if denom <= 1e-12:
        return 0.0
    return float(float(arr.sum()) ** 2 / denom)


def correlation(values_a: np.ndarray, values_b: np.ndarray) -> float:
    a = np.asarray(values_a, dtype=np.float64).reshape(-1)
    b = np.asarray(values_b, dtype=np.float64).reshape(-1)
    if a.size != b.size or a.size < 2:
        return 0.0
    a_std = float(a.std())
    b_std = float(b.std())
    if a_std <= 1e-12 or b_std <= 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _accumulate_agent_vector(
    current: np.ndarray | None,
    info: dict,
    key: str,
) -> np.ndarray | None:
    if key not in info:
        return current
    values = np.asarray(info[key], dtype=np.float64)
    return values if current is None else current + values


def summarize_episode(infos: list[dict], rewards: list[np.ndarray]) -> dict[str, Any]:
    if not infos:
        return {}
    reward_arr = np.asarray(rewards, dtype=np.float32)
    total_reward = float(reward_arr.sum())
    mean_reward = float(reward_arr.mean())
    total_cost = 0.0
    total_emission = 0.0
    total_lccoins = 0.0
    total_p2p = 0.0
    total_attempted_p2p = 0.0
    total_grid_buy_cost = 0.0
    total_social_cost = 0.0
    total_participant_payment_cost = 0.0
    total_p2p_transfer_payment = 0.0
    total_action_bound_deviation = 0.0
    total_action_bound_penalty = 0.0
    total_emergency_cost = 0.0
    total_load_shed = 0.0
    total_pv_curtailment = 0.0
    total_grid_carbon = 0.0
    total_pv_credit = 0.0
    total_allowance_need = 0.0
    total_allowance_buy = 0.0
    total_low_carbon_sell = 0.0
    total_carbon_reduction = 0.0
    total_pv_generation = 0.0
    total_pv_used = 0.0
    total_repair_deviation = 0.0
    total_storage_repair_deviation = 0.0
    total_safety_adjustment = 0.0
    total_repair_iterations = 0.0
    total_lccoins_reward_raw = 0.0
    total_lccoins_reward_clipped = 0.0
    total_lccoins_reward_weight = 0.0
    total_clear_seconds = 0.0
    total_settlement_seconds = 0.0
    total_reward_seconds = 0.0
    feasible = 0
    violation_totals = {"voltage": 0.0, "line": 0.0, "soc": 0.0, "trade": 0.0}
    settled = 0
    rejected = 0
    constraint_rejections = 0
    hash_rejections = 0
    execution_reverts = 0
    max_violation = 0.0
    agent_profit = None
    agent_lccoins = None
    agent_carbon = None
    agent_q_lc = None
    agent_c_offset = None
    agent_c_sell = None
    agent_lccoins_candidate = None
    for info in infos:
        total_cost += float(info.get("system_cost", 0.0))
        total_social_cost += float(info.get("system_social_cost", info.get("system_cost", 0.0)))
        total_participant_payment_cost += float(info.get("participant_payment_cost", 0.0))
        total_p2p_transfer_payment += float(info.get("p2p_transfer_payment", 0.0))
        total_action_bound_deviation += float(info.get("action_bound_deviation", 0.0))
        total_action_bound_penalty += float(info.get("action_bound_penalty", 0.0))
        total_emission += float(info.get("carbon_emission", 0.0))
        total_lccoins += float(info.get("lccoins", 0.0))
        total_p2p += float(info.get("p2p_energy", 0.0))
        total_attempted_p2p += float(info.get("attempted_p2p_energy", 0.0))
        total_grid_buy_cost += float(info.get("grid_buy_cost", 0.0))
        total_emergency_cost += float(info.get("emergency_cost", 0.0))
        total_load_shed += float(info.get("load_shed", 0.0))
        total_pv_curtailment += float(info.get("pv_curtailment", 0.0))
        total_grid_carbon += float(info.get("grid_carbon_emission", 0.0))
        total_pv_credit += float(info.get("pv_credit", 0.0))
        total_allowance_need += float(info.get("net_carbon_allowance_need", 0.0))
        total_allowance_buy += float(info.get("allowance_buy", 0.0))
        total_low_carbon_sell += float(info.get("low_carbon_sell", 0.0))
        total_carbon_reduction += float(info.get("carbon_reduction", 0.0))
        total_pv_generation += float(info.get("pv_generation", 0.0))
        total_pv_used += float(info.get("pv_used", 0.0))
        total_repair_deviation += float(info.get("trade_repair_deviation", 0.0))
        total_storage_repair_deviation += float(
            info.get("storage_repair_deviation", 0.0)
        )
        total_safety_adjustment += float(info.get("safety_adjustment", 0.0))
        total_repair_iterations += float(info.get("repair_iterations", 0.0))
        total_lccoins_reward_raw += float(info.get("lccoins_reward_raw", 0.0))
        total_lccoins_reward_clipped += float(
            info.get("lccoins_reward_clipped", 0.0)
        )
        total_lccoins_reward_weight += float(
            info.get("lccoins_reward_weight", 0.0)
        )
        total_clear_seconds += float(info.get("clear_seconds", 0.0))
        total_settlement_seconds += float(info.get("settlement_seconds", 0.0))
        total_reward_seconds += float(info.get("reward_seconds", 0.0))
        settled += int(info.get("settled", False))
        feasible += int(info.get("feasible", False))
        rejected += int(not info.get("settled", False))
        reason = str(info.get("record_reason", "")).lower()
        state = str(info.get("record_state", "")).lower()
        constraint_rejections += int("constraint" in reason)
        hash_rejections += int("hash" in reason)
        execution_reverts += int(state == "reverted")
        max_violation = max(max_violation, float(info.get("max_violation", 0.0)))
        for key in violation_totals:
            violation_totals[key] += float(info.get("violations", {}).get(key, 0.0))
        agent_profit = _accumulate_agent_vector(agent_profit, info, "agent_profit")
        agent_lccoins = _accumulate_agent_vector(agent_lccoins, info, "agent_lccoins")
        agent_carbon = _accumulate_agent_vector(agent_carbon, info, "agent_carbon_emission")
        agent_q_lc = _accumulate_agent_vector(agent_q_lc, info, "agent_q_lc")
        agent_c_offset = _accumulate_agent_vector(agent_c_offset, info, "agent_c_offset")
        agent_c_sell = _accumulate_agent_vector(agent_c_sell, info, "agent_c_sell")
        agent_lccoins_candidate = _accumulate_agent_vector(
            agent_lccoins_candidate, info, "agent_lccoins_candidate"
        )
    renewable_consumption_rate = total_pv_used / max(total_pv_generation, 1e-12)
    repair_rate = total_repair_deviation / max(total_attempted_p2p, 1e-12)
    agent_profit_arr = (
        np.zeros(reward_arr.shape[-1], dtype=np.float64)
        if agent_profit is None
        else np.asarray(agent_profit, dtype=np.float64)
    )
    agent_lccoins_arr = (
        np.zeros_like(agent_profit_arr)
        if agent_lccoins is None
        else np.asarray(agent_lccoins, dtype=np.float64)
    )
    agent_carbon_arr = (
        np.zeros_like(agent_profit_arr)
        if agent_carbon is None
        else np.asarray(agent_carbon, dtype=np.float64)
    )
    agent_q_lc_arr = (
        np.zeros_like(agent_profit_arr)
        if agent_q_lc is None
        else np.asarray(agent_q_lc, dtype=np.float64)
    )
    agent_c_offset_arr = (
        np.zeros_like(agent_profit_arr)
        if agent_c_offset is None
        else np.asarray(agent_c_offset, dtype=np.float64)
    )
    agent_c_sell_arr = (
        np.zeros_like(agent_profit_arr)
        if agent_c_sell is None
        else np.asarray(agent_c_sell, dtype=np.float64)
    )
    agent_lccoins_candidate_arr = (
        np.zeros_like(agent_profit_arr)
        if agent_lccoins_candidate is None
        else np.asarray(agent_lccoins_candidate, dtype=np.float64)
    )
    return {
        "total_reward": total_reward,
        "mean_reward": mean_reward,
        "system_cost": total_cost,
        "system_social_cost": total_social_cost,
        "participant_payment_cost": total_participant_payment_cost,
        "p2p_transfer_payment": total_p2p_transfer_payment,
        "carbon_emission": total_emission,
        "grid_carbon_emission": total_grid_carbon,
        "pv_credit": total_pv_credit,
        "net_carbon_allowance_need": total_allowance_need,
        "allowance_buy": total_allowance_buy,
        "low_carbon_sell": total_low_carbon_sell,
        "q_lc": total_pv_used,
        "pv_used": total_pv_used,
        "carbon_offset": float(agent_c_offset_arr.sum()),
        "carbon_reduction": total_carbon_reduction,
        "lccoins": total_lccoins,
        "lccoins_candidate": float(agent_lccoins_candidate_arr.sum()),
        "p2p_energy": total_p2p,
        "attempted_p2p_energy": total_attempted_p2p,
        "grid_buy_cost": total_grid_buy_cost,
        "emergency_cost": total_emergency_cost,
        "load_shed": total_load_shed,
        "pv_curtailment": total_pv_curtailment,
        "renewable_consumption_rate": renewable_consumption_rate,
        "trade_repair_deviation": total_repair_deviation,
        "storage_repair_deviation": total_storage_repair_deviation,
        "action_bound_deviation": total_action_bound_deviation,
        "action_bound_penalty": total_action_bound_penalty,
        "safety_adjustment": total_safety_adjustment,
        "repair_iterations": total_repair_iterations,
        "trade_repair_rate": repair_rate,
        "settlement_success_rate": settled / max(len(infos), 1),
        "feasible_rate": feasible / max(len(infos), 1),
        "rejected_records": float(rejected),
        "constraint_rejection_records": float(constraint_rejections),
        "hash_rejection_records": float(hash_rejections),
        "execution_revert_records": float(execution_reverts),
        "lccoins_reward_raw": total_lccoins_reward_raw,
        "lccoins_reward_clipped": total_lccoins_reward_clipped,
        "lccoins_reward_weight_mean": total_lccoins_reward_weight / max(len(infos), 1),
        "clear_seconds": total_clear_seconds,
        "settlement_seconds": total_settlement_seconds,
        "reward_seconds": total_reward_seconds,
        "max_violation": max_violation,
        "voltage_violation": violation_totals["voltage"],
        "line_violation": violation_totals["line"],
        "soc_violation": violation_totals["soc"],
        "trade_violation": violation_totals["trade"],
        "agent_profit_mean": float(agent_profit_arr.mean()) if agent_profit_arr.size else 0.0,
        "agent_profit_std": float(agent_profit_arr.std()) if agent_profit_arr.size else 0.0,
        "agent_profit_gini": gini(agent_profit_arr),
        "agent_profit_jain": jain_index(agent_profit_arr),
        "agent_lccoins_gini": gini(agent_lccoins_arr),
        "agent_carbon_gini": gini(agent_carbon_arr),
        "agent_q_lc": agent_q_lc_arr.astype(float).tolist(),
        "agent_c_offset": agent_c_offset_arr.astype(float).tolist(),
        "agent_c_sell": agent_c_sell_arr.astype(float).tolist(),
        "agent_lccoins": agent_lccoins_arr.astype(float).tolist(),
        "agent_lccoins_candidate": agent_lccoins_candidate_arr.astype(float).tolist(),
        "agent_lccoins_q_lc_corr": correlation(agent_lccoins_arr, agent_q_lc_arr),
        "agent_lccoins_c_offset_corr": correlation(agent_lccoins_arr, agent_c_offset_arr),
        "agent_lccoins_candidate_corr": correlation(
            agent_lccoins_arr, agent_lccoins_candidate_arr
        ),
    }


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)
