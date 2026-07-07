from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tecsf.benchmark_cases import BENCHMARK_CASES, BenchmarkCase, get_benchmark_case
from tecsf.benchmark_sources import (
    BenchmarkSourceUnavailable,
    compare_benchmark_cases,
    failed_checks,
    load_authoritative_benchmark_case,
)


SYNTHETIC33_BRANCHES = np.asarray(
    [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        (8, 9),
        (9, 10),
        (10, 11),
        (11, 12),
        (12, 13),
        (13, 14),
        (14, 15),
        (15, 16),
        (16, 17),
        (1, 18),
        (18, 19),
        (19, 20),
        (20, 21),
        (2, 22),
        (22, 23),
        (23, 24),
        (5, 25),
        (25, 26),
        (26, 27),
        (27, 28),
        (28, 29),
        (29, 30),
        (30, 31),
        (31, 32),
    ],
    dtype=np.int64,
)


def _daily_profile(horizon: int, peak_hour: float, width: float) -> np.ndarray:
    hours = np.arange(horizon, dtype=np.float32) * 24.0 / max(horizon, 1)
    distance = np.minimum(np.abs(hours - peak_hour), 24.0 - np.abs(hours - peak_hour))
    return np.exp(-0.5 * (distance / width) ** 2).astype(np.float32)


def _time_series_shapes(horizon: int, day_type: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    morning = _daily_profile(horizon, peak_hour=8.0, width=3.2)
    evening = _daily_profile(horizon, peak_hour=19.0, width=4.0)
    midday = np.maximum(0.0, np.sin(np.pi * np.arange(horizon) / max(horizon - 1, 1))).astype(
        np.float32
    )
    weekend_scale = 0.86 if day_type == "weekend" else 1.0
    cloud_scale = 0.45 if day_type == "cloudy" else 1.0
    load_shape = weekend_scale * (0.55 + 0.25 * morning + 0.55 * evening)
    load_shape = load_shape / max(float(load_shape.mean()), 1e-8)
    pv_shape = cloud_scale * midday
    price_shape = 0.82 + 0.28 * evening + 0.10 * morning
    return load_shape.astype(np.float32), pv_shape.astype(np.float32), price_shape.astype(np.float32)


def _line_capacity_mw(case: BenchmarkCase) -> np.ndarray:
    return (
        np.sqrt(3.0)
        * float(case.base_kv)
        * (case.line_current_limit_a.astype(np.float32) / 1000.0)
    ).astype(np.float32)


def _model_resistance(case: BenchmarkCase) -> tuple[np.ndarray, np.ndarray]:
    z_base_ohm = float(case.base_kv) ** 2 / float(case.base_mva)
    scale = z_base_ohm * float(case.base_mva) * 100.0
    return (
        (case.resistance_ohm / scale).astype(np.float32),
        (case.reactance_ohm / scale).astype(np.float32),
    )


def _resolve_standard_case(
    case_name: str,
    standard_source: str,
    case69_m: str | Path | None,
) -> tuple[BenchmarkCase, str, str]:
    locked = get_benchmark_case(case_name)
    if standard_source == "locked":
        return (
            locked,
            "locked_standard_profile",
            "Locked benchmark table; validate against authoritative sources with "
            "scripts/validate_benchmark_sources.py before paper-grade runs.",
        )

    try:
        source_case = load_authoritative_benchmark_case(case_name, case69_m=case69_m)
    except BenchmarkSourceUnavailable as exc:
        if standard_source == "auto":
            return (
                locked,
                "locked_after_authoritative_unavailable",
                f"Authoritative source unavailable locally: {exc}",
            )
        raise

    checks = compare_benchmark_cases(locked, source_case)
    problems = failed_checks(checks)
    if problems:
        detail = "; ".join(
            f"{check.name} expected {check.expected} observed {check.observed}"
            for check in problems[:3]
        )
        raise ValueError(f"{case_name} authoritative source does not match locked table: {detail}")
    return (
        source_case,
        "authoritative_source_validated",
        "Authoritative pandapower/MATPOWER source loaded locally and matched the locked "
        "benchmark topology, load, and impedance tables.",
    )


def build_standard_profile(
    case_name: str,
    agents: int | None,
    horizon: int,
    seed: int,
    day_type: str,
    pv_penetration: float,
    standard_source: str = "locked",
    case69_m: str | Path | None = None,
) -> dict[str, np.ndarray]:
    case, source_mode, source_validation = _resolve_standard_case(
        case_name, standard_source, case69_m
    )
    rng = np.random.default_rng(seed)
    load_shape, pv_shape, price_shape = _time_series_shapes(horizon, day_type)
    load_nodes = np.arange(1, case.num_buses, dtype=np.int64)
    agent_count = int(agents) if agents is not None else int(load_nodes.shape[0])
    if agent_count < 1:
        raise ValueError("--agents must be positive")
    agent_nodes = load_nodes[np.arange(agent_count) % load_nodes.shape[0]]
    base_load = case.bus_load_p_mw[agent_nodes]
    load_noise = rng.normal(0.0, 0.015, size=(agent_count, horizon)).astype(np.float32)
    load = np.maximum(
        0.0,
        base_load.reshape(agent_count, 1) * load_shape.reshape(1, horizon) + load_noise,
    ).astype(np.float32)
    pv_capacity = np.maximum(base_load * float(pv_penetration), 0.02)
    pv_jitter = rng.uniform(0.85, 1.15, size=(agent_count, 1)).astype(np.float32)
    pv_noise = rng.normal(0.0, 0.01, size=(agent_count, horizon)).astype(np.float32)
    pv = np.maximum(
        0.0,
        pv_capacity.reshape(agent_count, 1) * pv_jitter * pv_shape.reshape(1, horizon) + pv_noise,
    ).astype(np.float32)
    resistance, reactance = _model_resistance(case)
    return {
        "load": load,
        "pv": pv,
        "num_nodes": np.asarray(case.num_buses, dtype=np.int64),
        "agent_nodes": agent_nodes.astype(np.int64),
        "line_from": case.line_from.astype(np.int64),
        "line_to": case.line_to.astype(np.int64),
        "resistance": resistance,
        "reactance": reactance,
        "raw_resistance_ohm": case.resistance_ohm.astype(np.float32),
        "raw_reactance_ohm": case.reactance_ohm.astype(np.float32),
        "line_capacity": _line_capacity_mw(case),
        "line_current_limit_a": case.line_current_limit_a.astype(np.float32),
        "bus_load_p_mw": case.bus_load_p_mw.astype(np.float32),
        "bus_load_q_mvar": case.bus_load_q_mvar.astype(np.float32),
        "grid_buy_price": (0.82 * price_shape).astype(np.float32),
        "grid_sell_price": (0.35 * (0.92 + 0.10 * pv_shape)).astype(np.float32),
        "carbon_allowance_price": np.full(horizon, 0.06, dtype=np.float32),
        "low_carbon_sell_price": np.full(horizon, 0.03, dtype=np.float32),
        "grid_emission_factor": (0.58 * (1.0 + 0.15 * price_shape - 0.10 * pv_shape)).astype(
            np.float32
        ),
        "case_name": np.asarray(case.name),
        "case_display_name": np.asarray(case.display_name),
        "case_source": np.asarray(case.source),
        "case_source_url": np.asarray(case.source_url),
        "case_notes": np.asarray(case.notes),
        "benchmark_source_mode": np.asarray(source_mode),
        "benchmark_source_validation": np.asarray(source_validation),
        "benchmark_source_policy": np.asarray(
            "Hybrid source workflow: standard IEEE data are loaded or validated from "
            "pandapower/MATPOWER when available, then frozen into NPZ profiles for "
            "reproducible RL training and evaluation."
        ),
        "model_parameter_notes": np.asarray(
            "Raw branch ohm values are preserved; resistance/reactance fields are "
            "scaled equivalents for the linear radial voltage approximation."
        ),
        "base_kv": np.asarray(case.base_kv, dtype=np.float32),
        "base_mva": np.asarray(case.base_mva, dtype=np.float32),
        "benchmark_total_p_mw": np.asarray(case.total_load_p_mw, dtype=np.float32),
        "benchmark_total_q_mvar": np.asarray(case.total_load_q_mvar, dtype=np.float32),
        "benchmark_num_buses": np.asarray(case.num_buses, dtype=np.int64),
        "benchmark_num_branches": np.asarray(case.num_branches, dtype=np.int64),
    }


def build_synthetic33_profile(
    agents: int | None,
    horizon: int,
    seed: int,
    day_type: str,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    agent_count = 32 if agents is None else int(agents)
    if agent_count < 1:
        raise ValueError("--agents must be positive")
    num_nodes = 33
    agent_nodes = 1 + (np.arange(agent_count, dtype=np.int64) % (num_nodes - 1))
    load_shape, pv_shape, price_shape = _time_series_shapes(horizon, day_type)
    household_scale = rng.uniform(0.6, 1.7, size=(agent_count, 1)).astype(np.float32)
    pv_capacity = rng.uniform(0.2, 2.6, size=(agent_count, 1)).astype(np.float32)
    load_noise = rng.normal(0.0, 0.035, size=(agent_count, horizon)).astype(np.float32)
    pv_noise = rng.normal(0.0, 0.025, size=(agent_count, horizon)).astype(np.float32)
    load = np.maximum(
        0.12,
        household_scale * load_shape.reshape(1, horizon) + load_noise,
    ).astype(np.float32)
    pv = np.maximum(0.0, pv_capacity * pv_shape.reshape(1, horizon) + pv_noise).astype(
        np.float32
    )
    line_count = SYNTHETIC33_BRANCHES.shape[0]
    depth_scale = np.linspace(1.0, 1.8, line_count, dtype=np.float32)
    return {
        "load": load,
        "pv": pv,
        "num_nodes": np.asarray(num_nodes, dtype=np.int64),
        "agent_nodes": agent_nodes,
        "line_from": SYNTHETIC33_BRANCHES[:, 0],
        "line_to": SYNTHETIC33_BRANCHES[:, 1],
        "resistance": (0.00035 * depth_scale).astype(np.float32),
        "reactance": (0.00020 * depth_scale).astype(np.float32),
        "line_capacity": np.full(line_count, 18.0, dtype=np.float32),
        "grid_buy_price": (0.82 * price_shape).astype(np.float32),
        "grid_sell_price": (0.35 * (0.92 + 0.10 * pv_shape)).astype(np.float32),
        "carbon_allowance_price": np.full(horizon, 0.06, dtype=np.float32),
        "low_carbon_sell_price": np.full(horizon, 0.03, dtype=np.float32),
        "grid_emission_factor": (0.58 * (1.0 + 0.15 * price_shape - 0.10 * pv_shape)).astype(
            np.float32
        ),
        "case_name": np.asarray("synthetic33"),
        "case_display_name": np.asarray("IEEE-33-style synthetic profile"),
        "case_source": np.asarray("synthetic 33-node radial profile"),
        "case_source_url": np.asarray(""),
        "case_notes": np.asarray("Synthetic/derived profile; not a standard IEEE 33-bus case."),
        "base_kv": np.asarray(12.66, dtype=np.float32),
        "base_mva": np.asarray(10.0, dtype=np.float32),
        "benchmark_total_p_mw": np.asarray(float(load.mean(axis=1).sum()), dtype=np.float32),
        "benchmark_total_q_mvar": np.asarray(0.0, dtype=np.float32),
        "benchmark_num_buses": np.asarray(num_nodes, dtype=np.int64),
        "benchmark_num_branches": np.asarray(line_count, dtype=np.int64),
    }


def build_profile(
    agents: int | None,
    horizon: int,
    seed: int,
    day_type: str,
    case: str = "synthetic33",
    pv_penetration: float = 0.5,
    standard_source: str = "locked",
    case69_m: str | Path | None = None,
) -> dict[str, np.ndarray]:
    if case == "synthetic33":
        return build_synthetic33_profile(agents, horizon, seed, day_type)
    return build_standard_profile(
        case,
        agents,
        horizon,
        seed,
        day_type,
        pv_penetration,
        standard_source=standard_source,
        case69_m=case69_m,
    )


def main() -> None:
    cases = ["synthetic33", *sorted(BENCHMARK_CASES)]
    parser = argparse.ArgumentParser(
        description="从合成或标准配电网算例创建场景 profile."
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--case", choices=cases, default="synthetic33")
    parser.add_argument("--agents", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--pv-penetration", type=float, default=0.5)
    parser.add_argument(
        "--standard-source",
        choices=["locked", "authoritative", "auto"],
        default="locked",
        help=(
            "Source mode for IEEE benchmark cases: locked uses frozen project tables; "
            "authoritative loads pandapower/MATPOWER sources and requires local inputs; "
            "auto tries authoritative first and falls back to locked tables."
        ),
    )
    parser.add_argument(
        "--case69-m",
        default=None,
        help="Path to MATPOWER case69.m when --case ieee69 uses authoritative/auto source mode.",
    )
    parser.add_argument(
        "--day-type",
        choices=["weekday", "weekend", "cloudy"],
        default="weekday",
    )
    args = parser.parse_args()
    if args.horizon < 2:
        raise ValueError("--horizon must be at least 2")

    profile = build_profile(
        agents=args.agents,
        horizon=args.horizon,
        seed=args.seed,
        day_type=args.day_type,
        case=args.case,
        pv_penetration=args.pv_penetration,
        standard_source=args.standard_source,
        case69_m=args.case69_m,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **profile)
    print(f"profile={output}")
    print(
        f"case={args.case} agents={profile['load'].shape[0]} "
        f"nodes={int(profile['num_nodes'])} horizon={args.horizon} "
        f"day_type={args.day_type} seed={args.seed}"
    )
    if args.case != "synthetic33":
        print(
            "benchmark="
            f"{float(profile['benchmark_total_p_mw']):.4f} MW/"
            f"{float(profile['benchmark_total_q_mvar']):.4f} MVAr"
        )
        print(f"source_mode={str(np.asarray(profile['benchmark_source_mode']).item())}")


if __name__ == "__main__":
    main()
