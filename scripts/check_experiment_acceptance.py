from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _find_group(summary: dict[str, Any], label: str, variant: str) -> dict[str, Any] | None:
    rows = summary.get("by_setting_variant") or summary.get("by_variant") or []
    for row in rows:
        if row.get("variant") != variant:
            continue
        if "label" in row and row.get("label") != label:
            continue
        return row
    return None


def _gate(row: dict[str, Any], min_success: float, max_violation: float) -> bool:
    return (
        float(row.get("eval_settlement_success_rate_mean", 0.0)) >= min_success
        and float(row.get("eval_max_violation_mean", float("inf"))) <= max_violation
    )


def _require_numeric(row: dict[str, Any], key: str, failures: list[str], context: str) -> None:
    if not isinstance(row.get(key), (int, float)):
        failures.append(f"{context} missing numeric metric {key}")


def _require_post_outputs(root: Path, suite: str, failures: list[str]) -> None:
    if not (root / suite / "statistics" / "paired_comparisons.json").exists():
        failures.append(f"missing statistics output for {suite}")
    if not (root / suite / "pareto" / "pareto_runs.json").exists():
        failures.append(f"missing pareto output for {suite}")
    if not (root / suite / "figures").exists():
        failures.append(f"missing figures output for {suite}")


def _check_file(path: Path, failures: list[str]) -> dict[str, Any] | None:
    if not path.exists():
        failures.append(f"missing {path}")
        return None
    return _load(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check improved TECSF experiment outputs against acceptance gates."
    )
    parser.add_argument("output_dir")
    parser.add_argument("--min-success", type=float, default=0.95)
    parser.add_argument("--max-violation", type=float, default=0.1)
    parser.add_argument("--min-lccoins-kappa01-reward", type=float, default=-1.0)
    parser.add_argument(
        "--benchmark-cases",
        nargs="+",
        default=None,
        help="Benchmark suites expected under output_dir as benchmark_<case>. Defaults to discovered benchmark_* directories.",
    )
    args = parser.parse_args()

    root = Path(args.output_dir)
    failures: list[str] = []
    benchmark_cases = (
        args.benchmark_cases
        if args.benchmark_cases is not None
        else sorted(
            path.name.removeprefix("benchmark_")
            for path in root.glob("benchmark_*")
            if path.is_dir() and path.name != "benchmark_configs" and path.name != "benchmark_profiles"
        )
    )

    formal = _check_file(root / "formal_multiseed" / "summary.json", failures)
    lccoins = _check_file(root / "lccoins_sensitivity" / "summary.json", failures)
    network = _check_file(root / "network_stress" / "summary.json", failures)
    system = _check_file(root / "system_stress" / "summary.json", failures)
    scalability = _check_file(root / "scalability" / "summary.json", failures)
    settlement = _check_file(root / "settlement_stress" / "summary.json", failures)
    benchmarks = {
        case: _check_file(root / f"benchmark_{case}" / "summary.json", failures)
        for case in benchmark_cases
    }

    if formal:
        row = _find_group(formal, "", "tecsf")
        if not row or not _gate(row, args.min_success, args.max_violation):
            failures.append("formal_multiseed TECSF does not pass feasibility gate")
        elif row:
            for metric in [
                "eval_q_lc_mean",
                "eval_carbon_offset_mean",
                "eval_low_carbon_sell_mean",
                "eval_lccoins_mean",
                "eval_lccoins_candidate_mean",
                "eval_agent_lccoins_q_lc_corr_mean",
                "eval_agent_lccoins_c_offset_corr_mean",
                "eval_agent_lccoins_candidate_corr_mean",
                "eval_participant_payment_cost_mean",
                "eval_p2p_transfer_payment_mean",
            ]:
                _require_numeric(row, metric, failures, "formal_multiseed TECSF")
    if lccoins:
        row = _find_group(lccoins, "kappa_0p1__aq_1__ao_0p5", "tecsf")
        if not row or not _gate(row, args.min_success, args.max_violation):
            failures.append("lccoins kappa=0.1 TECSF does not pass feasibility gate")
        elif float(row.get("eval_mean_reward_mean", -999.0)) < args.min_lccoins_kappa01_reward:
            failures.append("lccoins kappa=0.1 TECSF reward still indicates collapse")
        elif row:
            for metric in [
                "eval_q_lc_mean",
                "eval_carbon_offset_mean",
                "eval_low_carbon_sell_mean",
                "eval_lccoins_mean",
                "eval_lccoins_candidate_mean",
                "eval_agent_lccoins_candidate_corr_mean",
            ]:
                _require_numeric(row, metric, failures, "lccoins kappa=0.1 TECSF")
    if network:
        for label in ["line_0p5__trade_1", "line_0p5__trade_1p3"]:
            row = _find_group(network, label, "tecsf")
            if not row or not _gate(row, args.min_success, args.max_violation):
                failures.append(f"network_stress {label} TECSF does not pass feasibility gate")
    if system:
        for row in system.get("by_setting_variant", []):
            if row.get("variant") == "tecsf" and not _gate(row, args.min_success, args.max_violation):
                failures.append(
                    f"system_stress {row.get('label', '<missing>')} TECSF does not pass feasibility gate"
                )
    if scalability:
        row = _find_group(scalability, "agents_16__nodes_9", "tecsf")
        if not row or not _gate(row, args.min_success, args.max_violation):
            failures.append("scalability agents_16__nodes_9 TECSF does not pass feasibility gate")
    if settlement:
        if not settlement.get("all_passed", False):
            failures.append("settlement_stress did not pass all cases")
    for case, summary in benchmarks.items():
        if not summary:
            continue
        row = _find_group(summary, "", "tecsf")
        if not row or not _gate(row, args.min_success, args.max_violation):
            failures.append(f"benchmark_{case} TECSF does not pass feasibility gate")

    for suite in [
        "formal_multiseed",
        "lccoins_sensitivity",
        "network_stress",
        "system_stress",
        "scalability",
        *[f"benchmark_{case}" for case in benchmark_cases],
    ]:
        _require_post_outputs(root, suite, failures)
    if not (root / "settlement_stress" / "figures").exists():
        failures.append("missing figures output for settlement_stress")

    if failures:
        for failure in failures:
            print(f"FAIL {failure}")
        raise SystemExit(1)
    print("acceptance=passed")


if __name__ == "__main__":
    main()
