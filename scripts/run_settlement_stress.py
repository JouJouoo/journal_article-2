from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tecsf.chain import REJECTED, REVERTED, SETTLED, SettlementEngine, SimulatedTECSChain
from tecsf.config import ExperimentConfig, ScenarioConfig
from tecsf.data import generate_synthetic_scenario
from tecsf.market import ActionBatch, clear_market
from tecsf.metrics import write_json
from tecsf.utils import stable_hash

from _experiment_utils import write_csv


def _package(seed: int):
    config = ExperimentConfig(scenario=ScenarioConfig(num_agents=3, num_nodes=3, horizon=4, seed=seed))
    scenario = generate_synthetic_scenario(config, seed=seed)
    actions = ActionBatch(
        q_buy=np.asarray([1.0, 0.5, 0.0], dtype=np.float32),
        q_sell=np.asarray([0.0, 0.0, 1.5], dtype=np.float32),
        price_buy=np.asarray([0.9, 0.8, 0.1], dtype=np.float32),
        price_sell=np.asarray([0.1, 0.1, 0.4], dtype=np.float32),
        charge=np.zeros(3, dtype=np.float32),
        discharge=np.zeros(3, dtype=np.float32),
    )
    package = clear_market(
        scenario,
        config,
        0,
        np.full(3, config.storage.init_soc, dtype=np.float32),
        actions,
    )
    return config, package


def _balance_error(before: np.ndarray, after: np.ndarray) -> float:
    return float(np.max(np.abs(before - after))) if before.size else 0.0


def _case_rows(seed: int) -> list[dict]:
    rows: list[dict] = []

    config, package = _package(seed)
    engine = SettlementEngine(config, num_agents=3)
    record = engine.settle(package)
    rows.append(
        {
            "seed": seed,
            "case": "normal_settlement",
            "state": record.state,
            "passed": record.state == SETTLED,
            "reason": record.reason,
            "rollback_energy_error": 0.0,
            "rollback_carbon_error": 0.0,
            "rollback_lccoins_error": 0.0,
        }
    )

    config, package = _package(seed + 10_000)
    tampered = copy.deepcopy(package)
    tampered.package_hash = "bad"
    engine = SettlementEngine(config, num_agents=3)
    record = engine.settle(tampered)
    rows.append(
        {
            "seed": seed,
            "case": "hash_tamper_rejection",
            "state": record.state,
            "passed": record.state == REJECTED and "hash" in record.reason,
            "reason": record.reason,
            "rollback_energy_error": 0.0,
            "rollback_carbon_error": 0.0,
            "rollback_lccoins_error": 0.0,
        }
    )

    config, package = _package(seed + 20_000)
    violated = copy.deepcopy(package)
    violated.violations["line"] = config.clearing.network_tolerance + 1.0
    violated.package_hash = stable_hash(violated.payload(include_hash=False))
    engine = SettlementEngine(config, num_agents=3)
    record = engine.settle(violated)
    rows.append(
        {
            "seed": seed,
            "case": "constraint_violation_rejection",
            "state": record.state,
            "passed": record.state == REJECTED and "constraint" in record.reason,
            "reason": record.reason,
            "rollback_energy_error": 0.0,
            "rollback_carbon_error": 0.0,
            "rollback_lccoins_error": 0.0,
        }
    )

    config, package = _package(seed + 30_000)
    chain = SimulatedTECSChain(config, num_agents=3)
    energy_before = chain.energy_balances.copy()
    carbon_before = chain.carbon_balances.copy()
    lccoins_before = chain.lccoins_balances.copy()
    record = chain.settle(package, force_execution_failure=True)
    rows.append(
        {
            "seed": seed,
            "case": "forced_execution_revert",
            "state": record.state,
            "passed": record.state == REVERTED,
            "reason": record.reason,
            "rollback_energy_error": _balance_error(energy_before, chain.energy_balances),
            "rollback_carbon_error": _balance_error(carbon_before, chain.carbon_balances),
            "rollback_lccoins_error": _balance_error(lccoins_before, chain.lccoins_balances),
        }
    )

    config, package = _package(seed + 40_000)
    engine = SettlementEngine(config, num_agents=3)
    first = engine.settle(package)
    energy_before = engine.energy_balances.copy()
    carbon_before = engine.carbon_balances.copy()
    lccoins_before = engine.lccoins_balances.copy()
    second = engine.settle(package)
    rows.append(
        {
            "seed": seed,
            "case": "duplicate_lccoins_revert",
            "state": second.state,
            "passed": first.state == SETTLED and second.state == REVERTED,
            "reason": second.reason,
            "rollback_energy_error": _balance_error(energy_before, engine.energy_balances),
            "rollback_carbon_error": _balance_error(carbon_before, engine.carbon_balances),
            "rollback_lccoins_error": _balance_error(lccoins_before, engine.lccoins_balances),
        }
    )

    config, package = _package(seed + 50_000)
    chain = SimulatedTECSChain(config, num_agents=3)
    record = chain.settle(package)
    ledger = chain.ledger_payload()
    block = ledger["blocks"][-1]
    receipt = block["receipts"][0]
    audit_passed = (
        record.state == SETTLED
        and receipt["record_id"] == record.record_id
        and receipt["state"] == SETTLED
        and block["prev_hash"] == chain.genesis_hash
        and bool(block["state_root"])
        and bool(block["merkle_root"])
    )
    rows.append(
        {
            "seed": seed,
            "case": "audit_linkage_consistency",
            "state": record.state,
            "passed": audit_passed,
            "reason": record.reason,
            "rollback_energy_error": 0.0,
            "rollback_carbon_error": 0.0,
            "rollback_lccoins_error": 0.0,
        }
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stress-test TECS-Chain settlement rejection, rollback, minting, and audit invariants."
    )
    parser.add_argument("--output-dir", default="outputs/settlement_stress")
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 42, 100, 2026, 3407])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in args.seeds:
        rows.extend(_case_rows(seed))
    summary = {
        "seeds": args.seeds,
        "case_count": len(rows),
        "passed_count": sum(bool(row["passed"]) for row in rows),
        "all_passed": all(bool(row["passed"]) for row in rows),
        "max_rollback_energy_error": max(float(row["rollback_energy_error"]) for row in rows),
        "max_rollback_carbon_error": max(float(row["rollback_carbon_error"]) for row in rows),
        "max_rollback_lccoins_error": max(float(row["rollback_lccoins_error"]) for row in rows),
        "rows": rows,
    }
    write_json(output_dir / "summary.json", summary)
    write_csv(output_dir / "settlement_stress.csv", rows)
    print(f"summary={output_dir / 'summary.json'}")
    print(f"csv={output_dir / 'settlement_stress.csv'}")
    print(f"passed={summary['passed_count']}/{summary['case_count']}")


if __name__ == "__main__":
    main()
