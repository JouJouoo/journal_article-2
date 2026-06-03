from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tecsf.benchmark_cases import BENCHMARK_CASES
from tecsf.benchmark_sources import (
    BenchmarkSourceUnavailable,
    compare_benchmark_cases,
    failed_checks,
    load_authoritative_benchmark_case,
)


def _validate_case(case_name: str, case69_m: str | None) -> dict:
    locked = BENCHMARK_CASES[case_name]
    try:
        source = load_authoritative_benchmark_case(case_name, case69_m=case69_m)
    except BenchmarkSourceUnavailable as exc:
        return {
            "case": case_name,
            "status": "skipped",
            "reason": str(exc),
        }

    checks = compare_benchmark_cases(locked, source)
    problems = failed_checks(checks)
    return {
        "case": case_name,
        "status": "failed" if problems else "passed",
        "source": source.source,
        "source_url": source.source_url,
        "checks": [
            {
                "name": check.name,
                "passed": check.passed,
                "expected": check.expected,
                "observed": check.observed,
            }
            for check in checks
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate locked TECSF IEEE benchmark tables against local authoritative "
            "pandapower/MATPOWER sources."
        )
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=sorted(BENCHMARK_CASES),
        default=sorted(BENCHMARK_CASES),
    )
    parser.add_argument(
        "--case69-m",
        default=None,
        help="Path to local MATPOWER case69.m for IEEE 69-bus source validation.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat unavailable authoritative sources as failures instead of skips.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = parser.parse_args()

    results = [_validate_case(case_name, args.case69_m) for case_name in args.cases]
    failed = [item for item in results if item["status"] == "failed"]
    skipped = [item for item in results if item["status"] == "skipped"]
    if args.strict:
        failed.extend(skipped)

    if args.json:
        print(json.dumps({"results": results}, ensure_ascii=False, indent=2))
    else:
        for result in results:
            status = result["status"]
            case_name = result["case"]
            if status == "skipped":
                print(f"{case_name}: skipped - {result['reason']}")
                continue
            print(f"{case_name}: {status} - {result['source']}")
            for check in result["checks"]:
                marker = "ok" if check["passed"] else "FAIL"
                print(
                    f"  {marker} {check['name']}: "
                    f"expected={check['expected']} observed={check['observed']}"
                )

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
