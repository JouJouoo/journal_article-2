from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_OBJECTIVES = [
    ("eval_system_cost", "lower"),
    ("eval_grid_carbon_emission", "lower"),
    ("eval_mean_reward", "higher"),
    ("eval_feasible_rate", "higher"),
    ("eval_max_violation", "lower"),
]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _group_key(row: dict[str, Any]) -> str:
    return str(row.get("label", ""))


def _value(row: dict[str, Any], metric: str) -> float | None:
    value = row.get(metric)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _is_feasible(row: dict[str, Any], min_success: float, max_violation: float) -> bool:
    success = _value(row, "eval_settlement_success_rate")
    violation = _value(row, "eval_max_violation")
    if success is None or violation is None:
        return False
    return success >= min_success and violation <= max_violation


def _dominates(a: dict[str, Any], b: dict[str, Any], objectives: list[tuple[str, str]]) -> bool:
    better_once = False
    for metric, direction in objectives:
        av = _value(a, metric)
        bv = _value(b, metric)
        if av is None or bv is None:
            return False
        if direction == "lower":
            if av > bv:
                return False
            better_once = better_once or av < bv
        else:
            if av < bv:
                return False
            better_once = better_once or av > bv
    return better_once


def analyze(
    summary: dict[str, Any],
    objectives: list[tuple[str, str]],
    min_success: float,
    max_violation: float,
) -> list[dict[str, Any]]:
    runs = list(summary.get("runs", []))
    rows: list[dict[str, Any]] = []
    for label in sorted({_group_key(row) for row in runs}):
        group = [row for row in runs if _group_key(row) == label]
        feasible = [row for row in group if _is_feasible(row, min_success, max_violation)]
        for row in group:
            dominated_by = []
            if row in feasible:
                dominated_by = [
                    other.get("variant", "")
                    for other in feasible
                    if other is not row and _dominates(other, row, objectives)
                ]
            out = {
                "label": label,
                "variant": row.get("variant", ""),
                "seed": row.get("seed", ""),
                "passes_feasibility_gate": row in feasible,
                "pareto_efficient": row in feasible and not dominated_by,
                "dominated_by": ";".join(sorted(str(item) for item in dominated_by)),
            }
            for metric, _ in objectives:
                out[metric] = row.get(metric, "")
            rows.append(out)
    return rows


def _parse_objectives(items: list[str]) -> list[tuple[str, str]]:
    out = []
    for item in items:
        if ":" not in item:
            raise ValueError(f"Objective must be metric:direction, got {item!r}")
        metric, direction = item.split(":", 1)
        if direction not in {"higher", "lower"}:
            raise ValueError(f"Objective direction must be higher or lower, got {direction!r}")
        out.append((metric, direction))
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="分析可行集 Pareto 效率."
    )
    parser.add_argument("summary", help="summary.json produced by an experiment suite.")
    parser.add_argument("--min-success", type=float, default=0.95)
    parser.add_argument("--max-violation", type=float, default=0.1)
    parser.add_argument(
        "--objectives",
        nargs="+",
        default=[f"{metric}:{direction}" for metric, direction in DEFAULT_OBJECTIVES],
    )
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    summary_path = Path(args.summary)
    output_dir = Path(args.output_dir) if args.output_dir else summary_path.parent / "pareto"
    objectives = _parse_objectives(args.objectives)
    rows = analyze(_read_json(summary_path), objectives, args.min_success, args.max_violation)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "pareto_runs.csv", rows)
    with (output_dir / "pareto_runs.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=True, indent=2)
    print(f"rows={len(rows)}")
    print(f"csv={output_dir / 'pareto_runs.csv'}")
    print(f"json={output_dir / 'pareto_runs.json'}")


if __name__ == "__main__":
    main()
