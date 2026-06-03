from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_METRICS = [
    "eval_feasible_rate",
    "eval_settlement_success_rate",
    "eval_max_violation",
    "eval_mean_reward",
    "eval_system_cost",
    "eval_grid_carbon_emission",
    "eval_net_carbon_allowance_need",
]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _mean(values: list[float]) -> float:
    return float(sum(values) / max(len(values), 1))


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = _mean(values)
    return math.sqrt(sum((item - avg) ** 2 for item in values) / (len(values) - 1))


def _normal_quantile_975() -> float:
    return 1.959963984540054


def _paired_permutation_pvalue(diffs: list[float]) -> float:
    nonzero = [float(item) for item in diffs if abs(float(item)) > 1e-12]
    if not nonzero:
        return 1.0
    observed = abs(_mean(nonzero))
    total = 0
    extreme = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(nonzero)):
        total += 1
        candidate = abs(_mean([value * sign for value, sign in zip(nonzero, signs)]))
        if candidate + 1e-12 >= observed:
            extreme += 1
    return float(extreme / max(total, 1))


def _holm_adjust(rows: list[dict[str, Any]]) -> None:
    indexed = sorted(
        [(idx, float(row["p_value"])) for idx, row in enumerate(rows)],
        key=lambda item: item[1],
    )
    m = len(indexed)
    running = 0.0
    adjusted = [1.0] * len(rows)
    for rank, (idx, p_value) in enumerate(indexed):
        value = min(1.0, (m - rank) * p_value)
        running = max(running, value)
        adjusted[idx] = running
    for idx, value in enumerate(adjusted):
        rows[idx]["p_holm"] = float(value)


def _paired_rows(
    runs: list[dict[str, Any]],
    metric: str,
    baseline: str,
    variant: str,
    label: str,
) -> tuple[list[float], list[float]]:
    base_by_seed = {
        int(row["seed"]): float(row[metric])
        for row in runs
        if row.get("variant") == baseline
        and row.get("label", "") == label
        and isinstance(row.get(metric), (int, float))
    }
    variant_by_seed = {
        int(row["seed"]): float(row[metric])
        for row in runs
        if row.get("variant") == variant
        and row.get("label", "") == label
        and isinstance(row.get(metric), (int, float))
    }
    seeds = sorted(set(base_by_seed) & set(variant_by_seed))
    return [base_by_seed[seed] for seed in seeds], [variant_by_seed[seed] for seed in seeds]


def _comparison_direction(metric: str) -> str:
    if "cost" in metric or "emission" in metric or "violation" in metric or "allowance" in metric:
        return "lower"
    return "higher"


def analyze(summary: dict[str, Any], baseline: str, metrics: list[str]) -> list[dict[str, Any]]:
    runs = summary.get("runs", [])
    labels = sorted({str(row.get("label", "")) for row in runs})
    variants = sorted({str(row.get("variant", "")) for row in runs if row.get("variant") != baseline})
    rows: list[dict[str, Any]] = []
    for label in labels:
        for variant in variants:
            for metric in metrics:
                base_values, variant_values = _paired_rows(
                    runs, metric, baseline=baseline, variant=variant, label=label
                )
                if len(base_values) < 2:
                    continue
                diffs = [v - b for b, v in zip(base_values, variant_values)]
                diff_mean = _mean(diffs)
                diff_std = _std(diffs)
                se = diff_std / math.sqrt(len(diffs)) if len(diffs) > 1 else 0.0
                ci = _normal_quantile_975() * se
                effect = diff_mean / diff_std if diff_std > 1e-12 else 0.0
                p_value = _paired_permutation_pvalue(diffs)
                rows.append(
                    {
                        "label": label,
                        "baseline": baseline,
                        "variant": variant,
                        "metric": metric,
                        "direction": _comparison_direction(metric),
                        "n_pairs": len(diffs),
                        "baseline_mean": _mean(base_values),
                        "variant_mean": _mean(variant_values),
                        "mean_diff_variant_minus_baseline": diff_mean,
                        "ci95_low": diff_mean - ci,
                        "ci95_high": diff_mean + ci,
                        "cohen_dz": effect,
                        "p_value": p_value,
                    }
                )
    _holm_adjust(rows)
    for row in rows:
        direction = row["direction"]
        diff = float(row["mean_diff_variant_minus_baseline"])
        better = diff > 0.0 if direction == "higher" else diff < 0.0
        row["significant_better_after_holm"] = bool(better and row["p_holm"] < 0.05)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run paired statistical comparisons for TECSF experiment summaries."
    )
    parser.add_argument("summary", help="summary.json produced by an experiment suite.")
    parser.add_argument("--baseline", default="tecsf")
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    summary_path = Path(args.summary)
    output_dir = Path(args.output_dir) if args.output_dir else summary_path.parent / "statistics"
    summary = _read_json(summary_path)
    rows = analyze(summary, baseline=args.baseline, metrics=args.metrics)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "paired_comparisons.csv", rows)
    with (output_dir / "paired_comparisons.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=True, indent=2)
    print(f"comparisons={len(rows)}")
    print(f"csv={output_dir / 'paired_comparisons.csv'}")
    print(f"json={output_dir / 'paired_comparisons.json'}")


if __name__ == "__main__":
    main()
