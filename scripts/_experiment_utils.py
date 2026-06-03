from __future__ import annotations

import csv
import json
import sys
import tracemalloc
from time import perf_counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tecsf.config import ExperimentConfig
from tecsf.metrics import write_json
from tecsf.rl.mappo import evaluate_policy, train


@dataclass(frozen=True)
class TrainEvalJob:
    config: str | ExperimentConfig
    variant: str
    seed: int
    episodes: int
    eval_episodes: int
    output_dir: str
    device: str
    eval_seed_start: int
    label: str = ""


def _mean_numeric(rows: list[dict[str, Any]], prefix: str = "") -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted({key for row in rows for key, value in row.items() if isinstance(value, (int, float))})
    out = {}
    for key in keys:
        values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
        if values:
            out[f"{prefix}{key}"] = float(np.mean(values))
    return out


def summarize_train_eval(
    job: TrainEvalJob,
    train_metrics: list[dict[str, Any]],
    eval_result: dict[str, Any],
) -> dict[str, Any]:
    last = train_metrics[-1] if train_metrics else {}
    last_100 = train_metrics[-100:] if train_metrics else []
    row: dict[str, Any] = {
        "label": job.label,
        "variant": job.variant,
        "seed": job.seed,
        "episodes": job.episodes,
        "eval_episodes": job.eval_episodes,
        "device": job.device,
        "output_dir": job.output_dir,
    }
    for key, value in last.items():
        if isinstance(value, (int, float)):
            row[f"train_last_{key}"] = float(value)
    row.update(_mean_numeric(last_100, prefix="train_last100_"))
    row.update(_mean_numeric(eval_result.get("metrics", []), prefix="eval_"))
    row["eval_passes_feasibility_gate"] = (
        float(row.get("eval_settlement_success_rate", 0.0)) >= 0.95
        and float(row.get("eval_max_violation", float("inf"))) <= 0.1
    )
    return row


def run_train_eval_job(job: TrainEvalJob) -> dict[str, Any]:
    output_dir = Path(job.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "run_summary.json"
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    tracemalloc.start()
    start = perf_counter()
    try:
        result = train(
            config=job.config,
            variant=job.variant,
            output_dir=output_dir,
            episodes=job.episodes,
            seed=job.seed,
            device=job.device,
        )
        train_seconds = perf_counter() - start
        eval_start = perf_counter()
        eval_result = evaluate_policy(
            result.checkpoint_path,
            config=job.config,
            episodes=job.eval_episodes,
            device=job.device,
            seed_start=job.eval_seed_start,
        )
        eval_seconds = perf_counter() - eval_start
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    write_json(output_dir / "evaluation.json", eval_result)
    summary = summarize_train_eval(job, result.episode_metrics, eval_result)
    summary["train_seconds"] = float(train_seconds)
    summary["eval_seconds"] = float(eval_seconds)
    summary["total_seconds"] = float(train_seconds + eval_seconds)
    summary["peak_python_memory_mb"] = float(peak_bytes / (1024.0 * 1024.0))
    write_json(summary_path, summary)
    return summary


def run_jobs(jobs: list[TrainEvalJob], workers: int) -> list[dict[str, Any]]:
    if workers <= 1:
        rows = []
        for job in jobs:
            print(f"running label={job.label or '-'} variant={job.variant} seed={job.seed} device={job.device}")
            rows.append(run_train_eval_job(job))
        return rows

    rows = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_train_eval_job, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            row = future.result()
            print(f"finished label={job.label or '-'} variant={job.variant} seed={job.seed} device={job.device}")
            rows.append(row)
    return sorted(rows, key=lambda row: (row.get("label", ""), row.get("variant", ""), row.get("seed", 0)))


def aggregate_rows(rows: list[dict[str, Any]], group_keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row.get(item, "") for item in group_keys)
        groups.setdefault(key, []).append(row)
    summaries = []
    for key, items in sorted(groups.items(), key=lambda kv: kv[0]):
        summary = {group_keys[idx]: key[idx] for idx in range(len(group_keys))}
        summary["runs"] = len(items)
        numeric_keys = sorted(
            {
                item_key
                for item in items
                for item_key, value in item.items()
                if isinstance(value, (int, float)) and item_key not in group_keys
            }
        )
        for item_key in numeric_keys:
            values = [float(item[item_key]) for item in items if isinstance(item.get(item_key), (int, float))]
            if not values:
                continue
            summary[f"{item_key}_mean"] = float(np.mean(values))
            summary[f"{item_key}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        summary["passes_feasibility_gate"] = (
            float(summary.get("eval_settlement_success_rate_mean", 0.0)) >= 0.95
            and float(summary.get("eval_max_violation_mean", float("inf"))) <= 0.1
        )
        summaries.append(summary)
    return summaries


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        target.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
