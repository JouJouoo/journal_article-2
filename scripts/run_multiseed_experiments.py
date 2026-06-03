from __future__ import annotations

import argparse
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tecsf.config import load_config
from tecsf.device import assign_parallel_device, cuda_device_count
from tecsf.metrics import write_json
from tecsf.variants import VARIANTS

from _experiment_utils import TrainEvalJob, aggregate_rows, run_jobs, write_csv


DEFAULT_VARIANTS = [
    "tecsf",
    "no_chain",
    "no_lccoins",
    "no_feedback",
    "mappo",
    "constrained_mappo",
    "safety_only",
    "myopic_opt",
    "greedy_feasible",
    "no_lagrange",
    "preset_low_carbon",
    "heuristic",
]


def _scalar_text(data: np.lib.npyio.NpzFile, key: str) -> str:
    if key not in data:
        return ""
    return str(np.asarray(data[key]).item())


def _scalar_float(data: np.lib.npyio.NpzFile, key: str) -> float | None:
    if key not in data:
        return None
    return float(np.asarray(data[key]).item())


def _profile_metadata(config_path: str) -> dict:
    cfg = load_config(config_path)
    profile_path = cfg.scenario.profile_path
    if not profile_path:
        return {}
    path = Path(profile_path)
    if not path.exists():
        return {"profile_path": profile_path, "profile_missing": True}
    with np.load(path) as data:
        return {
            "profile_path": str(path),
            "case_name": _scalar_text(data, "case_name"),
            "case_display_name": _scalar_text(data, "case_display_name"),
            "source": _scalar_text(data, "case_source"),
            "source_url": _scalar_text(data, "case_source_url"),
            "base_kv": _scalar_float(data, "base_kv"),
            "base_mva": _scalar_float(data, "base_mva"),
            "benchmark_total_p_mw": _scalar_float(data, "benchmark_total_p_mw"),
            "benchmark_total_q_mvar": _scalar_float(data, "benchmark_total_q_mvar"),
            "benchmark_num_buses": int(np.asarray(data["benchmark_num_buses"]).item())
            if "benchmark_num_buses" in data
            else None,
            "benchmark_num_branches": int(np.asarray(data["benchmark_num_branches"]).item())
            if "benchmark_num_branches" in data
            else None,
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run formal multi-seed TECSF baseline and ablation experiments."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-dir", default="outputs/formal_multiseed")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-seed-start", type=int, default=100000)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 42, 100, 2026, 3407])
    parser.add_argument("--variants", nargs="+", choices=sorted(VARIANTS), default=DEFAULT_VARIANTS)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:<index>")
    parser.add_argument("--jobs", type=int, default=1, help="Parallel worker processes.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    job_index = 0
    for variant in args.variants:
        for seed in args.seeds:
            device = assign_parallel_device(args.device, job_index)
            jobs.append(
                TrainEvalJob(
                    config=args.config,
                    variant=variant,
                    seed=seed,
                    episodes=args.episodes,
                    eval_episodes=args.eval_episodes,
                    output_dir=str(output_dir / variant / f"seed_{seed}"),
                    device=device,
                    eval_seed_start=args.eval_seed_start,
                    label="formal_multiseed",
                )
            )
            job_index += 1

    print(
        f"jobs={len(jobs)} workers={args.jobs} requested_device={args.device} "
        f"cuda_devices={cuda_device_count()}"
    )
    rows = run_jobs(jobs, workers=args.jobs)
    by_variant = aggregate_rows(rows, group_keys=("variant",))
    payload = {
        "config": args.config,
        "benchmark_case": _profile_metadata(args.config),
        "episodes": args.episodes,
        "eval_episodes": args.eval_episodes,
        "seeds": args.seeds,
        "variants": args.variants,
        "requested_device": args.device,
        "cuda_devices": cuda_device_count(),
        "runs": rows,
        "by_variant": by_variant,
    }
    write_json(output_dir / "summary.json", payload)
    write_csv(output_dir / "runs.csv", rows)
    write_csv(output_dir / "summary_by_variant.csv", by_variant)
    print(f"summary={output_dir / 'summary.json'}")
    print(f"runs_csv={output_dir / 'runs.csv'}")
    print(f"summary_csv={output_dir / 'summary_by_variant.csv'}")


if __name__ == "__main__":
    main()
