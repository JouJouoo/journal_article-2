from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "outputs" / "full_experiments_20260703_no_ablation_logs"
STATUS_PATH = LOG_DIR / "status.txt"
SHORT_DIR = "outputs/full_experiments_20260703_paper_aligned_short100_no_ablation"
FULL_DIR = "outputs/full_experiments_20260703_paper_aligned_no_ablation"


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def write_status(message: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with STATUS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(f"{_timestamp()} {message}\n")


def run_step(name: str, args: list[str], env: dict[str, str]) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{name}.log"
    write_status(f"START {name}")
    with log_path.open("ab") as log:
        proc = subprocess.run(
            [sys.executable, *args],
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    write_status(f"END {name} exit={proc.returncode} log={log_path}")
    if proc.returncode != 0:
        raise SystemExit(f"{name} failed with exit code {proc.returncode}; see {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the 2026-07-03 LC-MAPPO short and full experiment workflow without ablations."
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"

    if args.validate_only:
        write_status(f"VALIDATE_ONLY python={sys.executable} device={args.device} jobs={args.jobs}")
        return

    try:
        run_step(
            "short100_suite",
            [
                "scripts/run_improved_experiment_suite.py",
                "--quick",
                "--episodes",
                "100",
                "--eval-episodes",
                "5",
                "--seeds",
                "7",
                "42",
                "--device",
                args.device,
                "--jobs",
                str(args.jobs),
                "--benchmark-cases",
                "ieee33bw",
                "ieee69",
                "--benchmark-standard-source",
                "authoritative",
                "--case69-m",
                "data/case69.m",
                "--skip-ablations",
                "--output-dir",
                SHORT_DIR,
            ],
            env,
        )
        run_step(
            "full1000_suite",
            [
                "scripts/run_improved_experiment_suite.py",
                "--episodes",
                "1000",
                "--eval-episodes",
                "20",
                "--seeds",
                "7",
                "42",
                "100",
                "2026",
                "3407",
                "--device",
                args.device,
                "--jobs",
                str(args.jobs),
                "--benchmark-cases",
                "ieee33bw",
                "ieee69",
                "--benchmark-standard-source",
                "authoritative",
                "--case69-m",
                "data/case69.m",
                "--skip-ablations",
                "--output-dir",
                FULL_DIR,
            ],
            env,
        )
        run_step(
            "paper_figures",
            [
                "scripts/plot_paper_figures.py",
                FULL_DIR,
                "--skip-ablations",
                "--output-dir",
                f"{FULL_DIR}/paper_figures",
            ],
            env,
        )
        run_step(
            "ieee_benchmark_paper_figure",
            [
                "scripts/plot_ieee_benchmark_paper_figure.py",
                "--root",
                FULL_DIR,
                "--output-dir",
                f"{FULL_DIR}/paper_figures",
            ],
            env,
        )
        run_step(
            "tecsf_multiseed_curve",
            [
                "scripts/plot_multiseed_training.py",
                "--run-dir",
                f"{FULL_DIR}/formal_multiseed/tecsf",
                "--output-dir",
                f"{FULL_DIR}/formal_multiseed/tecsf/figures",
            ],
            env,
        )
    except BaseException as exc:
        write_status(f"FAILED {exc}")
        raise
    write_status("ALL_COMPLETE")


if __name__ == "__main__":
    main()
