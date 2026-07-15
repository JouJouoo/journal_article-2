"""
TECS-CHAIN IEEE Parallel Training Launcher
===========================================
Spawns two independent Python processes for IEEE 33 and IEEE 69 training.
Uses DETACHED_PROCESS (Windows) or nohup-like detachment (Linux/macOS) to
fully decouple from parent console.
Writes PIDs to files for the monitoring script.

Usage:
    python scripts/launch_ieee_parallel.py --base-dir outputs/ieee_benchmark \
        --episodes 5000 --device cpu
"""
import argparse
import subprocess
import sys
import os
from pathlib import Path

# Windows: DETACHED_PROCESS = the child gets NO console and does NOT inherit
# the parent console — survives even if the parent session terminates.
DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATION_FLAGS = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0

# Use the same Python interpreter that runs this script
PYTHON = sys.executable
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
SCRIPT = str(SCRIPT_DIR / "run_multiseed_experiments.py")


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch IEEE benchmark training in parallel.")
    parser.add_argument("--base-dir", default=str(PROJECT_DIR / "outputs" / "ieee_benchmark"),
                        help="Base output directory for benchmark results.")
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--seeds", default="7")
    parser.add_argument("--variants", default="tecsf")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()

    BASE_OUT = Path(args.base_dir)
    LOG_DIR = BASE_OUT / "parallel_logs"
    PID_DIR = BASE_OUT / "parallel_pids"

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    PID_DIR.mkdir(parents=True, exist_ok=True)

    jobs = [
        {
            "name": "ieee33",
            "config": str(BASE_OUT / "benchmark_configs" / "ieee33bw.yaml"),
            "out_dir": str(BASE_OUT / "benchmark_ieee33bw"),
            "log": LOG_DIR / "ieee33.log",
            "pid_file": PID_DIR / "ieee33.pid",
        },
        {
            "name": "ieee69",
            "config": str(BASE_OUT / "benchmark_configs" / "ieee69.yaml"),
            "out_dir": str(BASE_OUT / "benchmark_ieee69"),
            "log": LOG_DIR / "ieee69.log",
            "pid_file": PID_DIR / "ieee69.pid",
        },
    ]

    import shutil

    for job in jobs:
        # Remove old output dir
        shutil.rmtree(job["out_dir"], ignore_errors=True)

        log_fh = open(str(job["log"]), "w")

        proc = subprocess.Popen(
            [
                PYTHON, "-u", SCRIPT,   # -u = unbuffered stdout
                "--config", job["config"],
                "--episodes", str(args.episodes),
                "--eval-episodes", str(args.eval_episodes),
                "--seeds", args.seeds,
                "--variants", args.variants,
                "--output-dir", job["out_dir"],
                "--device", args.device,
                "--jobs", str(args.jobs),
            ],
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            creationflags=CREATION_FLAGS,
            close_fds=True,
        )
        job["pid_file"].write_text(str(proc.pid))
        print(f"[{job['name']}] PID={proc.pid}  log={job['log']}")

    print("DONE — both processes launched and detached.")


if __name__ == "__main__":
    main()
