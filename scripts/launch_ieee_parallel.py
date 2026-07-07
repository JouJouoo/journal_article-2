"""
TECS-CHAIN IEEE Parallel Training Launcher
===========================================
Spawns two independent Python processes for IEEE 33 and IEEE 69 training.
Uses DETACHED_PROCESS (Windows) to fully decouple from parent console.
Writes PIDs to files for the monitoring script.
"""
import subprocess
import sys
import os
from pathlib import Path

# Windows: DETACHED_PROCESS = the child gets NO console and does NOT inherit
# the parent console — survives even if the parent session terminates.
DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATION_FLAGS = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0

PYTHON = r"C:\Users\zrway\.conda\envs\DP-LCRL\python.exe"
SCRIPT = r"C:\Users\zrway\Desktop\期刊论文-2\scripts\run_multiseed_experiments.py"
BASE_OUT = Path(r"C:\Users\zrway\Desktop\期刊论文-2\outputs\ieee_benchmark_lcmappo_20260706")
LOG_DIR = BASE_OUT / "parallel_5000_logs"
PID_DIR = BASE_OUT / "parallel_5000_pids"

LOG_DIR.mkdir(parents=True, exist_ok=True)
PID_DIR.mkdir(parents=True, exist_ok=True)

jobs = [
    {
        "name": "ieee33",
        "config": str(BASE_OUT / "benchmark_configs" / "ieee33bw.yaml"),
        "out_dir": str(BASE_OUT / "benchmark_ieee33bw_5000"),
        "log": LOG_DIR / "ieee33_5000.log",
        "pid_file": PID_DIR / "ieee33.pid",
    },
    {
        "name": "ieee69",
        "config": str(BASE_OUT / "benchmark_configs" / "ieee69.yaml"),
        "out_dir": str(BASE_OUT / "benchmark_ieee69_5000"),
        "log": LOG_DIR / "ieee69_5000.log",
        "pid_file": PID_DIR / "ieee69.pid",
    },
]

for job in jobs:
    # Remove old output dir
    import shutil
    shutil.rmtree(job["out_dir"], ignore_errors=True)

    log_fh = open(str(job["log"]), "w")

    proc = subprocess.Popen(
        [
            PYTHON, "-u", SCRIPT,   # -u = unbuffered stdout
            "--config", job["config"],
            "--episodes", "5000",
            "--eval-episodes", "50",
            "--seeds", "7",
            "--variants", "tecsf",
            "--output-dir", job["out_dir"],
            "--device", "cpu",
            "--jobs", "1",
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
