"""
TECS-CHAIN Training Monitor (output-based)
============================================
Checks IEEE 33 & IEEE 69 training status by monitoring output files.
Does NOT rely on PID tracking (processes may be managed by the IDE).

Detection strategy:
  - Check if metrics files exist and are being updated
  - If latest file modification > 1 hour old, assume crashed
  - Read episode progress from run_multiseed stdout logs
  - If crashed, restart using subprocess

Usage:
    python scripts/monitor_ieee_training.py --base-dir outputs/ieee_benchmark
"""

import argparse
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Use the same Python interpreter that runs this script
PYTHON = sys.executable
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
SCRIPT = str(SCRIPT_DIR / "run_multiseed_experiments.py")

# Windows process creation flags for detached processes
DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATION_FLAGS = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0


def build_jobs(base_dir: Path) -> dict:
    """Build job configs relative to the given base directory."""
    BASE_OUT = Path(base_dir)
    LOG_DIR = BASE_OUT / "parallel_logs"
    PID_DIR = BASE_OUT / "parallel_pids"
    MON_LOG = LOG_DIR / "monitor.log"

    return {
        "ieee33": {
            "name": "IEEE 33 (ieee33bw)",
            "log_file": LOG_DIR / "ieee33.log",
            "config": str(BASE_OUT / "benchmark_configs" / "ieee33bw.yaml"),
            "out_dir": str(BASE_OUT / "benchmark_ieee33bw"),
            "metrics": BASE_OUT / "benchmark_ieee33bw" / "tecsf" / "seed_7" / "tecsf_metrics.json",
            "pid_file": PID_DIR / "ieee33.pid",
            "log_dir": LOG_DIR,
            "pid_dir": PID_DIR,
            "mon_log": MON_LOG,
        },
        "ieee69": {
            "name": "IEEE 69 (ieee69)",
            "log_file": LOG_DIR / "ieee69.log",
            "config": str(BASE_OUT / "benchmark_configs" / "ieee69.yaml"),
            "out_dir": str(BASE_OUT / "benchmark_ieee69"),
            "metrics": BASE_OUT / "benchmark_ieee69" / "tecsf" / "seed_7" / "tecsf_metrics.json",
            "pid_file": PID_DIR / "ieee69.pid",
            "log_dir": LOG_DIR,
            "pid_dir": PID_DIR,
            "mon_log": MON_LOG,
        },
    }


def log(msg: str, log_dir: Path, mon_log: Path):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        with open(mon_log, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def is_training_active(cfg: dict) -> bool:
    """Check if training is active by examining output file freshness."""
    mf = cfg["metrics"]
    if mf.exists():
        mtime = mf.stat().st_mtime
        age_s = time.time() - mtime
        if age_s < 3600:
            return True
    lf = cfg["log_file"]
    if lf.exists():
        mtime = lf.stat().st_mtime
        age_s = time.time() - mtime
        if age_s < 3600:
            return True
    return False


def read_progress_from_log(log_path: Path) -> dict:
    """Parse training log for episode progress."""
    result = {"episode": None, "total": None, "reward": None, "eta_s": None}
    if not log_path.exists():
        return result
    try:
        content = log_path.read_text(errors="replace")
        matches = re.findall(
            r"episode\s+(\d+)/(\d+)\s+reward=([\d.\-]+).*?eta=(\d+)s", content
        )
        if matches:
            last = matches[-1]
            result["episode"] = int(last[0])
            result["total"] = int(last[1])
            result["reward"] = float(last[2])
            result["eta_s"] = int(last[3])
    except Exception:
        pass
    return result


def read_progress_from_metrics(metrics_path: Path) -> dict:
    """Read last episode from metrics JSON."""
    result = {"episode": None, "reward": None}
    if not metrics_path.exists():
        return result
    try:
        import json
        with open(metrics_path) as f:
            data = json.load(f)
        if isinstance(data, list) and len(data) > 0:
            last = data[-1]
            result["episode"] = last.get("episode", len(data))
            result["reward"] = last.get("total_reward")
    except Exception:
        pass
    return result


def restart_job(key: str, cfg: dict):
    """Restart a crashed training job via subprocess."""
    log(f"[{cfg['name']}] RESTART triggered — launching...", cfg["log_dir"], cfg["mon_log"])
    import shutil
    shutil.rmtree(cfg["out_dir"], ignore_errors=True)

    try:
        proc = subprocess.Popen(
            [
                PYTHON, "-u", SCRIPT,
                "--config", cfg["config"],
                "--episodes", "5000",
                "--eval-episodes", "50",
                "--seeds", "7",
                "--variants", "tecsf",
                "--output-dir", cfg["out_dir"],
                "--device", "cpu",
                "--jobs", "1",
            ],
            stdout=open(str(cfg["log_file"]), "w"),
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            creationflags=CREATION_FLAGS,
        )
        cfg["pid_dir"].mkdir(parents=True, exist_ok=True)
        cfg["pid_file"].write_text(str(proc.pid))
        log(f"[{cfg['name']}] Relaunched PID={proc.pid}", cfg["log_dir"], cfg["mon_log"])
    except Exception as e:
        log(f"[{cfg['name']}] RESTART FAILED: {e}", cfg["log_dir"], cfg["mon_log"])


def run_check(base_dir: Path):
    JOBS = build_jobs(base_dir)
    for key, cfg in JOBS.items():
        name = cfg["name"]
        active = is_training_active(cfg)

        log_progress = read_progress_from_log(cfg["log_file"])
        met_progress = read_progress_from_metrics(cfg["metrics"])

        ep_log = f"{log_progress['episode']}/{log_progress['total']}" if log_progress["episode"] else "?"
        ep_met = met_progress.get("episode", "?")
        rw_log = f"{log_progress['reward']:.4f}" if log_progress["reward"] is not None else "?"
        rw_met = f"{met_progress['reward']:.4f}" if met_progress.get("reward") is not None else "?"

        status = "ACTIVE" if active else "STALE (>1h no update)"
        log(f"[{name}] status={status}  log_ep={ep_log}  metrics_ep={ep_met}  "
            f"log_rw={rw_log}  metrics_rw={rw_met}", cfg["log_dir"], cfg["mon_log"])

        if not active:
            ep = log_progress["episode"] or met_progress.get("episode")
            total = log_progress["total"]
            if total and ep and ep >= total:
                log(f"[{name}] COMPLETED — skipped restart", cfg["log_dir"], cfg["mon_log"])
            else:
                restart_job(key, cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor IEEE benchmark training progress.")
    parser.add_argument("--base-dir", default=str(PROJECT_DIR / "outputs" / "ieee_benchmark"),
                        help="Base output directory for benchmark results.")
    args = parser.parse_args()
    run_check(Path(args.base_dir))
