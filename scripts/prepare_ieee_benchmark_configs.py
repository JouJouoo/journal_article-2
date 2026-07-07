"""为 IEEE 33/69 基准算例生成配置文件，并运行 lc_mappo（tecsf）训练。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np
import yaml

from create_benchmark_profile import build_profile

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "ieee_benchmark_lcmappo_20260706"
PROFILE_DIR = OUTPUT_DIR / "benchmark_profiles"
CONFIG_DIR = OUTPUT_DIR / "benchmark_configs"

EPISODES = 1000
SEED = 7
CASE69_M = ROOT / "data" / "case69.m"


def generate_config(case: str) -> Path:
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    profile_path = (PROFILE_DIR / f"{case}.npz").resolve()
    print(f"Generating profile for {case} ...")
    profile = build_profile(
        agents=None,
        horizon=24,
        seed=SEED,
        day_type="weekday",
        case=case,
        pv_penetration=0.5,
        standard_source="authoritative",
        case69_m=str(CASE69_M) if CASE69_M.exists() else None,
    )
    np.savez(profile_path, **profile)
    print(f"  profile saved to {profile_path}")

    with (ROOT / "configs" / "default.yaml").open("r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f) or {}

    cfg = dict(base_config)
    cfg["scenario"] = dict(base_config.get("scenario", {}))
    cfg["scenario"]["profile_path"] = str(profile_path)
    # 确保种子设置正确
    cfg["scenario"]["seed"] = SEED
    cfg["rl"]["seed"] = SEED
    cfg["rl"]["episodes"] = EPISODES

    config_path = CONFIG_DIR / f"{case}.yaml"
    with config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"  config saved to {config_path}")
    return config_path


def main() -> None:
    cases = ["ieee33bw", "ieee69"]
    config_paths: dict[str, Path] = {}
    for case in cases:
        config_paths[case] = generate_config(case)

    print("\n=== 配置文件已生成，请手动运行以下命令：===\n")
    for case, config_path in config_paths.items():
        out_dir = OUTPUT_DIR / f"benchmark_{case}"
        print(
            f"\"{sys.executable}\" scripts/run_multiseed_experiments.py "
            f"--config {config_path} "
            f"--episodes {EPISODES} "
            f"--eval-episodes 20 "
            f"--seeds {SEED} "
            f"--variants tecsf "
            f"--output-dir {out_dir} "
            f"--device cpu "
            f"--jobs 1"
        )
    print("\n或在 Python 中直接调用 train() 函数。")


if __name__ == "__main__":
    main()
