from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

from create_benchmark_profile import build_profile


PYTHON = sys.executable


def _run(cmd: list[str], cwd: Path, dry_run: bool) -> None:
    printable = " ".join(f'"{item}"' if " " in item else item for item in cmd)
    print(printable)
    if dry_run:
        return
    subprocess.run(cmd, cwd=cwd, check=True)


def _suite_commands(
    args: argparse.Namespace,
    root: Path,
    benchmark_configs: dict[str, Path] | None = None,
) -> list[list[str]]:
    out = Path(args.output_dir)
    benchmark_configs = benchmark_configs or {}
    common = [
        "--episodes",
        str(args.episodes),
        "--eval-episodes",
        str(args.eval_episodes),
        "--seeds",
        *[str(seed) for seed in args.seeds],
        "--device",
        args.device,
        "--jobs",
        str(args.jobs),
    ]
    if args.benchmark_only:
        return _benchmark_commands(common, out, benchmark_configs)
    if args.quick:
        commands = [
            [
                PYTHON,
                "scripts/run_multiseed_experiments.py",
                *common,
                "--variants",
                "tecsf",
                "constrained_mappo",
                "myopic_opt",
                "safety_only",
                "heuristic",
                "--output-dir",
                str(out / "formal_multiseed"),
            ],
            [
                PYTHON,
                "scripts/run_lccoins_sensitivity.py",
                *common,
                "--variants",
                "tecsf",
                "no_lccoins",
                "--kappa-values",
                "0.1",
                "--output-dir",
                str(out / "lccoins_sensitivity"),
            ],
            [
                PYTHON,
                "scripts/run_network_stress.py",
                *common,
                "--variants",
                "tecsf",
                "myopic_opt",
                "no_lagrange",
                "heuristic",
                "--line-capacity-scales",
                "0.5",
                "--trade-power-scales",
                "1.0",
                "1.3",
                "--output-dir",
                str(out / "network_stress"),
            ],
            [
                PYTHON,
                "scripts/run_system_stress.py",
                *common,
                "--variants",
                "tecsf",
                "myopic_opt",
                "heuristic",
                "--load-scales",
                "1.3",
                "--pv-scales",
                "0.7",
                "--grid-price-scales",
                "1.0",
                "--carbon-price-scales",
                "2.0",
                "--line-capacity-scales",
                "0.7",
                "--output-dir",
                str(out / "system_stress"),
            ],
            [
                PYTHON,
                "scripts/run_scalability_experiment.py",
                *common,
                "--variants",
                "tecsf",
                "myopic_opt",
                "heuristic",
                "--agent-counts",
                "16",
                "--node-counts",
                "9",
                "--output-dir",
                str(out / "scalability"),
            ],
            [
                PYTHON,
                "scripts/run_settlement_stress.py",
                "--seeds",
                *[str(seed) for seed in args.seeds],
                "--output-dir",
                str(out / "settlement_stress"),
            ],
        ]
        commands.extend(_benchmark_commands(common, out, benchmark_configs))
        return commands
    commands = [
        [
            PYTHON,
            "scripts/run_multiseed_experiments.py",
            *common,
            "--output-dir",
            str(out / "formal_multiseed"),
        ],
        [
            PYTHON,
            "scripts/run_lccoins_sensitivity.py",
            *common,
            "--kappa-values",
            "0",
            "0.1",
            "0.2",
            "0.5",
            "--output-dir",
            str(out / "lccoins_sensitivity"),
        ],
        [
            PYTHON,
            "scripts/run_network_stress.py",
            *common,
            "--line-capacity-scales",
            "1.0",
            "0.7",
            "0.5",
            "--trade-power-scales",
            "1.0",
            "1.3",
            "--output-dir",
            str(out / "network_stress"),
        ],
        [
            PYTHON,
            "scripts/run_system_stress.py",
            *common,
            "--load-scales",
            "1.0",
            "1.3",
            "--pv-scales",
            "0.7",
            "1.0",
            "1.3",
            "--grid-price-scales",
            "1.0",
            "1.3",
            "--carbon-price-scales",
            "1.0",
            "2.0",
            "--line-capacity-scales",
            "1.0",
            "0.7",
            "--output-dir",
            str(out / "system_stress"),
        ],
        [
            PYTHON,
            "scripts/run_scalability_experiment.py",
            *common,
            "--agent-counts",
            "8",
            "16",
            "32",
            "--node-counts",
            "5",
            "9",
            "17",
            "--output-dir",
            str(out / "scalability"),
        ],
        [
            PYTHON,
            "scripts/run_settlement_stress.py",
            "--seeds",
            *[str(seed) for seed in args.seeds],
            "--output-dir",
            str(out / "settlement_stress"),
        ],
    ]
    commands.extend(_benchmark_commands(common, out, benchmark_configs))
    return commands


def _benchmark_commands(
    common: list[str],
    output_dir: Path,
    benchmark_configs: dict[str, Path],
) -> list[list[str]]:
    commands: list[list[str]] = []
    for case, config_path in benchmark_configs.items():
        commands.append(
            [
                PYTHON,
                "scripts/run_multiseed_experiments.py",
                "--config",
                str(config_path),
                *common,
                "--variants",
                "tecsf",
                "myopic_opt",
                "heuristic",
                "--output-dir",
                str(output_dir / f"benchmark_{case}"),
            ]
        )
    return commands


def _post_commands(
    output_dir: Path,
    benchmark_cases: list[str] | None = None,
    benchmark_only: bool = False,
) -> list[list[str]]:
    commands: list[list[str]] = []
    suites = []
    if not benchmark_only:
        suites.extend(
            [
                "formal_multiseed",
                "lccoins_sensitivity",
                "network_stress",
                "system_stress",
                "scalability",
            ]
        )
    suites.extend(f"benchmark_{case}" for case in (benchmark_cases or []))
    for suite in suites:
        summary = output_dir / suite / "summary.json"
        commands.extend(
            [
                [PYTHON, "scripts/plot_experiment_results.py", str(summary)],
                [PYTHON, "scripts/analyze_experiment_statistics.py", str(summary), "--baseline", "tecsf"],
                [PYTHON, "scripts/analyze_pareto_front.py", str(summary)],
            ]
        )
    if not benchmark_only:
        commands.append(
            [
                PYTHON,
                "scripts/plot_settlement_stress.py",
                str(output_dir / "settlement_stress" / "summary.json"),
            ]
        )
        commands.append(
            [
                PYTHON,
                "scripts/check_experiment_acceptance.py",
                str(output_dir),
            ]
        )
    return commands


def _prepare_benchmark_configs(args: argparse.Namespace, root: Path) -> dict[str, Path]:
    if not args.benchmark_cases or args.dry_run:
        return {case: Path(args.output_dir) / "benchmark_configs" / f"{case}.yaml" for case in args.benchmark_cases}
    output_dir = Path(args.output_dir)
    profile_dir = output_dir / "benchmark_profiles"
    config_dir = output_dir / "benchmark_configs"
    profile_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    with (root / args.config).open("r", encoding="utf-8") as handle:
        base_config = yaml.safe_load(handle) or {}
    configs: dict[str, Path] = {}
    for case in args.benchmark_cases:
        profile_path = (profile_dir / f"{case}.npz").resolve()
        profile = build_profile(
            agents=None,
            horizon=int(base_config.get("scenario", {}).get("horizon", 24)),
            seed=int(args.seeds[0]),
            day_type=args.benchmark_day_type,
            case=case,
            pv_penetration=args.benchmark_pv_penetration,
        )
        np.savez(profile_path, **profile)
        cfg = dict(base_config)
        cfg["scenario"] = dict(base_config.get("scenario", {}))
        cfg["scenario"]["profile_path"] = str(profile_path)
        config_path = config_dir / f"{case}.yaml"
        with config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(cfg, handle, sort_keys=False)
        configs[case] = config_path
    return configs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the improved TECSF experiment suite and post-analysis tools."
    )
    parser.add_argument("--output-dir", default="outputs/report_experiments_improved")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a small end-to-end grid for smoke validation of the improved suite.",
    )
    parser.add_argument(
        "--benchmark-cases",
        nargs="*",
        choices=["ieee33bw", "ieee69"],
        default=[],
        help="Also run formal benchmark-case suites with generated standard profile configs.",
    )
    parser.add_argument(
        "--benchmark-only",
        action="store_true",
        help="Run and post-process only the requested benchmark-case suites.",
    )
    parser.add_argument("--benchmark-day-type", choices=["weekday", "weekend", "cloudy"], default="weekday")
    parser.add_argument("--benchmark-pv-penetration", type=float, default=0.5)
    parser.add_argument("--skip-post", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.episodes is None:
        args.episodes = 3 if args.quick else 1000
    if args.eval_episodes is None:
        args.eval_episodes = 2 if args.quick else 20
    if args.seeds is None:
        args.seeds = [7, 42] if args.quick else [7, 42, 100, 2026, 3407]
    if args.benchmark_only and not args.benchmark_cases:
        parser.error("--benchmark-only requires at least one --benchmark-cases value")

    root = Path(__file__).resolve().parents[1]
    output_dir = Path(args.output_dir)
    benchmark_configs = _prepare_benchmark_configs(args, root)
    for command in _suite_commands(args, root, benchmark_configs):
        _run(command, root, args.dry_run)
    if not args.skip_post:
        for command in _post_commands(output_dir, args.benchmark_cases, args.benchmark_only):
            _run(command, root, args.dry_run)


if __name__ == "__main__":
    main()
