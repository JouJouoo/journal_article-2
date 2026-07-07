from __future__ import annotations

import argparse
import copy
import sys
from dataclasses import asdict
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tecsf.config import load_config
from tecsf.device import assign_parallel_device, cuda_device_count
from tecsf.metrics import write_json
from tecsf.variants import VARIANTS

from _experiment_utils import TrainEvalJob, aggregate_rows, run_jobs, write_csv


def _slug(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run LCCoins asset-aware sensitivity experiments."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-dir", default="outputs/lccoins_sensitivity")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-seed-start", type=int, default=120000)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 42, 100])
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=sorted(VARIANTS),
        default=["tecsf"],
    )
    parser.add_argument(
        "--asset-utility-weights",
        nargs="+",
        type=float,
        default=None,
        help="Set stock and increment utility weights to the same values.",
    )
    parser.add_argument("--stock-utility-weights", nargs="+", type=float, default=None)
    parser.add_argument("--increment-utility-weights", nargs="+", type=float, default=None)
    parser.add_argument("--clean-energy-weights", nargs="+", type=float, default=None)
    parser.add_argument("--carbon-reduction-weights", nargs="+", type=float, default=None)
    parser.add_argument("--minting-coefficients", nargs="+", type=float, default=None)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:<index>")
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()

    base_config = load_config(args.config)
    if args.asset_utility_weights is not None:
        asset_weight_pairs = [(value, value) for value in args.asset_utility_weights]
    else:
        stock_weights = (
            args.stock_utility_weights
            if args.stock_utility_weights is not None
            else [base_config.lccoins.stock_utility_weight]
        )
        increment_weights = (
            args.increment_utility_weights
            if args.increment_utility_weights is not None
            else [base_config.lccoins.increment_utility_weight]
        )
        asset_weight_pairs = list(product(stock_weights, increment_weights))
    clean_energy_weights = (
        args.clean_energy_weights
        if args.clean_energy_weights is not None
        else [base_config.lccoins.clean_energy_weight]
    )
    carbon_reduction_weights = (
        args.carbon_reduction_weights
        if args.carbon_reduction_weights is not None
        else [base_config.lccoins.carbon_reduction_weight]
    )
    minting_coefficients = (
        args.minting_coefficients
        if args.minting_coefficients is not None
        else [base_config.lccoins.minting_coefficient]
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    configs = {}
    job_index = 0
    for (stock_weight, increment_weight), clean_weight, carbon_weight, minting_coeff in product(
        asset_weight_pairs,
        clean_energy_weights,
        carbon_reduction_weights,
        minting_coefficients,
    ):
        cfg = copy.deepcopy(base_config)
        cfg.lccoins.stock_utility_weight = stock_weight
        cfg.lccoins.increment_utility_weight = increment_weight
        cfg.lccoins.clean_energy_weight = clean_weight
        cfg.lccoins.carbon_reduction_weight = carbon_weight
        cfg.lccoins.minting_coefficient = minting_coeff
        label = (
            f"stock_{_slug(stock_weight)}__inc_{_slug(increment_weight)}"
            f"__ce_{_slug(clean_weight)}__cr_{_slug(carbon_weight)}"
        )
        if abs(float(minting_coeff) - 1.0) > 1e-12:
            label += f"__mint_{_slug(minting_coeff)}"
        configs[label] = asdict(cfg)
        for variant in args.variants:
            for seed in args.seeds:
                device = assign_parallel_device(args.device, job_index)
                jobs.append(
                    TrainEvalJob(
                        config=cfg,
                        variant=variant,
                        seed=seed,
                        episodes=args.episodes,
                        eval_episodes=args.eval_episodes,
                        output_dir=str(output_dir / label / variant / f"seed_{seed}"),
                        device=device,
                        eval_seed_start=args.eval_seed_start,
                        label=label,
                    )
                )
                job_index += 1

    print(
        f"jobs={len(jobs)} workers={args.jobs} requested_device={args.device} "
        f"cuda_devices={cuda_device_count()}"
    )
    rows = run_jobs(jobs, workers=args.jobs)
    by_setting_variant = aggregate_rows(rows, group_keys=("label", "variant"))
    payload = {
        "base_config": args.config,
        "episodes": args.episodes,
        "eval_episodes": args.eval_episodes,
        "seeds": args.seeds,
        "variants": args.variants,
        "configs": configs,
        "runs": rows,
        "by_setting_variant": by_setting_variant,
    }
    write_json(output_dir / "summary.json", payload)
    write_csv(output_dir / "runs.csv", rows)
    write_csv(output_dir / "summary_by_setting_variant.csv", by_setting_variant)
    print(f"summary={output_dir / 'summary.json'}")
    print(f"summary_csv={output_dir / 'summary_by_setting_variant.csv'}")


if __name__ == "__main__":
    main()
