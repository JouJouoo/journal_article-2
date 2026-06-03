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
        description="Run LCCoins reward-weight sensitivity experiments."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-dir", default="outputs/lccoins_sensitivity")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-seed-start", type=int, default=120000)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 42, 100])
    parser.add_argument("--variants", nargs="+", choices=sorted(VARIANTS), default=["tecsf", "no_lccoins", "no_feedback", "preset_low_carbon"])
    parser.add_argument("--kappa-values", nargs="+", type=float, default=[0.0, 0.05, 0.1, 0.2, 0.5])
    parser.add_argument("--alpha-q-values", nargs="+", type=float, default=[1.0])
    parser.add_argument("--alpha-offset-values", nargs="+", type=float, default=[0.5])
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:<index>")
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()

    base_config = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    configs = {}
    job_index = 0
    for kappa, alpha_q, alpha_offset in product(
        args.kappa_values, args.alpha_q_values, args.alpha_offset_values
    ):
        cfg = copy.deepcopy(base_config)
        cfg.lccoins.kappa = kappa
        cfg.lccoins.alpha_q = alpha_q
        cfg.lccoins.alpha_offset = alpha_offset
        label = f"kappa_{_slug(kappa)}__aq_{_slug(alpha_q)}__ao_{_slug(alpha_offset)}"
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
