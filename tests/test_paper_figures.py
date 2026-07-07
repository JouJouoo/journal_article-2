from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_paper_module():
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    path = scripts_dir / "plot_paper_figures.py"
    spec = importlib.util.spec_from_file_location("plot_paper_figures", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _write_summary(root: Path, suite: str, rows: list[dict]) -> None:
    out = root / suite
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps({"runs": rows}, indent=2), encoding="utf-8")


def _run_row(label: str, variant: str, seed: int, scale: float = 1.0) -> dict:
    return {
        "label": label,
        "variant": variant,
        "seed": seed,
        "eval_mean_reward": -0.3 * scale,
        "eval_system_cost": 70.0 * scale,
        "eval_grid_carbon_emission": 50.0 * scale,
        "eval_net_carbon_allowance_need": 20.0 * scale,
        "eval_renewable_consumption_rate": 0.99,
        "eval_settlement_success_rate": min(1.0, 0.96 / scale),
        "eval_max_violation": max(0.0, scale - 1.0),
        "eval_lccoins": 30.0 / scale,
        "total_seconds": 100.0 * scale,
    }


def _build_synthetic_report(root: Path) -> None:
    formal = []
    variants = ["tecsf", "mappo", "no_chain"]
    for idx, variant in enumerate(variants, start=1):
        for seed in [1, 2]:
            formal.append(_run_row("formal_multiseed", variant, seed, scale=1.0 + idx * 0.05 + seed * 0.01))
    _write_summary(root, "formal_multiseed", formal)

    lccoins = []
    for asset_weight in ["0", "0p1", "0p2"]:
        for idx, variant in enumerate(["tecsf"], start=1):
            for seed in [1, 2]:
                label = f"stock_{asset_weight}__inc_{asset_weight}__ce_1__cr_0p5"
                lccoins.append(_run_row(label, variant, seed, scale=1.0 + idx * 0.03))
    _write_summary(root, "lccoins_sensitivity", lccoins)

    network = []
    for line in ["0p5", "0p7"]:
        for trade in ["1", "1p3"]:
            for idx, variant in enumerate(["tecsf", "mappo"], start=1):
                for seed in [1, 2]:
                    network.append(_run_row(f"line_{line}__trade_{trade}", variant, seed, scale=1.0 + idx * 0.1))
    _write_summary(root, "network_stress", network)

    scalability = []
    for agents in [8, 16]:
        for nodes in [5, 9]:
            for variant, scale in [("mappo", 1.0), ("tecsf", 1.6)]:
                for seed in [1, 2]:
                    scalability.append(_run_row(f"agents_{agents}__nodes_{nodes}", variant, seed, scale=scale))
    _write_summary(root, "scalability", scalability)

    benchmark = []
    for idx, variant in enumerate(["tecsf", "mappo"], start=1):
        for seed in [1, 2]:
            benchmark.append(_run_row("formal_multiseed", variant, seed, scale=1.0 + idx * 0.04))
    _write_summary(root, "benchmark_ieee33bw", benchmark)
    _write_summary(root, "benchmark_ieee69", benchmark)

    settlement_rows = []
    for case, state in [("hash_tamper_rejection", "Rejected"), ("normal_settlement", "Settled")]:
        for seed in [1, 2]:
            settlement_rows.append(
                {
                    "case": case,
                    "seed": seed,
                    "state": state,
                    "passed": True,
                    "rollback_energy_error": 0.0,
                    "rollback_carbon_error": 0.0,
                    "rollback_lccoins_error": 0.0,
                }
            )
    out = root / "settlement_stress"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps({"rows": settlement_rows}, indent=2), encoding="utf-8")


def test_plot_all_generates_four_publication_figures(tmp_path):
    paper = _load_paper_module()
    _build_synthetic_report(tmp_path)

    outputs = paper.plot_all(tmp_path, tmp_path / "paper_figures", formats=("png",), dpi=90)

    assert set(outputs) == {
        "fig1_main_comparison",
        "fig2_safety_stress",
        "fig3_lccoins_sensitivity",
        "fig4_generalization_scalability",
    }
    for paths in outputs.values():
        assert len(paths) == 1
        assert paths[0].suffix == ".png"
        assert paths[0].exists()
        assert paths[0].stat().st_size > 0
