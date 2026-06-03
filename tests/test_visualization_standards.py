from __future__ import annotations

from pathlib import Path


PLOTTING_SCRIPTS = [
    "scripts/run_experiments.py",
    "scripts/plot_multiseed_training.py",
    "scripts/plot_experiment_results.py",
    "scripts/plot_settlement_stress.py",
    "scripts/visualize_ledger.py",
    "scripts/plot_paper_figures.py",
]


def test_plotting_scripts_use_shared_publication_exporter():
    root = Path(__file__).resolve().parents[1]
    for relative in PLOTTING_SCRIPTS:
        text = (root / relative).read_text(encoding="utf-8")
        assert "save_publication_figure" in text, relative
        assert ".savefig(" not in text, relative
        assert "plt.savefig(" not in text, relative
        assert "dpi=160" not in text, relative
        assert "dpi=180" not in text, relative
        assert "dpi=220" not in text, relative


def test_visualization_text_uses_lccoins_consistently():
    root = Path(__file__).resolve().parents[1]
    for relative in PLOTTING_SCRIPTS + ["README.md"]:
        text = (root / relative).read_text(encoding="utf-8")
        assert "Lccoins" not in text, relative
