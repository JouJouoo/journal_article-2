from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_style_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "figure_style.py"
    spec = importlib.util.spec_from_file_location("figure_style", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_label_parsing_and_display_names():
    style = _load_style_module()

    assert style.display_variant("no_chain") == "w/o chain"
    assert style.display_variant("tecsf") == "LC-MAPPO"
    assert style.display_variant("lc_mappo") == "LC-MAPPO"
    assert style.variant_color("lc_mappo") == style.variant_color("tecsf")

    parsed = style.parse_label("stock_0p2__inc_0p1__ce_1__cr_0p5")
    assert parsed["stock"] == 0.2
    assert parsed["inc"] == 0.1
    assert parsed["ce"] == 1
    assert parsed["cr"] == 0.5
    assert style.format_label("stock_0p2__inc_0p1__ce_1__cr_0p5") == "stock=0.2, inc=0.1"
    assert style.format_label("agents_16__nodes_9") == "16 agents / 9 nodes"
    assert style.format_label("line_0p7__trade_1p3") == "line=0.7, trade=1.3"


def test_save_publication_figure_writes_requested_formats(tmp_path):
    style = _load_style_module()
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])

    paths = style.save_publication_figure(fig, tmp_path / "figure.png", formats=("png", "svg"), dpi=120)
    plt.close(fig)

    assert [path.suffix for path in paths] == [".png", ".svg"]
    assert all(path.exists() and path.stat().st_size > 0 for path in paths)
