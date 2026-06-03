from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_plot_module():
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    path = scripts_dir / "plot_experiment_results.py"
    spec = importlib.util.spec_from_file_location("plot_experiment_results", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_metric_panel_grid_can_show_all_defined_metrics():
    plot = _load_plot_module()

    assert len(plot.METRIC_PANELS) == 7
    assert plot._grid_shape(len(plot.METRIC_PANELS), max_cols=3) == (3, 3)
