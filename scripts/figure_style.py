from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


OKABE_ITO = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#F0E442",
    "#000000",
]

METHOD_DISPLAY_NAMES = {
    "tecsf": "LC-MAPPO",
    "lc_mappo": "LC-MAPPO",
    "mappo": "MAPPO",
    "no_chain": "w/o chain",
}

METHOD_COLORS = {
    "tecsf": "#0072B2",
    "lc_mappo": "#0072B2",
    "mappo": "#D55E00",
    "no_chain": "#56B4E9",
}

METHOD_ORDER = [
    "tecsf",
    "lc_mappo",
    "mappo",
    "no_chain",
]

STATE_COLORS = {
    "Settled": "#009E73",
    "Verified": "#56B4E9",
    "Pending": "#F0E442",
    "Rejected": "#D55E00",
    "Reverted": "#CC79A7",
}

CONSTRAINT_COLORS = {
    "Voltage": "#0072B2",
    "Line": "#E69F00",
    "SOC": "#009E73",
    "Trade": "#CC79A7",
}


def apply_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "SimHei"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "axes.titleweight": "bold",
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.unicode_minus": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 120,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.prop_cycle": plt.cycler(color=OKABE_ITO),
        }
    )


def save_publication_figure(
    fig,
    path_base: str | Path,
    formats: Iterable[str] = ("pdf", "svg", "png"),
    dpi: int = 600,
) -> list[Path]:
    base = Path(path_base)
    formats = tuple(fmt.lower().lstrip(".") for fmt in formats)
    if base.suffix.lower().lstrip(".") in formats:
        base = base.with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for fmt in formats:
        path = base.with_suffix(f".{fmt}")
        kwargs = {"bbox_inches": "tight"}
        if fmt in {"png", "tif", "tiff", "jpg", "jpeg"}:
            kwargs["dpi"] = dpi
        fig.savefig(path, **kwargs)
        paths.append(path)
    return paths


def display_variant(variant: str) -> str:
    return METHOD_DISPLAY_NAMES.get(str(variant), str(variant).replace("_", " "))


def variant_color(variant: str) -> str:
    return METHOD_COLORS.get(str(variant), OKABE_ITO[variant_sort_key(str(variant)) % len(OKABE_ITO)])


def variant_sort_key(variant: str) -> int:
    try:
        return METHOD_ORDER.index(str(variant))
    except ValueError:
        return len(METHOD_ORDER) + abs(hash(str(variant))) % 1000


def clean_label(value: str) -> str:
    return str(value).replace("_", " ")


def _parse_token_value(value: str) -> float | int | str:
    text = str(value).replace("p", ".")
    try:
        number = float(text)
    except ValueError:
        return str(value)
    if number.is_integer():
        return int(number)
    return number


def parse_label(label: str) -> dict[str, float | int | str]:
    parsed: dict[str, float | int | str] = {}
    for part in str(label).split("__"):
        if "_" not in part:
            continue
        key, raw = part.split("_", 1)
        parsed[key] = _parse_token_value(raw)
    return parsed


def label_value(label: str, key: str, default: float | int | str | None = None):
    return parse_label(label).get(key, default)


def format_label(label: str) -> str:
    parsed = parse_label(label)
    if "stock" in parsed and "inc" in parsed:
        return f"stock={parsed['stock']}, inc={parsed['inc']}"
    if "line" in parsed and "trade" in parsed:
        return f"line={parsed['line']}, trade={parsed['trade']}"
    if "agents" in parsed and "nodes" in parsed:
        return f"{parsed['agents']} agents / {parsed['nodes']} nodes"
    if str(label) == "formal_multiseed":
        return "formal"
    return clean_label(str(label))


def style_axes(ax, grid_axis: str | None = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color="#D9D9D9", linewidth=0.55, alpha=0.75)
        ax.set_axisbelow(True)


def mean_std_ci(values: list[float] | np.ndarray) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    mean = float(arr.mean())
    if arr.size < 2:
        return mean, 0.0, 0.0
    std = float(arr.std(ddof=1))
    ci95 = float(1.959963984540054 * std / np.sqrt(arr.size))
    return mean, std, ci95


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size == 0:
        return values.astype(float, copy=True)
    out = np.empty_like(values, dtype=float)
    cumsum = np.cumsum(np.insert(values.astype(float), 0, 0.0))
    for idx in range(values.size):
        start = max(0, idx - window + 1)
        out[idx] = (cumsum[idx + 1] - cumsum[start]) / (idx - start + 1)
    return out
