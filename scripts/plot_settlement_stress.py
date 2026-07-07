from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from figure_style import apply_publication_style, clean_label, save_publication_figure


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="绘制联盟链结算压力测试结果."
    )
    parser.add_argument("summary", help="summary.json from scripts/run_settlement_stress.py")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    summary_path = Path(args.summary)
    payload = _read_json(summary_path)
    rows = payload.get("rows", [])
    if not rows:
        raise SystemExit(f"No rows found in {summary_path}")
    output_dir = Path(args.output_dir) if args.output_dir else summary_path.parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    apply_publication_style()

    cases = sorted({row["case"] for row in rows})
    checks = [
        ("passed", "Pass"),
        ("rollback_energy_error", "Energy rollback"),
        ("rollback_carbon_error", "Carbon rollback"),
        ("rollback_lccoins_error", "LCCoins rollback"),
    ]
    matrix = np.zeros((len(cases), len(checks)), dtype=float)
    for case in cases:
        case_rows = [row for row in rows if row["case"] == case]
        matrix[cases.index(case), 0] = 1.0 if all(bool(row["passed"]) for row in case_rows) else 0.0
        for idx, (field, _) in enumerate(checks[1:], start=1):
            matrix[cases.index(case), idx] = (
                1.0 if max(abs(float(row[field])) for row in case_rows) <= 1e-9 else 0.0
            )

    fig, ax = plt.subplots(figsize=(7.2, max(2.4, len(cases) * 0.42)))
    cmap = ListedColormap(["#D55E00", "#009E73"])
    ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_title("Settlement stress-test outcomes", loc="left", fontweight="bold")
    ax.set_xticks(np.arange(len(checks)), [label for _, label in checks], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(cases)), [clean_label(case) for case in cases])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, "pass" if matrix[i, j] else "fail", ha="center", va="center", fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    paths = save_publication_figure(fig, output_dir / "settlement_stress_outcomes")
    plt.close(fig)
    for path in paths:
        print(f"figure={path}")


if __name__ == "__main__":
    main()
