from __future__ import annotations

import argparse
import csv
import html
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from figure_style import STATE_COLORS, apply_publication_style, save_publication_figure


def _short_hash(value: str, length: int = 10) -> str:
    return value[:length] if value else ""


def _read_ledger(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _block_rows(ledger: dict) -> list[dict]:
    rows = []
    for block in ledger.get("blocks", []):
        receipt = block.get("receipts", [{}])[0] if block.get("receipts") else {}
        tx = block.get("transactions", [{}])[0] if block.get("transactions") else {}
        rows.append(
            {
                "height": int(block["height"]),
                "block_hash": block["block_hash"],
                "prev_hash": block["prev_hash"],
                "tx_id": tx.get("tx_id", ""),
                "record_id": receipt.get("record_id", ""),
                "state": receipt.get("state", ""),
                "reason": receipt.get("reason", ""),
                "lccoins_total": float(receipt.get("lccoins_total", 0.0)),
                "energy_payment_count": int(receipt.get("energy_payment_count", 0)),
                "carbon_entry_count": int(receipt.get("carbon_entry_count", 0)),
                "state_root": block["state_root"],
                "merkle_root": block["merkle_root"],
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_chain_overview(rows: list[dict], output: Path) -> None:
    if not rows:
        fig, ax = plt.subplots(figsize=(7, 2.5))
        ax.text(0.5, 0.5, "No blocks in ledger", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        save_publication_figure(fig, output)
        plt.close(fig)
        return

    heights = np.asarray([row["height"] for row in rows])
    lccoins = np.asarray([row["lccoins_total"] for row in rows], dtype=float)
    cumulative_lccoins = np.cumsum(lccoins)
    colors = [STATE_COLORS.get(row["state"], "#999999") for row in rows]
    max_per_row = 12
    lane_count = int(np.ceil(len(rows) / max_per_row))
    display_width = min(max_per_row, len(rows))

    fig = plt.figure(figsize=(12, 7.2 + 0.45 * lane_count))
    gs = fig.add_gridspec(3, 1, height_ratios=[0.8 * lane_count + 0.2, 1.0, 1.2], hspace=0.45)

    ax_chain = fig.add_subplot(gs[0, 0])
    ax_chain.set_title("A. Simulated TECS-Chain block linkage", loc="left", fontweight="bold")
    ax_chain.set_xlim(-0.85, display_width + 0.1)
    ax_chain.set_ylim(-0.2, lane_count)
    ax_chain.axis("off")
    for idx, row in enumerate(rows):
        lane = idx // max_per_row
        x = idx % max_per_row
        y = lane_count - lane - 0.58
        ax_chain.add_patch(
            plt.Rectangle(
                (x - 0.36, y - 0.22),
                0.72,
                0.44,
                facecolor="#F7F7F7",
                edgecolor="#333333",
                linewidth=0.8,
            )
        )
        ax_chain.text(x, y + 0.12, f"H{row['height']}", ha="center", va="center", fontsize=7)
        ax_chain.text(
            x,
            y,
            _short_hash(row["block_hash"], 8),
            ha="center",
            va="center",
            fontsize=6,
            family="monospace",
        )
        ax_chain.text(
            x,
            y - 0.12,
            _short_hash(row["state_root"], 8),
            ha="center",
            va="center",
            fontsize=6,
            color="#666666",
            family="monospace",
        )
        if idx < len(rows) - 1:
            next_lane = (idx + 1) // max_per_row
            if next_lane == lane:
                ax_chain.annotate(
                    "",
                    xy=(x + 0.62, y),
                    xytext=(x + 0.38, y),
                    arrowprops=dict(arrowstyle="->", color="#333333", linewidth=0.8),
                )
            else:
                next_y = lane_count - next_lane - 0.58
                gap_y = (y + next_y) / 2
                right_margin = display_width - 0.05
                left_margin = -0.62
                ax_chain.plot(
                    [x + 0.38, right_margin, right_margin, left_margin],
                    [y, y, gap_y, gap_y],
                    color="#333333",
                    linewidth=0.8,
                )
                ax_chain.annotate(
                    "",
                    xy=(-0.38, next_y + 0.12),
                    xytext=(left_margin, gap_y),
                    arrowprops=dict(arrowstyle="->", color="#333333", linewidth=0.8),
                )
    ax_chain.text(
        display_width - 0.35,
        -0.12,
        "Each arrow denotes prev_hash linkage; lower code in each block is state_root.",
        ha="right",
        va="bottom",
        fontsize=7,
        color="#666666",
    )

    ax_state = fig.add_subplot(gs[1, 0])
    ax_state.set_title("B. Settlement receipt state by block", loc="left", fontweight="bold")
    ax_state.bar(heights, np.ones_like(heights), color=colors, edgecolor="#333333", linewidth=0.3)
    ax_state.set_yticks([])
    ax_state.set_xlabel("Block height")
    ax_state.set_xlim(-0.6, len(rows) - 0.4)
    present_states = {row["state"] for row in rows}
    legend_handles = [
        Patch(facecolor=color, edgecolor="none", label=state)
        for state, color in STATE_COLORS.items()
        if state in present_states
    ]
    ax_state.legend(handles=legend_handles, frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.35))

    ax_lc = fig.add_subplot(gs[2, 0])
    ax_lc.set_title("C. LCCoins mint per block and cumulative total", loc="left", fontweight="bold")
    ax_lc.bar(heights, lccoins, color="#0072B2", alpha=0.78, label="Minted per block")
    ax_lc.set_xlabel("Block height")
    ax_lc.set_ylabel("LCCoins minted")
    ax2 = ax_lc.twinx()
    ax2.plot(heights, cumulative_lccoins, color="#D55E00", linewidth=2, label="Cumulative")
    ax2.set_ylabel("Cumulative LCCoins")
    lines, labels = ax_lc.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax_lc.legend(lines + lines2, labels + labels2, frameon=False, loc="upper left")

    fig.subplots_adjust(left=0.08, right=0.92, top=0.93, bottom=0.08, hspace=0.58)
    save_publication_figure(fig, output)
    plt.close(fig)


def _plot_balances(ledger: dict, output: Path) -> None:
    state = ledger.get("state", {})
    energy = np.asarray(state.get("energy_balances", []), dtype=float)
    carbon = np.asarray(state.get("carbon_balances", []), dtype=float)
    lccoins = np.asarray(state.get("lccoins_balances", []), dtype=float)
    agents = np.arange(max(len(energy), len(carbon), len(lccoins)))

    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    datasets = [
        ("A. Final energy balances", energy, "#4C78A8", "Energy balance"),
        ("B. Final carbon balances", carbon, "#59A14F", "Carbon balance"),
        ("C. Final LCCoins balances", lccoins, "#F28E2B", "LCCoins"),
    ]
    for ax, (title, values, color, ylabel) in zip(axes, datasets):
        ax.set_title(title, loc="left", fontweight="bold")
        ax.bar(agents[: len(values)], values, color=color)
        ax.set_ylabel(ylabel)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[-1].set_xlabel("Agent")
    axes[-1].set_xticks(agents, [f"A{i}" for i in agents])
    fig.tight_layout()
    save_publication_figure(fig, output)
    plt.close(fig)


def _plot_receipt_metrics(rows: list[dict], output: Path) -> None:
    heights = np.asarray([row["height"] for row in rows])
    energy_counts = np.asarray([row["energy_payment_count"] for row in rows], dtype=float)
    carbon_counts = np.asarray([row["carbon_entry_count"] for row in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    width = 0.38
    ax.bar(heights - width / 2, energy_counts, width=width, label="Energy payment entries", color="#4C78A8")
    ax.bar(heights + width / 2, carbon_counts, width=width, label="Carbon entries", color="#59A14F")
    ax.set_title("Receipt contents by block")
    ax.set_xlabel("Block height")
    ax.set_ylabel("Entry count")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_publication_figure(fig, output)
    plt.close(fig)


def _write_html_report(
    ledger_path: Path,
    output_dir: Path,
    rows: list[dict],
    chain_fig: Path,
    balances_fig: Path,
    receipts_fig: Path,
) -> Path:
    settled = sum(row["state"] == "Settled" for row in rows)
    total_lccoins = sum(row["lccoins_total"] for row in rows)
    head_hash = rows[-1]["block_hash"] if rows else ""

    table_rows = "\n".join(
        "<tr>"
        f"<td>{row['height']}</td>"
        f"<td>{html.escape(row['state'])}</td>"
        f"<td><code>{_short_hash(row['block_hash'], 12)}</code></td>"
        f"<td><code>{_short_hash(row['prev_hash'], 12)}</code></td>"
        f"<td><code>{_short_hash(row['tx_id'], 12)}</code></td>"
        f"<td>{row['lccoins_total']:.4f}</td>"
        f"<td>{row['energy_payment_count']}</td>"
        f"<td>{row['carbon_entry_count']}</td>"
        "</tr>"
        for row in rows
    )
    report = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>TECS-Chain ledger report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 28px; color: #222; }}
    h1, h2 {{ margin-bottom: 0.3rem; }}
    .summary {{ display: grid; grid-template-columns: repeat(4, minmax(140px, 1fr)); gap: 12px; margin: 18px 0; }}
    .card {{ border: 1px solid #ddd; border-radius: 6px; padding: 12px; background: #fafafa; }}
    .value {{ font-size: 1.3rem; font-weight: 700; }}
    img {{ max-width: 100%; border: 1px solid #ddd; margin: 12px 0 24px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 0.9rem; }}
    th, td {{ border-bottom: 1px solid #e5e5e5; padding: 7px 8px; text-align: left; }}
    th {{ background: #f2f2f2; }}
    code {{ font-size: 0.85rem; }}
  </style>
</head>
<body>
  <h1>TECS-Chain ledger report</h1>
  <p>Source ledger: <code>{html.escape(str(ledger_path))}</code></p>
  <div class="summary">
    <div class="card"><div>Blocks</div><div class="value">{len(rows)}</div></div>
    <div class="card"><div>Settled receipts</div><div class="value">{settled}/{len(rows)}</div></div>
    <div class="card"><div>Total LCCoins</div><div class="value">{total_lccoins:.4f}</div></div>
    <div class="card"><div>Head hash</div><div class="value"><code>{_short_hash(head_hash, 12)}</code></div></div>
  </div>
  <h2>Blockchain linkage and settlement effects</h2>
  <img src="{html.escape(chain_fig.name)}" alt="Chain overview">
  <h2>Final ledger state</h2>
  <img src="{html.escape(balances_fig.name)}" alt="Final balances">
  <h2>Receipt contents</h2>
  <img src="{html.escape(receipts_fig.name)}" alt="Receipt metrics">
  <h2>Block table</h2>
  <table>
    <thead>
      <tr>
        <th>Height</th><th>State</th><th>Block hash</th><th>Prev hash</th>
        <th>Tx</th><th>LCCoins</th><th>Energy entries</th><th>Carbon entries</th>
      </tr>
    </thead>
    <tbody>
      {table_rows}
    </tbody>
  </table>
</body>
</html>
"""
    report_path = output_dir / "ledger_report.html"
    report_path.write_text(report, encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize a simulated TECS-Chain ledger JSON.")
    parser.add_argument("ledger", help="Path to ledger_ep*.json exported by scripts/export_ledger.py.")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    apply_publication_style()

    ledger_path = Path(args.ledger)
    output_dir = Path(args.output_dir) if args.output_dir else ledger_path.parent / "visuals"
    output_dir.mkdir(parents=True, exist_ok=True)

    ledger = _read_ledger(ledger_path)
    rows = _block_rows(ledger)
    _write_csv(output_dir / "block_receipts.csv", rows)

    chain_fig = output_dir / "chain_overview.png"
    balances_fig = output_dir / "final_balances.png"
    receipts_fig = output_dir / "receipt_metrics.png"
    _plot_chain_overview(rows, chain_fig)
    _plot_balances(ledger, balances_fig)
    _plot_receipt_metrics(rows, receipts_fig)
    report_path = _write_html_report(
        ledger_path,
        output_dir,
        rows,
        chain_fig,
        balances_fig,
        receipts_fig,
    )

    print(f"report={report_path}")
    print(f"chain_overview={chain_fig}")
    print(f"final_balances={balances_fig}")
    print(f"receipt_metrics={receipts_fig}")
    print(f"block_receipts={output_dir / 'block_receipts.csv'}")


if __name__ == "__main__":
    sys.exit(main())
