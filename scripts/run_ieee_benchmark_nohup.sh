#!/bin/bash
# IEEE 33/69 benchmark runner - survives session restarts via nohup
# Usage: nohup bash scripts/run_ieee_benchmark_nohup.sh > benchmark_run.log 2>&1 &
#
# Configurable via environment variables:
#   BASE_DIR   - base output directory (default: outputs/ieee_benchmark)
#   EPISODES   - number of episodes (default: 200)
#   PYTHON     - Python interpreter path (default: auto-detect)

set -e

BASE_DIR="${BASE_DIR:-outputs/ieee_benchmark}"
EPISODES="${EPISODES:-200}"

PYTHON="${PYTHON:-$(which python3 || which python)}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_DIR="$PROJECT_DIR/$BASE_DIR/benchmark_configs"
OUT_DIR="$PROJECT_DIR/$BASE_DIR"
SCRIPTS="$SCRIPT_DIR"

echo "=== BENCHMARK RUN STARTED $(date) ==="
echo "  BASE_DIR  = $BASE_DIR"
echo "  EPISODES  = $EPISODES"
echo "  PYTHON    = $PYTHON"

# --- IEEE 33 ---
echo "[1/2] Starting IEEE 33 (ieee33bw) $EPISODES episodes..."
rm -rf "$OUT_DIR/benchmark_ieee33bw"
"$PYTHON" "$SCRIPTS/run_multiseed_experiments.py" \
  --config "$CONFIG_DIR/ieee33bw.yaml" \
  --episodes "$EPISODES" --eval-episodes 20 --seeds 7 --variants tecsf \
  --output-dir "$OUT_DIR/benchmark_ieee33bw" --device cpu --jobs 1

echo "[1/2] IEEE 33 DONE at $(date)"

# --- IEEE 69 ---
echo "[2/2] Starting IEEE 69 (ieee69) $EPISODES episodes..."
rm -rf "$OUT_DIR/benchmark_ieee69"
"$PYTHON" "$SCRIPTS/run_multiseed_experiments.py" \
  --config "$CONFIG_DIR/ieee69.yaml" \
  --episodes "$EPISODES" --eval-episodes 20 --seeds 7 --variants tecsf \
  --output-dir "$OUT_DIR/benchmark_ieee69" --device cpu --jobs 1

echo "[2/2] IEEE 69 DONE at $(date)"
echo "=== ALL BENCHMARK RUNS COMPLETE ==="
