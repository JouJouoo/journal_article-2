#!/bin/bash
# IEEE 33/69 benchmark runner - survives session restarts via nohup
# Usage: nohup bash scripts/run_ieee_benchmark_nohup.sh > benchmark_run.log 2>&1 &

PYTHON="/c/Users/zrway/.conda/envs/DP-LCRL/python.exe"
ROOT="C:/Users/zrway/Desktop/期刊论文-2"
CONFIG_DIR="$ROOT/outputs/ieee_benchmark_lcmappo_20260706/benchmark_configs"
OUT_DIR="$ROOT/outputs/ieee_benchmark_lcmappo_20260706"
SCRIPTS="$ROOT/scripts"

echo "=== BENCHMARK RUN STARTED $(date) ==="

# --- IEEE 33 ---
echo "[1/2] Starting IEEE 33 (ieee33bw) 200 episodes..."
rm -rf "$OUT_DIR/benchmark_ieee33bw"
"$PYTHON" "$SCRIPTS/run_multiseed_experiments.py" \
  --config "$CONFIG_DIR/ieee33bw.yaml" \
  --episodes 200 --eval-episodes 20 --seeds 7 --variants tecsf \
  --output-dir "$OUT_DIR/benchmark_ieee33bw" --device cpu --jobs 1

echo "[1/2] IEEE 33 DONE at $(date)"

# --- IEEE 69 ---
echo "[2/2] Starting IEEE 69 (ieee69) 200 episodes..."
rm -rf "$OUT_DIR/benchmark_ieee69"
"$PYTHON" "$SCRIPTS/run_multiseed_experiments.py" \
  --config "$CONFIG_DIR/ieee69.yaml" \
  --episodes 200 --eval-episodes 20 --seeds 7 --variants tecsf \
  --output-dir "$OUT_DIR/benchmark_ieee69" --device cpu --jobs 1

echo "[2/2] IEEE 69 DONE at $(date)"
echo "=== ALL BENCHMARK RUNS COMPLETE ==="
