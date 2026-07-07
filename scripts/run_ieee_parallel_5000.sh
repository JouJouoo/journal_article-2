#!/bin/bash
# ===========================================================================
# Launch IEEE 33 + IEEE 69 LC-MAPPO training in parallel (5000 episodes each)
# Saves PIDs for monitoring; uses nohup to survive session restarts.
# ===========================================================================
set -e

PYTHON="/c/Users/zrway/.conda/envs/DP-LCRL/python.exe"
SCRIPT="C:/Users/zrway/Desktop/期刊论文-2/scripts/run_multiseed_experiments.py"
BASE_OUT="C:/Users/zrway/Desktop/期刊论文-2/outputs/ieee_benchmark_lcmappo_20260706"
CONFIG33="${BASE_OUT}/benchmark_configs/ieee33bw.yaml"
CONFIG69="${BASE_OUT}/benchmark_configs/ieee69.yaml"
LOG_DIR="${BASE_OUT}/parallel_5000_logs"
PID_DIR="${BASE_OUT}/parallel_5000_pids"

# ── Prepare directories ────────────────────────────────────────────
mkdir -p "$LOG_DIR" "$PID_DIR"
rm -rf "${BASE_OUT}/benchmark_ieee33bw_5000" "${BASE_OUT}/benchmark_ieee69_5000"

# Write a timestamped metadata file
echo "start_time=$(date '+%Y-%m-%dT%H:%M:%S%z')" > "${PID_DIR}/metadata.txt"
echo "ieee33_episodes=5000" >> "${PID_DIR}/metadata.txt"
echo "ieee69_episodes=5000" >> "${PID_DIR}/metadata.txt"
echo "config33=${CONFIG33}" >> "${PID_DIR}/metadata.txt"
echo "config69=${CONFIG69}" >> "${PID_DIR}/metadata.txt"

# ── Launch IEEE 33 ─────────────────────────────────────────────────
echo "[$(date '+%H:%M:%S')] Launching IEEE 33 (5000 episodes)..."
nohup "$PYTHON" "$SCRIPT" \
    --config "$CONFIG33" \
    --episodes 5000 \
    --eval-episodes 50 \
    --seeds 7 \
    --variants tecsf \
    --output-dir "${BASE_OUT}/benchmark_ieee33bw_5000" \
    --device cpu \
    --jobs 1 \
    > "${LOG_DIR}/ieee33_5000.log" 2>&1 &
PID33=$!
echo "$PID33" > "${PID_DIR}/ieee33.pid"
echo "[$(date '+%H:%M:%S')] IEEE 33 launched — PID=${PID33}"

# ── Launch IEEE 69 ─────────────────────────────────────────────────
echo "[$(date '+%H:%M:%S')] Launching IEEE 69 (5000 episodes)..."
nohup "$PYTHON" "$SCRIPT" \
    --config "$CONFIG69" \
    --episodes 5000 \
    --eval-episodes 50 \
    --seeds 7 \
    --variants tecsf \
    --output-dir "${BASE_OUT}/benchmark_ieee69_5000" \
    --device cpu \
    --jobs 1 \
    > "${LOG_DIR}/ieee69_5000.log" 2>&1 &
PID69=$!
echo "$PID69" > "${PID_DIR}/ieee69.pid"
echo "[$(date '+%H:%M:%S')] IEEE 69 launched — PID=${PID69}"

# ── Summary ────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "  Both jobs launched successfully"
echo "  IEEE 33  PID = ${PID33}  |  log = ${LOG_DIR}/ieee33_5000.log"
echo "  IEEE 69  PID = ${PID69}  |  log = ${LOG_DIR}/ieee69_5000.log"
echo "  PID files in ${PID_DIR}/"
echo "=========================================="
