#!/bin/bash
# ===========================================================================
# Launch IEEE 33 + IEEE 69 LC-MAPPO training in parallel
# Saves PIDs for monitoring; uses nohup to survive session restarts.
#
# Usage:
#   bash scripts/run_ieee_parallel_5000.sh [--base-dir DIR] [--episodes N]
# ===========================================================================
set -e

# Configurable via arguments or environment
BASE_DIR="${BASE_DIR:-outputs/ieee_benchmark}"
EPISODES="${EPISODES:-5000}"

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --base-dir) BASE_DIR="$2"; shift 2 ;;
        --episodes) EPISODES="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; shift ;;
    esac
done

PYTHON="${PYTHON:-$(which python3 || which python)}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SCRIPT="$SCRIPT_DIR/run_multiseed_experiments.py"
BASE_OUT="$PROJECT_DIR/$BASE_DIR"
CONFIG33="${BASE_OUT}/benchmark_configs/ieee33bw.yaml"
CONFIG69="${BASE_OUT}/benchmark_configs/ieee69.yaml"
LOG_DIR="${BASE_OUT}/parallel_logs"
PID_DIR="${BASE_OUT}/parallel_pids"

# ── Prepare directories ────────────────────────────────────────────
mkdir -p "$LOG_DIR" "$PID_DIR"
rm -rf "${BASE_OUT}/benchmark_ieee33bw" "${BASE_OUT}/benchmark_ieee69"

# Write a timestamped metadata file
echo "start_time=$(date '+%Y-%m-%dT%H:%M:%S%z')" > "${PID_DIR}/metadata.txt"
echo "ieee33_episodes=$EPISODES" >> "${PID_DIR}/metadata.txt"
echo "ieee69_episodes=$EPISODES" >> "${PID_DIR}/metadata.txt"
echo "config33=${CONFIG33}" >> "${PID_DIR}/metadata.txt"
echo "config69=${CONFIG69}" >> "${PID_DIR}/metadata.txt"

# ── Launch IEEE 33 ─────────────────────────────────────────────────
echo "[$(date '+%H:%M:%S')] Launching IEEE 33 ($EPISODES episodes)..."
nohup "$PYTHON" "$SCRIPT" \
    --config "$CONFIG33" \
    --episodes "$EPISODES" \
    --eval-episodes 50 \
    --seeds 7 \
    --variants tecsf \
    --output-dir "${BASE_OUT}/benchmark_ieee33bw" \
    --device cpu \
    --jobs 1 \
    > "${LOG_DIR}/ieee33.log" 2>&1 &
PID33=$!
echo "$PID33" > "${PID_DIR}/ieee33.pid"
echo "[$(date '+%H:%M:%S')] IEEE 33 launched — PID=${PID33}"

# ── Launch IEEE 69 ─────────────────────────────────────────────────
echo "[$(date '+%H:%M:%S')] Launching IEEE 69 ($EPISODES episodes)..."
nohup "$PYTHON" "$SCRIPT" \
    --config "$CONFIG69" \
    --episodes "$EPISODES" \
    --eval-episodes 50 \
    --seeds 7 \
    --variants tecsf \
    --output-dir "${BASE_OUT}/benchmark_ieee69" \
    --device cpu \
    --jobs 1 \
    > "${LOG_DIR}/ieee69.log" 2>&1 &
PID69=$!
echo "$PID69" > "${PID_DIR}/ieee69.pid"
echo "[$(date '+%H:%M:%S')] IEEE 69 launched — PID=${PID69}"

# ── Summary ────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "  Both jobs launched successfully"
echo "  IEEE 33  PID = ${PID33}  |  log = ${LOG_DIR}/ieee33.log"
echo "  IEEE 69  PID = ${PID69}  |  log = ${LOG_DIR}/ieee69.log"
echo "  PID files in ${PID_DIR}/"
echo "=========================================="
