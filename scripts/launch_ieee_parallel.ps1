# ===========================================================================
# Launch IEEE 33 + IEEE 69 training in parallel (5000 episodes each)
# Uses Start-Process to detach from the current console session.
# ===========================================================================

$PYTHON = "C:\Users\zrway\.conda\envs\DP-LCRL\python.exe"
$SCRIPT = "C:\Users\zrway\Desktop\期刊论文-2\scripts\run_multiseed_experiments.py"
$BASE_OUT = "C:\Users\zrway\Desktop\期刊论文-2\outputs\ieee_benchmark_lcmappo_20260706"
$LOG_DIR = "$BASE_OUT\parallel_5000_logs"
$PID_DIR = "$BASE_OUT\parallel_5000_pids"

New-Item -ItemType Directory -Force -Path $LOG_DIR | Out-Null
New-Item -ItemType Directory -Force -Path $PID_DIR | Out-Null

# Clean old output
Remove-Item -Recurse -Force "$BASE_OUT\benchmark_ieee33bw_5000" -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force "$BASE_OUT\benchmark_ieee69_5000" -ErrorAction SilentlyContinue

# ── IEEE 33 ────────────────────────────────────────────────────────
Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Launching IEEE 33 (5000 episodes)..."
$proc33 = Start-Process -FilePath $PYTHON -ArgumentList @(
    $SCRIPT,
    "--config", "$BASE_OUT\benchmark_configs\ieee33bw.yaml",
    "--episodes", "5000",
    "--eval-episodes", "50",
    "--seeds", "7",
    "--variants", "tecsf",
    "--output-dir", "$BASE_OUT\benchmark_ieee33bw_5000",
    "--device", "cpu",
    "--jobs", "1"
) -NoNewWindow -PassThru -RedirectStandardOutput "$LOG_DIR\ieee33_5000.log" -RedirectStandardError "$LOG_DIR\ieee33_5000.log"
$proc33.Id | Out-File -FilePath "$PID_DIR\ieee33.pid" -NoNewline
Write-Host "  IEEE 33  PID = $($proc33.Id)"

# ── IEEE 69 ────────────────────────────────────────────────────────
Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Launching IEEE 69 (5000 episodes)..."
$proc69 = Start-Process -FilePath $PYTHON -ArgumentList @(
    $SCRIPT,
    "--config", "$BASE_OUT\benchmark_configs\ieee69.yaml",
    "--episodes", "5000",
    "--eval-episodes", "50",
    "--seeds", "7",
    "--variants", "tecsf",
    "--output-dir", "$BASE_OUT\benchmark_ieee69_5000",
    "--device", "cpu",
    "--jobs", "1"
) -NoNewWindow -PassThru -RedirectStandardOutput "$LOG_DIR\ieee69_5000.log" -RedirectStandardError "$LOG_DIR\ieee69_5000.log"
$proc69.Id | Out-File -FilePath "$PID_DIR\ieee69.pid" -NoNewline
Write-Host "  IEEE 69  PID = $($proc69.Id)"

Write-Host ""
Write-Host "Both jobs launched. Use scripts\monitor_ieee_training.py to check."
