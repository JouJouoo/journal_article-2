@echo off
set PYTHON=C:\Users\zrway\.conda\envs\DP-LCRL\python.exe
set SCRIPT=C:\Users\zrway\Desktop\期刊论文-2\scripts\run_multiseed_experiments.py
set BASE=C:\Users\zrway\Desktop\期刊论文-2\outputs\ieee_benchmark_lcmappo_20260706
set LOG=C:\Users\zrway\Desktop\期刊论文-2\outputs\ieee_benchmark_lcmappo_20260706\parallel_5000_logs
set PID=C:\Users\zrway\Desktop\期刊论文-2\outputs\ieee_benchmark_lcmappo_20260706\parallel_5000_pids

rem Clean old output
rmdir /s /q "%BASE%\benchmark_ieee33bw_5000" 2>nul
rmdir /s /q "%BASE%\benchmark_ieee69_5000" 2>nul

echo [%time%] Starting IEEE33 training...
start "IEEE33_Training" /MIN cmd /c "%PYTHON% %SCRIPT% --config %BASE%\benchmark_configs\ieee33bw.yaml --episodes 5000 --eval-episodes 50 --seeds 7 --variants tecsf --output-dir %BASE%\benchmark_ieee33bw_5000 --device cpu --jobs 1 > %LOG%\ieee33_5000.log 2>&1"

echo [%time%] Starting IEEE69 training...
start "IEEE69_Training" /MIN cmd /c "%PYTHON% %SCRIPT% --config %BASE%\benchmark_configs\ieee69.yaml --episodes 5000 --eval-episodes 50 --seeds 7 --variants tecsf --output-dir %BASE%\benchmark_ieee69_5000 --device cpu --jobs 1 > %LOG%\ieee69_5000.log 2>&1"

echo [%time%] Both launched.
echo Use scripts\monitor_ieee_training.py to check status.
