@echo off
cd /d "%~dp0\.."
if not exist outputs\full_experiments_20260703_no_ablation_logs mkdir outputs\full_experiments_20260703_no_ablation_logs
"C:\Users\zrway\.conda\envs\DP-LCRL\python.exe" scripts\run_full_experiments_20260703.py --device cuda --jobs 3 %* >> outputs\full_experiments_20260703_no_ablation_logs\cmd_launcher.log 2>&1
