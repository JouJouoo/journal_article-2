# TECSF

This repository implements a reproducible Python prototype for the Trusted
Energy-Carbon Co-Settlement Feedback Framework described in the project notes.

The first version is intentionally local and lightweight:

- off-chain P2P energy-carbon clearing with double auction matching;
- linear radial-network feasibility checks and repair;
- carbon responsibility, carbon allowance purchase, low-carbon contribution sale;
- local TECS-Chain settlement simulator with atomic state transitions;
- LCCoins minted only from settled records;
- low-carbon-asset-aware MAPPO training in PyTorch;
- synthetic scenario generation for out-of-the-box experiments.

## Quick Start

Use the conda Python that is already available on this machine:

```powershell
& "C:\Users\zrway\.conda\envs\DP-LCRL\python.exe" -m pytest
& "C:\Users\zrway\.conda\envs\DP-LCRL\python.exe" scripts\train.py --episodes 3 --output-dir outputs\smoke
```

Run the default experiment variants:

```powershell
& "C:\Users\zrway\.conda\envs\DP-LCRL\python.exe" scripts\run_experiments.py --episodes 5 --output-dir outputs\experiments
```

Run the formal multi-seed baseline/ablation suite with automatic CUDA fallback:

```powershell
& "C:\Users\zrway\.conda\envs\DP-LCRL\python.exe" scripts\run_multiseed_experiments.py --episodes 1000 --eval-episodes 20 --device auto --jobs 1 --output-dir outputs\formal_multiseed
```

If multiple GPUs are available, increase `--jobs`; workers are assigned to
`cuda:<index>` round-robin. On a single GPU, keep `--jobs 1` unless GPU memory
has been checked.

Run the targeted experiment suites:

```powershell
& "C:\Users\zrway\.conda\envs\DP-LCRL\python.exe" scripts\run_lccoins_sensitivity.py --device auto --output-dir outputs\lccoins_sensitivity
& "C:\Users\zrway\.conda\envs\DP-LCRL\python.exe" scripts\run_network_stress.py --device auto --output-dir outputs\network_stress
& "C:\Users\zrway\.conda\envs\DP-LCRL\python.exe" scripts\run_system_stress.py --device auto --output-dir outputs\system_stress
& "C:\Users\zrway\.conda\envs\DP-LCRL\python.exe" scripts\run_scalability_experiment.py --device auto --output-dir outputs\scalability
& "C:\Users\zrway\.conda\envs\DP-LCRL\python.exe" scripts\run_settlement_stress.py --output-dir outputs\settlement_stress
```

Create figures from any generated `summary.json`:

```powershell
& "C:\Users\zrway\.conda\envs\DP-LCRL\python.exe" scripts\plot_experiment_results.py outputs\formal_multiseed\summary.json
& "C:\Users\zrway\.conda\envs\DP-LCRL\python.exe" scripts\plot_settlement_stress.py outputs\settlement_stress\summary.json
```

Create publication-oriented paper figures from an already completed report
directory. This does not re-run training or evaluation; it only redraws figures
from existing `summary.json`, metrics, statistics, and Pareto outputs:

```powershell
& "C:\Users\zrway\.conda\envs\DP-LCRL\python.exe" scripts\plot_paper_figures.py outputs\report_experiments_20260601_fixed --output-dir outputs\report_experiments_20260601_fixed\paper_figures
```

The paper-figure exporter writes `fig1_main_comparison`,
`fig2_safety_stress`, `fig3_lccoins_sensitivity`, and
`fig4_generalization_scalability` as PDF, SVG, and 600 DPI PNG files.

Run paired statistical analysis for a completed multi-seed suite:

```powershell
& "C:\Users\zrway\.conda\envs\DP-LCRL\python.exe" scripts\analyze_experiment_statistics.py outputs\formal_multiseed\summary.json --baseline tecsf
& "C:\Users\zrway\.conda\envs\DP-LCRL\python.exe" scripts\analyze_pareto_front.py outputs\formal_multiseed\summary.json
```

Run the improved full-suite orchestrator. Use `--quick` for code-level
end-to-end smoke validation; omit it for the full 1000-episode paper run.
Pass `--benchmark-cases ieee33bw ieee69` to add strict standard distribution
benchmark profiles to the suite. If the core paper suites have already been
run, use `--benchmark-only` to add only the IEEE benchmark evidence without
re-running the earlier grids:

```powershell
& "C:\Users\zrway\.conda\envs\DP-LCRL\python.exe" scripts\run_improved_experiment_suite.py --quick --benchmark-cases ieee33bw ieee69 --device cpu --jobs 1 --output-dir outputs\improved_suite_quick
& "C:\Users\zrway\.conda\envs\DP-LCRL\python.exe" scripts\run_improved_experiment_suite.py --benchmark-cases ieee33bw ieee69 --device cpu --jobs 3 --output-dir outputs\improved_suite_1000
& "C:\Users\zrway\.conda\envs\DP-LCRL\python.exe" scripts\run_improved_experiment_suite.py --benchmark-only --benchmark-cases ieee33bw ieee69 --episodes 1000 --eval-episodes 20 --seeds 7 42 100 2026 3407 --device cpu --jobs 3 --output-dir outputs\report_experiments_20260601_fixed
```

The protocol split in `configs/experiment_protocol.yaml` separates calibration,
validation, and locked final-test seeds/scenarios. Use calibration only for
hyperparameter exploration, validation for model selection, and final_test for
paper evidence.

External load/PV/network profiles can be supplied through
`scenario.profile_path` in the config. The `.npz` file must contain `load` and
`pv` arrays shaped `(agents, horizon)` and may optionally include
`agent_nodes`, `line_from`, `line_to`, `line_capacity`, prices, and carbon
factor arrays.

Generate benchmark profiles for external-validity runs. `ieee33bw` and
`ieee69` now follow a hybrid source workflow: pandapower/MATPOWER are treated
as the authoritative standard-case sources, while TECSF training and evaluation
use frozen `.npz` profiles for reproducibility. `synthetic33` is a derived
IEEE-33-style synthetic profile and must not be reported as a standard case:

```powershell
& "C:\Users\zrway\.conda\envs\DP-LCRL\python.exe" scripts\create_benchmark_profile.py --case ieee33bw --output outputs\profiles\ieee33bw_weekday.npz --day-type weekday
& "C:\Users\zrway\.conda\envs\DP-LCRL\python.exe" scripts\create_benchmark_profile.py --case ieee69 --output outputs\profiles\ieee69_weekday.npz --day-type weekday
```

Before paper-grade benchmark runs, validate the locked source tables against
local authoritative sources. IEEE 33-bus validation requires the optional
`benchmark-sources` extra for `pandapower`; IEEE 69-bus validation requires a
local MATPOWER `case69.m`:

```powershell
& "C:\Users\zrway\.conda\envs\DP-LCRL\python.exe" -m pip install ".[benchmark-sources]"
& "C:\Users\zrway\.conda\envs\DP-LCRL\python.exe" scripts\validate_benchmark_sources.py --case69-m data\case69.m --strict
```

To generate profiles directly from locally available authoritative sources, use
`--standard-source authoritative`; `--standard-source auto` attempts the
authoritative path and falls back to the locked tables when local dependencies
or `case69.m` are unavailable:

```powershell
& "C:\Users\zrway\.conda\envs\DP-LCRL\python.exe" scripts\create_benchmark_profile.py --case ieee33bw --standard-source authoritative --output outputs\profiles\ieee33bw_weekday.npz
& "C:\Users\zrway\.conda\envs\DP-LCRL\python.exe" scripts\create_benchmark_profile.py --case ieee69 --standard-source authoritative --case69-m data\case69.m --output outputs\profiles\ieee69_weekday.npz
```

Export a simulated TECS-Chain ledger from a trained checkpoint:

```powershell
& "C:\Users\zrway\.conda\envs\DP-LCRL\python.exe" scripts\export_ledger.py outputs\<run>\tecsf_checkpoint.pt --episodes 1 --output-dir outputs\ledger_export
```

The exported ledger contains blocks, settlement transactions, execution receipts,
state roots, event logs, final balances, and LCCoins mint records.

## Main Variants

- `tecsf`: full low-carbon-asset-aware MAPPO framework.
- `no_chain`: bypasses TECS-Chain atomic settlement.
- `no_lccoins`: keeps settlement but removes LCCoins reward.
- `mappo`: uses the same environment without LCCoins asset observation or recurrent policy state.
- `constrained_mappo`: Lagrangian/safety-shield MAPPO without TECS-Chain,
  LCCoins, or low-carbon asset utility, used as a constrained RL baseline.
- `safety_only`: keeps learning and safety penalties but removes TECS-Chain,
  LCCoins, and low-carbon asset utility to isolate the safety shield contribution.
- `myopic_opt`: deterministic one-step dispatch optimizer over storage and grid
  exchange, used as a lightweight optimization baseline.
- `greedy_feasible`: deterministic non-learning baseline that greedily uses
  storage and P2P bids before the same feasibility and settlement checks.
- `no_lagrange`: removes dynamic Lagrange penalties.
- `preset_low_carbon`: uses a model-internal low-carbon reward without TECS-Chain or LCCoins.
- `heuristic`: deterministic non-learning baseline.

## Project Layout

- `configs/default.yaml`: default synthetic scenario and training settings.
- `src/tecsf/`: implementation modules.
- `scripts/`: command-line entry points and dated workflow launchers.
- `tests/`: unit and smoke tests.
- `docs/`: manuscript, experiment protocol/report, and literature materials.
- `outputs/`: generated experiment artifacts. This directory is ignored by Git; keep only paper-facing result sets that still need local inspection.

## Safety and Reporting Improvements

The environment includes a configurable safety shield (`clearing.enable_safety_shield`)
that projects unsafe storage actions toward lower feeder net injections before
settlement. LCCoins training rewards can be clipped and adaptively down-weighted
after constraint violations or settlement failures while keeping ledger minting
auditable. Experiment summaries now include feasible-rate, safety-adjustment,
storage-repair, repair-iteration, rejection-reason, runtime, peak-memory, and
LCCoins reward-scale metrics.
