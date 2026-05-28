# TECSF

This repository implements a reproducible Python prototype for the Trusted
Energy-Carbon Co-Settlement Feedback Framework described in the project notes.

The first version is intentionally local and lightweight:

- off-chain P2P energy-carbon clearing with double auction matching;
- linear radial-network feasibility checks and repair;
- carbon responsibility, carbon allowance purchase, low-carbon contribution sale;
- local TECS-Chain settlement simulator with atomic state transitions;
- Lccoins minted only from settled records;
- settlement-feedback recurrent MAPPO training in PyTorch;
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

Export a simulated TECS-Chain ledger from a trained checkpoint:

```powershell
& "C:\Users\zrway\.conda\envs\DP-LCRL\python.exe" scripts\export_ledger.py outputs\tecsf_only_fixed_env_1000_20260527_151553\tecsf_checkpoint.pt --episodes 1 --output-dir outputs\ledger_export
```

The exported ledger contains blocks, settlement transactions, execution receipts,
state roots, event logs, final balances, and Lccoins mint records.

## Main Variants

- `tecsf`: full settlement-feedback framework.
- `no_chain`: bypasses TECS-Chain atomic settlement.
- `no_lccoins`: keeps settlement but removes Lccoins reward.
- `no_feedback`: removes previous confirmed settlement feedback from observations.
- `mappo`: uses the same environment without recurrent hidden-state feedback.
- `no_lagrange`: removes dynamic Lagrange penalties.
- `heuristic`: deterministic non-learning baseline.

## Project Layout

- `configs/default.yaml`: default synthetic scenario and training settings.
- `src/tecsf/`: implementation modules.
- `scripts/`: command-line entry points.
- `tests/`: unit and smoke tests.
