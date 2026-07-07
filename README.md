# 面向 P2P 低碳能源交易的低碳资产感知 MAPPO

本仓库实现了论文"面向 P2P 低碳能源交易的低碳资产感知的 MAPPO 算法"的可复现 Python 原型。

## 核心模块

- **P2P 低碳能源交易环境**：链下双向拍卖匹配、线性辐射状网络可行性检查与修复、碳责任核算与碳市场结算。
- **LCCoins 生成机制**：基于联盟链共识验证的低碳贡献可信结算与代币铸造。验证节点由市场清算、计量数据、碳核算和账本维护节点组成，对低碳贡献进行独立确认，通过 2/3 拜占庭容错阈值后触发铸造。
- **低碳资产优势分解 MAPPO**：将优势估计分解为经济优势和 LCCoins 低碳资产优势，引入低碳资产信用分配 critic 和 LCCoins 资产状态自适应 PPO 裁剪机制。
- **基于 LCCoins 动态资产效用的奖励函数**：采用 CRRA 效用函数刻画低碳资产存量效用与增量效用，使策略兼顾当前低碳收益与长期资产积累。

## 快速开始

使用本机已有的 conda Python 环境：

```powershell
& "C:\Users\zrway\.conda\envs\DP-LCRL\python.exe" -m pytest
& "C:\Users\zrway\.conda\envs\DP-LCRL\python.exe" scripts\train.py --episodes 3 --output-dir outputs\smoke
```

运行默认实验变体：

```powershell
& "C:\Users\zrway\.conda\envs\DP-LCRL\python.exe" scripts\run_experiments.py --episodes 5 --output-dir outputs\experiments
```

运行多种子正式基线/消融实验（自动 CUDA 回退）：

```powershell
& "C:\Users\zrway\.conda\envs\DP-LCRL\python.exe" scripts\run_multiseed_experiments.py --episodes 1000 --eval-episodes 20 --device auto --jobs 1 --output-dir outputs\formal_multiseed
```

多 GPU 环境可增加 `--jobs`，工作进程以轮转方式分配到 `cuda:<index>`。单 GPU 建议保持 `--jobs 1`。

## 实验套件

```powershell
& "C:\Users\zrway\.conda\envs\DP-LCRL\python.exe" scripts\run_lccoins_sensitivity.py --device auto --output-dir outputs\lccoins_sensitivity
& "C:\Users\zrway\.conda\envs\DP-LCRL\python.exe" scripts\run_network_stress.py --device auto --output-dir outputs\network_stress
& "C:\Users\zrway\.conda\envs\DP-LCRL\python.exe" scripts\run_system_stress.py --device auto --output-dir outputs\system_stress
& "C:\Users\zrway\.conda\envs\DP-LCRL\python.exe" scripts\run_scalability_experiment.py --device auto --output-dir outputs\scalability
& "C:\Users\zrway\.conda\envs\DP-LCRL\python.exe" scripts\run_settlement_stress.py --output-dir outputs\settlement_stress
```

从 `summary.json` 生成图表：

```powershell
& "C:\Users\zrway\.conda\envs\DP-LCRL\python.exe" scripts\plot_experiment_results.py outputs\formal_multiseed\summary.json
& "C:\Users\zrway\.conda\envs\DP-LCRL\python.exe" scripts\plot_settlement_stress.py outputs\settlement_stress\summary.json
```

从已有报告目录生成论文级图表（不重新训练，仅从现有数据重绘）：

```powershell
& "C:\Users\zrway\.conda\envs\DP-LCRL\python.exe" scripts\plot_paper_figures.py outputs\report_experiments_20260601_fixed --output-dir outputs\report_experiments_20260601_fixed\paper_figures
```

配对统计分析：

```powershell
& "C:\Users\zrway\.conda\envs\DP-LCRL\python.exe" scripts\analyze_experiment_statistics.py outputs\formal_multiseed\summary.json --baseline tecsf
& "C:\Users\zrway\.conda\envs\DP-LCRL\python.exe" scripts\analyze_pareto_front.py outputs\formal_multiseed\summary.json
```

改进版全量实验编排器（`--quick` 为代码级端到端冒烟验证）：

```powershell
& "C:\Users\zrway\.conda\envs\DP-LCRL\python.exe" scripts\run_improved_experiment_suite.py --quick --benchmark-cases ieee33bw ieee69 --device cpu --jobs 1 --output-dir outputs\improved_suite_quick
& "C:\Users\zrway\.conda\envs\DP-LCRL\python.exe" scripts\run_improved_experiment_suite.py --benchmark-cases ieee33bw ieee69 --device cpu --jobs 3 --output-dir outputs\improved_suite_1000
```

## 联盟链账本导出

从训练好的检查点导出联盟链账本：

```powershell
& "C:\Users\zrway\.conda\envs\DP-LCRL\python.exe" scripts\export_ledger.py outputs\<run>\tecsf_checkpoint.pt --episodes 1 --output-dir outputs\ledger_export
```

导出的账本包含区块、结算交易、执行回执、状态根、事件日志、最终余额、LCCoins 铸造记录和验证节点确认信息。

## 标准算例

`ieee33bw` 和 `ieee69` 采用混合来源工作流：pandapower/MATPOWER 作为权威标准算例来源，训练和评估使用冻结的 `.npz` profile 以保证可复现性。`synthetic33` 为派生的 IEEE-33 风格合成 profile，不能报告为标准算例。

```powershell
& "C:\Users\zrway\.conda\envs\DP-LCRL\python.exe" scripts\create_benchmark_profile.py --case ieee33bw --output outputs\profiles\ieee33bw_weekday.npz --day-type weekday
& "C:\Users\zrway\.conda\envs\DP-LCRL\python.exe" scripts\validate_benchmark_sources.py --case69-m data\case69.m --strict
```

## 实验变体

| 变体名 | 说明 |
|--------|------|
| `tecsf` / `lc_mappo` | 完整低碳资产感知 MAPPO（论文主方法） |
| `no_chain` | 绕过联盟链原子结算 |
| `no_lccoins` | 保留结算但不铸造 LCCoins |
| `mappo` | 标准 MAPPO，无 LCCoins 资产观测和循环网络 |
| `constrained_mappo` | 拉格朗日/安全屏蔽 MAPPO，无联盟链和 LCCoins |
| `safety_only` | 保留学习和安全惩罚，去除联盟链和 LCCoins |
| `no_lagrange` | 去除动态拉格朗日惩罚 |
| `preset_low_carbon` | 使用模型内生低碳奖励，不经联盟链确认 |
| `myopic_opt` | 确定性单步调度优化基线 |
| `greedy_feasible` | 确定性贪心可行基线 |
| `heuristic` | 确定性启发式基线 |

> 变体名 `tecsf` 和 `lc_mappo` 均等价于论文中的"低碳资产感知的 MAPPO 算法"，保留 `tecsf` 仅为向后兼容已有实验输出。

## 目录结构

```
configs/      默认配置和实验协议
data/         标准算例数据（如 case69.m）
docs/         论文、实验协议、文献材料
outputs/      实验产物（Git 忽略，仅保留论文级结果集）
scripts/      命令行入口和实验编排脚本
src/tecsf/    核心实现模块
tests/        单元测试和冒烟测试
```

## 安全与报告

环境包含可配置安全屏蔽（`clearing.enable_safety_shield`），在结算前将不安全的储能动作投影到更低的馈线净注入。LCCoins 训练奖励可裁剪并在约束违反或结算失败后自适应降权，同时保持账本铸造的可审计性。实验摘要包含可行率、安全调整、储能修复、拒绝原因、运行时间、峰值内存和 LCCoins 奖励尺度等指标。
