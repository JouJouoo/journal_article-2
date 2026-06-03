# TECSF 1000-episode 修复版实验报告

## 1. 运行配置

本报告只使用 `outputs/report_experiments_20260601_fixed/` 下的新结果。正式实验命令为：

```powershell
& 'C:\Users\zrway\.conda\envs\DP-LCRL\python.exe' -u scripts\run_improved_experiment_suite.py --episodes 1000 --eval-episodes 20 --seeds 7 42 100 2026 3407 --device cpu --jobs 40 --benchmark-cases ieee33bw ieee69 --output-dir outputs\report_experiments_20260601_fixed
```

重跑时设置 `OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`、`OPENBLAS_NUM_THREADS=1`、`NUMEXPR_NUM_THREADS=1` 和 `TORCH_NUM_THREADS=1`，用于提高多进程 CPU 利用率并降低线程过度竞争。

## 2. 输出完整性

新目录包含主对比、LCCoins 敏感性、网络压力、系统压力、可扩展性、结算压力、IEEE 33-bus、IEEE 69-bus 和 ledger 审计案例。主对比、LCCoins 敏感性、网络压力、系统压力、可扩展性、IEEE 33-bus 和 IEEE 69-bus 均生成 `statistics/paired_comparisons.json` 与 `pareto/pareto_runs.json`。结算压力生成 `figures/settlement_stress_outcomes.png`，ledger 案例生成 `ledger_case/visuals/chain_overview.png`。

最终验收命令：

```powershell
& 'C:\Users\zrway\.conda\envs\DP-LCRL\python.exe' scripts\check_experiment_acceptance.py outputs\report_experiments_20260601_fixed
```

返回 `acceptance=passed`。单元测试命令 `& 'C:\Users\zrway\.conda\envs\DP-LCRL\python.exe' -m pytest` 返回 `33 passed in 14.48 s`。

## 3. 成本口径

本报告保留 `system_cost` 作为社会成本口径；在当前实现中它与 `system_social_cost` 一致，不把 P2P 内部转移支付计入社会成本。参与者支付口径由 `participant_payment_cost` 表示，P2P 内部转移规模由 `p2p_transfer_payment` 表示。默认主对比中 TECSF 的 `system_cost`、`participant_payment_cost` 和 `p2p_transfer_payment` 分别为 57.69、57.69 和 0.00；heuristic 分别为 52.38、58.79 和 6.41；myopic_opt 分别为 48.40、57.45 和 9.04。

## 4. 主对比结果

| 方法 | 系统成本 | 购电碳排 | LCCoins | 平均回报 | 结算成功率 | 最大违背 |
| --- | --- | --- | --- | --- | --- | --- |
| TECSF | 57.69 ± 0.73 | 41.41 ± 0.58 | 36.72 ± 0.08 | -0.266 ± 0.009 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| no_lccoins | 57.63 ± 0.44 | 41.35 ± 0.34 | 0.00 ± 0.00 | -0.305 ± 0.011 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| no_lagrange | 57.69 ± 0.73 | 41.41 ± 0.58 | 36.72 ± 0.08 | -0.266 ± 0.009 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| heuristic | 52.38 ± 0.00 | 38.17 ± 0.00 | 35.99 ± 0.00 | -0.228 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| myopic_opt | 48.40 ± 0.00 | 35.32 ± 0.00 | 0.00 ± 0.00 | -0.238 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 |

主对比支持 TECSF 的可信结算和低碳反馈闭环，但不支持“成本或碳排全面最优”的表述。

## 5. LCCoins 行为证据

| 方法 | 低碳电量 q_lc | 碳抵扣 | 剩余低碳售卖 | LCCoins | corr(LCCoins,q_lc) | corr(LCCoins,碳抵扣) | corr(LCCoins,候选量) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TECSF | 162.30 ± 0.00 | 25.57 ± 0.49 | 70.37 ± 0.49 | 36.72 ± 0.08 | 0.969 ± 0.001 | -0.413 ± 0.003 | 1.000 ± 0.000 |
| no_lccoins | 162.30 ± 0.00 | 25.60 ± 0.42 | 70.34 ± 0.42 | 0.00 ± 0.00 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| heuristic | 162.30 ± 0.00 | 21.18 ± 0.00 | 74.77 ± 0.00 | 35.99 ± 0.00 | 0.978 ± 0.000 | -0.406 ± 0.000 | 1.000 ± 0.000 |

LCCoins 与候选铸造量一致，并与低碳电量贡献高度相关。因此，论文中将其解释为结算锚定低碳反馈信号，而不是直接保证低成本或低碳排的奖励项。

## 6. 压力与扩展性

网络压力中 `line=0.5, trade=1` 的 TECSF 结果为系统成本 66.42 ± 2.01、购电碳排 45.19 ± 1.08、结算成功率 1.000 ± 0.000、最大违背 0.000 ± 0.000。`line=0.5, trade=1.3` 的 TECSF 结果为系统成本 68.77 ± 1.96、购电碳排 46.19 ± 0.54、结算成功率 1.000 ± 0.000、最大违背 0.000 ± 0.000。

系统压力中 TECSF 的最高系统成本场景为 `load_1p3__pv_0p7__price_1p3__carbon_2__line_0p7`，成本 306.28、购电碳排 86.57、结算成功率 1.000、最大违背 0.000。最高购电碳排场景为 `load_1p3__pv_0p7__price_1__carbon_1__line_0p7`，购电碳排 87.87、结算成功率 1.000、最大违背 0.000。

可扩展性中 16 agents / 9 nodes 的 TECSF 结果为系统成本 233.64 ± 9.67、购电碳排 77.32 ± 2.90、结算成功率 1.000 ± 0.000、最大违背 0.000 ± 0.000。32 agents / 17 nodes 的 TECSF 结果为系统成本 889.09 ± 5.17、购电碳排 88.82 ± 1.28、运行时间 8446.2 ± 68.9 s、结算成功率 1.000 ± 0.000、最大违背 0.000 ± 0.000。该场景通过可行性验收，但运行时间和成本仍是扩展性边界。

## 7. IEEE 33/69 标准算例

| 算例 | 方法 | 系统成本 | 购电碳排 | LCCoins | 结算成功率 | 最大违背 |
| --- | --- | --- | --- | --- | --- | --- |
| IEEE 33-bus | TECSF | 14.14 ± 0.00 | 10.06 ± 0.00 | 6.36 ± 0.00 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| IEEE 33-bus | myopic_opt | 14.08 ± 0.00 | 10.01 ± 0.00 | 0.00 ± 0.00 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| IEEE 69-bus | TECSF | 26.30 ± 0.01 | 19.47 ± 0.01 | 9.40 ± 0.00 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| IEEE 69-bus | myopic_opt | 19.56 ± 0.00 | 14.30 ± 0.00 | 0.00 ± 0.00 | 1.000 ± 0.000 | 0.000 ± 0.000 |

两套标准算例均通过可行性门槛。IEEE 33-bus 中 TECSF 接近但略弱于 myopic_opt；IEEE 69-bus 中 TECSF 优于 heuristic 但弱于 myopic_opt。

## 8. Ledger 审计

使用 `formal_multiseed/tecsf/seed_7/tecsf_checkpoint.pt` 导出的 ledger 案例包含 24 个 block、24 条 transaction、24/24 条 settled records，总 LCCoins 为 36.94，head hash 为 `86aedb9c549497a61b836939a2f28f4c886b0af6c7617719fa26474971535529`。可视化位于 `ledger_case/visuals/ledger_report.html` 和 `ledger_case/visuals/chain_overview.png`。

## 9. 统计与 Pareto

正式后处理已为主对比、LCCoins 敏感性、网络压力、系统压力、可扩展性、IEEE 33-bus 和 IEEE 69-bus 生成统计与 Pareto 文件。本次成对比较在 Holm 校正后均未通过 0.05 门槛，因此报告和论文按观测结果、机制证据和验收门槛解释差异。
