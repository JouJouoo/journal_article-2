# 低碳资产感知 MAPPO 代码对齐计划

## Summary
- 任务类别：论文内容驱动的代码改造；不启动正式实验，不改论文稿，不提交 Git。
- 目标：把现有 `tecsf` 主算法对齐为论文中的低碳资产感知 MAPPO：`LCCoins` 链上累计资产进入观测、奖励、优势分解、信用分配和 PPO 自适应裁剪。
- 保留现有 P2P 清算、储能约束、碳核算、可行性投影和现有消融脚本入口；新增/调整变体用于论文对照。

## Key Changes
- 配置与变体：
  - 在 `ExperimentConfig` 中新增论文参数：清洁电量权重、碳减排权重、铸币系数、CRRA `rho/B0`、存量/增量效用权重、验证节点数、共识阈值、`clip_eps_min/max`。
  - 默认让主算法 `tecsf` 等价于论文版 `lc_mappo`；保留旧名称兼容脚本。
  - 给 `VariantSpec` 增加开关：是否使用资产观测、资产效用、双优势分解、信用分配、自适应裁剪。

- 环境与链上资产：
  - 将 `LCCoins` 余额默认纳入 observation，并加入论文中的 P2P 参考价信号；`no_lccoins`/标准 `mappo` 不使用该资产状态。
  - 将 `CarbonResult.lccoins_candidate` 改为论文公式：`C = omega_e * E_clean + omega_c * R_co2`，`M = chi * lambda * C`。
  - 在 `SimulatedTECSChain` 中加入联盟链式验证节点和 `ceil(2V/3)` 阈值共识；只有通过共识的记录才铸造 `LCCoins`。
  - `info` 暴露 `agent_reward_eco`、`agent_reward_coin`、`agent_utility_stock`、`agent_utility_increment`、`agent_lccoins_balance`、`consensus_confirmed` 等训练和统计字段。

- MAPPO 训练：
  - 将 `CentralizedCritic` 改为双价值头：`V_eco(s)` 与 `V_coin(s)`。
  - `RolloutBuffer` 存储 per-agent 经济回报、低碳资产回报、资产状态和 P2P 交易关系。
  - 分别计算 `A_eco` 与 `A_coin` 的 GAE；用信用分配网络 `h_psi(z_i,z_j,e_ij)` 生成 `kappa_ij`，得到 `bar_A_coin`。
  - 用资产门控网络 `g_omega(B_i)` 融合经济优势和低碳资产优势。
  - 用裁剪门控 `g_nu(B_i)` 生成逐智能体 `epsilon_i,t`，替代固定 `clip_eps`。
  - 检查点保存 actor、双头 critic、信用分配网络、优势门控和裁剪门控参数。

## Public Interfaces
- `train(..., variant="tecsf")` 继续可用，但其语义变为论文版低碳资产感知 MAPPO。
- 新增推荐入口：`variant="lc_mappo"`。
- 旧 checkpoint 可能因 observation 维度和 critic 输出变化不兼容；实现中不做旧模型热加载迁移。
- `configs/default.yaml` 会更新为论文版默认配置；旧对照通过变体开关控制，而不是靠默认关闭核心机制。

## Test Plan
- 新增/更新单元测试：
  - LCCoins 公式：清洁电量、碳减排、铸币量与非负约束。
  - 链上共识：阈值通过才铸币，失败不更新余额。
  - 环境：observation 维度、余额更新、CRRA 存量/增量效用、reward component 字段。
  - MAPPO：双价值头输出、优势分解、信用权重归一化、自适应 clip 范围。
  - 训练 smoke：`lc_mappo` 1 episode 可跑通并写出 checkpoint/metrics。
- 验证命令：
  - `cd 实验代码 && PYTHONPATH=src python3 -m pytest tests/test_env.py tests/test_chain.py tests/test_training_smoke.py tests/test_metrics.py`
  - 若时间允许，再跑 `cd 实验代码 && PYTHONPATH=src python3 -m pytest`
  - smoke 输出写到 `/private/tmp`，不写入论文实验输出目录。
- 不运行 `100/200 episode` 或 `1000 episode`；如后续要做论文实验，先补 1 页实验协议并按项目规则推进。

## Assumptions
- 这次只改 `实验代码`，不改 `论文文稿`、`参考文献`、已有 `outputs`。
- 主算法按用户选择采用“完整算法”实现，不做最小补丁。
- 当前目录不是 Git 仓库；后续若需要提交，需先确认真实仓库位置和提交范围。
