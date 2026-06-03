# TECSF 论文实验方案精简优化建议

生成日期：2026-05-31
依据代码：当前工作区最新代码
依据结果：`outputs/report_experiments_20260528_1000/TECSF_1000episode实验报告.md`

## 1. 总体判断

实验方案不宜继续扩成“大而全”的工程验证。对当前 TECSF 论文而言，核心目标是证明：

1. TECSF 能形成可审计的电碳协同结算闭环。
2. LCCoins 和结算反馈确实参与策略学习。
3. 拉格朗日约束和结算拒绝机制能提高安全可行性。
4. 方法能迁移到 IEEE 33/69 标准配电网 profile，但不声称成本和碳排全面最优。

因此，最终论文实验只需要保留能支撑上述四点的最小充分实验包。复杂的大规模系统压力、真实多天气年度数据、更多优化器/MILP 基线、32 agents/33 nodes 扩展等，可以作为后续工作或补充实验，不应成为当前投稿版本的主线。

## 2. 建议保留的最小实验包

| 实验 | 是否保留 | 目的 | 最小设置 |
| --- | --- | --- | --- |
| 主对比与核心消融 | 必须保留 | 证明完整框架与关键模块的作用 | TECSF、MAPPO、constrained_mappo、myopic_opt、heuristic、no_lccoins、no_feedback、no_lagrange |
| LCCoins 敏感性 | 保留但压缩 | 说明奖励权重不是任意设定 | 只扫 `kappa={0,0.1,0.2,0.5}`，重点展示 TECSF 与 no_lccoins |
| 网络约束压力 | 保留但压缩 | 证明安全机制在紧约束下有必要 | 只保留 `line=0.7, trade=1` 和 `line=0.5, trade=1` 两个代表场景 |
| TECS-Chain 结算压力 | 必须保留 | 支撑可信结算、拒绝、回滚、重复铸造防护 | 6 类 case x 5 seeds 即可 |
| Ledger 案例 | 保留为图示 | 让审稿人看到 receipt、block、余额、LCCoins 记录 | 1 个 TECSF checkpoint，24 时段 |
| IEEE 33/69 标准算例 | 必须保留 | 提供外部有效性边界 | TECSF、myopic_opt、heuristic，5 seeds |
| 可扩展性实验 | 降级为简短边界分析 | 不主张强可扩展性，只说明当前边界 | 保留 8/16 agents、5/9 nodes；不再扩到 32 agents |
| 系统压力大网格 | 暂不作为主文必需 | 覆盖真实扰动，但成本高且不直接支撑核心贡献 | 可放后续工作或补充材料 |

这个组合已经足够支撑一篇方法型学术论文。不要再把实验主线扩成“所有场景、所有参数、所有基线都验证”。

## 3. 当前已有结果应如何使用

当前 `outputs/report_experiments_20260528_1000/` 已经覆盖主对比、LCCoins 敏感性、网络压力、可扩展性、结算压力、ledger 案例和 IEEE 33/69 标准算例，数量上已经足够。

但需要注意：最新代码已经补入若干修正，包括：

1. 动态动作边界和 storage safety projection。
2. LCCoins 自适应奖励权重与裁剪。
3. `system_social_cost` 与 `participant_payment_cost` 的成本口径拆分。
4. `myopic_opt`、`constrained_mappo`、`safety_only` 等更强基线。
5. 统计检验、Pareto、验收门槛和 IEEE benchmark profile 脚本。

因此，旧报告可以作为“问题诊断”和“初版结果”，但如果论文要写最终数值，建议只重跑一个精简最终包，不必重跑所有扩展网格。

## 4. 建议最终重跑的精简包

### E1. 主对比与消融

保留 5 seeds、1000 episodes、20 eval episodes。方法建议为：

```text
tecsf
mappo
constrained_mappo
myopic_opt
heuristic
no_lccoins
no_feedback
no_lagrange
```

不必再把所有可选变体都放入主表。`safety_only` 可作为补充表或消融附录。

必须报告：

```text
system_social_cost
participant_payment_cost
grid_carbon_emission
net_carbon_allowance_need
lccoins
mean_reward
settlement_success_rate
feasible_rate
max_violation
runtime
```

### E2. LCCoins 敏感性

只保留 `kappa={0,0.1,0.2,0.5}`。主文中不需要展示所有组合参数，只回答一个问题：LCCoins 权重如何影响收益和安全。

如果空间有限，主文放 TECSF 表格，no_lccoins 作为参照线即可。

### E3. 网络约束压力

只保留两个代表场景：

```text
line=0.7, trade=1.0    中等紧约束
line=0.5, trade=1.0    最紧约束
```

方法保留：

```text
tecsf
no_lagrange
heuristic 或 myopic_opt
```

这已经足以证明：去掉约束机制会严重失效，而 TECSF 在紧约束下有安全收益和成本代价。

### E4. TECS-Chain 结算压力与 ledger 案例

保留现有 6 类 case：

```text
正常结算
哈希篡改
约束越限
执行失败回滚
重复铸造
审计一致性
```

这部分不需要扩大。只要报告 30/30 passed、回滚误差为 0，并给出 ledger 图即可。

### E5. IEEE 33/69 标准算例

保留 IEEE 33-bus 和 IEEE 69-bus。方法只需：

```text
tecsf
myopic_opt
heuristic
```

这部分的论文作用是外部有效性和边界说明，不是证明 TECSF 支配优化基线。因此结论应写成：

```text
TECSF 的可信结算和 LCCoins 反馈链路能够迁移到标准配电网 profile；
但在成本和碳排上不全面优于 myopic_opt。
```

## 5. 可以降级或删除的复杂项

以下内容不建议作为当前论文主文必需实验：

| 项目 | 建议 | 原因 |
| --- | --- | --- |
| 32 agents / 17 或 33 nodes | 暂不做主文 | 当前 16 agents / 9 nodes 已暴露边界，继续扩展只会拉长工作量 |
| 多天气、多季节、真实全年时间序列 | 后续工作 | 对外部有效性有帮助，但不是本文机制证明的必要条件 |
| 完整 MILP/MIQP/AC-OPF 基线 | 可选补充 | 实现成本高，且会把论文重心转成调度最优性比较 |
| 大量 carbon price、PV penetration、storage capacity 网格 | 附录或后续 | 容易稀释主线，当前 LCCoins 与网络压力已足够说明敏感性 |
| 复杂攻击模型、隐私保护、共识机制对比 | 不做 | 当前 TECS-Chain 是本地可审计状态机，不主张真实链部署或隐私协议 |
| 过多统计方法 | 保留最小统计 | 95% CI、配对检验、Holm 校正即可，不需要复杂贝叶斯或多层模型 |

## 6. 论文结论边界

当前论文最稳妥的结论不是：

```text
TECSF 在经济成本和碳排放上全面优于所有方法。
```

而应该是：

```text
TECSF 能在 P2P 电碳交易中提供可审计的低碳激励和可信结算闭环；
相比去掉 LCCoins 或拉格朗日约束的消融模型，TECSF 提高了结算安全性、约束可行性和低碳反馈完整性；
标准 IEEE 33/69 算例进一步表明该机制可迁移到公开配电网 profile，但成本和碳排优势并不全面成立。
```

这个口径更符合现有数据，也更不容易被审稿人抓住“成本不最优”“myopic_opt 更低碳”的问题。

## 7. 最小验收标准

最终论文实验只需要满足以下门槛：

1. 主对比中 TECSF 的 `settlement_success_rate >= 0.95`，`max_violation <= 0.1`。
2. no_lagrange 在紧约束或主实验中表现出明显安全退化，用于证明约束机制必要。
3. no_lccoins 或 no_feedback 相比 TECSF 在回报、LCCoins 或反馈完整性上有可解释差异。
4. TECS-Chain 压力测试全部通过，回滚误差为 0。
5. IEEE 33/69 至少证明 TECSF 可运行、可结算、可审计，并如实报告弱项。
6. 所有“显著优于”的表述必须有配对统计或置信区间支持；没有统计支持的指标只写“相当”“接近”或“未观察到优势”。

## 8. 推荐执行顺序

1. 先用当前最新代码跑 quick suite，确认脚本和指标没有断裂。
2. 重跑 E1 主对比与消融，生成主表。
3. 重跑 E5 IEEE 33/69，确认动作投影和成本口径修正后的标准算例表现。
4. 只重跑压缩版 E2/E3，不跑完整大网格。
5. 复用或轻量重跑 E4 结算压力和 ledger 案例。
6. 最后生成统计检验和 Pareto/feasible-gate 标记，用于约束论文表述。

这样可以把实验工作控制在论文需要的范围内，同时保留足够的证据链。
