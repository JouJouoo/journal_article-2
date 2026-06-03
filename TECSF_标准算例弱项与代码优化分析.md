# TECSF 标准算例弱项与代码优化分析

## 1. 结论摘要

当前论文中“标准算例结果说明 TECSF 的结算与激励链路具有一定迁移能力，但在成本和碳排方面并未全面优于 myopic_opt；IEEE 33-bus 中个别 seed 出现可行性弱项”的表述是准确且必要的。进一步检查代码和逐 seed 结果后，可以确认该问题不是单纯训练轮数不足，也不是 IEEE 33-bus 标准算例本身不能运行，而是由三类因素共同造成：标准算例下动作尺度与负荷尺度不匹配，当前安全投影不能充分修复过度储能动作，系统成本指标对 P2P 内部转移支付存在口径偏差。

最直接的证据来自 `outputs/report_experiments_20260528_1000/benchmark_ieee33bw/summary.json`。IEEE 33-bus 的 TECSF 只有 seed 3407 出现明显弱项，其评估结果为：系统成本 59.66、购电碳排 34.09、结算成功率 0.9583、最大约束违背 203.52、线路违背 203.52、constraint rejection 记录数为 1。复现逐时段评估后发现，该 seed 在 20 个评估 episode 的首时段 `t=0` 都触发同一条约束拒绝；由于每个 episode 有 24 个时段，因此聚合后表现为 `23/24 = 0.9583` 的结算成功率。该时段 actor 输出几乎完全饱和，32 个 agent 的总放电功率达到约 47.95 MW，而 `t=0` 总负荷仅约 3.49 MW、总 PV 仅约 0.09 MW，导致 IEEE 33-bus 首段线路潮流约 -44.46 MW，远超 8.77 MW 的线路容量。

因此，代码层面有明确优化空间，而且优先级较高。建议先修复动作尺度和安全投影，再重新跑 IEEE 33/69 标准算例；同时修正“系统成本”统计口径，避免把 P2P 内部支付当作系统社会成本。

## 2. 证据定位

### 2.1 IEEE 33-bus 弱项集中在 seed 3407

IEEE 33-bus 的 TECSF 五个 seed 中，7、42、100、2026 四个 seed 评估阶段均无约束违背，只有 3407 失效：

| seed | 系统成本 | 购电碳排 | 结算成功率 | 最大违背 | 线路违背 | rejected records |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | 23.79 | 17.32 | 1.0000 | 0.00 | 0.00 | 0 |
| 42 | 23.78 | 17.32 | 1.0000 | 0.00 | 0.00 | 0 |
| 100 | 23.79 | 17.33 | 1.0000 | 0.00 | 0.00 | 0 |
| 2026 | 29.08 | 20.10 | 1.0000 | 0.00 | 0.00 | 0 |
| 3407 | 59.66 | 34.09 | 0.9583 | 203.52 | 203.52 | 1 |

这说明 IEEE 33-bus 的失败不是普遍性崩溃，而是某个随机初始化学到的策略在首时段产生极端动作，并且当前投影无法兜底。

### 2.2 失败时段的行为特征

复现 `seed=3407` 的 IEEE 33-bus 评估后，20 个 eval episode 的 `t=0` 都出现同一问题。首时段关键数值如下：

| 项目 | 数值 |
| --- | ---: |
| 总负荷 | 3.49 MW |
| 总 PV | 0.09 MW |
| raw action 均值绝对值 | 0.999 |
| 总购电申报 | 95.91 MW |
| 总售电申报 | 95.91 MW |
| 总充电动作 | 0.00 MW |
| 总放电动作 | 47.95 MW |
| 初始 P2P 成交 | 95.91 MW |
| 修正后 P2P 成交 | 12.96 MW |
| 最严重线路 | 0 -> 1 |
| 该线路潮流 | -44.46 MW |
| 该线路容量 | 8.77 MW |
| 总线路越限 | 203.52 |

这里的关键不是 P2P 成交本身，而是过度储能放电形成了巨大反向净注入。当前代码会缩减 P2P 成交，但在当前物理潮流建模中，P2P 交易经电网购售电兜底后对节点物理净注入会抵消，单纯缩小 P2P 矩阵不能修复线路潮流。

## 3. 代码层面的主要问题

### 3.1 标准算例动作尺度与负荷尺度不匹配

默认配置来自小规模合成场景：`max_buy_power=3.0`、`max_sell_power=3.0`、`max_discharge_power=1.5`。这些参数在 `configs/default.yaml` 和 `outputs/report_experiments_20260528_1000/benchmark_configs/ieee33bw.yaml` 中保持不变。但 IEEE 33-bus profile 中每个 bus 的标准负荷通常只有几十 kW 到数百 kW，首时段单 agent 负荷可低至约 0.04-0.10 MW。

这导致一个明显不合理的动作空间：每个 agent 可以申报 3 MW 买卖功率和 1.5 MW 储能放电，而其本地负荷可能只有 0.05 MW 量级。对于 32 个 agent，总放电上限接近 48 MW，而 IEEE 33-bus 首段线路容量约 8.77 MW。TECSF actor 一旦饱和，就能产生远超标准配电网容量的物理注入。

相关代码位置：

| 问题 | 位置 |
| --- | --- |
| 动作缩放直接使用全局 `max_buy_power/max_sell_power/max_discharge_power` | `src/tecsf/market.py:88` |
| benchmark config 继承默认动作上限 | `outputs/report_experiments_20260528_1000/benchmark_configs/ieee33bw.yaml:23` |
| 标准 profile 负荷来自 IEEE bus load，未同步缩放动作上限 | `scripts/create_benchmark_profile.py:91` |

### 3.2 安全储能投影不能抑制过度放电

`_storage_safety_dispatch` 的目标是根据本地净负荷调整储能动作。当某 agent 为净负荷时，它会将放电提高到至少覆盖净负荷；当某 agent 为净剩余时，它会将充电提高到吸收剩余电量。然而当前逻辑不会把已经过大的放电压低到本地净负荷以内。对于 IEEE 33-bus seed 3407，actor 在所有 agent 上输出接近最大放电，`_storage_safety_dispatch` 认为这些 agent 属于净负荷节点，但没有削减过度放电，最终形成大规模反向潮流。

相关代码位置：

| 问题 | 位置 |
| --- | --- |
| safety dispatch 只提升 deficit 节点放电，不限制过度放电 | `src/tecsf/market.py:200` |
| SOC 限制只保证不越 SOC，不保证网络安全 | `src/tecsf/market.py:158` |
| 失败时 `storage_repair_deviation=0`，说明投影没有改变过度放电 | `benchmark_ieee33bw` seed 3407 评估记录 |

更合理的安全投影应当把 deficit 节点的放电上限限制为本地净负荷或下游线路余量，把 surplus 节点的充电上限限制为本地净剩余或上游线路余量。至少需要先做本地净负荷级别的保守裁剪：

```text
discharge_i <= max(load_i - pv_i, 0)
charge_i <= max(pv_i - load_i, 0)
```

更强的版本则应沿径向馈线计算每条线路的容量余量，按灵敏度或子树贡献进行投影。

### 3.3 P2P 修正无法修复当前物理潮流

在 `_balance_agents` 和 `_network_state` 的组合下，节点物理净注入实际等价于：

```text
pv + discharge - load - charge
```

原因是 P2P 售出、P2P 买入、电网买入和电网售出在节点注入计算中相互抵消。因此，只缩减 `p2p_power` 并不会改变实际线路潮流。当前 `clear_market` 在网络违背时会反复执行 `p2p_power *= repair_shrink`，但如果违背来自储能过度放电、PV 过剩或负荷缺口，该修正对潮流无效。

相关代码位置：

| 问题 | 位置 |
| --- | --- |
| `grid_buy/grid_sell` 兜底后 P2P 对物理注入抵消 | `src/tecsf/market.py:269` 与 `src/tecsf/market.py:284` |
| 网络违背时主要缩小 P2P 成交 | `src/tecsf/market.py:331` |
| seed 3407 中 P2P 从 95.91 修到 12.96，但线路越限仍为 203.52 | 逐时段复现结果 |

这意味着当前“交易修正”更像结算量修正，而不是物理可行性投影。论文中如果继续写“P2P 成交量修正保证配电网约束”，会有风险。代码应改为“先投影物理净注入，再修正 P2P 结算量”，或者明确 P2P 修正只处理交易匹配，不直接保证潮流约束。

### 3.4 系统成本指标可能高估 TECSF

当前 `system_cost` 计算包含 `p2p_cost.sum()`：

```python
grid_cost.sum()
+ p2p_cost.sum()
+ op_cost.sum()
+ np.maximum(carbon_cost, 0.0).sum()
+ emergency_cost.sum()
```

该口径把 P2P 买方支付计入系统成本，但没有扣除 P2P 卖方收入。对于“系统社会成本”或“系统运行成本”，P2P 支付属于参与者之间的内部转移，应该在系统层面相互抵消。保留 `p2p_cost.sum()` 会系统性惩罚 P2P 成交更多的方法，而 TECSF 正是倾向于产生更多 P2P 交易和 LCCoins 反馈的机制，因此这会削弱 TECSF 在“系统成本”上的表现。

相关代码位置：

| 问题 | 位置 |
| --- | --- |
| `system_cost` 包含 P2P 内部支付 | `src/tecsf/env.py:448` |
| agent profit 中 P2P 收益和成本是主体级真实转移 | `src/tecsf/env.py:373` |

建议将指标拆成两个：

```text
social_system_cost = grid_buy_cost + storage_op_cost + positive_carbon_cost + emergency_cost
participant_payment = p2p_cost + grid_buy_cost + positive_carbon_cost
```

论文中“系统成本”应使用前者；若使用后者，应改名为“总支付支出”或“买方总支出”。

### 3.5 拉格朗日乘子每个 episode 重置，约束记忆较弱

环境 reset 时会执行 `self.lagrange.fill(0.0)`。这意味着拉格朗日乘子只在单个 24 时段 episode 内积累，而不会跨 episode 形成持久约束价格。对于训练中的重复首时段越限，策略在每个 episode 的 `t=0` 面对的拉格朗日乘子都是 0，只能依赖固定惩罚和采样梯度进行纠正。

相关代码位置：

| 问题 | 位置 |
| --- | --- |
| reset 时清空拉格朗日乘子 | `src/tecsf/env.py:76`、`src/tecsf/env.py:89` |
| 乘子在 step 后更新，但当前时段动作已经执行 | `src/tecsf/env.py:120` |

如果想增强约束学习，应考虑跨 episode 保留或学习化约束乘子，或者至少把上一 episode 的首时段约束统计转化为 action mask / dynamic bound。

### 3.6 PPO 训练缺少动作可行性先验

当前 actor 输出 6 维连续动作，经过 `tanh` 和 `scale_raw_actions` 映射为物理动作。买电、卖电、充电、放电之间没有强制互斥，也没有基于本地负荷、PV、SOC 或线路容量的动态动作上限。虽然 `_apply_storage_arrays` 会禁止同时充放电，但不会限制“远超本地负荷的单向放电”或“同时大买大卖”的策略倾向。seed 3407 评估中，6 个动作维度几乎全部饱和，说明策略已经进入边界解。

相关代码位置：

| 问题 | 位置 |
| --- | --- |
| actor 输出后直接按全局上限缩放 | `src/tecsf/market.py:88` |
| PPO 使用 episode 级平均 reward 优势，未显式加入 per-agent safety shaping | `src/tecsf/rl/mappo.py:101` |
| deterministic evaluation 使用 actor mean，饱和策略会稳定复现 | `src/tecsf/rl/mappo.py:248` |

## 4. 为什么 myopic_opt 更优

myopic_opt 在 IEEE 33/69 标准算例上成本和碳排更低，并不说明 TECSF 的可信结算机制没有价值，而是说明当前 TECSF 学习策略在标准算例下没有学到更好的物理调度。myopic_opt 的优势来自三个实现特征：

第一，myopic_opt 是确定性局部贪心策略，不探索，不会产生极端饱和动作。第二，它在每个 agent 内部把充电限制在本地 PV 剩余内，把放电限制在本地负荷缺口内，因此天然避免了 seed 3407 这种“低负荷时全体大放电”的反向潮流。第三，它的目标直接包含电网购电价格、碳排因子、上网电价和低碳贡献售卖价格，因此更接近单步经济/碳排最小化。

相关代码位置是 `src/tecsf/env.py:202`。该函数中：

```text
max_charge_local = min(max_charge, local_surplus)
max_discharge_local = min(max_discharge, local_deficit)
```

这两个局部约束正是当前 TECSF actor 缺少的安全先验。

## 5. 建议的代码改进优先级

### P0：先修动作尺度和安全投影

这是最直接解决 IEEE 33-bus 可行性弱项的改动。

建议新增 benchmark profile 的动态动作上限，或在环境中根据 profile 负荷自动设置：

```text
max_buy_power_i = c_buy * max_t(load_i,t)
max_sell_power_i = c_sell * max_t(pv_i,t)
max_discharge_power_i = min(default, c_dis * max_t(load_i,t))
max_charge_power_i = min(default, c_ch * max_t(pv_i,t))
```

如果暂时不支持逐 agent 上限，也应在标准算例 config 中将全局 `max_buy_power/max_sell_power/max_discharge_power` 缩放到 IEEE bus 负荷量级，而不是沿用合成场景默认值。

同时修改 `_storage_safety_dispatch`：当网络存在越限风险时，不只增加 deficit 节点放电，而要裁剪过度放电和过度充电。最低限度可以先采用本地保守投影：

```text
safe_discharge_i = min(discharge_i, max(load_i - pv_i, 0), max_discharge_by_soc_i)
safe_charge_i = min(charge_i, max(pv_i - load_i, 0), max_charge_by_soc_i)
```

更理想的方案是用径向 feeder 的子树灵敏度做线路容量投影。

### P0：修正 system_cost 口径

建议把当前 `system_cost` 改为两个指标。论文中的主“系统成本”使用社会成本，不计 P2P 内部转移支付：

```text
system_social_cost = grid_buy_cost + op_cost + max(carbon_cost, 0) + emergency_cost
```

另行保留：

```text
participant_payment_cost = p2p_cost + grid_buy_cost + op_cost + max(carbon_cost, 0) + emergency_cost
```

这样可以避免 TECSF 因 P2P 成交更多而在“系统成本”指标上被结构性惩罚。修正后需要重新生成 summary 和论文表格。

### P1：将 P2P 修正与物理可行性投影解耦

当前 `p2p_power *= repair_shrink` 不能修复储能或净注入导致的线路越限。建议把 `clear_market` 改成两层：

1. 物理动作投影：先投影 `charge/discharge/pv_curtail/load_shed/grid_exchange`，保证节点净注入满足线路容量和电压边界。
2. 交易结算修正：再在已可行的物理状态上修正 P2P 成交量，使结算量不超过可交易余缺。

如果短期不实现优化器，可以先做启发式径向投影：从根线路开始检查每条线路的子树净注入，若超限，则按子树内 agent 的过度放电或过度售电比例削减。

### P1：加入动作 mask 或 barrier penalty

建议对 actor 输出前后增加动作可行性先验：

```text
q_sell_i <= max(pv_i + discharge_i - load_i - charge_i, 0)
q_buy_i <= max(load_i + charge_i - pv_i - discharge_i, 0)
discharge_i <= local_deficit_i + allowed_export_i
charge_i <= local_surplus_i + allowed_import_i
```

对于学习算法，可以加入 barrier penalty 或 action saturation penalty，惩罚长期贴边的买卖/充放电动作。seed 3407 的 raw action 均值绝对值为 0.999，说明 saturation penalty 很有必要。

### P1：跨 episode 约束记忆或约束 critic

如果继续使用拉格朗日项，建议不要在每个 episode reset 后完全清零约束乘子，至少可以做以下之一：

1. 训练级全局乘子：跨 episode 保留移动平均乘子。
2. 分时段乘子：为 24 个时段维护独立约束风险，专门处理 `t=0` 这类重复失败。
3. 约束 critic：单独学习可行性风险，用于 actor loss 中的 safety term。

### P2：改进 TECSF 与 myopic_opt 的公平比较

myopic_opt 本质上使用了强局部可行性先验。若 TECSF 不使用同等 action projection，比较会偏向 myopic_opt。建议增加两个对照：

1. `tecsf_projected`：TECSF actor + 与 myopic_opt 同等级别的本地动作投影。
2. `myopic_plus_chain`：myopic_opt + TECS-Chain + LCCoins，用于区分“策略质量”和“可信结算机制”的贡献。

这能让论文回答更清楚：TECSF 的贡献到底来自可信结算/低碳反馈，还是来自底层调度策略。

## 6. 论文表述建议

当前论文不要声称 TECSF 在 IEEE 标准算例上全面优于 myopic_opt。更稳妥的表述是：

> IEEE 33/69 标准配电网算例表明，TECSF 的可信结算、receipt 审计和 LCCoins 反馈链路能够迁移到标准网络 profile；但当前学习策略在标准算例下的经济性和碳排放并未支配 myopic_opt，IEEE 33-bus 中 seed 3407 还暴露了动作尺度、储能投影和线路容量约束处理不足。该结果说明本文机制层贡献成立，但底层可行性投影和经济调度策略仍需进一步优化。

论文中可以把这部分写成“局限性与改进方向”，而不是回避。这样反而更符合学术论文写作标准，因为它清楚地区分了机制验证和调度最优性的证据边界。

## 7. 建议的验证实验

改代码后不要直接重跑全部 1000 episode。建议采用分层验证：

第一步，做 unit/smoke test。构造 IEEE 33-bus `t=0`、全体最大放电动作，验证投影后 `line_violation=0` 或至少显著低于容差，且不出现 SOC 越界。

第二步，重放旧 checkpoint。用 seed 3407 的旧 TECSF checkpoint 重新评估 IEEE 33-bus。如果只改投影，不重训，理想结果应是 `t=0` 不再 rejected，结算成功率从 0.9583 恢复到 1.000，但成本/碳排可能变化。

第三步，短训验证。对 IEEE 33-bus 只跑 `seed=3407`、100-200 episode，比较旧版与新版的 `eval_max_violation`、`eval_system_social_cost`、`eval_grid_carbon_emission`。

第四步，正式重跑。若短训有效，再跑 IEEE 33/69 的 5 seeds x 1000 episode，并更新论文标准算例结果。

建议验收门槛如下：

| 指标 | 目标 |
| --- | --- |
| IEEE 33-bus TECSF 结算成功率 | 1.000 或至少不低于 0.995 |
| IEEE 33-bus TECSF 最大约束违背 | 0.000 或低于 0.1 |
| seed 3407 首时段 rejected | 0 |
| 系统社会成本 | 不因 P2P 内部支付被重复计入 |
| IEEE 69-bus 可行性 | 不退化 |

## 8. 是否值得优化

值得优化，而且优先级高。原因有三点。第一，当前弱项有明确可复现触发点，不是模糊的训练不稳定。第二，修复动作尺度和投影后，很可能直接消除 IEEE 33-bus seed 3407 的可行性失败。第三，修正系统成本口径后，TECSF 的经济性结论可能会更公平，至少不会因为 P2P 内部转移支付被计入系统成本而被低估。

不过，即使完成这些优化，也不应预设 TECSF 一定会全面优于 myopic_opt。更合理的目标是：先证明 TECSF 在标准算例上稳定可行，再比较其在可信结算、低碳激励和多目标权衡上的优势。如果要追求成本/碳排同时优于 myopic_opt，还需要进一步改进策略学习目标、动作投影和经济调度基线。
