# 实验协议：mappo 基线方法 10000 episodes 训练

**日期**: 2026-07-05  
**实验目的**: 使用基线方法 mappo 进行 10000 episodes 训练，生成训练曲线用于论文

## 1. 研究问题

- mappo 基线方法在 P2P 低碳能源交易环境中的训练收敛性如何？
- 训练奖励、P2P 交易量、电网交易量的收敛曲线特征是什么？

## 2. 实验配置

### 2.1 方法变体
- **variant**: `mappo` (基线方法)
- **特性**: 标准 MAPPO，无区块链、无 LCCoins、无循环网络、无资产感知

### 2.2 训练参数
- **episodes**: 10000
- **config**: `configs/default.yaml`
- **seed**: 7 (默认)
- **device**: cpu (默认，可根据需要改为 cuda)

### 2.3 环境配置
- **num_agents**: 8
- **num_nodes**: 5
- **horizon**: 24
- **scene**: P2P 低碳能源交易

## 3. 数据来源

- 使用合成 profile (非 IEEE 标准算例)
- 配置文件中 `profile_path: ""` 表示使用默认合成数据

## 4. 输出和评估

### 4.1 输出目录
- `outputs/experiments/mappo/`

### 4.2 输出文件
- 训练指标: `outputs/experiments/mappo/episode_metrics.json`
- 训练曲线图: `outputs/experiments/mappo/reward_curves.png/pdf/svg`
- 模型检查点: `outputs/experiments/mappo/checkpoints/`

### 4.3 评估指标
- **训练奖励** (total_reward / mean_reward)
- **P2P 交易量** (p2p_energy)
- **电网交易量** (grid_buy_cost / grid_sell_cost)
- **收敛性**: 观察 10000 episodes 内是否收敛

## 5. 验收标准

- 训练完成 10000 episodes 无报错
- 生成完整的训练曲线图（奖励、P2P、电网）
- 观察是否收敛（最后 1000 episodes 方差稳定）

## 6. 支持论文结论

本实验结果为论文提供:
- mappo 基线方法的训练收敛性证据
- 与其他变体（tecsf, no_chain）的对比基准
- 证明训练框架的有效性

## 7. 执行命令

```bash
cd C:\Users\zrway\Desktop\期刊论文-2
& "C:\Users\zrway\.conda\envs\DP-LCRL\python.exe" scripts/run_experiments.py \
  --variants mappo \
  --episodes 10000 \
  --output-dir outputs/experiments \
  --smooth-window 100
```

## 8. 时间估算

- 假设每 episode 约 1-5 秒（取决于环境复杂度）
- 10000 episodes ≈ 3-14 小时（CPU）
- 建议使用 GPU 加速或后台运行

## 9. 风险

- 长时间训练可能被中断
- 建议定期检查输出日志
- 确保磁盘空间充足（checkpoints 和日志）

---
**协议状态**: 待执行  
**创建时间**: 2026-07-05 11:34
