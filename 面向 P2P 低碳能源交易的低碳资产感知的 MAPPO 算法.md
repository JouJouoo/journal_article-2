# 面向 P2P 低碳能源交易的低碳资产感知的 MAPPO 算法

## 摘要

现有 P2P 低碳能源交易强化学习方法主要通过在奖励函数中加入当期碳排放、低碳电量或碳交易成本等低碳项来促进低碳行为。然而，这类设计通常将低碳贡献视为即时激励，忽略了低碳资产的可累积性，容易导致策略学习短视化和平均化。为此，本文提出一种低碳资产感知的 MAPPO 算法，使策略具备面向长期低碳收益的跨期决策能力。

首先，设计了一种锚定用户低碳贡献的区块链代币（LCCoins）。将用户碳减排量与实际消纳的清洁电量在区块链中做可信结算，然后铸造为相应的LCCoins分配在账户中，为低碳资产的累积提供基础。

其次，提出一种低碳资产优势分解 MAPPO 策略学习机制。将标准的单一优势估计分解为经济优势、 LCCoins 低碳资产优势。在此基础上，引入低碳资产信用分配 critic 和 LCCoins 资产状态自适应 PPO 裁剪机制，使策略更新既能识别不同产消者对低碳资产增长的贡献，也能学到何种更新强度更有利于长期低碳资产积累。

然后，设计了一种基于LCCoins动态资产效用的奖励函数。一方面采用 CRRA 效用函数将账户余额映射为低碳资产存量效用，另一方面将相邻时段 LCCoins 变化量映射为低碳资产增量效用。使奖励函数能够同时反映低碳资产的累积水平与动态变化，策略在及时响应当前决策带来的低碳收益变化的同时，也关注长期低碳资产积累。

## 1 引言

在分布式新能源高比例接入的配电侧，P2P 能源交易将产消者的本地购售电行为与分布式能源消纳直接连接起来[1], [2]。产消者依托屋顶光伏、用户侧储能和柔性负荷参与本地交易，需要在交易价格、负荷需求、储能状态和清洁电量利用之间进行动态权衡[2], [3]。与传统以经济成本为主的交易优化不同，P2P 低碳能源交易还需要反映清洁电量消纳、碳减排贡献、碳配额成本和低碳行为激励等目标[4]–[7]。因此，如何在动态交易过程中同时兼顾经济收益和低碳收益，是 P2P 能源交易低碳化运行中的关键问题[4]–[8]。

强化学习为 P2P 低碳能源交易中的序贯决策提供了有效工具。由于产消者交易行为具有多主体互动、储能时序耦合和供需状态动态变化等特征，强化学习尤其是多智能体强化学习常被用于优化产消者的购售电、储能充放电和协同交易策略[3], [9]–[11]。现有研究通常通过在奖励函数中加入碳排放惩罚、低碳电量奖励、清洁能源消纳奖励或碳交易成本等即时低碳激励项，引导智能体在当前时段选择低碳行为[12], [13]。这类方法能够使策略学习直接感知低碳目标，但其低碳激励通常仍以当期收益为中心，缺少对低碳贡献累积价值的表达。当低碳贡献只作为即时奖励进入策略更新时，智能体难以将当前低碳行为与后续资产状态和长期收益联系起来，容易形成短视化或平均化的低碳策略。

区块链技术为低碳贡献的可信记录和透明结算提供了另一类解决思路。已有研究将区块链应用于 P2P 能源交易结算、能源与碳市场一体化交易、碳配额交易、能源代币交易以及多时段市场验证等场景[14]–[18]。其优势在于能够为用户低碳行为提供可验证、可追溯的链上凭证，使碳减排量、碳配额流转和清洁电量消纳等低碳贡献不再只是临时性行为记录，而具备形成链上确认记录或代币化激励的基础[14]–[16]。然而，现有区块链低碳应用多停留在交易结算、凭证记录或激励分配层面，主要解决低碳贡献如何被可信确认的问题[14]–[18]。这些链上低碳资产通常没有进一步反馈到强化学习决策过程中，也没有作为智能体状态感知和奖励更新的重要依据。

由此可见，现有研究在低碳交易决策和低碳贡献记录两个方面已形成一定基础，但二者之间仍存在割裂。一方面，强化学习方法能够处理 P2P 交易中的动态策略优化，却多将低碳需求转化为即时奖励项，难以体现低碳贡献在长期决策中的持续作用[3], [9]–[13]；另一方面，区块链方法能够记录和累计低碳贡献，却较少与智能体策略学习过程耦合[14]–[18]。因此，现有研究仍缺少一种将链上低碳贡献、资产状态表征和强化学习决策相结合的建模框架。

为弥合这一缺口，本文提出一种低碳资产感知的 MAPPO 算法。本文的基本思路是：利用区块链将用户低碳贡献转化为可累计的 LCCoins，并将其作为低碳资产引入多智能体强化学习，使策略学习不仅响应当期低碳激励，也能够关注低碳贡献的持续积累。围绕这一思路，本文从低碳资产生成、低碳资产优势分解策略学习和奖励函数设计三个方面展开方法构建。

本文的主要贡献如下。

1. 设计一种锚定用户低碳贡献的区块链代币 LCCoins，将碳减排量与实际消纳的清洁电量进行可信结算，并形成可累计的链上低碳资产。现有区块链 P2P 能源交易、电碳市场集成、碳配额交易和能源代币交易研究已经能够将能源交易结果、碳配额或低碳贡献记录为链上凭证或可交易资产[14]–[18]，但这些链上资产通常作为结算、交易或激励结果存在，较少进一步回流到智能体策略学习过程。LCCoins 的区别在于将经可信确认的低碳资产余额作为可观测状态和奖励反馈引入 MAPPO，使链上低碳资产不仅用于结算确认，也直接参与后续交易策略更新。

2. 提出低碳资产优势分解 MAPPO 策略学习机制，将经共识确认的 LCCoins 账户余额作为低碳资产纳入策略更新。不同于现有 P2P 多智能体强化学习通常将低碳目标并入单一标量奖励，本文在 PPO/MAPPO 的近端策略更新和集中式训练、分布式执行框架基础上[19], [20]，将 MAPPO 的优势估计分解为经济优势和 LCCoins 低碳资产优势，并通过低碳资产信用分配 critic 和 LCCoins 资产状态自适应 PPO 裁剪机制，引导策略学习哪些动作、哪些产消者交互以及何种更新强度更有利于长期低碳资产积累。

3. 设计基于 LCCoins 动态资产效用的奖励函数，将 LCCoins 余额和 LCCoins 变化量共同引入奖励函数，引导策略兼顾当前低碳收益与长期低碳资产积累。不同于现有低碳 P2P 交易和能碳联合强化学习通常以碳配额成本、碳交易收益或当期低碳目标作为优化项或即时奖励信号[6], [12], [13]，本文同时刻画低碳资产存量效用和低碳资产增量效用，使奖励函数能够反映低碳贡献的累积水平及其动态变化。

## 2 方法

本文将 P2P 低碳能源交易建模为多产消者参与的序贯决策问题，并在 MAPPO 框架中引入可累计的链上低碳资产。方法部分的核心逻辑是：首先由交易与运行结果计算用户低碳贡献，并将其铸造为 LCCoins；其次将 LCCoins 余额作为低碳资产状态回流至 MAPPO 策略学习，并将优势估计分解为经济优势和低碳资产优势；然后结合低碳资产信用分配 critic 与 LCCoins 资产状态自适应 PPO 裁剪机制完成策略更新；最后通过 LCCoins 存量效用和增量效用构造奖励函数，使智能体在优化当期交易收益的同时关注长期低碳资产积累。

### 2.1 P2P 低碳能源交易场景建模

![image-20260703150402074](/Users/joujou/Library/Application Support/typora-user-images/image-20260703150402074.png)

考虑由 $N$ 个产消者组成的本地能源社区，产消者集合记为 $\mathcal{N}=\{1,2,\cdots,N\}$，调度周期被划分为 $T$ 个离散时段，时段集合记为 $\mathcal{T}=\{1,2,\cdots,T\}$。在时段 $t$，产消者 $i$ 具有本地负荷 $L_{i,t}$、光伏出力 $G_{i,t}$ 和储能荷电状态 $SOC_{i,t}$。产消者可以通过 P2P 市场与其他用户交易电能，也可以在 P2P 交易未完全匹配时由上级电网进行补充购售电。其主要决策包括向 P2P 市场提交的购电申报量 $\hat{P}^{p2p,b}_{i,t}$、售电申报量 $\hat{P}^{p2p,s}_{i,t}$、买入报价 $\hat{\pi}^{p2p,b}_{i,t}$、卖出报价 $\hat{\pi}^{p2p,s}_{i,t}$ 以及储能充放电功率 $P^{ch}_{i,t}$、$P^{dis}_{i,t}$。

P2P 市场根据各产消者提交的报价和申报量进行双边拍卖匹配。记 $Q^{p2p}_{ij,t}$ 为时段 $t$ 产消者 $i$ 从产消者 $j$ 购买的成交功率，$\Pi^{p2p}_{ij,t}$ 为对应的成交价格，则清算后的 P2P 购售电功率为：

$$
P^{p2p,b}_{i,t}=\sum_{j\in\mathcal{N},j\ne i}Q^{p2p}_{ij,t},\quad
P^{p2p,s}_{i,t}=\sum_{j\in\mathcal{N},j\ne i}Q^{p2p}_{ji,t}.
$$

其中，$\Pi^{p2p}_{t}\in\mathbb{R}^{N\times N}$ 为清算后形成的双边成交价格矩阵，不同买卖双方可以具有不同成交价格。未被 P2P 市场匹配的剩余功率由上级电网兜底结算，得到 $P^{g,b}_{i,t}$ 和 $P^{g,s}_{i,t}$。该清算环节仅作为环境交易执行机制，本文不将其作为算法创新点展开。

产消者在每个时段需要满足功率平衡约束：

$$
G_{i,t}+P^{dis}_{i,t}+P^{p2p,b}_{i,t}+P^{g,b}_{i,t}
=L_{i,t}+P^{ch}_{i,t}+P^{p2p,s}_{i,t}+P^{g,s}_{i,t}.
$$

储能状态随充放电行为动态变化：

$$
SOC_{i,t+1}=SOC_{i,t}
+\frac{\eta^{ch}_{i}P^{ch}_{i,t}\Delta t}{E_i}
-\frac{P^{dis}_{i,t}\Delta t}{\eta^{dis}_{i}E_i},
$$

其中，$E_i$ 为储能容量，$\eta^{ch}_{i}$ 和 $\eta^{dis}_{i}$ 分别为充电效率和放电效率，$\Delta t$ 为时段长度。储能运行需要满足 $SOC_i^{min}\le SOC_{i,t}\le SOC_i^{max}$、$0\le P^{ch}_{i,t}\le P_i^{ch,max}$ 和 $0\le P^{dis}_{i,t}\le P_i^{dis,max}$。P2P 交易还需满足市场清算约束：

$$
\sum_{i\in\mathcal{N}}P^{p2p,b}_{i,t}
=
\sum_{i\in\mathcal{N}}P^{p2p,s}_{i,t}.
$$

在该交易环境中，低碳目标来自两个方面：一是光伏电量被本地负荷、储能或其他产消者实际消纳形成的清洁电量贡献；二是相对于基准供电方式减少的碳排放贡献。本文不将二者仅作为当期奖励项处理，而是进一步将其转化为可累计的链上低碳资产，为后续策略学习提供跨期状态反馈。

### 2.2 锚定低碳贡献的 LCCoins 生成机制

为使低碳贡献具备可记录、可追溯和可累计属性，本文设计 LCCoins 作为锚定用户低碳贡献的链上代币。时段 $t$ 结束后，系统根据已清算的 P2P 交易结果、光伏消纳结果和电网购售电结果计算产消者 $i$ 的低碳贡献。记 $E^{clean}_{i,t}$ 为产消者 $i$ 在时段 $t$ 实际消纳或向社区提供并被消纳的清洁电量，$R^{co2}_{i,t}$ 为相对于基准排放因子得到的碳减排量，则用户低碳贡献可表示为：

$$
C_{i,t}=\omega_e E^{clean}_{i,t}+\omega_c R^{co2}_{i,t},
$$

其中，$\omega_e$ 和 $\omega_c$ 分别为清洁电量贡献和碳减排贡献的折算系数。碳减排量按照基准排放与实际排放的差额计算：

$$
R^{co2}_{i,t}=\max\{0,\xi^{base}E^{load}_{i,t}-\xi^{act}_{i,t}E^{load}_{i,t}\},
$$

其中，$E^{load}_{i,t}=L_{i,t}\Delta t$ 为用户用电量，$\xi^{base}$ 为基准供电排放因子，$\xi^{act}_{i,t}$ 为考虑清洁电量消纳和外部购电后的实际等效排放因子。该定义保证 LCCoins 只对应正向低碳贡献，避免高排放行为被折算为负资产。

区块链模块负责记录时段、用户、交易电量、清洁电量、碳减排量和代币铸造结果。由于 LCCoins 的铸造依赖低碳贡献的可信确认，本文进一步设计面向 LCCoins 铸造的独立共识机制。考虑 P2P 能源社区中交易主体和运行管理节点身份相对明确，本文采用联盟链式授权验证节点集合 $\mathcal{V}$ 对低碳贡献进行确认。验证节点可由市场清算节点、计量数据节点、碳核算节点和账本维护节点组成，其职责不是重新优化交易决策，而是对交易结果、清洁电量归属和碳减排核算进行一致性验证。

时段 $t$ 结束后，市场清算结果、计量数据和碳核算结果被打包为候选低碳资产记录。每个验证节点 $v\in\mathcal{V}$ 独立检查以下条件：P2P 购售电量是否满足清算平衡，清洁电量是否存在重复计入，碳减排量是否由基准排放因子和实际等效排放因子一致计算，LCCoins 铸造量是否符合预设铸造规则。若验证通过，节点生成确认标识 $\sigma^v_{i,t}=1$；否则生成 $\sigma^v_{i,t}=0$。当确认节点数达到阈值 $q$ 时，低碳贡献记录被写入区块并触发代币铸造：

$$
\chi_{i,t}=
\begin{cases}
1, & \sum_{v\in\mathcal{V}}\sigma^v_{i,t}\ge q,\\
0, & \sum_{v\in\mathcal{V}}\sigma^v_{i,t}<q,
\end{cases}
$$

其中，$\chi_{i,t}$ 为产消者 $i$ 在时段 $t$ 的低碳贡献共识结果。对于包含 $|\mathcal{V}|$ 个验证节点的联盟链，可取 $q=\lceil 2|\mathcal{V}|/3\rceil$ 作为拜占庭容错意义下的确认阈值。智能合约仅对通过共识确认的低碳贡献铸造 LCCoins：

$$
M_{i,t}=\chi_{i,t}\lambda C_{i,t},
$$

其中，$M_{i,t}$ 为时段 $t$ 给产消者 $i$ 铸造的 LCCoins 数量，$\lambda$ 为代币铸造系数。当低碳贡献未通过共识确认时，$\chi_{i,t}=0$，对应时段不触发 LCCoins 铸造。产消者链上账户余额更新为：

$$
B_{i,t+1}=B_{i,t}+M_{i,t},
$$

其中，$B_{i,t}$ 为时段 $t$ 开始时产消者 $i$ 的 LCCoins 余额。本文通过上述共识机制保证只有经多节点一致确认的低碳贡献才能进入代币铸造过程，但不进一步展开链上吞吐量、确认时延和通信开销等交易性能建模。LCCoins 的关键作用在于将原本分散在各时段的低碳贡献转化为连续可观测的低碳资产状态，并进一步反馈到 MAPPO 的状态输入和奖励更新中。

### 2.3 低碳资产优势分解 MAPPO 策略学习机制

在本文中，储能 $SOC$ 作为常规运行状态进入智能体观测，用于反映储能时序耦合和充放电可行性。低碳资产专指经区块链确认并累计到用户账户中的 LCCoins。为使策略能够利用该链上低碳资产信息，产消者 $i$ 在时段 $t$ 的局部观测定义为：

$$
o_{i,t}=\left[
L_{i,t},G_{i,t},\Pi^{p2p}_{t-1},\pi^{g,b}_{t},\pi^{g,s}_{t},
SOC_{i,t},\tilde{B}_{i,t}
\right],
$$

其中，$\Pi^{p2p}_{t-1}$ 为上一时段 P2P 市场清算后形成的双边成交价格矩阵，作为公开历史价格信息进入观测；$\pi^{g,b}_{t}$ 和 $\pi^{g,s}_{t}$ 分别为电网购电价格和售电价格，$\tilde{B}_{i,t}$ 为归一化后的 LCCoins 余额。当前时段的 P2P 报价价格由智能体动作给出，清算后形成的 $\Pi^{p2p}_{t}$ 作为环境结果进入下一时段观测，从而避免将当前成交价格提前泄露给策略。本文将低碳资产状态定义为：

$$
x^{lc}_{i,t}=\tilde{B}_{i,t}.
$$

标准 PPO 通过裁剪代理目标限制策略更新幅度[19]，MAPPO 在此基础上采用集中式训练、分布式执行结构，并通常基于单一优势函数更新 actor[20]。在常规标量奖励下，经济收益和低碳资产积累会被混合为一个标量回报。多目标强化学习研究表明，过早标量化可能造成不同目标学习信号之间的相互干扰[21]。为避免低碳资产信号被经济收益主导，本文将优势估计分解为经济优势和 LCCoins 低碳资产优势。经济回报 $r^{eco}_{i,t}$ 由第 2.4 节中的交易和运行收益给出，低碳资产回报定义为：

$$
r^{coin}_{i,t}=\alpha u^{stock}_{i,t}+\beta u^{inc}_{i,t}.
$$

集中式 critic 输出两个价值头：

$$
V_\phi(s_t)=
\left[
V^{eco}_\phi(s_t),
V^{coin}_\phi(s_t)
\right].
$$

对 $m\in\{eco,coin\}$，分别计算时序差分误差和优势函数：

$$
\delta^m_{i,t}=r^m_{i,t}
+\gamma V^m_\phi(s_{t+1})-V^m_\phi(s_t),
$$

$$
\hat{A}^m_{i,t}
=\sum_{l=0}^{T-t}(\gamma\lambda_{gae})^l\delta^m_{i,t+l}.
$$

多智能体信用分配研究通常通过奖励解耦或注意力式贡献权重识别不同智能体对联合回报的影响[22], [23]。为处理 P2P 交易中“一个产消者的 LCCoins 增长可能由其他产消者的清洁电量供给共同促成”的信用分配问题，本文进一步引入低碳资产信用分配 critic。该 critic 根据产消者状态、LCCoins 余额和交易关系生成贡献权重：

$$
\kappa_{ij,t}
=
\frac{\exp(h_\psi(z_{i,t},z_{j,t},e_{ij,t}))}
{\sum_{j'\in\mathcal{N}}\exp(h_\psi(z_{i,t},z_{j',t},e_{ij',t}))},
$$

其中，$z_{i,t}=[o_{i,t},x^{lc}_{i,t}]$ 为产消者 $i$ 的状态-低碳资产特征，$e_{ij,t}$ 表示产消者 $i$ 与 $j$ 之间的 P2P 交易关系，$h_\psi(\cdot)$ 为可学习的信用评分函数。链上低碳资产优势经信用分配后得到：

$$
\bar{A}^{coin}_{i,t}
=\sum_{j\in\mathcal{N}}\kappa_{ij,t}\hat{A}^{coin}_{j,t}.
$$

随后，低碳资产状态门控网络根据 $x^{lc}_{i,t}$ 生成经济优势和低碳资产优势的融合权重：

$$
[w^{eco}_{i,t},w^{coin}_{i,t}]
=\operatorname{softmax}(g_\omega(x^{lc}_{i,t})).
$$

最终用于 actor 更新的低碳资产分解优势为：

$$
\hat{A}^{LC}_{i,t}
=w^{eco}_{i,t}\hat{A}^{eco}_{i,t}
+w^{coin}_{i,t}\bar{A}^{coin}_{i,t}.
$$

进一步地，本文引入 LCCoins 资产状态自适应 PPO 裁剪机制。传统 PPO 使用固定裁剪系数 $\epsilon$[19]，难以区分不同低碳资产状态下策略更新的风险；已有协调式 PPO 研究则主要关注多智能体策略更新步长的协同约束[24]。本文将裁剪强度与 LCCoins 资产状态关联，根据 LCCoins 余额生成时变裁剪系数：

$$
\epsilon_{i,t}
=\epsilon^{min}
+(\epsilon^{max}-\epsilon^{min})
\sigma(g_\nu(x^{lc}_{i,t})),
$$

其中，$\epsilon^{min}$ 和 $\epsilon^{max}$ 分别为最小和最大裁剪边界，$g_\nu(\cdot)$ 为低碳资产状态门控函数，$\sigma(\cdot)$ 为 Sigmoid 函数。当 LCCoins 余额较低、策略仍需探索低碳资产增长路径时，裁剪范围可适度放宽；当 LCCoins 余额较高或资产状态波动较大时，裁剪范围收紧以抑制过度更新。

MAPPO 仍采用集中式训练和分布式执行结构[20]。执行阶段，每个产消者根据自身观测 $o_{i,t}$ 输出动作 $a_{i,t}$：

$$
a_{i,t}\sim \pi_{\theta_i}(a_{i,t}|o_{i,t}),
$$

其中，$a_{i,t}$ 包括 P2P 购售电申报量、P2P 买卖报价和储能充放电决策，可写为：

$$
a_{i,t}=\left[
\hat{P}^{p2p,b}_{i,t},\hat{P}^{p2p,s}_{i,t},
\hat{\pi}^{p2p,b}_{i,t},\hat{\pi}^{p2p,s}_{i,t},
P^{ch}_{i,t},P^{dis}_{i,t}
\right].
$$

P2P 交易经市场清算后得到实际成交功率 $Q^{p2p}_{ij,t}$ 和成交价格矩阵 $\Pi^{p2p}_{t}$；未匹配电量由电网兜底，因此电网购售电量为环境清算后的结果。训练阶段，actor 采用低碳资产分解优势和自适应裁剪系数更新：

$$
\mathcal{L}^{LC-MAPPO}(\theta)=
\mathbb{E}_t\left[
\min\left(
\rho_{i,t}(\theta)\hat{A}^{LC}_{i,t},
\operatorname{clip}(\rho_{i,t}(\theta),1-\epsilon_{i,t},1+\epsilon_{i,t})\hat{A}^{LC}_{i,t}
\right)
\right],
$$

其中，$\rho_{i,t}(\theta)=\pi_{\theta}(a_{i,t}|o_{i,t})/\pi_{\theta^{old}}(a_{i,t}|o_{i,t})$。由此，本文的算法创新集中在 LCCoins 低碳资产对优势分解、信用分配和 PPO 近端更新约束的作用。

### 2.4 基于 LCCoins 动态资产效用的奖励函数

奖励塑形研究表明，附加奖励信号需要关注其对原策略目标的影响，potential-based 形式可在特定条件下保持策略不变性[25], [26]。本文并不将 LCCoins 效用作为原奖励的等价势函数变换，而是将其作为低碳资产偏好的显式目标项。为避免低碳目标仅以当期即时激励形式进入策略更新，本文构造基于 LCCoins 动态资产效用的奖励函数。产消者 $i$ 在时段 $t$ 的奖励由经济收益、低碳资产存量效用和低碳资产增量效用三部分组成：

$$
r_{i,t}=r^{eco}_{i,t}
+\alpha u^{stock}_{i,t}
+\beta u^{inc}_{i,t},
$$

其中，$r^{eco}_{i,t}$ 为交易和运行经济收益，$u^{stock}_{i,t}$ 为低碳资产存量效用，$u^{inc}_{i,t}$ 为低碳资产增量效用，$\alpha$ 和 $\beta$ 分别为两类低碳资产效用的权重。

经济收益项反映 P2P 交易、电网购售电和储能运行成本：

$$
r^{eco}_{i,t}=
\left(
\sum_{j\in\mathcal{N},j\ne i}\Pi^{p2p}_{ji,t}Q^{p2p}_{ji,t}
-
\sum_{j\in\mathcal{N},j\ne i}\Pi^{p2p}_{ij,t}Q^{p2p}_{ij,t}
\right)\Delta t
+\pi^{g,s}_{t}P^{g,s}_{i,t}\Delta t
-\pi^{g,b}_{t}P^{g,b}_{i,t}\Delta t
-c^{bat}_{i}(P^{ch}_{i,t}+P^{dis}_{i,t})\Delta t,
$$

其中，第一项为产消者 $i$ 在双边 P2P 成交中的售电收益与购电成本之差，$c^{bat}_{i}$ 为储能充放电退化成本系数。低碳资产存量效用采用 CRRA 函数刻画 LCCoins 余额带来的长期资产价值：

$$
U(B)=
\begin{cases}
\dfrac{(B+B_0)^{1-\rho}-1}{1-\rho}, & \rho\ne 1,\\
\ln(B+B_0), & \rho=1,
\end{cases}
$$

其中，$B_0>0$ 为避免零余额奇异的平移常数，$\rho$ 为风险厌恶系数。存量效用定义为：

$$
u^{stock}_{i,t}=U(B_{i,t+1}).
$$

该项使策略持续关注已经形成的低碳资产水平，避免低碳贡献只在产生当期发挥作用。为同时保留当前决策带来的即时反馈，本文进一步定义 LCCoins 增量效用：

$$
u^{inc}_{i,t}=U(B_{i,t+1})-U(B_{i,t}).
$$

由于 $B_{i,t+1}-B_{i,t}=M_{i,t}$，增量效用直接反映时段 $t$ 的低碳贡献铸币结果。存量效用强调低碳资产的长期累积水平，增量效用强调当前行为对低碳资产变化的边际贡献。二者共同进入奖励函数，使策略既不会只追逐当期清洁电量奖励，也不会因只关注累计余额而削弱对当前低碳行为的响应。

### 2.5 低碳资产优势分解 MAPPO 算法流程

基于上述交易环境、LCCoins 生成机制、低碳资产状态和动态资产效用奖励函数，本文形成低碳资产优势分解 MAPPO 算法。每个 episode 中，环境首先初始化负荷、光伏、电网价格、历史 P2P 成交价格矩阵、储能 SOC 和 LCCoins 账户余额；随后各产消者根据包含常规运行状态、历史成交价格矩阵和 LCCoins 余额的观测选择 P2P 报价、申报交易量和储能动作；环境执行动作并完成 P2P 市场清算、未匹配电量的电网兜底、储能状态更新和功率平衡检查；区块链共识模块根据清算结果计算低碳贡献，并在验证节点达成确认后铸造 LCCoins、更新账户余额；最后分别计算经济回报和 LCCoins 低碳资产回报，并通过优势分解、信用分配和 LCCoins 资产状态自适应 PPO 裁剪完成策略更新。

**算法 1 低碳资产优势分解 MAPPO 训练流程**

输入：产消者集合 $\mathcal{N}$，训练轮数 $K$，时段数 $T$，折扣因子 $\gamma$，GAE 参数 $\lambda_{gae}$，PPO 裁剪边界 $\epsilon^{min}$ 和 $\epsilon^{max}$。

输出：各产消者 actor 参数 $\theta_i$，双价值头 critic 参数 $\phi$，信用分配 critic 参数 $\psi$，优势融合门控参数 $\omega$ 和裁剪门控参数 $\nu$。

1. 初始化 actor 网络 $\{\pi_{\theta_i}\}_{i\in\mathcal{N}}$、双价值头 critic $V_\phi$、信用分配 critic $h_\psi$、优势融合门控 $g_\omega$、裁剪门控 $g_\nu$ 和 LCCoins 账户余额 $B_{i,1}$。
2. 对每个训练轮次 $k=1,2,\cdots,K$：
3. 初始化 episode 中的负荷、光伏、电网价格、历史 P2P 成交价格矩阵和储能状态。
4. 对每个时段 $t=1,2,\cdots,T$：
5. 每个产消者构造观测 $o_{i,t}$，其中包含负荷、光伏、历史 P2P 成交价格矩阵、电网价格、$SOC_{i,t}$ 和 LCCoins 余额 $\tilde{B}_{i,t}$。
6. 每个 actor 根据 $\pi_{\theta_i}(a_{i,t}|o_{i,t})$ 采样 P2P 购售电申报量、买卖报价和储能动作。
7. 环境执行动作，完成 P2P 清算，得到 $Q^{p2p}_{ij,t}$ 和 $\Pi^{p2p}_{t}$，并对未匹配电量进行电网兜底、功率平衡约束处理和 $SOC_{i,t+1}$ 更新。
8. 区块链共识模块计算 $E^{clean}_{i,t}$、$R^{co2}_{i,t}$ 和 $C_{i,t}$，由验证节点确认 $\chi_{i,t}$，再铸造 $M_{i,t}$ 并更新 $B_{i,t+1}$。
9. 根据交易收益计算 $r^{eco}_{i,t}$，根据 LCCoins 存量效用和增量效用计算 $r^{coin}_{i,t}$。
10. 存储联合状态、局部观测、低碳资产状态、动作、两类回报、下一状态和动作概率。
11. episode 结束后，双价值头 critic 分别计算经济和 LCCoins 低碳资产两类时序差分误差：

$$
\delta^m_{i,t}=r^m_{i,t}
+\gamma V^m_\phi(s_{t+1})-V^m_\phi(s_t),
\quad m\in\{eco,coin\}.
$$

12. 对两类时序差分误差分别采用 GAE 计算 $\hat{A}^{eco}_{i,t}$ 和 $\hat{A}^{coin}_{i,t}$。
13. 信用分配 critic 根据交易关系和资产状态生成 $\kappa_{ij,t}$，并得到链上资产信用分配优势 $\bar{A}^{coin}_{i,t}$。
14. 优势融合门控根据 $x^{lc}_{i,t}$ 生成 $w^{eco}_{i,t}$ 和 $w^{coin}_{i,t}$，得到低碳资产分解优势：

$$
\hat{A}^{LC}_{i,t}
=w^{eco}_{i,t}\hat{A}^{eco}_{i,t}
+w^{coin}_{i,t}\bar{A}^{coin}_{i,t}.
$$

15. 裁剪门控根据 $x^{lc}_{i,t}$ 生成 LCCoins 资产状态自适应裁剪系数 $\epsilon_{i,t}$，并更新 actor：

$$
\mathcal{L}^{LC-MAPPO}(\theta)=
\mathbb{E}_t\left[
\min\left(
\rho_{i,t}(\theta)\hat{A}^{LC}_{i,t},
\operatorname{clip}(\rho_{i,t}(\theta),1-\epsilon_{i,t},1+\epsilon_{i,t})\hat{A}^{LC}_{i,t}
\right)
\right],
$$

其中，$\rho_{i,t}(\theta)=\pi_{\theta}(a_{i,t}|o_{i,t})/\pi_{\theta^{old}}(a_{i,t}|o_{i,t})$。

16. 分别最小化两类价值函数误差，更新双价值头 critic：

$$
\mathcal{L}^{critic}(\phi)=
\mathbb{E}_t\left[
\sum_{m\in\{eco,coin\}}
\left(V^m_\phi(s_t)-\hat{R}^m_t\right)^2
\right],
$$

其中，$\hat{R}^m_t$ 为第 $m$ 类回报对应的折扣回报。训练完成后，分布式执行阶段只保留各产消者 actor，产消者根据本地运行观测和自身 LCCoins 余额独立生成交易与储能动作。由此，本文提出的算法围绕 LCCoins 低碳资产状态进行优势分解、信用分配和 PPO 更新约束。

## 参考文献

[1] W. Tushar, T. K. Saha, C. Yuen, D. Smith, and H. V. Poor, "Peer-to-Peer Trading in Electricity Networks: An Overview," *IEEE Transactions on Smart Grid*, vol. 11, no. 4, pp. 3185-3200, 2020.

[2] W. Tushar, C. Yuen, T. K. Saha, T. Morstyn, A. C. Chapman, M. J. E. Alam, S. Hanif, and H. V. Poor, "Peer-to-Peer Energy Systems for Connected Communities: A Review of Recent Advances and Emerging Challenges," *Applied Energy*, vol. 282, Art. no. 116131, 2021.

[3] C. Feng and A. L. Liu, "Peer-to-Peer Energy Trading of Solar and Energy Storage: A Networked Multiagent Reinforcement Learning Approach," *Applied Energy*, vol. 383, Art. no. 125283, 2025.

[4] Z. Lu, L. Bai, J. Wang, J. Wei, Y. Xiao, and Y. Chen, "Peer-to-Peer Joint Electricity and Carbon Trading Based on Carbon-Aware Distribution Locational Marginal Pricing," *IEEE Transactions on Power Systems*, vol. 38, no. 1, pp. 835-852, 2023.

[5] J. Li, S. Ge, Z. Xu, H. Liu, J. Li, C. Wang, and X. Cheng, "A Network-Secure Peer-to-Peer Trading Framework for Electricity-Carbon Integrated Market Among Local Prosumers," *Applied Energy*, vol. 335, Art. no. 120420, 2023.

[6] C. Wu, X. Chen, H. Hua, K. Yu, L. Gan, J. Shen, and Y. Ding, "Peer-to-Peer Energy Trading Optimization for Community Prosumers Considering Carbon Cap-and-Trade," *Applied Energy*, vol. 358, Art. no. 122611, 2024.

[7] J. Wang, X. Jin, H. Jia, M. Tostado-Véliz, Y. Mu, X. Yu, and S. Liang, "Joint Electricity and Carbon Sharing With PV and Energy Storage: A Low-Carbon DR-Based Game Theoretic Approach," *IEEE Transactions on Sustainable Energy*, vol. 15, no. 4, pp. 2703-2717, 2024.

[8] X. Wei, Y. Xu, H. Sun, and W. K. Chan, "Peer-to-Peer Energy Trading of Carbon-Aware Prosumers: An Online Accelerated Distributed Approach With Differential Privacy," *IEEE Transactions on Smart Grid*, vol. 15, no. 6, pp. 5595-5609, 2024.

[9] T. Chen, S. Bu, X. Liu, J. Kang, F. R. Yu, and Z. Han, "Peer-to-Peer Energy Trading and Energy Conversion in Interconnected Multi-Energy Microgrids Using Multi-Agent Deep Reinforcement Learning," *IEEE Transactions on Smart Grid*, vol. 13, no. 1, pp. 715-727, 2022.

[10] D. Qiu, Y. Ye, D. Papadaskalopoulos, and G. Strbac, "Scalable Coordinated Management of Peer-to-Peer Energy Trading: A Multi-Cluster Deep Reinforcement Learning Approach," *Applied Energy*, vol. 292, Art. no. 116940, 2021.

[11] Y. Ye, D. Papadaskalopoulos, Q. Yuan, Y. Tang, and G. Strbac, "Multi-Agent Deep Reinforcement Learning for Coordinated Energy Trading and Flexibility Services Provision in Local Electricity Markets," *IEEE Transactions on Smart Grid*, vol. 14, no. 2, pp. 1541-1554, 2023.

[12] D. Qiu, J. Xue, T. Zhang, J. Wang, and M. Sun, "Federated Reinforcement Learning for Smart Building Joint Peer-to-Peer Energy and Carbon Allowance Trading," *Applied Energy*, vol. 333, Art. no. 120526, 2023.

[13] Y. Zhou, Z. Ma, T. Wang, J. Zhang, X. Shi, and S. Zou, "Joint Energy and Carbon Trading for Multi-Microgrid System Based on Multi-Agent Deep Reinforcement Learning," *IEEE Transactions on Power Systems*, vol. 39, no. 6, pp. 7376-7388, 2024.

[14] W. Hua, J. Jiang, H. Sun, and J. Wu, "A Blockchain Based Peer-to-Peer Trading Framework Integrating Energy and Carbon Markets," *Applied Energy*, vol. 279, Art. no. 115539, 2020.

[15] M. Yan, M. Shahidehpour, A. Alabdulwahab, A. Abusorrah, N. Gurung, H. Zheng, O. Ogunnubi, A. Vukojevic, and E. A. Paaso, "Blockchain for Transacting Energy and Carbon Allowance in Networked Microgrids," *IEEE Transactions on Smart Grid*, vol. 12, no. 6, pp. 4702-4714, 2021.

[16] M. Mehdinejad, H. Shayanfar, and B. Mohammadi-Ivatloo, "Decentralized Blockchain-Based Peer-to-Peer Energy-Backed Token Trading for Active Prosumers," *Energy*, vol. 244, Art. no. 122713, 2022.

[17] C.-T. Huang and I. J. Scott, "Peer-to-Peer Multi-Period Energy Market With Flexible Scheduling on a Scalable and Cost-Effective Blockchain," *Applied Energy*, vol. 367, Art. no. 123331, 2024.

[18] A. K. Mazrae, H. Naderian, H. R. Baghaee, M. K. Sheikh-El-Eslami, and M. Karimi, "Transactive Energy and Peer-to-Peer Energy Trading Based on Blockchain: A Comprehensive Review and a Generalized Cyber-Physical Framework," *Energy Strategy Reviews*, vol. 62, Art. no. 101949, 2025.

[19] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, "Proximal Policy Optimization Algorithms," arXiv preprint arXiv:1707.06347, 2017.

[20] C. Yu, A. Velu, E. Vinitsky, J. Gao, Y. Wang, A. Bayen, and Y. Wu, "The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games," *Advances in Neural Information Processing Systems*, vol. 35, 2022.

[21] T. Ambadkar, S. Panda, S. Kale, J. Dodge, and A. Verma, "Preference Conditioned Multi-Objective Reinforcement Learning: Decomposed, Diversity-Driven Policy Optimization," arXiv preprint arXiv:2602.07764, 2026.

[22] A. Kapoor, B. Freed, H. Choset, and J. Schneider, "Assigning Credit with Partial Reward Decoupling in Multi-Agent Proximal Policy Optimization," arXiv preprint arXiv:2408.04295, 2024.

[23] B. Freed, A. Kapoor, I. Abraham, J. Schneider, and H. Choset, "Learning Cooperative Multi-Agent Policies with Partial Reward Decoupling," arXiv preprint arXiv:2112.12740, 2021.

[24] Z. Wu, C. Yu, D. Ye, J. Zhang, H. Piao, and H. H. Zhuo, "Coordinated Proximal Policy Optimization," arXiv preprint arXiv:2111.04051, 2021.

[25] A. Y. Ng, D. Harada, and S. Russell, "Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping," in *Proceedings of the Sixteenth International Conference on Machine Learning*, pp. 278-287, 1999.

[26] X. Lu, H. M. Schwartz, and S. N. Givigi Jr., "Policy Invariance under Reward Transformations for General-Sum Stochastic Games," *Journal of Artificial Intelligence Research*, vol. 41, pp. 397-406, 2011.
