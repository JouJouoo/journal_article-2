# P2P能源交易、强化学习与区块链文献调研

> 说明：本文围绕“P2P能源交易—电碳协同结算—低碳激励—强化学习策略优化—区块链可信结算”这一研究链条，筛选与拟提出的可信电碳协同结算链（Trusted Energy-Carbon Co-Settlement Chain, TECS-Chain）高度相关的期刊论文 20 篇。筛选时优先选择中科院 1 区期刊中的英文期刊论文，剔除中文期刊、会议论文和预印本；同时考虑到“P2P能源交易、强化学习、区块链”三者同时出现在同一篇中科院 1 区期刊论文中的文献较少，本文采用“主题强相关”口径，即分别覆盖 P2P能源交易、电碳耦合交易、碳责任分配、区块链可信结算、低碳激励和多智能体强化学习策略优化等关键模块。正式投稿或学位论文使用时，期刊分区建议以学校认可的中科院分区表年份为准。

## 一、筛选说明

- 文献数量：20 篇。
- 期刊类型：英文 SCI 期刊论文。
- 分区口径：优先选择中科院 1 区或常被认定为 1 区的能源、电力、自动化与工程技术类期刊，如 *Applied Energy*、*IEEE Transactions on Smart Grid*、*IEEE Transactions on Power Systems*、*IEEE Transactions on Sustainable Energy*、*IEEE Transactions on Industrial Informatics*、*Engineering* 等。
- 主题覆盖：P2P能源交易、碳责任/碳配额/电碳联合交易、区块链可信交易、低碳激励机制、多智能体强化学习、联邦强化学习、安全强化学习、储能与多微网协同调度。
- 排序原则：优先排列与本文 TECS-Chain 直接相关的“电碳耦合与链上结算”文献，其次排列“强化学习交易策略优化”文献，最后排列 P2P 交易机制、网络约束和激励兼容基础文献。

## 二、文献主要内容总结

### [1] W. Hua 等：融合能源市场与碳市场的区块链 P2P 交易框架

该文提出一种基于区块链的 P2P 交易框架，用于同时交易电能和碳排放许可。文章指出，未来产消者不仅会产生电能交易行为，也会产生与用能行为对应的碳排放责任，因此需要一种能够同时追踪电能交易、碳配额流转和主体行为记录的去中心化机制。

该文利用区块链的透明性、可追溯性和不可篡改性，实现交易记录、碳排放许可和低碳激励的可信管理。其核心贡献在于将“能源交易”和“碳市场”统一到同一 P2P 平台中，使电能交易不再只是经济结算，而是同时包含环境责任结算。

该文与本文 TECS-Chain 的关系最直接：本文拟进一步解决“链下碳责任核算结果如何进入链上协同结算”的问题，并将结算确认后的低碳贡献转化为 Lccoins 奖励信号，嵌入强化学习回报函数中。

### [2] Z. Lu 等：基于碳感知 DLMP 的 P2P 电力与碳联合交易

该文提出基于碳感知配电节点边际电价（Carbon-aware Distribution Locational Marginal Pricing, CDLMP）的 P2P 电力与碳联合交易机制。上层由配电系统运营商基于最优潮流和碳排放流计算电力与碳服务价格，下层由产消者分布式完成电力和碳许可交易。

文章将物理潮流、电力价格和碳责任统一到节点边际价格框架中，使不同节点的电力交易能够反映其真实网络影响和碳排放责任。相比仅在交易结束后进行碳排放统计，该文强调碳信息应内嵌于 P2P 交易出清和价格形成过程。

该文可支撑本文关于“P2P 电量结算与碳责任结算需要显式锚定”的论述。本文在此基础上进一步关注链上协同结算，即将电量结算结果和碳责任结果同步写入可信账本，避免电量收益与碳责任归属脱节。

### [3] H. Ma 等：配电网中电—碳耦合的 P2P 交易

该文提出面向配电网的 P2P 电力—碳耦合交易机制。模型采用上下层结构：上层由配电系统运营商计算碳耦合网络费用，下层由产消者进行电力交易和碳排放权交易决策。

文章结合碳排放流和最优潮流模型，量化 P2P 交易对碳责任和网络运行的影响，并通过分布式优化方法求解双层交互模型。研究表明，将碳责任嵌入 P2P 交易机制后，系统可在提高经济收益的同时降低碳排放。

该文与本文的“电碳协同结算”问题高度相关。不同的是，本文更强调链下联合出清结果与链上原子结算机制之间的严格对应，并希望通过 Lccoins 将经可信确认的低碳贡献转化为可兑现的交易收益。

### [4] D. Qiu 等：智能建筑能源与碳配额联合交易的联邦强化学习

该文研究智能建筑之间的 P2P 能源交易与碳配额交易联合优化问题，提出联邦强化学习方法 Fed-JPC。该方法在不直接共享建筑私有数据的前提下，学习能源交易和碳配额交易的联合策略。

文章将 P2P 能源交易与碳配额市场放入同一决策框架，使智能建筑既根据电价和能源需求调整交易策略，也根据碳配额成本和碳排放责任调整行为。联邦学习机制进一步保护了建筑主体的隐私。

该文对本文具有重要启示：低碳 P2P 交易中的强化学习回报不应只由模型内部设定，而应与真实能源结算和碳责任结算相联系。本文提出将 Lccoins 嵌入 RL 回报，正是对这一思路的进一步深化。

### [5] Y. Zhou 等：基于多智能体深度强化学习的多微网能碳联合交易

该文研究多微电网系统中的能量交易和碳交易联合决策问题，构建一对一直接交易机制，避免集中式多方市场的复杂清算过程。作者将多微网交易建模为马尔可夫决策过程，并提出改进的 MAPPO 方法进行策略学习。

该文将能量交易、碳交易和多智能体强化学习紧密结合。各微网在学习过程中不仅追求自身经济收益，也受到碳交易成本和低碳运行目标影响，因此能够在交易策略层面体现低碳约束。

该文可作为本文“Lccoins 低碳奖励嵌入强化学习回报设计”的直接支撑。不同的是，本文强调低碳奖励信号应来自链上可信结算结果，而不是仅由仿真环境或模型内部奖励函数预设。

### [6] X. Wei 等：面向碳感知产消者的在线加速分布式 P2P 交易

该文研究碳感知产消者的 P2P 能源交易问题，在模型中显式考虑电力碳强度、碳排放约束和隐私保护。作者提出带重球加速的在线分布式算法，并结合差分隐私和遗忘因子处理不确定性与隐私问题。

文章说明，产消者的交易策略会受到碳强度、排放约束和价格信号共同影响。碳责任并不是交易后的附属统计结果，而是影响主体交易决策的重要变量。

该文可支撑本文关于“碳责任需要转化为后续交易策略优化中的反馈信号”的论述。TECS-Chain 通过链上结算确认碳责任，再通过 Lccoins 形成可学习、可兑现的低碳反馈信号。

### [7] J. Wang 等：含光伏和储能的电力与碳共享 Stackelberg 博弈

该文提出含光伏和储能的电力与碳共享框架，设置产消者聚合商同时承担碳聚合商、储能运营商和能源共享服务商的角色。文章构建低碳需求响应模型，并利用 Stackelberg 博弈刻画聚合商定价与产消者响应之间的互动。

该文的重要贡献在于揭示储能在低碳 P2P 交易中的双重作用：储能既参与电量跨时转移，也会影响碳责任在不同时段之间的归属。低碳交易中不能只记录即时电量，还需考虑储能充放电造成的碳责任转移。

该文为本文处理“储能碳责任记忆”和“电碳协同结算”提供支撑。TECS-Chain 可进一步将储能相关电量与碳责任同步结算，避免储能收益与碳责任不匹配。

### [8] C. Wu 等：考虑碳上限交易的社区产消者 P2P 交易优化

该文针对碳上限—交易机制下的社区产消者能源交易问题，提出基于博弈论的优化方法。文章考虑产消者是否从绿色电力市场购电，或是否承担电力市场与碳市场共同成本，并建立非合作博弈、演化博弈和 Stackelberg 博弈相结合的交易模型。

研究表明，将碳上限交易机制引入社区 P2P 交易后，产消者的收益和系统碳排放水平都会发生明显变化。碳约束能够改变主体的交易偏好，使其更倾向于低碳电源和低碳交易路径。

该文可用于支撑本文“低碳 P2P 交易已从单一电量优化扩展到碳成本、碳约束和碳责任协同优化”的研究背景。

### [9] C. Feng 和 A. L. Liu：太阳能与储能 P2P 交易的网络化多智能体强化学习

该文面向含太阳能和储能资源的 P2P 能量交易，提出网络化多智能体强化学习方法。文章指出，普通用户缺乏反复参与 P2P 交易和管理光伏储能资源的专业知识，而可再生能源零边际成本也使公平定价更加困难。

作者采用基于供需比率的市场清算机制，并设计 MARL 框架自动学习用户竞价和储能管理策略。同时，算法将电压约束等物理网络约束纳入学习过程，以保证 P2P 交易的物理可行性。

该文与本文的强化学习交易决策部分高度相关。本文进一步将低碳结算结果作为奖励信号，使强化学习不仅优化经济收益和网络可行性，还持续优化低碳目标。

### [10] T. Chen 等：互联多能源微电网中的 P2P 交易与能量转换 MADRL

该文研究住宅、商业和工业多能源微电网之间的 P2P 能源交易与内部能量转换问题，系统包含电、热、天然气、储能和转换设备。作者提出基于多智能体 actor-critic 和 TD3 的 MATD3 方法，处理连续动作空间和多主体耦合问题。

该文说明，P2P 能源交易通常不是孤立的电力市场行为，而是与微电网内部调度、储能运行、多能转换以及外部购能行为同时发生。强化学习能够在复杂多能耦合系统中学习交易与调度协同策略。

该文可为本文低碳 P2P 交易中的多主体策略学习提供方法基础，尤其适合支撑“交易策略应与资源调度和低碳目标同步优化”的论述。

### [11] Y. Ye 等：本地电力市场中能量交易与灵活性服务协同的 MADRL 方法

该文研究本地电力市场中产消者同时参与本地能源交易和灵活性服务提供的问题。作者提出多智能体注意力 critic 与优先经验回放结合的方法，在包含大量住宅产消者的真实数据场景中学习协调策略。

文章表明，P2P 或本地能源市场中的主体决策往往同时面向多个收益来源，包括电量交易收益、灵活性服务收益和系统调节价值。多智能体强化学习能够刻画主体间相互影响和动态响应过程。

该文对本文的启示在于：低碳激励 Lccoins 也可以被视为一种新的收益来源，应与电量交易收益一起进入主体策略学习过程。

### [12] S. Liu 等：区域互联微电网能源交易与管理的强化学习方法

该文研究区域互联微电网之间的能源交易和运行管理问题，采用强化学习方法在可再生能源和负荷不确定条件下学习交易及调度策略。文章强调，区域微电网通过互联交易可以降低运行成本并提升可再生能源利用水平。

该文说明，微电网之间的能源交易本质上是一个动态决策问题，受到可再生能源出力、负荷需求、交易价格和储能状态等因素影响。强化学习适合处理这类不确定性和时序决策问题。

该文可作为本文“P2P 交易策略需要在动态环境中持续演化”的支撑文献。本文进一步将动态演化目标从经济优化扩展到电碳协同优化。

### [13] D. Qiu 等：可扩展 P2P 能源交易协调管理的多簇深度强化学习

该文针对大规模 P2P 能源交易的可扩展性问题，提出多簇深度强化学习方法。通过将大量产消者划分为多个簇，降低全局协调复杂度，并在局部学习和整体协调之间取得平衡。

文章表明，当 P2P 交易主体数量增加时，直接进行全局多智能体学习会面临维度灾难、训练不稳定和通信成本过高等问题。分簇学习能够增强模型的可扩展性。

该文对本文有直接启示：TECS-Chain 面向动态参与主体时，需要考虑主体数量变化和跨规模泛化问题，多簇或分层强化学习可作为实现路径。

### [14] L. Yan 等：面向智能住户社区的分层深度强化学习交易机制

该文研究智能住户社区中的能源交易与用能调度问题，提出分层深度强化学习框架。内层使用多智能体强化学习实现住户侧家电调度，外层使用深度强化学习根据历史净负荷和电价信息形成社区内部交易价格。

该方法将“用户用能决策”和“社区市场定价”分层处理，适合刻画居民侧异质性、随机性和隐私约束。对于 P2P 交易而言，定价策略和用能策略互相影响，应在统一框架中协同优化。

该文为本文低碳激励设计提供启示：Lccoins 可以作为外层价格或收益信号的一部分，影响内层用户策略，从而形成电碳协同的层级反馈机制。

### [15] M. I. Azim 等：基于硬件在环验证的合作式 P2P 交易框架

该文面向配电网内产消者的 P2P 能源交易，提出一种合作博弈交易框架，将交易收益分配、网络运行约束和主体参与激励统一考虑。文章不仅在软件仿真中验证机制有效性，还通过实时数字仿真和硬件在环平台检验其实际运行可行性。

该文的重要价值在于将 P2P 交易从纯算法设计推进到接近工程验证的层面。合作联盟形成、稳定收益分摊和物理网络约束处理，是构建可落地 P2P 交易机制的关键。

该文对本文的启示在于，TECS-Chain 不应只停留在概念层面，还需要考虑链上结算延迟、交易执行可靠性和实际系统验证。

### [16] Z. Guo 等：高效率且激励兼容的 P2P 能源交易机制

该文提出一种高效率、激励兼容的 P2P 能源交易机制，目标是在缺乏完全信任的去中心化环境中鼓励产消者如实报价和参与合作。文章关注真实性、个体理性、预算平衡和效率问题，并设计匹配与价格机制以提升 P2P 交易可实施性。

该文可用于支撑本文中“低碳激励必须具有可信基础和可兑现收益”的论述。若低碳奖励无法与真实结算结果对应，主体可能缺乏如实申报和持续参与的动力。

本文设计 Lccoins 的目的，正是在链上可信结算的基础上建立差异化、可兑现的低碳激励，从而增强激励兼容性。

### [17] N. Tarashandeh 和 A. Karimi：保持主体独立性的网络约束 P2P 交易

该文研究在考虑配电网络约束的同时，如何保持 P2P 交易主体的独立性。文章指出，完全去除配电系统运营商可能威胁网络安全，而过度集中控制又会削弱代理自治性。

作者提出基于 ADMM 的分布式 P2P 市场框架，并利用电压、电流对有功和无功功率变化的灵敏度系数，将网络安全约束嵌入每个主体的子问题中。

该文对本文的启示在于：P2P 交易需要在“去中心化主体自治”和“系统安全约束”之间取得平衡。TECS-Chain 的链上结算也应避免过度中心化，同时确保交易与碳责任可验证。

### [18] G. Belgioioso 等：运行安全的 P2P 能源交易博弈清算机制

该文将配电网中的 P2P 能源交易建模为广义聚合博弈，由网络运营商负责执行系统运行约束。文章设计了一个分布式市场清算机制，并证明其能够收敛到经济高效、策略稳定且运行安全的变分广义纳什均衡。

该文的重要贡献在于将市场清算和配电网安全运行统一到一个博弈论框架中，使 P2P 交易不仅满足经济性，也满足网络电压、电流等物理约束。

该文可作为本文链下联合出清模块的理论支撑。TECS-Chain 可在链下完成安全可行的电碳联合出清，并将结果映射到链上协同结算。

### [19] C. Feng 等：基于广义快速对偶上升的网络约束 P2P 交易

该文提出事件驱动的本地 P2P 电力市场框架，用于支持短期或即时本地电能交易。文章通过节点电压和网络损耗对节点注入功率的灵敏度分析，将 P2P 交易对配电网的影响内生化。

作者进一步采用广义快速对偶上升方法实现高效分布式市场出清。结果表明，该方法能够保证配电系统安全运行，并具有较好的收敛性能。

该文对本文的启示在于：链下联合出清应具备实时性和分布式可扩展性，否则链上结算即使可信，也难以满足 P2P 交易的运行需求。

### [20] S. C. Das 等：区块链集成的软件定义去中心化 P2P 能源交易网络

该文提出一种区块链集成的软件定义去中心化 P2P 能源交易网络，用于可持续智能电网。框架将区块链与软件定义网络（SDN）结合，以实现安全交易、网络流控制和实时自适应，同时降低对中介机构的依赖。

文章还设计了矿工和领导节点选择算法，综合考虑能量水平、信誉值和停留时间，以提高交易验证效率并降低区块链运行开销。该文从通信网络、交易安全和交易延迟角度补充了区块链 P2P 能源交易的技术实现路径。

该文可为 TECS-Chain 的链上协同结算层提供架构启示，尤其是交易吞吐、结算延迟、节点信誉和共识机制设计。

## 三、标准格式参考文献

[1] W. Hua, J. Jiang, H. Sun, and J. Wu, “A Blockchain Based Peer-to-Peer Trading Framework Integrating Energy and Carbon Markets,” *Applied Energy*, vol. 279, Art. no. 115539, 2020. DOI: 10.1016/j.apenergy.2020.115539.

[2] Z. Lu, L. Bai, J. Wang, J. Wei, Y. Xiao, and Y. Chen, “Peer-to-Peer Joint Electricity and Carbon Trading Based on Carbon-Aware Distribution Locational Marginal Pricing,” *IEEE Transactions on Power Systems*, vol. 38, no. 1, pp. 835–852, 2023. DOI: 10.1109/TPWRS.2022.3167780.

[3] H. Ma, Y. Xiang, A. P. Zhao, S. Li, and J. Liu, “Optimal Peer-to-Peer Coupled Electricity and Carbon Trading in Distribution Networks,” *Engineering*, vol. 51, pp. 37–48, 2025.

[4] D. Qiu, J. Xue, T. Zhang, J. Wang, and M. Sun, “Federated Reinforcement Learning for Smart Building Joint Peer-to-Peer Energy and Carbon Allowance Trading,” *Applied Energy*, vol. 333, Art. no. 120526, 2023. DOI: 10.1016/j.apenergy.2022.120526.

[5] Y. Zhou, Z. Ma, T. Wang, J. Zhang, X. Shi, and S. Zou, “Joint Energy and Carbon Trading for Multi-Microgrid System Based on Multi-Agent Deep Reinforcement Learning,” *IEEE Transactions on Power Systems*, vol. 39, no. 6, pp. 7376–7388, 2024. DOI: 10.1109/TPWRS.2024.3380070.

[6] X. Wei, Y. Xu, H. Sun, and W. K. Chan, “Peer-to-Peer Energy Trading of Carbon-Aware Prosumers: An Online Accelerated Distributed Approach With Differential Privacy,” *IEEE Transactions on Smart Grid*, vol. 15, no. 6, pp. 5595–5609, 2024. DOI: 10.1109/TSG.2024.3398041.

[7] J. Wang, X. Jin, H. Jia, M. Tostado-Véliz, Y. Mu, X. Yu, and S. Liang, “Joint Electricity and Carbon Sharing With PV and Energy Storage: A Low-Carbon DR-Based Game Theoretic Approach,” *IEEE Transactions on Sustainable Energy*, vol. 15, no. 4, pp. 2703–2717, 2024. DOI: 10.1109/TSTE.2024.3439512.

[8] C. Wu, X. Chen, H. Hua, K. Yu, L. Gan, J. Shen, and Y. Ding, “Peer-to-Peer Energy Trading Optimization for Community Prosumers Considering Carbon Cap-and-Trade,” *Applied Energy*, vol. 358, Art. no. 122611, 2024. DOI: 10.1016/j.apenergy.2023.122611.

[9] C. Feng and A. L. Liu, “Peer-to-Peer Energy Trading of Solar and Energy Storage: A Networked Multiagent Reinforcement Learning Approach,” *Applied Energy*, vol. 383, Art. no. 125283, 2025. DOI: 10.1016/j.apenergy.2025.125283.

[10] T. Chen, S. Bu, X. Liu, J. Kang, F. R. Yu, and Z. Han, “Peer-to-Peer Energy Trading and Energy Conversion in Interconnected Multi-Energy Microgrids Using Multi-Agent Deep Reinforcement Learning,” *IEEE Transactions on Smart Grid*, vol. 13, no. 1, pp. 715–727, 2022. DOI: 10.1109/TSG.2021.3124465.

[11] Y. Ye, D. Papadaskalopoulos, Q. Yuan, Y. Tang, and G. Strbac, “Multi-Agent Deep Reinforcement Learning for Coordinated Energy Trading and Flexibility Services Provision in Local Electricity Markets,” *IEEE Transactions on Smart Grid*, vol. 14, no. 2, pp. 1541–1554, 2023. DOI: 10.1109/TSG.2022.3149266.

[12] S. Liu, S. Han, and S. Zhu, “Reinforcement Learning-Based Energy Trading and Management of Regional Interconnected Microgrids,” *IEEE Transactions on Smart Grid*, vol. 14, no. 3, pp. 2047–2059, 2023. DOI: 10.1109/TSG.2022.3214202.

[13] D. Qiu, Y. Ye, D. Papadaskalopoulos, and G. Strbac, “Scalable Coordinated Management of Peer-to-Peer Energy Trading: A Multi-Cluster Deep Reinforcement Learning Approach,” *Applied Energy*, vol. 292, Art. no. 116940, 2021. DOI: 10.1016/j.apenergy.2021.116940.

[14] L. Yan, X. Chen, Y. Chen, and J. Wen, “A Hierarchical Deep Reinforcement Learning-Based Community Energy Trading Scheme for a Neighborhood of Smart Households,” *IEEE Transactions on Smart Grid*, vol. 13, no. 6, pp. 4747–4758, 2022. DOI: 10.1109/TSG.2022.3181329.

[15] M. I. Azim, M. R. Alam, W. Tushar, T. K. Saha, and C. Yuen, “A Cooperative P2P Trading Framework: Developed and Validated Through Hardware-in-Loop,” *IEEE Transactions on Smart Grid*, vol. 14, no. 4, pp. 2999–3015, 2023. DOI: 10.1109/TSG.2022.3225520.

[16] Z. Guo, B. Qin, Z. Guan, Y. Wang, H. Zheng, and Q. Wu, “A High-Efficiency and Incentive-Compatible Peer-to-Peer Energy Trading Mechanism,” *IEEE Transactions on Smart Grid*, vol. 15, no. 1, pp. 1075–1088, 2024. DOI: 10.1109/TSG.2023.3275533.

[17] N. Tarashandeh and A. Karimi, “Peer-to-Peer Energy Trading under Distribution Network Constraints with Preserving Independent Nature of Agents,” *Applied Energy*, vol. 355, Art. no. 122240, 2024. DOI: 10.1016/j.apenergy.2023.122240.

[18] G. Belgioioso, W. W. Ananduta, S. Grammatico, and C. Ocampo-Martínez, “Operationally-Safe Peer-to-Peer Energy Trading in Distribution Grids: A Game-Theoretic Market-Clearing Mechanism,” *IEEE Transactions on Smart Grid*, vol. 13, no. 4, pp. 2897–2907, 2022. DOI: 10.1109/TSG.2022.3161273.

[19] C. Feng, B. Liang, Z. Li, W. Liu, and F. Wen, “Peer-to-Peer Energy Trading under Network Constraints Based on Generalized Fast Dual Ascent,” *IEEE Transactions on Smart Grid*, vol. 14, no. 2, pp. 1441–1453, 2023. DOI: 10.1109/TSG.2022.3149210.

[20] S. C. Das, U. K. Acharjee, M. J. Islam, M. A. Islam, A. Rahman, M. A. Kabir, and G. Muhammad, “Blockchain-Integrated Software Defined Decentralized Peer-to-Peer Energy Trading Network for Sustainable Smart Power Grid,” *Applied Energy*, vol. 411, Art. no. 127569, 2026. DOI: 10.1016/j.apenergy.2026.127569.
