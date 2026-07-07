"""低碳资产感知 MAPPO 原型包.

内部包名 tecsf 保留为技术标识符，等价于论文中的
'低碳资产感知的 MAPPO 算法'（LC-MAPPO）。
"""

from tecsf.config import ExperimentConfig, load_config
from tecsf.data import SyntheticScenario, generate_synthetic_scenario
from tecsf.env import EnergyCarbonEnv

__all__ = [
    "EnergyCarbonEnv",
    "ExperimentConfig",
    "SyntheticScenario",
    "generate_synthetic_scenario",
    "load_config",
]
