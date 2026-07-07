from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VariantSpec:
    name: str
    use_chain: bool = True
    use_lccoins: bool = True
    use_recurrence: bool = True
    use_lagrange: bool = True
    learning: bool = True
    use_asset_observation: bool = True
    use_asset_utility: bool = True
    use_dual_advantage: bool = True
    use_credit_assignment: bool = True
    use_adaptive_clip: bool = True


# 变体名 "tecsf" 和 "lc_mappo" 均等价于论文中的
# "低碳资产感知的 MAPPO 算法"（LC-MAPPO），保留 "tecsf" 仅为向后兼容。
#
# 消融设计（3 个变体）：
#   tecsf / lc_mappo — 完整方法（LC-MAPPO），全部组件开启。
#   no_chain         — 消融：关闭整个区块链部分（共识结算 + LCCoins 铸造
#                      + 资产感知反馈至 RL），保留循环网络和拉格朗日安全约束。
#   mappo            — 基线：标准 MAPPO，无区块链、无 LCCoins、无循环网络、
#                      无资产感知，保留拉格朗日安全约束（环境级）。
VARIANTS: dict[str, VariantSpec] = {
    "tecsf": VariantSpec("tecsf"),
    "lc_mappo": VariantSpec("lc_mappo"),
    "no_chain": VariantSpec(
        "no_chain",
        use_chain=False,
        use_lccoins=False,
        use_asset_observation=False,
        use_asset_utility=False,
        use_dual_advantage=False,
        use_credit_assignment=False,
        use_adaptive_clip=False,
    ),
    "mappo": VariantSpec(
        "mappo",
        use_chain=False,
        use_lccoins=False,
        use_recurrence=False,
        use_asset_observation=False,
        use_asset_utility=False,
        use_dual_advantage=False,
        use_credit_assignment=False,
        use_adaptive_clip=False,
    ),
}


def get_variant(name: str) -> VariantSpec:
    try:
        return VARIANTS[name]
    except KeyError as exc:
        known = ", ".join(sorted(VARIANTS))
        raise ValueError(f"Unknown variant '{name}'. Known variants: {known}") from exc
