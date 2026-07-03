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
    use_preset_low_carbon_reward: bool = False
    use_asset_observation: bool = True
    use_asset_utility: bool = True
    use_dual_advantage: bool = True
    use_credit_assignment: bool = True
    use_adaptive_clip: bool = True


VARIANTS: dict[str, VariantSpec] = {
    "tecsf": VariantSpec("tecsf"),
    "lc_mappo": VariantSpec("lc_mappo"),
    "no_chain": VariantSpec("no_chain", use_chain=False),
    "no_lccoins": VariantSpec(
        "no_lccoins",
        use_lccoins=False,
        use_asset_observation=False,
        use_asset_utility=False,
        use_dual_advantage=False,
        use_credit_assignment=False,
        use_adaptive_clip=False,
    ),
    "mappo": VariantSpec(
        "mappo",
        use_lccoins=False,
        use_recurrence=False,
        use_asset_observation=False,
        use_asset_utility=False,
        use_dual_advantage=False,
        use_credit_assignment=False,
        use_adaptive_clip=False,
    ),
    "constrained_mappo": VariantSpec(
        "constrained_mappo",
        use_chain=False,
        use_lccoins=False,
        use_recurrence=False,
        use_asset_observation=False,
        use_asset_utility=False,
        use_dual_advantage=False,
        use_credit_assignment=False,
        use_adaptive_clip=False,
    ),
    "safety_only": VariantSpec(
        "safety_only",
        use_chain=False,
        use_lccoins=False,
        use_asset_observation=False,
        use_asset_utility=False,
        use_dual_advantage=False,
        use_credit_assignment=False,
        use_adaptive_clip=False,
    ),
    "myopic_opt": VariantSpec(
        "myopic_opt",
        use_lccoins=False,
        use_recurrence=False,
        use_lagrange=False,
        learning=False,
        use_asset_observation=False,
        use_asset_utility=False,
        use_dual_advantage=False,
        use_credit_assignment=False,
        use_adaptive_clip=False,
    ),
    "greedy_feasible": VariantSpec(
        "greedy_feasible",
        use_recurrence=False,
        learning=False,
    ),
    "no_lagrange": VariantSpec("no_lagrange", use_lagrange=False),
    "heuristic": VariantSpec("heuristic", use_recurrence=False, learning=False),
    "preset_low_carbon": VariantSpec(
        "preset_low_carbon",
        use_chain=False,
        use_lccoins=False,
        use_preset_low_carbon_reward=True,
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
