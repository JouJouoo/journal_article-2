from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VariantSpec:
    name: str
    use_chain: bool = True
    use_lccoins: bool = True
    use_feedback: bool = True
    use_recurrence: bool = True
    use_lagrange: bool = True
    learning: bool = True
    use_preset_low_carbon_reward: bool = False


VARIANTS: dict[str, VariantSpec] = {
    "tecsf": VariantSpec("tecsf"),
    "no_chain": VariantSpec("no_chain", use_chain=False),
    "no_lccoins": VariantSpec("no_lccoins", use_lccoins=False),
    "no_feedback": VariantSpec("no_feedback", use_feedback=False),
    "mappo": VariantSpec("mappo", use_recurrence=False),
    "constrained_mappo": VariantSpec(
        "constrained_mappo",
        use_chain=False,
        use_lccoins=False,
        use_feedback=False,
        use_recurrence=False,
    ),
    "safety_only": VariantSpec(
        "safety_only",
        use_chain=False,
        use_lccoins=False,
        use_feedback=False,
    ),
    "myopic_opt": VariantSpec(
        "myopic_opt",
        use_lccoins=False,
        use_feedback=False,
        use_recurrence=False,
        use_lagrange=False,
        learning=False,
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
        use_feedback=False,
        use_preset_low_carbon_reward=True,
    ),
}


def get_variant(name: str) -> VariantSpec:
    try:
        return VARIANTS[name]
    except KeyError as exc:
        known = ", ".join(sorted(VARIANTS))
        raise ValueError(f"Unknown variant '{name}'. Known variants: {known}") from exc
