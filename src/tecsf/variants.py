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


VARIANTS: dict[str, VariantSpec] = {
    "tecsf": VariantSpec("tecsf"),
    "no_chain": VariantSpec("no_chain", use_chain=False),
    "no_lccoins": VariantSpec("no_lccoins", use_lccoins=False),
    "no_feedback": VariantSpec("no_feedback", use_feedback=False),
    "mappo": VariantSpec("mappo", use_recurrence=False),
    "no_lagrange": VariantSpec("no_lagrange", use_lagrange=False),
    "heuristic": VariantSpec("heuristic", use_recurrence=False, learning=False),
}


def get_variant(name: str) -> VariantSpec:
    try:
        return VARIANTS[name]
    except KeyError as exc:
        known = ", ".join(sorted(VARIANTS))
        raise ValueError(f"Unknown variant '{name}'. Known variants: {known}") from exc
