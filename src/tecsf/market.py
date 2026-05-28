from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tecsf.carbon import CarbonResult, compute_carbon_result
from tecsf.config import ExperimentConfig
from tecsf.data import SyntheticScenario
from tecsf.utils import stable_hash, to_jsonable


ACTION_DIM = 6
VIOLATION_KEYS = ("voltage", "line", "soc", "trade")


@dataclass
class ActionBatch:
    q_buy: np.ndarray
    q_sell: np.ndarray
    price_buy: np.ndarray
    price_sell: np.ndarray
    charge: np.ndarray
    discharge: np.ndarray


@dataclass
class ClearingPackage:
    epoch: int
    actions: ActionBatch
    p2p_power: np.ndarray
    p2p_price: np.ndarray
    grid_buy: np.ndarray
    grid_sell: np.ndarray
    charge: np.ndarray
    discharge: np.ndarray
    soc_next: np.ndarray
    node_injection: np.ndarray
    line_flow: np.ndarray
    voltages: np.ndarray
    violations: dict[str, float]
    carbon: CarbonResult
    package_hash: str

    def payload(self, include_hash: bool = False) -> dict:
        payload = {
            "epoch": self.epoch,
            "actions": self.actions,
            "p2p_power": self.p2p_power,
            "p2p_price": self.p2p_price,
            "grid_buy": self.grid_buy,
            "grid_sell": self.grid_sell,
            "charge": self.charge,
            "discharge": self.discharge,
            "soc_next": self.soc_next,
            "node_injection": self.node_injection,
            "line_flow": self.line_flow,
            "voltages": self.voltages,
            "violations": self.violations,
            "carbon": self.carbon,
        }
        if include_hash:
            payload["package_hash"] = self.package_hash
        return to_jsonable(payload)

    def max_violation(self) -> float:
        return max(float(v) for v in self.violations.values()) if self.violations else 0.0


def scale_raw_actions(raw_actions: np.ndarray, config: ExperimentConfig) -> ActionBatch:
    raw = np.asarray(raw_actions, dtype=np.float32)
    clipped = np.clip(raw, -1.0, 1.0)
    m = config.market
    s = config.storage
    price_span = m.p2p_price_max - m.p2p_price_min
    return ActionBatch(
        q_buy=(np.maximum(clipped[:, 0], 0.0) * m.max_buy_power).astype(np.float32),
        q_sell=(np.maximum(clipped[:, 1], 0.0) * m.max_sell_power).astype(np.float32),
        price_buy=(m.p2p_price_min + (clipped[:, 2] + 1.0) * 0.5 * price_span).astype(
            np.float32
        ),
        price_sell=(m.p2p_price_min + (clipped[:, 3] + 1.0) * 0.5 * price_span).astype(
            np.float32
        ),
        charge=(np.maximum(clipped[:, 4], 0.0) * s.max_charge_power).astype(np.float32),
        discharge=(np.maximum(clipped[:, 5], 0.0) * s.max_discharge_power).astype(np.float32),
    )


def normalize_physical_actions(actions: ActionBatch, config: ExperimentConfig) -> np.ndarray:
    m = config.market
    s = config.storage
    span = max(m.p2p_price_max - m.p2p_price_min, 1e-8)
    out = np.zeros((actions.q_buy.shape[0], ACTION_DIM), dtype=np.float32)
    out[:, 0] = actions.q_buy / max(m.max_buy_power, 1e-8)
    out[:, 1] = actions.q_sell / max(m.max_sell_power, 1e-8)
    out[:, 2] = (actions.price_buy - m.p2p_price_min) / span
    out[:, 3] = (actions.price_sell - m.p2p_price_min) / span
    out[:, 4] = actions.charge / max(s.max_charge_power, 1e-8)
    out[:, 5] = actions.discharge / max(s.max_discharge_power, 1e-8)
    return np.clip(out, 0.0, 1.0)


def double_auction(actions: ActionBatch) -> tuple[np.ndarray, np.ndarray]:
    n = actions.q_buy.shape[0]
    power = np.zeros((n, n), dtype=np.float32)
    price = np.zeros((n, n), dtype=np.float32)
    buy_order = sorted(range(n), key=lambda i: float(actions.price_buy[i]), reverse=True)
    sell_order = sorted(range(n), key=lambda i: float(actions.price_sell[i]))
    buy_rem = actions.q_buy.astype(np.float32).copy()
    sell_rem = actions.q_sell.astype(np.float32).copy()

    for buyer in buy_order:
        if buy_rem[buyer] <= 1e-8:
            continue
        for seller in sell_order:
            if seller == buyer or sell_rem[seller] <= 1e-8:
                continue
            if actions.price_buy[buyer] + 1e-8 < actions.price_sell[seller]:
                break
            qty = min(float(buy_rem[buyer]), float(sell_rem[seller]))
            if qty <= 1e-8:
                continue
            trade_price = 0.5 * (
                float(actions.price_buy[buyer]) + float(actions.price_sell[seller])
            )
            old_qty = float(power[seller, buyer])
            new_qty = old_qty + qty
            power[seller, buyer] = new_qty
            price[seller, buyer] = (
                (float(price[seller, buyer]) * old_qty + trade_price * qty) / new_qty
            )
            buy_rem[buyer] -= qty
            sell_rem[seller] -= qty
            if buy_rem[buyer] <= 1e-8:
                break
    return power, price


def _apply_storage_limits(
    soc: np.ndarray, actions: ActionBatch, config: ExperimentConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    s = config.storage
    charge = np.clip(actions.charge, 0.0, s.max_charge_power).astype(np.float32)
    discharge = np.clip(actions.discharge, 0.0, s.max_discharge_power).astype(np.float32)

    both = (charge > 1e-8) & (discharge > 1e-8)
    keep_charge = charge >= discharge
    discharge[both & keep_charge] = 0.0
    charge[both & ~keep_charge] = 0.0

    max_charge_by_soc = np.maximum(0.0, (s.soc_max - soc) / (s.charge_efficiency * config.scenario.delta_t))
    max_discharge_by_soc = np.maximum(
        0.0, (soc - s.soc_min) * s.discharge_efficiency / config.scenario.delta_t
    )
    requested_charge = charge.copy()
    requested_discharge = discharge.copy()
    charge = np.minimum(charge, max_charge_by_soc).astype(np.float32)
    discharge = np.minimum(discharge, max_discharge_by_soc).astype(np.float32)
    soc_next = (
        soc
        + s.charge_efficiency * charge * config.scenario.delta_t
        - discharge * config.scenario.delta_t / s.discharge_efficiency
    ).astype(np.float32)
    soc_next = np.clip(soc_next, s.soc_min, s.soc_max)
    violation = float(
        np.maximum(requested_charge - charge, 0.0).sum()
        + np.maximum(requested_discharge - discharge, 0.0).sum()
    )
    return charge, discharge, soc_next, violation


def _balance_agents(
    load: np.ndarray,
    pv: np.ndarray,
    charge: np.ndarray,
    discharge: np.ndarray,
    p2p_power: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    p2p_sold = p2p_power.sum(axis=1)
    p2p_bought = p2p_power.sum(axis=0)
    deficit = load + charge + p2p_sold - pv - discharge - p2p_bought
    grid_buy = np.maximum(deficit, 0.0).astype(np.float32)
    grid_sell = np.maximum(-deficit, 0.0).astype(np.float32)
    return grid_buy, grid_sell


def _network_state(
    scenario: SyntheticScenario,
    config: ExperimentConfig,
    p2p_power: np.ndarray,
    grid_buy: np.ndarray,
    grid_sell: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    n_nodes = scenario.num_nodes
    injection = np.zeros(n_nodes, dtype=np.float32)
    p2p_sold = p2p_power.sum(axis=1)
    p2p_bought = p2p_power.sum(axis=0)
    agent_inj = p2p_sold + grid_sell - p2p_bought - grid_buy
    for agent, node in enumerate(scenario.agent_nodes):
        injection[int(node)] += agent_inj[agent]

    children: dict[int, list[tuple[int, int]]] = {i: [] for i in range(n_nodes)}
    for idx, (parent, child) in enumerate(zip(scenario.line_from, scenario.line_to)):
        children[int(parent)].append((int(child), idx))

    subtree_inj = injection.copy()
    line_flow = np.zeros(len(scenario.line_to), dtype=np.float32)
    for child in range(n_nodes - 1, 0, -1):
        parent_candidates = np.where(scenario.line_to == child)[0]
        if parent_candidates.size == 0:
            continue
        line_idx = int(parent_candidates[0])
        parent = int(scenario.line_from[line_idx])
        flow = -subtree_inj[child]
        line_flow[line_idx] = flow
        subtree_inj[parent] += subtree_inj[child]

    voltages = np.full(n_nodes, config.network.voltage_ref, dtype=np.float32)
    for parent in range(n_nodes):
        for child, line_idx in children[parent]:
            voltages[child] = voltages[parent] - 2.0 * scenario.resistance[line_idx] * line_flow[line_idx]

    voltage_low = np.maximum(config.network.voltage_min - voltages, 0.0)
    voltage_high = np.maximum(voltages - config.network.voltage_max, 0.0)
    line_over = np.maximum(np.abs(line_flow) - scenario.line_capacity, 0.0)
    violations = {
        "voltage": float(voltage_low.sum() + voltage_high.sum()),
        "line": float(line_over.sum()),
        "trade": float(np.maximum(np.diag(p2p_power), 0.0).sum()),
    }
    return injection, line_flow, voltages, violations


def clear_market(
    scenario: SyntheticScenario,
    config: ExperimentConfig,
    t: int,
    soc: np.ndarray,
    actions: ActionBatch,
) -> ClearingPackage:
    idx = scenario.time_index(t)
    load = scenario.load[:, idx]
    pv = scenario.pv[:, idx]
    charge, discharge, soc_next, soc_violation = _apply_storage_limits(soc, actions, config)

    p2p_power, p2p_price = double_auction(actions)
    for _ in range(config.clearing.max_repair_iters + 1):
        grid_buy, grid_sell = _balance_agents(load, pv, charge, discharge, p2p_power)
        node_injection, line_flow, voltages, violations = _network_state(
            scenario, config, p2p_power, grid_buy, grid_sell
        )
        violations["soc"] = soc_violation
        if max(violations.values()) <= config.clearing.network_tolerance:
            break
        p2p_power = (p2p_power * config.clearing.repair_shrink).astype(np.float32)

    p2p_sold = p2p_power.sum(axis=1)
    carbon = compute_carbon_result(
        scenario=scenario,
        config=config,
        t=t,
        load=load,
        pv=pv,
        charge=charge,
        p2p_sell=p2p_sold,
        grid_buy=grid_buy,
        grid_sell=grid_sell,
    )
    package = ClearingPackage(
        epoch=t,
        actions=actions,
        p2p_power=p2p_power.astype(np.float32),
        p2p_price=p2p_price.astype(np.float32),
        grid_buy=grid_buy.astype(np.float32),
        grid_sell=grid_sell.astype(np.float32),
        charge=charge.astype(np.float32),
        discharge=discharge.astype(np.float32),
        soc_next=soc_next.astype(np.float32),
        node_injection=node_injection.astype(np.float32),
        line_flow=line_flow.astype(np.float32),
        voltages=voltages.astype(np.float32),
        violations={k: float(violations.get(k, 0.0)) for k in VIOLATION_KEYS},
        carbon=carbon,
        package_hash="",
    )
    package.package_hash = stable_hash(package.payload(include_hash=False))
    return package
