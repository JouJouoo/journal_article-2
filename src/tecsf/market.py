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
    attempted_p2p_power: np.ndarray
    attempted_charge: np.ndarray
    attempted_discharge: np.ndarray
    effective_load: np.ndarray
    effective_pv: np.ndarray
    load_shed: np.ndarray
    pv_curtailment: np.ndarray
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
    repair_iterations: int
    safety_adjustment: float
    action_bound_deviation: float
    package_hash: str

    def payload(self, include_hash: bool = False) -> dict:
        payload = {
            "epoch": self.epoch,
            "actions": self.actions,
            "attempted_p2p_power": self.attempted_p2p_power,
            "attempted_charge": self.attempted_charge,
            "attempted_discharge": self.attempted_discharge,
            "effective_load": self.effective_load,
            "effective_pv": self.effective_pv,
            "load_shed": self.load_shed,
            "pv_curtailment": self.pv_curtailment,
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
            "repair_iterations": self.repair_iterations,
            "safety_adjustment": self.safety_adjustment,
            "action_bound_deviation": self.action_bound_deviation,
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


def p2p_reference_price(
    scenario: SyntheticScenario, config: ExperimentConfig, t: int
) -> float:
    idx = scenario.time_index(t)
    reference = 0.5 * (
        float(scenario.grid_buy_price[idx])
        + float(scenario.grid_sell_price[idx])
    )
    return float(
        np.clip(
            reference,
            float(config.market.p2p_price_min),
            float(config.market.p2p_price_max),
        )
    )


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


def _apply_storage_arrays(
    soc: np.ndarray,
    requested_charge: np.ndarray,
    requested_discharge: np.ndarray,
    config: ExperimentConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    s = config.storage
    charge = np.clip(requested_charge, 0.0, s.max_charge_power).astype(np.float32)
    discharge = np.clip(requested_discharge, 0.0, s.max_discharge_power).astype(np.float32)

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


def _apply_storage_limits(
    soc: np.ndarray, actions: ActionBatch, config: ExperimentConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    return _apply_storage_arrays(soc, actions.charge, actions.discharge, config)


def _storage_headroom(
    soc: np.ndarray, config: ExperimentConfig
) -> tuple[np.ndarray, np.ndarray]:
    s = config.storage
    dt = max(config.scenario.delta_t, 1e-8)
    max_charge_by_soc = np.maximum(0.0, (s.soc_max - soc) / (s.charge_efficiency * dt))
    max_discharge_by_soc = np.maximum(0.0, (soc - s.soc_min) * s.discharge_efficiency / dt)
    max_charge = np.minimum(s.max_charge_power, max_charge_by_soc).astype(np.float32)
    max_discharge = np.minimum(s.max_discharge_power, max_discharge_by_soc).astype(np.float32)
    return max_charge, max_discharge


def _project_actions_to_local_bounds(
    load: np.ndarray,
    pv: np.ndarray,
    soc: np.ndarray,
    actions: ActionBatch,
    config: ExperimentConfig,
) -> tuple[ActionBatch, float]:
    """Project raw physical actions to bounds implied by local net demand.

    The global action space is intentionally broad for synthetic scenarios, but
    standard feeders can have bus-level loads far below those global limits.
    This projection prevents an actor from creating large exports by discharging
    storage far beyond local demand, which was the reproducible IEEE 33-bus
    failure mode.
    """

    if not config.clearing.enable_dynamic_action_bounds:
        return actions, 0.0

    trade_margin = max(float(config.clearing.local_trade_margin), 0.0)
    storage_margin = max(float(config.clearing.local_storage_margin), 0.0)
    tolerance = float(config.clearing.network_tolerance)
    local_deficit = np.maximum(load - pv, 0.0).astype(np.float32)
    local_surplus = np.maximum(pv - load, 0.0).astype(np.float32)
    max_charge, max_discharge = _storage_headroom(soc, config)

    charge_cap = np.minimum(max_charge, local_surplus * storage_margin + tolerance)
    discharge_cap = np.minimum(max_discharge, local_deficit * storage_margin + tolerance)
    charge = np.minimum(actions.charge, charge_cap).astype(np.float32)
    discharge = np.minimum(actions.discharge, discharge_cap).astype(np.float32)

    residual = load + charge - pv - discharge
    buy_cap = np.minimum(
        config.market.max_buy_power,
        np.maximum(residual, 0.0) * trade_margin + tolerance,
    )
    sell_cap = np.minimum(
        config.market.max_sell_power,
        np.maximum(-residual, 0.0) * trade_margin + tolerance,
    )
    q_buy = np.minimum(actions.q_buy, buy_cap).astype(np.float32)
    q_sell = np.minimum(actions.q_sell, sell_cap).astype(np.float32)

    deviation = float(
        np.maximum(actions.q_buy - q_buy, 0.0).sum()
        + np.maximum(actions.q_sell - q_sell, 0.0).sum()
        + np.maximum(actions.charge - charge, 0.0).sum()
        + np.maximum(actions.discharge - discharge, 0.0).sum()
    )
    return (
        ActionBatch(
            q_buy=q_buy,
            q_sell=q_sell,
            price_buy=actions.price_buy,
            price_sell=actions.price_sell,
            charge=charge,
            discharge=discharge,
        ),
        deviation,
    )


def _storage_safety_dispatch(
    load: np.ndarray,
    pv: np.ndarray,
    soc: np.ndarray,
    charge: np.ndarray,
    discharge: np.ndarray,
    config: ExperimentConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Project storage actions toward lower feeder net injections.

    P2P clearing redistributes energy payments, but radial line loading is driven
    by physical net injection after load, PV, and storage. When the network is
    already violating constraints, this shield uses available storage headroom to
    discharge at net-load agents and charge at net-surplus agents before the
    package is submitted for settlement.
    """

    fraction = float(np.clip(config.clearing.safety_storage_fraction, 0.0, 1.0))
    if fraction <= 0.0:
        return charge, discharge

    net_load = load - pv
    max_charge, max_discharge = _storage_headroom(soc, config)

    safe_charge = charge.astype(np.float32).copy()
    safe_discharge = discharge.astype(np.float32).copy()
    deficit = net_load > config.clearing.network_tolerance
    surplus = net_load < -config.clearing.network_tolerance

    target_discharge = np.minimum(np.maximum(net_load, 0.0) * fraction, max_discharge)
    target_charge = np.minimum(np.maximum(-net_load, 0.0) * fraction, max_charge)
    safe_charge[deficit] = 0.0
    safe_discharge[surplus] = 0.0
    safe_discharge[deficit] = np.minimum(
        np.maximum(safe_discharge[deficit], target_discharge[deficit]),
        target_discharge[deficit],
    )
    safe_charge[surplus] = np.minimum(
        np.maximum(safe_charge[surplus], target_charge[surplus]),
        target_charge[surplus],
    )
    safe_discharge = np.minimum(safe_discharge, np.maximum(net_load, 0.0) * fraction)
    safe_charge = np.minimum(safe_charge, np.maximum(-net_load, 0.0) * fraction)
    return safe_charge.astype(np.float32), safe_discharge.astype(np.float32)


def _emergency_balance_profiles(
    load: np.ndarray,
    pv: np.ndarray,
    charge: np.ndarray,
    discharge: np.ndarray,
    config: ExperimentConfig,
) -> tuple[np.ndarray, np.ndarray, float]:
    shrink = float(np.clip(config.clearing.emergency_balance_shrink, 0.0, 1.0))
    if shrink >= 1.0:
        return load, pv, 0.0
    net_injection = pv + discharge - load - charge
    target = shrink * net_injection
    new_load = load.astype(np.float32).copy()
    new_pv = pv.astype(np.float32).copy()

    surplus = net_injection > config.clearing.network_tolerance
    curtail = np.minimum((net_injection - target) * surplus, new_pv)
    new_pv = np.maximum(new_pv - curtail, 0.0).astype(np.float32)

    deficit = net_injection < -config.clearing.network_tolerance
    shed = np.minimum((target - net_injection) * deficit, new_load)
    new_load = np.maximum(new_load - shed, 0.0).astype(np.float32)
    adjustment = float(np.maximum(curtail, 0.0).sum() + np.maximum(shed, 0.0).sum())
    return new_load, new_pv, adjustment


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
    agent_injection: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    n_nodes = scenario.num_nodes
    injection = np.zeros(n_nodes, dtype=np.float32)
    for agent, node in enumerate(scenario.agent_nodes):
        injection[int(node)] += agent_injection[agent]

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
    raw_actions = actions
    actions, action_bound_deviation = _project_actions_to_local_bounds(
        load, pv, soc, raw_actions, config
    )
    effective_load = load.copy()
    effective_pv = pv.copy()
    charge, discharge, soc_next, soc_violation = _apply_storage_limits(soc, actions, config)
    attempted_charge = np.clip(raw_actions.charge, 0.0, config.storage.max_charge_power).astype(
        np.float32
    )
    attempted_discharge = np.clip(
        raw_actions.discharge, 0.0, config.storage.max_discharge_power
    ).astype(np.float32)
    if config.clearing.enable_safety_shield:
        soc_violation = 0.0

    p2p_power, p2p_price = double_auction(actions)
    attempted_p2p_power = p2p_power.copy()
    repair_iterations = 0
    safety_adjustment = 0.0
    safety_applied = False
    emergency_iterations = 0
    max_iterations = max(config.clearing.max_repair_iters, config.clearing.max_emergency_iters)
    for iteration in range(max_iterations + 1):
        grid_buy, grid_sell = _balance_agents(effective_load, effective_pv, charge, discharge, p2p_power)
        agent_injection = effective_pv + discharge - effective_load - charge
        node_injection, line_flow, voltages, violations = _network_state(
            scenario, config, agent_injection
        )
        violations["trade"] = float(np.maximum(np.diag(p2p_power), 0.0).sum())
        violations["soc"] = soc_violation
        if max(violations.values()) <= config.clearing.network_tolerance:
            break
        repair_iterations = iteration + 1
        network_violation = max(
            float(violations.get("voltage", 0.0)),
            float(violations.get("line", 0.0)),
        )
        if (
            config.clearing.enable_safety_shield
            and not safety_applied
            and network_violation > config.clearing.network_tolerance
        ):
            safe_charge, safe_discharge = _storage_safety_dispatch(
                load=load,
                pv=pv,
                soc=soc,
                charge=charge,
                discharge=discharge,
                config=config,
            )
            adjustment = float(
                np.abs(safe_charge - charge).sum()
                + np.abs(safe_discharge - discharge).sum()
            )
            safety_applied = True
            if adjustment > 1e-8:
                charge, discharge, soc_next, soc_violation = _apply_storage_arrays(
                    soc, safe_charge, safe_discharge, config
                )
                if config.clearing.enable_safety_shield:
                    soc_violation = 0.0
                safety_adjustment += adjustment
                continue
        if (
            config.clearing.enable_emergency_balancing
            and emergency_iterations < config.clearing.max_emergency_iters
            and network_violation > config.clearing.network_tolerance
        ):
            new_load, new_pv, emergency_adjustment = _emergency_balance_profiles(
                effective_load,
                effective_pv,
                charge,
                discharge,
                config,
            )
            if emergency_adjustment > 1e-8:
                effective_load = new_load
                effective_pv = new_pv
                safety_adjustment += emergency_adjustment
                emergency_iterations += 1
                continue
        if iteration < config.clearing.max_repair_iters:
            p2p_power = (p2p_power * config.clearing.repair_shrink).astype(np.float32)

    p2p_sold = p2p_power.sum(axis=1)
    carbon = compute_carbon_result(
        scenario=scenario,
        config=config,
        t=t,
        load=effective_load,
        pv=effective_pv,
        charge=charge,
        p2p_sell=p2p_sold,
        grid_buy=grid_buy,
        grid_sell=grid_sell,
    )
    package = ClearingPackage(
        epoch=t,
        actions=actions,
        attempted_p2p_power=attempted_p2p_power.astype(np.float32),
        attempted_charge=attempted_charge.astype(np.float32),
        attempted_discharge=attempted_discharge.astype(np.float32),
        effective_load=effective_load.astype(np.float32),
        effective_pv=effective_pv.astype(np.float32),
        load_shed=np.maximum(load - effective_load, 0.0).astype(np.float32),
        pv_curtailment=np.maximum(pv - effective_pv, 0.0).astype(np.float32),
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
        repair_iterations=int(repair_iterations),
        safety_adjustment=float(safety_adjustment),
        action_bound_deviation=float(action_bound_deviation),
        package_hash="",
    )
    package.package_hash = stable_hash(package.payload(include_hash=False))
    return package
