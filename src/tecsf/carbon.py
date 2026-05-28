from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tecsf.config import ExperimentConfig
from tecsf.data import SyntheticScenario


@dataclass
class CarbonResult:
    e_grid: np.ndarray
    e_pv_credit: np.ndarray
    c_offset: np.ndarray
    e_ca_need: np.ndarray
    a_buy: np.ndarray
    c_sell: np.ndarray
    e_resp: np.ndarray
    e_lc: np.ndarray
    carbon_reduction: np.ndarray
    q_lc: np.ndarray
    lccoins_candidate: np.ndarray


def compute_carbon_result(
    scenario: SyntheticScenario,
    config: ExperimentConfig,
    t: int,
    load: np.ndarray,
    pv: np.ndarray,
    charge: np.ndarray,
    p2p_sell: np.ndarray,
    grid_buy: np.ndarray,
    grid_sell: np.ndarray,
) -> CarbonResult:
    idx = scenario.time_index(t)
    dt = scenario.delta_t
    gamma_grid = float(scenario.grid_emission_factor[idx])
    gamma_pv = -config.market.pv_credit_beta * gamma_grid

    pv_use = np.minimum(pv, load + charge + p2p_sell + grid_sell)
    pv_use = np.maximum(pv_use, 0.0)
    e_grid = gamma_grid * grid_buy * dt
    e_pv_credit = gamma_pv * pv_use * dt
    c_pv = np.maximum(-e_pv_credit, 0.0)
    c_offset = np.minimum(c_pv, e_grid)
    e_ca_need = np.maximum(e_grid - c_offset, 0.0)
    a_buy = e_ca_need.copy()
    c_sell = np.maximum(c_pv - c_offset, 0.0)
    e_resp = e_grid - c_offset - a_buy
    e_lc = c_pv

    baseline_demand = load + charge
    e_base = gamma_grid * baseline_demand * dt
    carbon_reduction = np.maximum(0.0, e_base - e_grid)
    q_lc = pv_use * dt
    lccoins_candidate = (
        config.lccoins.alpha_q * (q_lc / max(config.lccoins.q_norm, 1e-8))
        + config.lccoins.alpha_offset
        * (c_offset / max(config.lccoins.offset_norm, 1e-8))
    )

    return CarbonResult(
        e_grid=e_grid.astype(np.float32),
        e_pv_credit=e_pv_credit.astype(np.float32),
        c_offset=c_offset.astype(np.float32),
        e_ca_need=e_ca_need.astype(np.float32),
        a_buy=a_buy.astype(np.float32),
        c_sell=c_sell.astype(np.float32),
        e_resp=e_resp.astype(np.float32),
        e_lc=e_lc.astype(np.float32),
        carbon_reduction=carbon_reduction.astype(np.float32),
        q_lc=q_lc.astype(np.float32),
        lccoins_candidate=lccoins_candidate.astype(np.float32),
    )
