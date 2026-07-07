from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np

from tecsf.benchmark_cases import BENCHMARK_CASES, BenchmarkCase, is_radial


class BenchmarkSourceUnavailable(RuntimeError):
    """Raised when an authoritative benchmark source cannot be loaded locally."""


@dataclass(frozen=True)
class SourceCheck:
    name: str
    passed: bool
    expected: str
    observed: str


def _as_text_array(value: np.ndarray, precision: int = 6) -> str:
    return np.array2string(np.asarray(value), precision=precision, separator=", ")


def _append_check(checks: list[SourceCheck], name: str, passed: bool, expected, observed) -> None:
    checks.append(SourceCheck(name, bool(passed), str(expected), str(observed)))


def _line_current_limit_from_rate_mva(rate_mva: np.ndarray, base_kv: float) -> np.ndarray:
    return (rate_mva * 1000.0 / (np.sqrt(3.0) * float(base_kv))).astype(np.float32)


def _align_lines_to_locked_topology(
    case_name: str,
    line_from: np.ndarray,
    line_to: np.ndarray,
    resistance: np.ndarray,
    reactance: np.ndarray,
    current_limits: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, bool]:
    locked = BENCHMARK_CASES.get(case_name)
    if locked is None:
        return line_from, line_to, resistance, reactance, current_limits, False
    if np.array_equal(line_from, locked.line_from) and np.array_equal(line_to, locked.line_to):
        return line_from, line_to, resistance, reactance, current_limits, False

    source_index: dict[tuple[int, int], int] = {}
    for idx, (src, dst) in enumerate(zip(line_from, line_to)):
        source_index.setdefault((int(src), int(dst)), idx)
        source_index.setdefault((int(dst), int(src)), idx)

    aligned_indices: list[int] = []
    for src, dst in zip(locked.line_from, locked.line_to):
        edge = (int(src), int(dst))
        if edge not in source_index:
            return line_from, line_to, resistance, reactance, current_limits, False
        aligned_indices.append(source_index[edge])

    index = np.asarray(aligned_indices, dtype=np.int64)
    aligned_limits = current_limits[index] if current_limits is not None else None
    return (
        locked.line_from.astype(np.int64),
        locked.line_to.astype(np.int64),
        resistance[index].astype(np.float32),
        reactance[index].astype(np.float32),
        aligned_limits.astype(np.float32) if aligned_limits is not None else None,
        True,
    )


def _resolve_current_limits(
    case_name: str,
    branch_count: int,
    source_rate_mva: np.ndarray | None = None,
) -> tuple[np.ndarray, str]:
    if source_rate_mva is not None and source_rate_mva.shape == (branch_count,):
        positive = np.asarray(source_rate_mva, dtype=np.float64) > 0.0
        if np.all(positive):
            locked = BENCHMARK_CASES.get(case_name)
            base_kv = locked.base_kv if locked is not None else 12.66
            return (
                _line_current_limit_from_rate_mva(source_rate_mva.astype(np.float64), base_kv),
                "Line current limits are converted from source RATE_A MVA values.",
            )

    locked = BENCHMARK_CASES.get(case_name)
    if locked is not None and locked.line_current_limit_a.shape == (branch_count,):
        return (
            locked.line_current_limit_a.astype(np.float32),
            "Line current limits are benchmark constraints because the source case "
            "does not provide complete thermal ratings.",
        )

    return (
        np.full(branch_count, 400.0, dtype=np.float32),
        "Line current limits use the fallback value because no source ratings exist.",
    )


def case_from_pandapower_case33bw() -> BenchmarkCase:
    try:
        import pandapower as pp
        import pandapower.networks as pn
    except ImportError as exc:
        raise BenchmarkSourceUnavailable(
            "pandapower is not installed; install the benchmark-sources extra to load case33bw."
        ) from exc

    net = pn.case33bw()
    pp.runpp(net, numba=False)
    if not bool(getattr(net, "converged", False)):
        raise BenchmarkSourceUnavailable("pandapower case33bw power flow did not converge")

    num_buses = int(len(net.bus))
    base_kv = float(net.bus.vn_kv.iloc[0]) if "vn_kv" in net.bus else 12.66
    base_mva = float(getattr(net, "sn_mva", 10.0))
    line_from = net.line.from_bus.to_numpy(dtype=np.int64)
    line_to = net.line.to_bus.to_numpy(dtype=np.int64)
    resistance = (
        net.line.r_ohm_per_km.to_numpy(dtype=np.float64)
        * net.line.length_km.to_numpy(dtype=np.float64)
    ).astype(np.float32)
    reactance = (
        net.line.x_ohm_per_km.to_numpy(dtype=np.float64)
        * net.line.length_km.to_numpy(dtype=np.float64)
    ).astype(np.float32)

    bus_load_p = np.zeros(num_buses, dtype=np.float32)
    bus_load_q = np.zeros(num_buses, dtype=np.float32)
    if hasattr(net, "load") and len(net.load) > 0:
        for _, row in net.load.iterrows():
            bus = int(row["bus"])
            bus_load_p[bus] += float(row["p_mw"])
            bus_load_q[bus] += float(row["q_mvar"])

    source_current_limits: np.ndarray | None = None
    if "max_i_ka" in net.line and np.all(np.isfinite(net.line.max_i_ka.to_numpy(dtype=float))):
        max_i_ka = net.line.max_i_ka.to_numpy(dtype=np.float64)
        if np.all((max_i_ka > 0.0) & (max_i_ka < 100.0)):
            source_current_limits = (max_i_ka * 1000.0).astype(np.float32)

    line_from, line_to, resistance, reactance, source_current_limits, aligned = (
        _align_lines_to_locked_topology(
            "ieee33bw",
            line_from,
            line_to,
            resistance,
            reactance,
            source_current_limits,
        )
    )
    if source_current_limits is not None:
        current_limits = source_current_limits.astype(np.float32)
        limit_note = "Line current limits are loaded from pandapower max_i_ka."
    else:
        current_limits, limit_note = _resolve_current_limits("ieee33bw", line_from.shape[0])
    align_note = (
        " Source lines are aligned to the locked 32-branch radial feeder; extra tie-lines "
        "from reconfiguration data are excluded."
        if aligned
        else ""
    )

    return BenchmarkCase(
        name="ieee33bw",
        display_name="IEEE 33-bus Baran-Wu radial distribution system",
        base_kv=base_kv,
        base_mva=base_mva,
        line_from=line_from,
        line_to=line_to,
        resistance_ohm=resistance,
        reactance_ohm=reactance,
        line_current_limit_a=current_limits,
        bus_load_p_mw=bus_load_p,
        bus_load_q_mvar=bus_load_q,
        source="pandapower.networks.case33bw() converted from MATPOWER/Baran-Wu data.",
        source_url=(
            "https://pandapower.readthedocs.io/en/develop/networks/"
            "power_system_test_cases.html#case-33bw"
        ),
        notes=(
            "Authoritative source load path; pp.runpp() is executed to confirm power-flow "
            f"convergence.{align_note} {limit_note}"
        ),
    )


def _strip_matlab_comments(text: str) -> str:
    return "\n".join(line.split("%", 1)[0] for line in text.splitlines())


def _extract_matpower_scalar(text: str, field: str) -> float:
    match = re.search(rf"mpc\.{re.escape(field)}\s*=\s*([0-9.eE+-]+)\s*;", text)
    if not match:
        raise ValueError(f"MATPOWER field mpc.{field} not found")
    return float(match.group(1))


def _extract_matpower_array(text: str, field: str) -> np.ndarray:
    match = re.search(rf"mpc\.{re.escape(field)}\s*=\s*\[(.*?)\]\s*;", text, re.S)
    if not match:
        raise ValueError(f"MATPOWER array mpc.{field} not found")
    rows: list[list[float]] = []
    for raw_line in match.group(1).splitlines():
        line = raw_line.strip().rstrip(";")
        if not line:
            continue
        rows.append([float(item) for item in line.split()])
    if not rows:
        raise ValueError(f"MATPOWER array mpc.{field} is empty")
    return np.asarray(rows, dtype=np.float64)


def case_from_matpower_file(case_name: str, path: str | Path) -> BenchmarkCase:
    if case_name != "ieee69":
        raise ValueError("MATPOWER file loading is currently supported for ieee69 only")

    source_path = Path(path)
    if not source_path.exists():
        raise BenchmarkSourceUnavailable(f"MATPOWER case file not found: {source_path}")

    raw_text = source_path.read_text(encoding="utf-8")
    text = _strip_matlab_comments(raw_text)
    base_mva = _extract_matpower_scalar(text, "baseMVA")
    bus = _extract_matpower_array(text, "bus")
    branch = _extract_matpower_array(text, "branch")
    if bus.shape[1] < 13:
        raise ValueError("MATPOWER bus array must have at least 13 columns")
    if branch.shape[1] < 4:
        raise ValueError("MATPOWER branch array must have at least 4 columns")

    bus_ids = bus[:, 0].astype(np.int64)
    expected_ids = np.arange(1, bus.shape[0] + 1, dtype=np.int64)
    if not np.array_equal(bus_ids, expected_ids):
        raise ValueError("MATPOWER bus ids must be contiguous and 1-based")

    bus_load_scale = 1.0
    if re.search(
        r"mpc\.bus\(:,\s*\[PD,\s*QD\]\)\s*=\s*mpc\.bus\(:,\s*\[PD,\s*QD\]\)\s*/\s*1e3",
        text,
    ):
        bus_load_scale = 1e-3
    bus_load_p = (bus[:, 2] * bus_load_scale).astype(np.float32)
    bus_load_q = (bus[:, 3] * bus_load_scale).astype(np.float32)
    base_kv = float(bus[0, 9]) if bus.shape[1] > 9 and bus[0, 9] > 0.0 else 12.66

    line_from = branch[:, 0].astype(np.int64) - 1
    line_to = branch[:, 1].astype(np.int64) - 1
    if np.any(line_from < 0) or np.any(line_to < 0):
        raise ValueError("MATPOWER branch bus ids must be positive")
    if np.any(line_from >= bus.shape[0]) or np.any(line_to >= bus.shape[0]):
        raise ValueError("MATPOWER branch endpoint outside bus count")

    source_rate = branch[:, 5].astype(np.float64) if branch.shape[1] > 5 else None
    current_limits, limit_note = _resolve_current_limits(case_name, branch.shape[0], source_rate)
    branch_unit_note = (
        "Branch resistance/reactance are parsed as the source-listed Ohm values before "
        "the MATPOWER p.u. conversion block."
    )
    if "mpc.branch(:, [BR_R BR_X])" not in text:
        branch_unit_note = "Branch resistance/reactance are parsed as listed in the source file."

    return BenchmarkCase(
        name=case_name,
        display_name="IEEE 69-bus radial distribution system",
        base_kv=base_kv,
        base_mva=base_mva,
        line_from=line_from,
        line_to=line_to,
        resistance_ohm=branch[:, 2].astype(np.float32),
        reactance_ohm=branch[:, 3].astype(np.float32),
        line_current_limit_a=current_limits,
        bus_load_p_mw=bus_load_p,
        bus_load_q_mvar=bus_load_q,
        source=f"MATPOWER {source_path.name} parsed from local source file.",
        source_url="https://github.com/MATPOWER/matpower/blob/master/data/case69.m",
        notes=(
            "Authoritative source load path. Loads are converted from kW/kVAr to MW/MVAr "
            f"when the MATPOWER conversion block is present. {branch_unit_note} {limit_note}"
        ),
    )


def load_authoritative_benchmark_case(
    case_name: str,
    case69_m: str | Path | None = None,
) -> BenchmarkCase:
    key = case_name.lower()
    if key == "ieee33bw":
        return case_from_pandapower_case33bw()
    if key == "ieee69":
        if case69_m is None:
            raise BenchmarkSourceUnavailable(
                "ieee69 authoritative loading requires --case69-m pointing to MATPOWER case69.m."
            )
        return case_from_matpower_file("ieee69", case69_m)
    known = ", ".join(sorted(BENCHMARK_CASES))
    raise ValueError(f"Unknown benchmark case '{case_name}'. Known cases: {known}")


def compare_benchmark_cases(
    locked: BenchmarkCase,
    source: BenchmarkCase,
    load_atol: float = 1e-4,
    impedance_atol: float = 1e-4,
) -> list[SourceCheck]:
    checks: list[SourceCheck] = []
    _append_check(checks, "case_name", locked.name == source.name, locked.name, source.name)
    _append_check(
        checks,
        "num_buses",
        locked.num_buses == source.num_buses,
        locked.num_buses,
        source.num_buses,
    )
    _append_check(
        checks,
        "num_branches",
        locked.num_branches == source.num_branches,
        locked.num_branches,
        source.num_branches,
    )
    _append_check(
        checks,
        "radial_topology",
        is_radial(source),
        "radial connected tree",
        f"radial={is_radial(source)}",
    )
    _append_check(
        checks,
        "line_from",
        np.array_equal(locked.line_from, source.line_from),
        _as_text_array(locked.line_from, 0),
        _as_text_array(source.line_from, 0),
    )
    _append_check(
        checks,
        "line_to",
        np.array_equal(locked.line_to, source.line_to),
        _as_text_array(locked.line_to, 0),
        _as_text_array(source.line_to, 0),
    )
    _append_check(
        checks,
        "total_load_p_mw",
        np.isclose(locked.total_load_p_mw, source.total_load_p_mw, atol=load_atol),
        f"{locked.total_load_p_mw:.6f}",
        f"{source.total_load_p_mw:.6f}",
    )
    _append_check(
        checks,
        "total_load_q_mvar",
        np.isclose(locked.total_load_q_mvar, source.total_load_q_mvar, atol=load_atol),
        f"{locked.total_load_q_mvar:.6f}",
        f"{source.total_load_q_mvar:.6f}",
    )
    _append_check(
        checks,
        "bus_load_p_mw",
        np.allclose(locked.bus_load_p_mw, source.bus_load_p_mw, atol=load_atol),
        _as_text_array(locked.bus_load_p_mw),
        _as_text_array(source.bus_load_p_mw),
    )
    _append_check(
        checks,
        "bus_load_q_mvar",
        np.allclose(locked.bus_load_q_mvar, source.bus_load_q_mvar, atol=load_atol),
        _as_text_array(locked.bus_load_q_mvar),
        _as_text_array(source.bus_load_q_mvar),
    )
    _append_check(
        checks,
        "resistance_ohm",
        np.allclose(locked.resistance_ohm, source.resistance_ohm, atol=impedance_atol),
        _as_text_array(locked.resistance_ohm),
        _as_text_array(source.resistance_ohm),
    )
    _append_check(
        checks,
        "reactance_ohm",
        np.allclose(locked.reactance_ohm, source.reactance_ohm, atol=impedance_atol),
        _as_text_array(locked.reactance_ohm),
        _as_text_array(source.reactance_ohm),
    )
    return checks


def failed_checks(checks: list[SourceCheck]) -> list[SourceCheck]:
    return [check for check in checks if not check.passed]
