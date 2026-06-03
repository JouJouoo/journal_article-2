from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    display_name: str
    base_kv: float
    base_mva: float
    line_from: np.ndarray
    line_to: np.ndarray
    resistance_ohm: np.ndarray
    reactance_ohm: np.ndarray
    line_current_limit_a: np.ndarray
    bus_load_p_mw: np.ndarray
    bus_load_q_mvar: np.ndarray
    source: str
    source_url: str
    notes: str

    @property
    def num_buses(self) -> int:
        return int(self.bus_load_p_mw.shape[0])

    @property
    def num_branches(self) -> int:
        return int(self.line_from.shape[0])

    @property
    def total_load_p_mw(self) -> float:
        return float(self.bus_load_p_mw.sum())

    @property
    def total_load_q_mvar(self) -> float:
        return float(self.bus_load_q_mvar.sum())


_IEEE33_ROWS = """
1 2 100 60 0.0922 0.0470 400
2 3 90 40 0.4930 0.2511 400
3 4 120 80 0.3660 0.1864 400
4 5 60 30 0.3811 0.1941 400
5 6 60 20 0.8190 0.7070 400
6 7 200 100 0.1872 0.6188 300
7 8 200 100 0.7114 0.2351 300
8 9 60 20 1.0300 0.7400 200
9 10 60 20 1.0440 0.7400 200
10 11 45 30 0.1966 0.0650 200
11 12 60 35 0.3744 0.1238 200
12 13 60 35 1.4680 1.1550 200
13 14 120 80 0.5416 0.7129 200
14 15 60 10 0.5910 0.5260 200
15 16 60 20 0.7463 0.5450 200
16 17 60 20 1.2890 1.7210 200
17 18 90 40 0.7320 0.5740 200
2 19 90 40 0.1640 0.1565 200
19 20 90 40 1.5042 1.3554 200
20 21 90 40 0.4095 0.4784 200
21 22 90 40 0.7089 0.9373 200
3 23 90 50 0.4512 0.3083 200
23 24 420 200 0.8980 0.7091 200
24 25 420 200 0.8960 0.7011 200
6 26 60 25 0.2030 0.1034 300
26 27 60 25 0.2842 0.1447 300
27 28 60 20 1.0590 0.9337 300
28 29 120 70 0.8042 0.7006 200
29 30 200 600 0.5075 0.2585 200
30 31 150 70 0.9744 0.9630 200
31 32 210 100 0.3105 0.3619 200
32 33 60 40 0.3410 0.5302 200
"""


_IEEE69_ROWS = """
1 2 0 0 0.0005 0.0012 400
2 3 0 0 0.0005 0.0012 400
3 4 0 0 0.0015 0.0036 400
4 5 0 0 0.0251 0.0294 400
5 6 2.6 2.2 0.366 0.1864 400
6 7 40.4 30 0.381 0.1941 400
7 8 75 54 0.0922 0.047 400
8 9 30 22 0.0493 0.0251 400
9 10 28 19 0.819 0.2707 400
10 11 145 104 0.1872 0.0619 200
11 12 145 104 0.7114 0.2351 200
12 13 8 5.5 1.03 0.34 200
13 14 8 5.5 1.044 0.34 200
14 15 0 0 1.058 0.3496 200
15 16 45.5 30 0.1966 0.065 200
16 17 60 35 0.3744 0.1238 200
17 18 60 35 0.0047 0.0016 200
18 19 0 0 0.3276 0.1083 200
19 20 1 0.6 0.2106 0.069 200
20 21 114 81 0.3416 0.1129 200
21 22 5.3 3.5 0.014 0.0046 200
22 23 0 0 0.1591 0.0526 200
23 24 28 20 0.3463 0.1145 200
24 25 0 0 0.7488 0.2475 200
25 26 14 10 0.3089 0.1021 200
26 27 14 10 0.1732 0.0572 200
3 28 26 18.6 0.0044 0.0108 200
28 29 26 18.6 0.064 0.1565 200
29 30 0 0 0.3978 0.1315 200
30 31 0 0 0.0702 0.0232 200
31 32 0 0 0.351 0.116 200
32 33 14 10 0.839 0.2816 200
33 34 19.5 14 1.708 0.5646 200
34 35 6 4 1.474 0.4873 200
3 36 26 18.6 0.0044 0.0108 200
36 37 26 18.6 0.064 0.1565 200
37 38 0 0 0.1053 0.123 200
38 39 24 17 0.0304 0.0355 200
39 40 24 17 0.0018 0.0021 200
40 41 1.2 1 0.7283 0.8509 200
41 42 0 0 0.31 0.3623 200
42 43 6 4.3 0.041 0.0478 200
43 44 0 0 0.0092 0.0116 200
44 45 39.2 26.3 0.1089 0.1373 200
45 46 39.2 26.3 0.0009 0.0012 200
4 47 0 0 0.0034 0.0084 300
47 48 79 56.4 0.0851 0.2083 300
48 49 384.7 274.5 0.2898 0.7091 300
49 50 384.7 274.5 0.0822 0.2011 300
8 51 40.5 28.3 0.0928 0.0473 200
51 52 3.6 2.7 0.3319 0.114 200
9 53 4.3 3.5 0.174 0.0886 300
53 54 26.4 19 0.203 0.1034 300
54 55 24 17.2 0.2842 0.1447 300
55 56 0 0 0.2813 0.1433 300
56 57 0 0 1.59 0.5337 300
57 58 0 0 0.7837 0.263 300
58 59 100 72 0.3042 0.1006 300
59 60 0 0 0.3861 0.1172 300
60 61 1244 888 0.5075 0.2585 300
61 62 32 23 0.0974 0.0496 300
62 63 0 0 0.145 0.0738 300
63 64 227 162 0.7105 0.3619 300
64 65 59 42 1.041 0.5302 300
11 66 18 13 0.2012 0.0611 200
66 67 18 13 0.0047 0.0014 200
12 68 28 20 0.7394 0.2444 200
68 69 28 20 0.0047 0.0016 200
"""


def _parse_rows(
    name: str,
    display_name: str,
    raw: str,
    num_buses: int,
    source: str,
    source_url: str,
    notes: str,
) -> BenchmarkCase:
    data = np.asarray(
        [[float(item) for item in line.split()] for line in raw.strip().splitlines()],
        dtype=np.float64,
    )
    if data.shape[1] != 7:
        raise ValueError(f"{name} data must have 7 columns")
    line_from = data[:, 0].astype(np.int64) - 1
    line_to = data[:, 1].astype(np.int64) - 1
    if np.any(line_from < 0) or np.any(line_to < 0):
        raise ValueError(f"{name} bus ids must be 1-based positive integers")
    if np.any(line_from >= num_buses) or np.any(line_to >= num_buses):
        raise ValueError(f"{name} branch endpoint outside bus count")
    bus_load_p = np.zeros(num_buses, dtype=np.float32)
    bus_load_q = np.zeros(num_buses, dtype=np.float32)
    for row in data:
        bus = int(row[1]) - 1
        bus_load_p[bus] += float(row[2]) / 1000.0
        bus_load_q[bus] += float(row[3]) / 1000.0
    return BenchmarkCase(
        name=name,
        display_name=display_name,
        base_kv=12.66,
        base_mva=10.0,
        line_from=line_from,
        line_to=line_to,
        resistance_ohm=data[:, 4].astype(np.float32),
        reactance_ohm=data[:, 5].astype(np.float32),
        line_current_limit_a=data[:, 6].astype(np.float32),
        bus_load_p_mw=bus_load_p,
        bus_load_q_mvar=bus_load_q,
        source=source,
        source_url=source_url,
        notes=notes,
    )


BENCHMARK_CASES: dict[str, BenchmarkCase] = {
    "ieee33bw": _parse_rows(
        name="ieee33bw",
        display_name="IEEE 33-bus Baran-Wu radial distribution system",
        raw=_IEEE33_ROWS,
        num_buses=33,
        source="MATPOWER/pandapower case33bw, Baran and Wu 1989; locked locally for reproducible profiles.",
        source_url="https://pandapower.readthedocs.io/en/v3.3.1/networks/power_system_test_cases.html#case-33bw",
        notes="Base load table is stored in kW/kVAr and converted to MW/MVAr.",
    ),
    "ieee69": _parse_rows(
        name="ieee69",
        display_name="IEEE 69-bus radial distribution system",
        raw=_IEEE69_ROWS,
        num_buses=69,
        source="MATPOWER case69, Baran and Wu 1989; locked locally for reproducible profiles.",
        source_url="https://github.com/MATPOWER/matpower/blob/master/data/case69.m",
        notes="Radial main feeder only; tie-lines are not closed in this model.",
    ),
}


def get_benchmark_case(name: str) -> BenchmarkCase:
    key = name.lower()
    try:
        return BENCHMARK_CASES[key]
    except KeyError as exc:
        known = ", ".join(sorted(BENCHMARK_CASES))
        raise ValueError(f"Unknown benchmark case '{name}'. Known cases: {known}") from exc


def is_radial(case: BenchmarkCase) -> bool:
    if case.num_branches != case.num_buses - 1:
        return False
    parent = list(range(case.num_buses))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for src, dst in zip(case.line_from, case.line_to):
        if src == dst:
            return False
        root_src = find(int(src))
        root_dst = find(int(dst))
        if root_src == root_dst:
            return False
        parent[root_dst] = root_src
    return len({find(idx) for idx in range(case.num_buses)}) == 1
