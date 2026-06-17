"""
Benchmark route algorithms on canonical neighborhoods for the three goals that
matter for this app:

  * COVERAGE     - % of required street length actually driven (customer visibility)
  * BACKTRACKING - overlap %: extra distance driven beyond covering each street once
  * DRIVABILITY  - U-turns and total maneuvers per km (lower = more human-like)

Run:
    python tests/benchmark_routes.py

It builds a few synthetic-but-realistic street graphs through the real
simplify_graph pipeline, runs each registered route algorithm, scores the node
path it produces, and prints a comparison table. Add algorithms to ALGORITHMS to
benchmark new ideas against the current profiles on the same yardstick.
"""
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.CRITICAL)

import networkx as nx

from find_route import (
    simplify_graph,
    clean_up_graph,
    find_route_max_coverage_optimized,
    find_route_eulerian_drivable,
    bearing,
    _turn_class,
)

SETTINGS_BASE = {
    "coverage_mode": "maximum",
    "min_street_length": 0,
    "node_snap_distance_m": 18,
    "filter_dead_ends": False,  # keep spurs in the required set so coverage is honest
}


# --------------------------------------------------------------------------- #
# Test neighborhoods (raw OSM-style nodes/ways -> real simplify_graph)
# --------------------------------------------------------------------------- #
def _grid(rows, cols, dlat=0.0009, dlon=0.0011, lat0=41.78, lon0=-88.24):
    nodes, ways = {}, []
    nid = lambda r, c: r * 1000 + c
    for r in range(rows):
        for c in range(cols):
            nodes[nid(r, c)] = (lat0 + r * dlat, lon0 + c * dlon)
    for r in range(rows):
        ways.append({"name": f"Cross St {r}", "highway": "residential",
                     "nodes": [nid(r, c) for c in range(cols)]})
    for c in range(cols):
        ways.append({"name": f"Avenue {c}", "highway": "residential",
                     "nodes": [nid(r, c) for r in range(rows)]})
    return nodes, ways, nid


def grid_clean():
    nodes, ways, nid = _grid(4, 5)
    return "grid 4x5", nodes, ways, nid(0, 0)


def grid_with_culdesacs():
    nodes, ways, nid = _grid(4, 5)
    # Add 3 short dead-end spurs hanging off interior nodes (force out-and-backs).
    spur_id = 90000
    for (r, c) in [(1, 1), (2, 3), (1, 3)]:
        base = nid(r, c)
        tip = spur_id
        spur_id += 1
        lat, lon = nodes[base]
        nodes[tip] = (lat + 0.0004, lon + 0.0004)
        ways.append({"name": f"Court {tip}", "highway": "residential",
                     "nodes": [base, tip]})
    return "grid+culdesacs", nodes, ways, nid(0, 0)


def organic():
    # Irregular spacing + a diagonal connector.
    nodes, ways, nid = _grid(4, 4, dlat=0.0011, dlon=0.0007)
    nodes[95000] = (nodes[nid(0, 0)][0] + 0.0005, nodes[nid(0, 0)][1] + 0.0004)
    ways.append({"name": "Diagonal Way", "highway": "residential",
                 "nodes": [nid(0, 0), 95000, nid(1, 1)]})
    return "organic 4x4", nodes, ways, nid(0, 0)


def barbell():
    """Two grids joined by a single long bridge street -- easy for a greedy
    walk to finish one side and strand the far side."""
    nodes, ways, nid = _grid(3, 3, lat0=41.78, lon0=-88.24)
    # Second cluster, offset east.
    off = 5000
    for r in range(3):
        for c in range(3):
            nodes[off + nid(r, c)] = (41.78 + r * 0.0009, -88.24 + (c + 6) * 0.0011)
    for r in range(3):
        ways.append({"name": f"E Cross {r}", "highway": "residential",
                     "nodes": [off + nid(r, c) for c in range(3)]})
    for c in range(3):
        ways.append({"name": f"E Avenue {c}", "highway": "residential",
                     "nodes": [off + nid(r, c) for r in range(3)]})
    # Bridge between the two clusters.
    ways.append({"name": "Bridge Road", "highway": "residential",
                 "nodes": [nid(1, 2), off + nid(1, 0)]})
    return "barbell", nodes, ways, nid(0, 0)


def comb():
    """A long spine with many dead-end teeth (lots of forced out-and-backs)."""
    nodes, ways = {}, []
    for i in range(9):
        nodes[i] = (41.78, -88.24 + i * 0.0011)
    ways.append({"name": "Spine Road", "highway": "residential", "nodes": list(range(9))})
    tip = 7000
    for i in range(1, 8):
        nodes[tip] = (41.78 + 0.0010, -88.24 + i * 0.0011)
        ways.append({"name": f"Tooth {i}", "highway": "residential", "nodes": [i, tip]})
        tip += 1
    return "comb", nodes, ways, 0


NEIGHBORHOODS = [grid_clean, grid_with_culdesacs, organic, barbell, comb]


def build_graph(nodes, ways, start):
    g = simplify_graph(nodes, ways, settings=SETTINGS_BASE, start_node=start, end_node=start)
    return clean_up_graph(g)


# --------------------------------------------------------------------------- #
# Route algorithms under test: fn(graph, start) -> list[node_id] (a path)
# --------------------------------------------------------------------------- #
def _profile(priority, **knobs):
    def run(graph, start):
        res = find_route_max_coverage_optimized(
            graph, start, start,
            settings={**SETTINGS_BASE, "speed_priority": priority, **knobs})
        return res.get("path", []) if isinstance(res, dict) else []
    return run


def _eulerian(graph, start):
    """Chinese-Postman backbone: minimal duplication for full coverage."""
    g_euler = nx.eulerize(graph.copy())
    try:
        edges = list(nx.eulerian_circuit(g_euler, source=start))
    except Exception:
        edges = list(nx.eulerian_circuit(g_euler))
    if not edges:
        return [start]
    return [edges[0][0]] + [v for _, v in edges]


def _euler_drivable(graph, start):
    res = find_route_eulerian_drivable(graph, start, start)
    return res.get("path", []) if isinstance(res, dict) else []


ALGORITHMS = [
    ("cur:fastest", _profile("fastest")),
    ("cur:balanced", _profile("balanced")),
    ("cur:thorough", _profile("thorough")),
    # B/C/D knobs on the balanced greedy route (default-off in production):
    ("bal+straight(B)", _profile("balanced", straight_bonus=15.0)),
    ("bal+recover(C)", _profile("balanced", stuck_recovery=True)),
    ("bal+bylength(D)", _profile("balanced", coverage_by_length=True)),
    ("bal+B+C+D", _profile("balanced", straight_bonus=15.0,
                           stuck_recovery=True, coverage_by_length=True)),
    ("eulerian/cpp", _eulerian),
    ("euler-drivable(A)", _euler_drivable),
]


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def evaluate(graph, path):
    total_len = sum(d.get("distance", 0) for _, _, d in graph.edges(data=True))
    if not path or len(path) < 2:
        return dict(coverage=0, driven_km=0, overlap=0, uturns=0, turns_per_km=0, straight=0)

    covered, driven = set(), 0.0
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        if graph.has_edge(u, v):
            covered.add(frozenset((u, v)))
            driven += graph[u][v].get("distance", 0)
    covered_len = sum(graph[list(e)[0]][list(e)[1]].get("distance", 0) for e in covered)

    straight = left = right = uturn = 0
    for i in range(1, len(path) - 1):
        try:
            p0 = graph.nodes[path[i - 1]]["coordinates"]
            p1 = graph.nodes[path[i]]["coordinates"]
            p2 = graph.nodes[path[i + 1]]["coordinates"]
        except KeyError:
            continue
        delta = (bearing(p1, p2) - bearing(p0, p1) + 540) % 360 - 180
        cls = _turn_class(delta)
        straight += cls == "straight"
        left += cls == "left"
        right += cls == "right"
        uturn += cls == "uturn"

    maneuvers = left + right + uturn
    total_turns = straight + maneuvers
    driven_km = driven / 1000
    return dict(
        coverage=100 * covered_len / total_len if total_len else 0,
        driven_km=driven_km,
        overlap=100 * (driven - covered_len) / covered_len if covered_len else 0,
        uturns=uturn,
        turns_per_km=maneuvers / driven_km if driven_km else 0,
        straight=100 * straight / total_turns if total_turns else 0,
    )


def main():
    hdr = f"{'algorithm':14} {'cover%':>7} {'driven_km':>9} {'overlap%':>8} {'uturns':>6} {'turns/km':>8} {'straight%':>9}"
    for nb in NEIGHBORHOODS:
        label, nodes, ways, start = nb()
        graph = build_graph(nodes, ways, start)
        print(f"\n=== {label}  ({graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges, "
              f"start={start}) ===")
        print(hdr)
        for name, fn in ALGORITHMS:
            m = evaluate(graph, fn(graph, start))
            print(f"{name:14} {m['coverage']:7.1f} {m['driven_km']:9.2f} {m['overlap']:8.1f} "
                  f"{m['uturns']:6d} {m['turns_per_km']:8.1f} {m['straight']:9.1f}")
    print("\nlower overlap% & uturns & turns/km = better drivability; higher cover% = more visibility")


if __name__ == "__main__":
    main()
