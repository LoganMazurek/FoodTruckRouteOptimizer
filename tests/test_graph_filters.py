import sys
import os

# Add the root directory to Python path so we can import top-level modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from types import SimpleNamespace

from find_route import simplify_graph
from get_street_data import extract_nodes_and_ways


class FakeNode:
    def __init__(self, node_id, lat, lon):
        self.id = node_id
        self.lat = lat
        self.lon = lon


class FakeWay:
    def __init__(self, name, highway, node_ids):
        self.tags = {"name": name, "highway": highway}
        self.nodes = [SimpleNamespace(id=node_id) for node_id in node_ids]


class FakeStreetData:
    def __init__(self, nodes, ways):
        self.nodes = nodes
        self.ways = ways


def test_extract_nodes_and_ways_strict_bbox_clips_and_splits_segments():
    street_data = FakeStreetData(
        nodes=[
            FakeNode(1, 41.0000, -88.0000),
            FakeNode(2, 41.0005, -88.0005),
            FakeNode(3, 41.0010, -88.0010),
            FakeNode(4, 41.0020, -88.0020),
        ],
        ways=[
            FakeWay("Main St", "residential", [1, 2, 3, 4]),
            FakeWay("Side St", "residential", [1, 4]),
        ],
    )

    nodes, ways = extract_nodes_and_ways(
        street_data,
        min_lat=41.0004,
        max_lat=41.0012,
        min_lng=-88.0012,
        max_lng=-88.0004,
    )

    assert set(nodes.keys()) == {2, 3}
    assert len(ways) == 1
    assert ways[0]["name"] == "Main St"
    assert ways[0]["nodes"] == [2, 3]


def test_simplify_graph_merges_very_close_nodes():
    nodes = {
        1: (41.000000, -88.000000),
        2: (41.000040, -88.000000),
        3: (41.002000, -88.000000),
    }
    ways = [
        {"name": "Street A", "nodes": [1, 3], "highway": "residential"},
        {"name": "Street B", "nodes": [2, 3], "highway": "residential"},
    ]

    graph = simplify_graph(
        nodes,
        ways,
        settings={
            "coverage_mode": "balanced",
            "min_street_length": 0,
            "node_snap_distance_m": 18,
        },
    )

    assert graph.number_of_nodes() == 2
    assert graph.number_of_edges() == 1


def test_expand_route_orients_intermediates_by_travel_direction():
    """Intermediate edge geometry must follow the direction of travel, not the
    order it was stored in. Otherwise reverse-direction traversal zig-zags and
    navigation apps insert spurious U-turns."""
    import networkx as nx
    from find_route import expand_route_with_geometry

    G = nx.Graph()
    G.add_node('A', coordinates=(0.0, 0.000))
    G.add_node('B', coordinates=(0.0, 0.004))
    arc = [(0.001, 0.001), (0.002, 0.002), (0.001, 0.003)]
    G.add_edge('A', 'B', intermediate_nodes=arc, geom_from='A')

    fwd = expand_route_with_geometry(['A', 'B'], G)
    rev = expand_route_with_geometry(['B', 'A'], G)

    # Reverse traversal is exactly the forward path reversed (no jump-back).
    assert rev == list(reversed(fwd))
    # And the arc is monotonic in lon for each direction (no backtrack spike).
    assert [p[1] for p in fwd] == sorted(p[1] for p in fwd)
    assert [p[1] for p in rev] == sorted((p[1] for p in rev), reverse=True)


def test_expand_route_drops_consecutive_duplicate_points():
    import networkx as nx
    from find_route import expand_route_with_geometry

    G = nx.Graph()
    G.add_node(1, coordinates=(0.0, 0.0))
    G.add_node(2, coordinates=(0.0, 0.001))
    G.add_node(3, coordinates=(0.0, 0.002))
    # Empty intermediates so the shared endpoint (node 2) is the only join.
    G.add_edge(1, 2, intermediate_nodes=[], geom_from=1)
    G.add_edge(2, 3, intermediate_nodes=[], geom_from=2)

    out = expand_route_with_geometry([1, 2, 3], G)
    assert out == [(0.0, 0.0), (0.0, 0.001), (0.0, 0.002)]
    # No consecutive duplicates anywhere.
    assert all(out[i] != out[i - 1] for i in range(1, len(out)))
