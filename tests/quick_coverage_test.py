
#!/usr/bin/env python
"""Direct coverage test runner - outputs results to console and file."""

import sys
import os
import math
import networkx as nx
from typing import Tuple, List

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def create_grid_graph(width: int, height: int, spacing: float = 0.001) -> Tuple[nx.Graph, str]:
    """Create a rectangular grid graph."""
    G = nx.Graph()
    node_id = 0
    node_map = {}
    
    for y in range(height):
        for x in range(width):
            lat = 41.8781 + (y * spacing)
            lon = -87.6298 + (x * spacing)
            G.add_node(node_id, coordinates=(lat, lon))
            node_map[(x, y)] = node_id
            node_id += 1
    
    for y in range(height):
        for x in range(width - 1):
            n1 = node_map[(x, y)]
            n2 = node_map[(x + 1, y)]
            distance = spacing * 111
            G.add_edge(n1, n2, weight=distance, distance=distance * 1000)
    
    for y in range(height - 1):
        for x in range(width):
            n1 = node_map[(x, y)]
            n2 = node_map[(x, y + 1)]
            distance = spacing * 111
            G.add_edge(n1, n2, weight=distance, distance=distance * 1000)
    
    return G, f"Grid {width}x{height}"


def create_branching_graph(num_branches: int = 4) -> Tuple[nx.Graph, str]:
    """Create a central hub with radial branches and cross-connections."""
    G = nx.Graph()
    node_id = 0

    center = node_id
    G.add_node(center, coordinates=(41.8781, -87.6298))
    node_id += 1

    branch_nodes = []

    for branch_idx in range(num_branches):
        angle = (2 * math.pi * branch_idx) / num_branches
        prev_node = center

        for distance_step in range(1, 5):
            lat = 41.8781 + (math.cos(angle) * distance_step * 0.001)
            lon = -87.6298 + (math.sin(angle) * distance_step * 0.001)

            G.add_node(node_id, coordinates=(lat, lon))

            edge_distance_km = 0.111
            G.add_edge(prev_node, node_id, weight=edge_distance_km, distance=edge_distance_km * 1000)

            prev_node = node_id
            branch_nodes.append(node_id)
            node_id += 1

        if branch_idx > 0 and branch_nodes:
            prev_branch_node = branch_nodes[-(4 + 1)]
            edge_distance_km = 0.111
            G.add_edge(prev_node, prev_branch_node, weight=edge_distance_km, distance=edge_distance_km * 1000)

    return G, f"Branching {num_branches}-branch"


def create_dense_neighborhood_graph() -> Tuple[nx.Graph, str]:
    """Create a neighborhood-like graph with mixed edge lengths."""
    G = nx.Graph()
    node_id = 0
    node_map = {}

    for y in range(5):
        for x in range(5):
            lat = 41.8781 + (y * 0.0008)
            lon = -87.6298 + (x * 0.0008)
            G.add_node(node_id, coordinates=(lat, lon))
            node_map[(x, y)] = node_id
            node_id += 1

    for y in range(5):
        for x in range(4):
            n1 = node_map[(x, y)]
            n2 = node_map[(x + 1, y)]
            G.add_edge(n1, n2, weight=0.09, distance=90)

    for y in range(4):
        for x in range(5):
            n1 = node_map[(x, y)]
            n2 = node_map[(x, y + 1)]
            G.add_edge(n1, n2, weight=0.09, distance=90)

    for y in range(4):
        for x in range(4):
            if (x + y) % 2 == 0:
                n1 = node_map[(x, y)]
                n2 = node_map[(x + 1, y + 1)]
                G.add_edge(n1, n2, weight=0.05, distance=50)

    return G, "Dense Neighborhood"


def graph_weight_stats(graph: nx.Graph) -> Tuple[float, float, int]:
    """Return min/max edge weight and unique weight count."""
    weights = [data.get('weight', 1.0) for _, _, data in graph.edges(data=True)]
    if not weights:
        return 0.0, 0.0, 0
    return min(weights), max(weights), len(set(weights))


def calculate_coverage_percentage(graph: nx.Graph, route: List[Tuple[float, float]]) -> float:
    """Calculate coverage percentage."""
    if not route or len(route) < 2:
        return 0.0
    
    coord_to_node = {graph.nodes[n]['coordinates']: n for n in graph.nodes()}
    used_edges = set()
    total_edges = set(frozenset(e) for e in graph.edges())
    
    for i in range(len(route) - 1):
        lat1, lon1 = route[i]
        lat2, lon2 = route[i + 1]
        
        node1 = None
        node2 = None
        
        for coords, node in coord_to_node.items():
            if abs(coords[0] - lat1) < 0.00001 and abs(coords[1] - lon1) < 0.00001:
                node1 = node
            if abs(coords[0] - lat2) < 0.00001 and abs(coords[1] - lon2) < 0.00001:
                node2 = node
        
        if node1 is not None and node2 is not None:
            edge = frozenset([node1, node2])
            if edge in total_edges:
                used_edges.add(edge)
    
    if not total_edges:
        return 0.0
    
    return len(used_edges) / len(total_edges) * 100


def main():
    try:
        from find_route import find_route_max_coverage_optimized
    except ImportError as e:
        print(f"ERROR: Could not import find_route: {e}")
        return 1
    
    # Define targets
    TARGETS = {
        'fastest': 30,
        'balanced': 70,
        'thorough': 95,
    }
    
    TOLERANCE = 15
    
    # Create test graphs
    test_graphs = [
        create_grid_graph(3, 3),
        create_grid_graph(5, 5),
        create_grid_graph(7, 7),
        create_branching_graph(4),
        create_branching_graph(8),
        create_dense_neighborhood_graph(),
    ]
    
    print("\n" + "="*90)
    print("COVERAGE TARGET VALIDATION - INITIAL TEST")
    print("="*90)
    print(f"\nTarget: Fastest=30%, Balanced=70%, Thorough=95% (±{TOLERANCE}% tolerance)\n")
    
    results = []
    
    for G, graph_name in test_graphs:
        min_w, max_w, unique_w = graph_weight_stats(G)
        print(f"\n{graph_name.upper()}")
        print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
        print(f"  Edge Weight Variance: min={min_w:.3f}, max={max_w:.3f}, unique_weights={unique_w}")
        print("-" * 90)
        
        for profile in ['fastest', 'balanced', 'thorough']:
            target = TARGETS[profile]
            
            try:
                route = find_route_max_coverage_optimized(
                    G, 0, settings={'speed_priority': profile}
                )
                coverage = calculate_coverage_percentage(G, route)
                deviation = abs(coverage - target)
                status = "✓ PASS" if deviation <= TOLERANCE else "✗ ADJUST"
                
                results.append({
                    'graph': graph_name,
                    'profile': profile,
                    'target': target,
                    'actual': coverage,
                    'deviation': deviation,
                    'status': status
                })
                
                print(f"  {profile.upper():10} | Target: {target:3.0f}% | Actual: {coverage:5.1f}% | Deviation: {deviation:5.1f}% | {status}")
                
            except Exception as e:
                print(f"  {profile.upper():10} | ERROR: {str(e)[:50]}")
    
    # Summary
    print("\n" + "="*90)
    print("SUMMARY")
    print("="*90)
    
    for profile in ['fastest', 'balanced', 'thorough']:
        profile_results = [r for r in results if r['profile'] == profile]
        if profile_results:
            avg_actual = sum(r['actual'] for r in profile_results) / len(profile_results)
            avg_deviation = sum(r['deviation'] for r in profile_results) / len(profile_results)
            pass_count = len([r for r in profile_results if r['status'].startswith('✓')])
            
            print(f"\n{profile.upper()}")
            print(f"  Target:         {TARGETS[profile]}%")
            print(f"  Average Actual: {avg_actual:.1f}%")
            print(f"  Avg Deviation:  {avg_deviation:.1f}%")
            print(f"  Pass Rate:      {pass_count}/{len(profile_results)}")
            
            if avg_deviation > TOLERANCE:
                if avg_actual < TARGETS[profile]:
                    print(f"  → Coverage is TOO LOW - need to increase settings")
                    print(f"    Try: increase coverage_threshold, unused_bonus, frontier_bonus")
                else:
                    print(f"  → Coverage is TOO HIGH - need to decrease settings")
                    print(f"    Try: decrease coverage_threshold, unused_bonus, frontier_bonus")
    
    print("\n" + "="*90)
    return 0


if __name__ == '__main__':
    sys.exit(main())
