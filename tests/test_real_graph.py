#!/usr/bin/env python
"""Test speed profiles on a real graph from production usage."""

import sys
import os
import pickle
import networkx as nx
from typing import Tuple, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def load_graph(graph_id: str):
    """Load nodes/ways and build baseline graph."""
    nodes_ways_path = f"temp/{graph_id}_nodes_ways.pkl"
    deleted_nodes_path = f"temp/{graph_id}_deleted_nodes.pkl"
    
    if not os.path.exists(nodes_ways_path):
        print(f"ERROR: nodes_ways file not found: {nodes_ways_path}")
        return None, None, None, None
    
    try:
        with open(nodes_ways_path, 'rb') as f:
            data = pickle.load(f)
        nodes = data['nodes']
        ways = data['ways']
        print(f"✓ Loaded nodes/ways data")
        
        # Load deleted nodes if they exist
        deleted_nodes = set()
        if os.path.exists(deleted_nodes_path):
            with open(deleted_nodes_path, 'rb') as f:
                deleted_nodes = set(pickle.load(f))
            print(f"✓ Loaded {len(deleted_nodes)} deleted nodes")
        
        # Import graph building functions
        from find_route import simplify_graph, clean_up_graph
        from app import apply_deleted_nodes_to_graph
        
        # Build baseline graph (min_street_length: 0, includes ALL streets)
        baseline_graph = simplify_graph(nodes, ways, settings={
            'coverage_mode': 'balanced',
            'min_street_length': 0
        })
        if deleted_nodes:
            baseline_graph = apply_deleted_nodes_to_graph(baseline_graph, deleted_nodes)
        baseline_graph = clean_up_graph(baseline_graph)
        
        print(f"✓ Built baseline graph: {baseline_graph.number_of_nodes()} nodes, {baseline_graph.number_of_edges()} edges")
        
        return nodes, ways, baseline_graph, deleted_nodes
        
    except Exception as e:
        print(f"ERROR loading graph: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None


def analyze_graph_variance(graph: nx.Graph):
    """Analyze edge weight distribution in the graph."""
    weights = []
    distances = []
    
    for u, v, data in graph.edges(data=True):
        if 'weight' in data:
            weights.append(data['weight'])
        if 'distance' in data:
            distances.append(data['distance'])
    
    if weights:
        print(f"  Edge weights: min={min(weights):.4f}, max={max(weights):.4f}, unique={len(set(weights))}")
    if distances:
        print(f"  Edge distances (m): min={min(distances):.1f}, max={max(distances):.1f}")


def calculate_coverage_percentage(baseline_graph: nx.Graph, route_graph: nx.Graph, route: List[Tuple[float, float]]) -> float:
    """Calculate coverage percentage by total edge length (matches web app calculation).
    
    Args:
        baseline_graph: Graph with all streets (min_street_length: 0) for baseline calculation
        route_graph: Graph used for routing (may have min_street_length filter)
        route: List of (lat, lon) coordinates
    """
    if not route or len(route) < 2:
        return 0.0
    
    # Calculate total available edge length from BASELINE graph (all streets)
    baseline_total_length = sum(
        data.get('distance', 0) for _, _, data in baseline_graph.edges(data=True)
    )
    
    if baseline_total_length == 0:
        return 0.0
    
    # Build coordinate lookup from route_graph (the graph actually used for routing)
    coord_to_node = {route_graph.nodes[n]['coordinates']: n for n in route_graph.nodes()}
    
    # Track covered edges and their lengths
    covered_edges = {}  # {frozenset([u,v]): length}
    
    for i in range(len(route) - 1):
        lat1, lon1 = route[i]
        lat2, lon2 = route[i + 1]
        
        node1 = node2 = None
        
        for coords, node in coord_to_node.items():
            if abs(coords[0] - lat1) < 0.00001 and abs(coords[1] - lon1) < 0.00001:
                node1 = node
            if abs(coords[0] - lat2) < 0.00001 and abs(coords[1] - lon2) < 0.00001:
                node2 = node
        
        if node1 is not None and node2 is not None and route_graph.has_edge(node1, node2):
            edge_key = frozenset([node1, node2])
            edge_length = route_graph[node1][node2].get('distance', 0)
            if edge_key not in covered_edges:
                covered_edges[edge_key] = edge_length
    
    # Sum covered edge lengths
    covered_length = sum(covered_edges.values())
    
    # Coverage = covered length / baseline total length
    return round((covered_length / baseline_total_length) * 100, 1)


def main():
    # Default to the graph just copied
    graph_id = "42534d92-e150-414d-bf0d-dff3bf7efd25"
    
    if len(sys.argv) > 1:
        graph_id = sys.argv[1]
    
    print("\n" + "="*80)
    print(f"TESTING REAL GRAPH: {graph_id}")
    print("="*80 + "\n")
    
    # Load nodes/ways and build baseline graph
    nodes, ways, baseline_graph, deleted_nodes = load_graph(graph_id)
    if baseline_graph is None:
        return 1
    
    # Analyze baseline graph characteristics
    print("\nBaseline Graph Analysis (min_street_length: 0):")
    analyze_graph_variance(baseline_graph)
    
    # Calculate baseline total length
    baseline_total_length = sum(
        data.get('distance', 0) for _, _, data in baseline_graph.edges(data=True)
    )
    print(f"  Total street length: {baseline_total_length/1000:.2f} km")
    
    # Try to load saved routes to get the start node used in web app
    routes_path = f"temp/{graph_id}_routes.pkl"
    saved_start_node = None
    if os.path.exists(routes_path):
        try:
            with open(routes_path, 'rb') as f:
                saved_routes = pickle.load(f)
            # Extract start node from first route if available
            if saved_routes and len(saved_routes) > 0:
                first_route_data = list(saved_routes.values())[0]
                if 'route' in first_route_data and len(first_route_data['route']) > 0:
                    first_coord = first_route_data['route'][0]
                    # Find node with matching coordinates
                    for node, data in baseline_graph.nodes(data=True):
                        if data.get('coordinates') == first_coord:
                            saved_start_node = node
                            print(f"✓ Using saved start node: {saved_start_node}")
                            break
        except Exception as e:
            print(f"Note: Could not load saved start node: {e}")
    
    # Import functions
    try:
        from find_route import find_route_max_coverage_optimized, simplify_graph, clean_up_graph
        from app import apply_deleted_nodes_to_graph
    except ImportError as e:
        print(f"\nERROR: Could not import find_route: {e}")
        return 1
    
    # Test all three profiles (matches web app logic)
    print("\n" + "-"*80)
    print("COVERAGE RESULTS (matches web app calculation)")
    print("-"*80)
    
    route_configs = [
        ('fastest', 30, 70),   # target 30%, min_street_length 70m
        ('balanced', 70, 100), # target 70%, min_street_length 100m
        ('thorough', 95, 0),   # target 95%, min_street_length 0m
    ]
    
    for profile, target, min_length in route_configs:
        # Build route-specific graph with min_street_length filter
        route_graph = simplify_graph(nodes, ways, settings={
            'coverage_mode': 'balanced',
            'min_street_length': min_length
        })
        if deleted_nodes:
            route_graph = apply_deleted_nodes_to_graph(route_graph, deleted_nodes)
        route_graph = clean_up_graph(route_graph)
        
        # Use saved start node if available, otherwise pick first available node
        if saved_start_node and saved_start_node in route_graph:
            start_node = saved_start_node
        else:
            start_node = list(route_graph.nodes())[0]
        
        try:
            route = find_route_max_coverage_optimized(
                route_graph, start_node, settings={'speed_priority': profile}
            )
            
            # Calculate coverage against baseline (all streets), not just route_graph
            coverage = calculate_coverage_percentage(baseline_graph, route_graph, route)
            delta = coverage - target
            
            status = "✓" if abs(delta) <= 5 else "!" if abs(delta) <= 15 else "✗"
            
            print(f"  {profile.upper():10} | Target: {target:2}% | Actual: {coverage:6.1f}% | Delta: {delta:+6.1f}% | Min: {min_length:3}m {status}")
            
        except Exception as e:
            print(f"  {profile.upper():10} | ERROR: {str(e)[:60]}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
