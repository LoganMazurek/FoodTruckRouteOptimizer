from collections import defaultdict
import itertools
import math
from matplotlib import pyplot as plt
import networkx as nx
import logging
from geopy.distance import geodesic
import re
import time

from visualization import selected_start_node

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Coverage Optimization Settings:
# - Reduced minimum street length from 200m to 50m to include smaller streets
# - Now includes residential and service roads for comprehensive coverage
# - Only excludes non-drivable paths (footways, tracks, pedestrian-only, cycleways)
# This significantly improves route coverage in residential areas
MIN_STREET_LENGTH_METERS = 50

def calculate_angle(p1, p2, p3):
    """
    Calculate the angle (in degrees) between three points p1 -> p2 -> p3.
    """
    def vector(a, b):
        return (float(b[0]) - float(a[0]), float(b[1]) - float(a[1]))

    v1 = vector(p2, p1)
    v2 = vector(p2, p3)

    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.hypot(*v1)
    mag2 = math.hypot(*v2)

    if mag1 == 0 or mag2 == 0:
        return 0

    cos_theta = dot_product / (mag1 * mag2)
    cos_theta = max(-1, min(1, cos_theta))  # Clamp value between -1 and 1
    angle_rad = math.acos(cos_theta)

    return math.degrees(angle_rad)

def euclidean_distance(lat1, lng1, lat2, lng2):
    """
    Calculate the Euclidean distance between two points (lat, lng) assuming
    a small area (up to a few miles)
    """
    lat_diff = lat2 - lat1
    lng_diff = lng2 - lng1

    return math.sqrt(lat_diff**2 + lng_diff**2)

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the distance between two points on the Earth's surface using the Haversine formula.
    Returns distance in kilometers.
    """
    R = 6371.0  # Earth radius in kilometers
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlng = lon2_rad - lon1_rad

    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlng / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c  # Distance in kilometers


def merge_nearby_nodes(graph, distance_threshold_m=18, protected_nodes=None):
    """
    Merge nodes that are very close together to reduce duplicate/stacked points.

    This helps collapse split carriageway endpoints and near-duplicate geometry
    nodes that create unnecessary route points.
    
    Args:
        graph: NetworkX graph
        distance_threshold_m: Minimum distance to treat nodes as separate
        protected_nodes: Set of node IDs to protect from merging (e.g., start/end nodes)
    """
    if graph.number_of_nodes() < 2 or distance_threshold_m <= 0:
        return graph

    protected_nodes = protected_nodes or set()
    lat_cell_size = distance_threshold_m / 111320.0
    if lat_cell_size <= 0:
        return graph

    buckets = defaultdict(list)
    coords = {}

    for node_id, data in graph.nodes(data=True):
        coord = data.get('coordinates')
        if not coord:
            continue

        lat, lon = coord
        coords[node_id] = (lat, lon)
        lat_idx = int(math.floor(lat / lat_cell_size))
        lon_idx = int(math.floor(lon / lat_cell_size))
        buckets[(lat_idx, lon_idx)].append(node_id)

    if len(coords) < 2:
        return graph

    mapping = {}
    processed = set()

    for node_id, (lat, lon) in coords.items():
        if node_id in processed:
            continue

        lat_idx = int(math.floor(lat / lat_cell_size))
        lon_idx = int(math.floor(lon / lat_cell_size))

        cluster = [node_id]
        for d_lat in (-1, 0, 1):
            for d_lon in (-1, 0, 1):
                for candidate in buckets.get((lat_idx + d_lat, lon_idx + d_lon), []):
                    if candidate == node_id or candidate in processed:
                        continue

                    c_lat, c_lon = coords[candidate]
                    if haversine(lat, lon, c_lat, c_lon) * 1000 <= distance_threshold_m:
                        cluster.append(candidate)

        if len(cluster) > 1:
            # Prefer protected nodes as representative
            protected_in_cluster = [n for n in cluster if n in protected_nodes]
            if protected_in_cluster:
                representative = protected_in_cluster[0]
            else:
                representative = max(cluster, key=lambda n: (graph.degree(n), str(n)))
            for member in cluster:
                mapping[member] = representative
            processed.update(cluster)
        else:
            processed.add(node_id)

    if not mapping:
        return graph

    merged = nx.relabel_nodes(graph, mapping, copy=True)

    collapsed = nx.Graph()
    for node_id, data in merged.nodes(data=True):
        collapsed.add_node(node_id, **data)

    for u, v, data in merged.edges(data=True):
        if u == v:
            continue

        if not collapsed.has_edge(u, v):
            collapsed.add_edge(u, v, **data)
            continue

        existing = collapsed[u][v]
        existing_weight = existing.get('weight', float('inf'))
        new_weight = data.get('weight', float('inf'))
        if new_weight < existing_weight:
            collapsed[u][v].update(data)

    logger.info(
        f"[MERGE_NODES] threshold={distance_threshold_m}m, "
        f"before={graph.number_of_nodes()} nodes/{graph.number_of_edges()} edges, "
        f"after={collapsed.number_of_nodes()} nodes/{collapsed.number_of_edges()} edges"
    )

    return collapsed


def compute_turn_penalty(degrees):
    """
    Compute a penalty based on the angle in degrees.
    """
    if degrees < 120:
        return 10
    elif degrees < 150:
        return 40
    else:
        return 50

def identify_intersections_and_end_nodes(nodes, ways):
    node_to_ways = {}

    # Map each node to the ways it's part of
    for way_id, node_list in ways.items():
        for node_id in node_list:
            node_to_ways.setdefault(node_id, set()).add(way_id)

    # Intersections: nodes part of 2+ ways
    intersection_node_ids = {nid for nid, way_set in node_to_ways.items() if len(way_set) > 1}

    # End nodes: first and last node of each way
    end_node_ids = set()
    for node_list in ways.values():
        end_node_ids.add(node_list[0])
        end_node_ids.add(node_list[-1])

    merged_nodes = intersection_node_ids | end_node_ids

    return {
        "intersection_node_ids": intersection_node_ids,
        "end_node_ids": end_node_ids,
        "merged_nodes": merged_nodes,
    }

def expand_route_with_geometry(route_path, graph):
    """
    Expand a simplified route by adding back intermediate nodes from the original geometry.
    
    Args:
        route_path: List of node IDs from the optimized route
        graph: NetworkX graph with 'intermediate_nodes' (as coordinates) stored on edges
    
    Returns:
        List of (lat, lon) tuples with all intermediate nodes included
    """
    if not route_path or len(route_path) < 2:
        return [(graph.nodes[n]['coordinates'][0], graph.nodes[n]['coordinates'][1]) for n in route_path]
    
    expanded_coords = []
    expanded_coords.append(tuple(graph.nodes[route_path[0]]['coordinates']))
    
    for i in range(1, len(route_path)):
        curr_node = route_path[i - 1]
        next_node = route_path[i]
        
        # Get intermediate node coordinates for this edge
        if graph.has_edge(curr_node, next_node):
            edge_data = graph[curr_node][next_node]
            intermediate_coords = edge_data.get('intermediate_nodes', [])
            
            # Add all intermediate node coordinates
            for coord in intermediate_coords:
                if coord:  # Skip None/empty entries
                    expanded_coords.append(tuple(coord))
        
        # Add the next endpoint
        expanded_coords.append(tuple(graph.nodes[next_node]['coordinates']))
    
    logger.debug(f"[EXPAND_ROUTE] Expanded route from {len(route_path)} to {len(expanded_coords)} points")
    return expanded_coords


def filter_dead_end_nodes(graph, protected_nodes=None, max_iterations=10):
    """
    Iteratively remove dead-end nodes (degree 1) and nodes that form short cul-de-sacs.
    Also filters out nodes from streets with names indicating courts/cul-de-sacs.
    
    Args:
        graph: NetworkX graph
        protected_nodes: Set of node IDs to protect from removal (e.g., start/end nodes)
        max_iterations: Maximum number of pruning passes (default 10)
    
    Returns:
        Filtered graph with dead-ends removed
    """
    G = graph.copy()
    protected_nodes = protected_nodes or set()
    
    # Identify court/cul-de-sac names
    court_keywords = ['court', 'ct', 'cul-de-sac', 'circle', 'cir', 'loop', 'place', 'pl']
    
    total_removed = 0
    iteration = 0
    
    for iteration in range(max_iterations):
        nodes_to_remove = set()
        
        for node in list(G.nodes()):
            # Never remove protected nodes (start/end)
            if node in protected_nodes:
                continue
                
            degree = G.degree(node)
            
            # Remove degree-1 nodes (dead ends)
            if degree == 1:
                nodes_to_remove.add(node)
                continue
            
            # Check if node is part of a court/cul-de-sac street
            if degree <= 2:
                for neighbor in G.neighbors(node):
                    if G.has_edge(node, neighbor):
                        edge_data = G[node][neighbor]
                        way_name = edge_data.get('way_name', '').lower()
                        # Check if street name contains court keywords
                        if any(keyword in way_name for keyword in court_keywords):
                            nodes_to_remove.add(node)
                            break
        
        if not nodes_to_remove:
            break
        
        G.remove_nodes_from(nodes_to_remove)
        total_removed += len(nodes_to_remove)
        
        # Keep only the largest connected component after removal
        if G.number_of_nodes() > 0 and not nx.is_connected(G):
            components = list(nx.connected_components(G))
            if components:
                # Prefer the component that contains the most protected nodes
                best_component = max(components, key=lambda c: sum(1 for n in c if n in protected_nodes))
                G = G.subgraph(best_component).copy()
    
    logger.debug(f"[FILTER_DEAD_ENDS] Removed {total_removed} dead-end/court nodes in {iteration + 1} iterations (protected {len(protected_nodes)} nodes)")
    return G

def simplify_graph(nodes, ways, settings=None, start_node=None, end_node=None):
    """
    Build a simplified graph from OSM data with configurable coverage settings.
    
    Args:
        nodes: Dictionary of node IDs to coordinates
        ways: List of way dictionaries with nodes and metadata
        settings: Optional dict with keys:
            - coverage_mode: 'balanced', 'maximum', or 'major-roads'
            - min_street_length: Minimum street length in meters (default 50)
            - speed_priority: 'balanced', 'fastest', or 'thorough'
            - filter_dead_ends: Whether to filter dead-ends (default False)
        start_node: Optional start node ID to protect from filtering
        end_node: Optional end node ID to protect from filtering
    """
    settings = settings or {}
    coverage_mode = settings.get('coverage_mode', 'balanced')
    min_length = settings.get('min_street_length', MIN_STREET_LENGTH_METERS)
    speed_priority = settings.get('speed_priority', 'balanced')
    node_snap_distance_m = settings.get('node_snap_distance_m', 18)
    filter_dead_ends = settings.get('filter_dead_ends', False)

    logger.info(
        f"[SIMPLIFY_GRAPH] Building graph with: coverage={coverage_mode}, "
        f"min_length={min_length}m, speed={speed_priority}, node_snap={node_snap_distance_m}m, "
        f"filter_dead_ends={filter_dead_ends}"
    )
    
    # Determine which road types to exclude based on coverage mode
    if coverage_mode == 'maximum':
        # Maximum coverage - only exclude truly non-drivable paths
        excluded_types = ["footway", "path", "cycleway", "steps"]
        logger.debug(f"[SIMPLIFY_GRAPH] Coverage=maximum: excluded types = {excluded_types}")
    elif coverage_mode == 'major-roads':
        # Major roads only - exclude residential and service roads
        excluded_types = ["footway", "track", "pedestrian", "path", "cycleway", "service", "residential", "unclassified"]
        logger.debug(f"[SIMPLIFY_GRAPH] Coverage=major-roads: excluded types = {excluded_types}")
    else:  # balanced
        # Balanced - exclude only non-drivable
        excluded_types = ["footway", "track", "pedestrian", "path", "cycleway"]
        logger.debug(f"[SIMPLIFY_GRAPH] Coverage=balanced: excluded types = {excluded_types}")
    
    node_to_ways = {}
    total_ways = len(ways)
    filtered_by_type = 0
    filtered_by_length = 0
    kept_ways = 0

    # Pre-compute segment lengths and aggregated street lengths.
    # IMPORTANT: OSM roads are often split into multiple way segments.
    # We filter by full aggregated street length (name + highway), not segment length.
    way_segment_lengths = {}
    street_total_lengths = defaultdict(float)

    for idx, way in enumerate(ways):
        node_list = way.get("nodes", [])
        segment_length = 0
        for i in range(1, len(node_list)):
            coord1 = nodes[node_list[i - 1]]
            coord2 = nodes[node_list[i]]
            segment_length += geodesic(coord1, coord2).meters

        way_segment_lengths[idx] = segment_length
        street_key = (way.get("name"), way.get("highway", "unclassified"))
        if street_key[0]:
            street_total_lengths[street_key] += segment_length

    for idx, way in enumerate(ways):
        way_id = way.get("name")
        node_list = way.get("nodes", [])

        # Filter based on coverage mode
        highway = way.get("highway")
        if not way_id or highway in excluded_types:
            filtered_by_type += 1
            continue
        
        street_key = (way_id, highway)
        effective_street_length = street_total_lengths.get(street_key, way_segment_lengths.get(idx, 0))

        if effective_street_length < min_length:
            filtered_by_length += 1
            continue  # Skip streets shorter than threshold
        
        kept_ways += 1
        for node_id in node_list:
            node_to_ways.setdefault(node_id, set()).add(way_id)

    logger.info(f"[SIMPLIFY_GRAPH] Way filtering: {total_ways} total, {filtered_by_type} by type, {filtered_by_length} by length (<{min_length}m), {kept_ways} kept")
    logger.debug(f"[SIMPLIFY_GRAPH] After filtering: {len(node_to_ways)} relevant nodes from {len(ways)} ways")

    # Intersections are nodes that belong to more than 1 way
    intersections = {nid for nid, ways_set in node_to_ways.items() if len(ways_set) > 1}
    
    # End nodes are the first and last nodes of each way
    end_nodes = set()
    for way in ways:
        node_list = way.get("nodes", [])
        if node_list:
            if node_list[0] not in intersections:
                end_nodes.add(node_list[0])
            if node_list[-1] not in intersections:
                end_nodes.add(node_list[-1])

    # Important nodes include intersections and end nodes
    important_nodes = intersections | end_nodes

    simplified_graph = nx.Graph()
    edges_added = 0
    ways_filtered_by_type_in_graph = 0
    ways_filtered_by_length_in_graph = 0

    # Loop through each way and build the graph
    for idx, way in enumerate(ways):
        way_id = way.get("name")
        node_list = way.get("nodes", [])

        # Apply same filtering for graph building
        highway = way.get("highway")
        if not way_id or highway in excluded_types:
            ways_filtered_by_type_in_graph += 1
            continue
        
        # Use aggregated street length (across all same-name segments),
        # not this single segment length.
        street_key = (way_id, highway)
        total_way_length = street_total_lengths.get(street_key, way_segment_lengths.get(idx, 0))

        if total_way_length < min_length:
            ways_filtered_by_length_in_graph += 1
            continue  # Skip streets shorter than threshold
        
        i = 0
        
        # Loop through the nodes in the current way (street)
        while i < len(node_list) - 1:
            if node_list[i] not in important_nodes:
                i += 1
                continue

            start_node = node_list[i]
            path = [start_node]
            total_distance = 0

            # Ensure the start node has its position data in the graph
            if start_node not in simplified_graph:
                simplified_graph.add_node(start_node, coordinates=nodes[start_node])

            j = i + 1
            while j < len(node_list):
                path.append(node_list[j])

                prev = nodes[node_list[j - 1]]
                curr = nodes[node_list[j]]
                total_distance += geodesic(prev, curr).meters

                # Ensure the current node has its position data in the graph
                if node_list[j] not in simplified_graph:
                    simplified_graph.add_node(node_list[j], coordinates=nodes[node_list[j]])

                if node_list[j] in important_nodes:
                    # Add edge between start_node and the current important node
                    # Store the intermediate node COORDINATES (not IDs) so they survive filtering
                    intermediate_coords = [(nodes[nid][0], nodes[nid][1]) for nid in path[1:-1]]
                    simplified_graph.add_edge(
                        start_node,
                        node_list[j],
                        distance=total_distance,
                        weight=total_distance,
                        way_id=way_id,
                        way_length=total_way_length,  # Store full way length for debugging
                        intermediate_nodes=intermediate_coords  # Store coordinates, not node IDs
                    )
                    edges_added += 1
                    break
                j += 1
            i = j

    logger.info(f"[SIMPLIFY_GRAPH] Graph building: {ways_filtered_by_type_in_graph} ways filtered by type, {ways_filtered_by_length_in_graph} by length, {edges_added} edges added")

    if simplified_graph.number_of_nodes() == 0:
        logger.warning("[SIMPLIFY_GRAPH] Empty graph after filtering")
        return simplified_graph

    # Prepare set of nodes to protect from merging (start and end nodes)
    protected_for_merge = set()
    if start_node is not None and start_node in simplified_graph:
        protected_for_merge.add(start_node)
    if end_node is not None and end_node in simplified_graph:
        protected_for_merge.add(end_node)

    simplified_graph = merge_nearby_nodes(
        simplified_graph,
        distance_threshold_m=node_snap_distance_m,
        protected_nodes=protected_for_merge
    )

    if simplified_graph.number_of_nodes() == 0:
        logger.warning("[SIMPLIFY_GRAPH] Empty graph after node merge")
        return simplified_graph

    # Retain the connected component containing the start node (required)
    # If end_node is specified, prefer component containing both start and end
    components = list(nx.connected_components(simplified_graph))
    
    if start_node not in simplified_graph:
        logger.warning(f"[SIMPLIFY_GRAPH] Start node {start_node} not in graph after merging")
        # Fallback to largest component
        largest_cc = max(components, key=len)
    elif end_node and end_node in simplified_graph:
        # Both start and end exist - prefer component with both
        for component in components:
            if start_node in component and end_node in component:
                largest_cc = component
                break
        else:
            # Couldn't find component with both - use component with start
            for component in components:
                if start_node in component:
                    largest_cc = component
                    break
    else:
        # Only start_node matters
        for component in components:
            if start_node in component:
                largest_cc = component
                break
    
    simplified_graph = simplified_graph.subgraph(largest_cc).copy()

    # Optional: Filter out dead-ends and cul-de-sacs
    if filter_dead_ends:
        nodes_before = simplified_graph.number_of_nodes()
        edges_before = simplified_graph.number_of_edges()
        
        # Protect start and end nodes from filtering
        protected = set()
        if start_node is not None and start_node in simplified_graph:
            protected.add(start_node)
        if end_node is not None and end_node in simplified_graph:
            protected.add(end_node)
        
        simplified_graph = filter_dead_end_nodes(simplified_graph, protected_nodes=protected)
        nodes_removed = nodes_before - simplified_graph.number_of_nodes()
        edges_removed = edges_before - simplified_graph.number_of_edges()
        logger.info(f"[FILTER_DEAD_ENDS] Removed {nodes_removed} dead-end nodes and {edges_removed} edges (protected {len(protected)} nodes)")

    # Validate: check for any edges from streets shorter than min_length
    edge_lengths = [data.get('way_length', 0) for u, v, data in simplified_graph.edges(data=True)]
    if edge_lengths:
        min_edge_way_length = min(edge_lengths)
        max_edge_way_length = max(edge_lengths)
        avg_edge_way_length = sum(edge_lengths) / len(edge_lengths)
        logger.info(f"[SIMPLIFY_GRAPH] Edge way lengths: min={min_edge_way_length:.1f}m, max={max_edge_way_length:.1f}m, avg={avg_edge_way_length:.1f}m")
        
        if min_edge_way_length < min_length:
            logger.warning(f"[SIMPLIFY_GRAPH] WARNING: Found edge from way shorter than {min_length}m threshold! Shortest: {min_edge_way_length:.1f}m")

    logger.info(f"[SIMPLIFY_GRAPH] Final graph: {simplified_graph.number_of_nodes()} nodes, {simplified_graph.number_of_edges()} edges")
    
    return simplified_graph


def check_graph_connectivity(G):
    if not nx.is_connected(G):
        print("Graph is not connected.")
        # Find the connected components
        components = list(nx.connected_components(G))
        print(f"Connected components: {components}")
    else:
        print("Graph is connected.")

def find_route_tsp(graph):
    G = graph.copy()  # Work on a copy to avoid modifying the original graph

    check_graph_connectivity(G)

    # Identify the furthest west node (node with the smallest longitude)
    furthest_west_node = min(G.nodes, key=lambda node: G.nodes[node]['coordinates'][1])

    # Nearest Neighbor TSP Approximation
    def nearest_neighbor_tsp(G, start):
        unvisited = set(G.nodes)
        current_node = start
        route = [current_node]
        unvisited.remove(current_node)

        total_distance = 0
        while unvisited:
            # Find the nearest unvisited node
            nearest_node = min(unvisited, key=lambda node: nx.dijkstra_path_length(G, current_node, node, weight='weight'))
            route.append(nearest_node)
            unvisited.remove(nearest_node)
            total_distance += nx.dijkstra_path_length(G, current_node, nearest_node, weight='weight')
            current_node = nearest_node

        # Return to the start node to complete the cycle (optional)
        route.append(start)
        total_distance += nx.dijkstra_path_length(G, current_node, start, weight='weight')

        return route, total_distance

    # Solve the TSP using the nearest neighbor approach
    optimized_route, total_distance = nearest_neighbor_tsp(G, furthest_west_node)

    logger.debug(f"Optimized Route: {optimized_route}")
    logger.debug(f"Total Distance: {total_distance}")

    if optimized_route:
        # Convert nodes to coordinates (lat, lon)
        coordinates_route = []
        for node in optimized_route:
            lat, lon = G.nodes[node]['coordinates']
            coordinates_route.append((lat, lon))

        # Expand route with intermediate nodes for better geometry detail
        expanded_route = expand_route_with_geometry(optimized_route, G)
        return expanded_route
    else:
        logger.error("No route found")
        return None
    
def find_route_greedy(graph, start_node, end_node, max_penalty=10000, visualize=False):
    path = [start_node]
    G = graph.copy()
    total_penalty = 0
    used_edges = set()
    current_node = start_node
    steps = []
    step_count = 0
    start_time = time.time()
    visit_counts = {node: 0 for node in G.nodes}
    visit_counts[start_node] = 1
    MAX_VISITS_PER_NODE = 3  # Allow up to 4 visits per node

    print(f"[DEBUG] Starting greedy route: start_node={start_node}, end_node={end_node}, total_nodes={len(G.nodes)}, total_edges={len(G.edges)}")

    while current_node != end_node:
        step_count += 1
        if step_count % 10 == 0:
            print(f"[DEBUG] Step {step_count}: current_node={current_node}, path_length={len(path)}")
        candidates = []
        for neighbor in G.neighbors(current_node):
            visits = visit_counts[neighbor]
            revisit_penalty = 200 * visits if visits > 0 else 0  # Penalty increases with each visit
            if visits >= MAX_VISITS_PER_NODE:
                continue  # Don't allow more than max visits
            edge = frozenset([current_node, neighbor])
            reuse_penalty = 50 if edge in used_edges else 0
            edge_length = G[current_node][neighbor]['weight']
            if len(path) >= 2:
                a = G.nodes[path[-2]]['coordinates']
                b = G.nodes[path[-1]]['coordinates']
                c = G.nodes[neighbor]['coordinates']
                angle_penalty = calculate_angle(a, b, c)
            else:
                angle_penalty = 0
            score = edge_length - reuse_penalty - angle_penalty - revisit_penalty
            candidates.append((score, neighbor, edge, reuse_penalty, angle_penalty, revisit_penalty, visits))
        if not candidates:
            logger.error(f"[DEBUG] No candidates found from node {current_node}. Stopping search. Path so far: {path}")
            break
        # Print candidate details for first 10 steps and every 25th step
        if step_count <= 10 or step_count % 25 == 0:
            print(f"[DEBUG] Candidates at step {step_count} from node {current_node}:")
            for cand in candidates:
                print(f"    neighbor={cand[1]}, score={cand[0]:.2f}, reuse_penalty={cand[3]}, angle_penalty={cand[4]:.2f}, revisit_penalty={cand[5]}, visits={cand[6]}")
        candidates.sort(reverse=True)
        best = candidates[0]
        _, next_node, edge, reuse_penalty, angle_penalty, revisit_penalty, visits = best
        used_edges.add(edge)
        path.append(next_node)
        visit_counts[next_node] += 1
        if visualize:
            steps.append({
                "current_node": current_node,
                "chosen": next_node,
                "path": list(path),
                "used_edges": list(used_edges)
            })
        current_node = next_node
        # Detect possible infinite loop or stuck condition
        if step_count > len(G.nodes) * MAX_VISITS_PER_NODE:
            break
    elapsed = time.time() - start_time
    print(f"[DEBUG] find_route_greedy finished in {elapsed:.2f} seconds, steps: {step_count}, path length: {len(path)}")
    coordinates_route = []
    if path:
        for node in path:
            coordinates_route.append((G.nodes[node]['coordinates'][0], G.nodes[node]['coordinates'][1]))
    if visualize:
        steps = [convert_frozensets(step) for step in steps]
        return coordinates_route, steps
    return coordinates_route

def clean_up_graph(G):
    isolated_nodes = list(nx.isolates(G))
    G.remove_nodes_from(isolated_nodes)
    return G

def is_turn(p1, p2, p3, threshold_degrees=30):
    """
    Returns True if the angle at p2 (between p1->p2 and p2->p3) is less than the threshold (i.e., a turn).
    """
    def vector(a, b):
        return (b[0] - a[0], b[1] - a[1])
    v1 = vector(p2, p1)
    v2 = vector(p2, p3)
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    mag1 = math.hypot(*v1)
    mag2 = math.hypot(*v2)
    if mag1 == 0 or mag2 == 0:
        return False
    cos_theta = dot / (mag1 * mag2)
    cos_theta = max(-1, min(1, cos_theta))  # Clamp value between -1 and 1
    angle = math.acos(cos_theta) * 180 / math.pi
    return angle > threshold_degrees

def filter_turns(route, threshold_degrees=70, way_ids=None, min_distance_meters=30, graph=None):
    """
    Keeps only the start, end, and nodes that are intersections (part of 2+ streets) or street ends.
    Optionally, also applies sharp turn/way change/distance logic if graph is not provided.
    """
    if len(route) <= 2:
        return route
    if graph is not None:
        # Build a set of intersection and end node coordinates
        node_to_ways = {}
        for n in graph.nodes:
            node_to_ways[n] = set()
        for u, v, data in graph.edges(data=True):
            way_id = data.get('way_id', None)
            node_to_ways[u].add(way_id)
            node_to_ways[v].add(way_id)
        intersection_nodes = {n for n, ways in node_to_ways.items() if len(ways) > 1}
        end_nodes = set()
        for way_id in set(w for ways in node_to_ways.values() for w in ways if w is not None):
            way_nodes = [n for n, ways in node_to_ways.items() if way_id in ways]
            if way_nodes:
                end_nodes.add(way_nodes[0])
                end_nodes.add(way_nodes[-1])
        important_nodes = intersection_nodes | end_nodes
        # Convert node IDs to coordinates for fast lookup
        important_coords = set(tuple(graph.nodes[n]['coordinates']) for n in important_nodes if 'coordinates' in graph.nodes[n])
        filtered = [route[0]]
        for pt in route[1:-1]:
            if tuple(pt) in important_coords:
                filtered.append(pt)
        filtered.append(route[-1])
        return filtered
    # Fallback: previous logic if graph is not provided
    filtered = [route[0]]
    last_kept = route[0]
    for i in range(1, len(route)-1):
        turn = is_turn(route[i-1], route[i], route[i+1], threshold_degrees)
        way_change = False
        if way_ids is not None and i < len(way_ids)-1:
            if way_ids[i] != way_ids[i+1]:
                way_change = True
        if turn or way_change:
            lat1, lon1 = last_kept
            lat2, lon2 = route[i]
            dist = haversine(lat1, lon1, lat2, lon2) * 1000
            if dist < min_distance_meters:
                continue
            filtered.append(route[i])
            last_kept = route[i]
    filtered.append(route[-1])
    return filtered

def find_route_drive_efficient(graph, start_node, end_node, max_iterations=None):
    """
    Efficient snaking route from start to end with no street repetition.
    
    Strategy:
    - Remove dead-ends from graph (they don't help with through-routing)
    - Each street traversed exactly once (no backtracking)
    - Snake back and forth towards the end node
    - When stuck, jump to nearest unexplored area closer to end
    
    Args:
        graph: NetworkX graph
        start_node: Starting node
        end_node: Ending node
        max_iterations: Optional iteration limit
    
    Returns:
        Dict with route and coverage metadata
    """
    G = graph.copy()
    
    if start_node not in G or end_node not in G:
        logger.warning(f"[DRIVE_EFFICIENT] Start or end node not in graph")
        return {
            'route': [],
            'path': [],
            'covered_edge_length_m': 0,
            'total_edge_length_m': 0,
            'covered_edge_count': 0,
            'total_edge_count': 0
        }
    
    # Remove dead-ends to create a cleaner routing network
    # Keep start and end nodes protected
    protected = {start_node, end_node}
    G_filtered = filter_dead_end_nodes(G, protected_nodes=protected, max_iterations=5)
    
    # If filtering removed too much, use original graph
    if G_filtered.number_of_nodes() < 3:
        G_filtered = G.copy()
    
    path = [start_node]
    used_edges = set()
    current_node = start_node
    
    total_edges = set(frozenset(e) for e in G_filtered.edges())
    total_edge_count = len(total_edges)
    
    if max_iterations is None:
        max_iterations = total_edge_count * 3
    
    end_coords = G_filtered.nodes[end_node]['coordinates']
    iteration = 0
    prev_node = None  # Track previous node for better path selection
    
    while iteration < max_iterations and current_node != end_node:
        iteration += 1
        neighbors = list(G_filtered.neighbors(current_node))
        if not neighbors:
            break
        
        # Find unexplored edges
        unexplored = []
        for neighbor in neighbors:
            edge = frozenset([current_node, neighbor])
            if edge not in used_edges:
                # Calculate distance to end node
                neighbor_coords = G_filtered.nodes[neighbor]['coordinates']
                dist_to_end = haversine(
                    neighbor_coords[0], neighbor_coords[1],
                    end_coords[0], end_coords[1]
                )
                
                # Calculate current distance to end
                curr_coords = G_filtered.nodes[current_node]['coordinates']
                curr_dist_to_end = haversine(
                    curr_coords[0], curr_coords[1],
                    end_coords[0], end_coords[1]
                )
                
                # Prefer edges that don't always go toward end (allows winding)
                # Score: lower is better
                # - Slight preference for moving toward end (not too aggressive)
                # - Penalize backtracking to previous node
                # - Favor longer edges (more coverage per turn)
                edge_length = G_filtered[current_node][neighbor].get('distance', 1)
                
                toward_end_score = (dist_to_end - curr_dist_to_end) / 1000.0  # Convert to km, can be negative
                backtrack_penalty = 1000 if neighbor == prev_node else 0
                length_bonus = -edge_length / 100.0  # Favor longer edges
                
                # Gentle bias toward end (0.3 weight), but allow lateral/winding movement
                score = (toward_end_score * 0.3) + backtrack_penalty + length_bonus
                
                unexplored.append((neighbor, score, edge, dist_to_end))
        
        if unexplored:
            # Sort by score (allows winding), with distance to end as tiebreaker
            unexplored.sort(key=lambda x: (x[1], x[3]))
            next_node = unexplored[0][0]
            edge = unexplored[0][2]
            
            path.append(next_node)
            used_edges.add(edge)
            prev_node = current_node
            current_node = next_node
        else:
            # No unexplored edges - need to find a way to unexplored areas
            # Find all nodes with unexplored edges, prefer closer to end
            candidates = []
            for node in G_filtered.nodes():
                if node == current_node:
                    continue
                # Check if this node has unexplored edges
                has_unexplored = False
                for neighbor in G_filtered.neighbors(node):
                    edge = frozenset([node, neighbor])
                    if edge not in used_edges:
                        has_unexplored = True
                        break
                
                if has_unexplored:
                    node_coords = G_filtered.nodes[node]['coordinates']
                    dist_to_end = haversine(
                        node_coords[0], node_coords[1],
                        end_coords[0], end_coords[1]
                    )
                    # Try to find shortest path to this node
                    try:
                        path_length = nx.shortest_path_length(G_filtered, current_node, node, weight='distance')
                        candidates.append((node, dist_to_end, path_length))
                    except nx.NetworkXNoPath:
                        continue
            
            if candidates:
                # Pick node closest to end with shortest path
                candidates.sort(key=lambda x: (x[1], x[2]))
                target_node = candidates[0][0]
                
                try:
                    bridge_path = nx.shortest_path(G_filtered, current_node, target_node, weight='distance')
                    # Add bridge path (skip first node since it's current)
                    for node in bridge_path[1:]:
                        path.append(node)
                    current_node = target_node
                except nx.NetworkXNoPath:
                    break
            else:
                # No more unexplored areas - navigate to end
                break
    
    # Navigate to end node if not there
    if current_node != end_node:
        try:
            shortest_path = nx.shortest_path(G_filtered, current_node, end_node, weight='distance')
            path.extend(shortest_path[1:])
            current_node = end_node
        except nx.NetworkXNoPath:
            logger.debug(f"[DRIVE_EFFICIENT] No path to end node from {current_node}")
    
    # Calculate coverage
    coverage_pct = len(used_edges) / total_edge_count if total_edge_count > 0 else 0
    
    # Calculate covered edge length
    total_edge_length = sum(G_filtered[u][v].get('distance', 0) for u, v in G_filtered.edges())
    covered_edge_length = 0
    for edge in used_edges:
        edge_tuple = tuple(edge)
        if len(edge_tuple) == 2:
            u, v = edge_tuple
            if G_filtered.has_edge(u, v):
                covered_edge_length += G_filtered[u][v].get('distance', 0)
    
    logger.debug(
        f"[DRIVE_EFFICIENT] Snaking route complete: {len(path)} nodes, {len(used_edges)}/{total_edge_count} edges ({coverage_pct*100:.1f}%), "
        f"iterations {iteration}/{max_iterations}"
    )
    
    # Expand with intermediate nodes
    expanded_route = expand_route_with_geometry(path, G_filtered)
    
    # Return route with coverage metadata
    return {
        'route': expanded_route,
        'path': path,  # Simplified node IDs for edge calculations
        'covered_edge_length_m': covered_edge_length,
        'total_edge_length_m': total_edge_length,
        'covered_edge_count': len(used_edges),
        'total_edge_count': total_edge_count
    }


def find_route_max_coverage_optimized(graph, start_node, end_node=None, forbid_uturns=True, settings=None):
    """
    Max-coverage variant with U-turn penalty at intersections (degree 3+).
    Uses greedy edge selection with a penalty (not hard block) for U-turns.
    This maintains coverage while gently preferring non-U-turn routes.
    
    If end_node is specified, the route will try to end at that node instead of
    returning to the start. This allows for A-to-B routes with maximum coverage.
    
    Args:
        graph: NetworkX graph
        start_node: Starting node
        end_node: Optional ending node
        forbid_uturns: Whether to penalize U-turns
        settings: Optional dict with:
            - speed_priority: 'fastest', 'balanced', or 'thorough'
    """
    settings = settings or {}
    speed_priority = settings.get('speed_priority', 'balanced')
    
    # For fastest speed priority, use the efficient driving algorithm
    if speed_priority == 'fastest':
        return find_route_drive_efficient(graph, start_node, end_node or start_node)
    
    import time
    G = graph.copy()

    if start_node not in G:
        logger.warning(f"[FIND_ROUTE] Start node {start_node} not present in graph")
        return []
    
    intersections = {n for n in G.nodes if G.degree(n) >= 3}
    
    path = [start_node]
    used_edges = set()
    edge_usage_count = defaultdict(int)
    current_node = start_node
    prev = None
    total_edges = set(frozenset(e) for e in G.edges)

    if not total_edges:
        return [(G.nodes[start_node]['coordinates'][0], G.nodes[start_node]['coordinates'][1])]

    # Distinct route profiles so each option has a clear, visible impact.
    # Lower score is better.
    # Key differences:
    # - fastest: Strongly prefers shorter segments and avoids backtracking (efficiency focused)
    # - balanced: Moderate exploration, balances coverage and efficiency  
    # - thorough: Strongly prefers frontier exploration for maximum coverage
    speed_profiles = {
        'fastest': {
            'coverage_threshold': 0.50,  # Exit at 50% coverage
            'max_edge_reuse': 2,
            'reuse_penalty': 200.0,  # Much higher - strongly avoid reuse
            'used_edge_penalty': 120.0,  # Much higher - strongly avoid used edges
            'unused_bonus': 3.0,  # Lower - less drive to find new edges
            'frontier_bonus': 1.0,  # Lower - less frontier exploration
            'backtrack_penalty': 150.0,  # Much higher - strongly avoid backtracking
            'uturn_penalty': 50.0,  # Higher - avoid U-turns
            'edge_length_weight': 0.4,  # Much higher - strongly prefer shorter edges
            'end_pull_start': 0.40,  # Pull toward end earlier
            'end_pull_weight': 0.50,  # Strong pull to end node
            'allow_early_end_exit': True,
        },
        'balanced': {
            'coverage_threshold': 0.85,  # Target ~85-90% coverage
            'max_edge_reuse': 3,  # Increased from 2 to allow more backtracking
            'reuse_penalty': 80.0,  # Reduced to allow more reuse
            'used_edge_penalty': 40.0,
            'unused_bonus': 25.0,  # Increased preference for new edges
            'frontier_bonus': 12.0,  # Increased frontier exploration
            'backtrack_penalty': 50.0,
            'uturn_penalty': 18.0,
            'edge_length_weight': 0.03,
            'end_pull_start': 0.75,
            'end_pull_weight': 0.03,
            'allow_early_end_exit': False,  # Must reach 85% target before stopping at end node
        },
        'thorough': {
            'coverage_threshold': 0.97,  # Target ~97-100% coverage
            'max_edge_reuse': 4,  # Increased from 3 to allow maximum backtracking
            'reuse_penalty': 20.0,  # Very low penalty for reuse
            'used_edge_penalty': 5.0,  # Minimal penalty for used edges
            'unused_bonus': 150.0,  # Huge preference for new edges
            'frontier_bonus': 120.0,  # Huge preference for frontier areas
            'backtrack_penalty': 3.0,  # Almost no penalty for backtracking
            'uturn_penalty': 1.0,  # Almost no U-turn penalty
            'edge_length_weight': -0.08,  # Stronger preference for longer edges
            'end_pull_start': 0.99,
            'end_pull_weight': 0.0005,  # Almost ignore end node until target met
            'allow_early_end_exit': False,  # Must reach coverage target before stopping
        },
    }
    profile = speed_profiles.get(speed_priority, speed_profiles['balanced'])

    COVERAGE_THRESHOLD = profile['coverage_threshold']
    MAX_EDGE_REUSE = profile['max_edge_reuse']
    logger.info(
        f"[FIND_ROUTE] Speed priority={speed_priority}: "
        f"coverage_target={COVERAGE_THRESHOLD*100:.1f}%, max_reuse={MAX_EDGE_REUSE}, "
        f"unused_bonus={profile['unused_bonus']}, frontier_bonus={profile['frontier_bonus']}, "
        f"end_pull_start={profile['end_pull_start']}, end_pull_weight={profile['end_pull_weight']}"
    )
    logger.debug(
        f"[FIND_ROUTE] Full profile: {profile}"
    )
    
    start_time = time.time()
    
    # Greedy traversal with U-turn penalty
    # Increase iteration budget for more thorough exploration
    # Higher max_edge_reuse requires proportionally more iterations
    if speed_priority == 'fastest':
        max_iterations = len(total_edges) * 3  # Quick exit, moderate budget
    elif speed_priority == 'balanced':
        max_iterations = len(total_edges) * 6  # Must reach 85%, generous budget
    else:  # thorough
        max_iterations = len(total_edges) * 12  # Maximum coverage, very generous budget
    
    iteration = 0
    reached_end_node = False
    
    while iteration < max_iterations:
        iteration += 1
        neighbors = list(G.neighbors(current_node))
        if not neighbors:
            break
        
        # Check if we're at the end node with good coverage
        if end_node and current_node == end_node:
            coverage_pct = len(used_edges) / len(total_edges)
            # For profiles that allow early exit (fastest/balanced), can stop at end node
            # For thorough, must reach coverage target before stopping
            should_stop = coverage_pct >= COVERAGE_THRESHOLD
            if not should_stop and profile.get('allow_early_end_exit', False):
                # Allow early exit if we've done a reasonable amount of exploration
                should_stop = iteration > max(100, len(total_edges))
            
            if should_stop:
                reached_end_node = True
                logger.debug(f"[FIND_ROUTE] Reached end node with {coverage_pct*100:.1f}% edge coverage (by count)")
                break
        
        best_candidate = None
        best_score = float('inf')
        best_tiebreaker = float('inf')
        
        for n in neighbors:
            edge = frozenset([current_node, n])
            if edge_usage_count[edge] >= MAX_EDGE_REUSE:
                continue
            
            # Calculate metrics for this edge
            edge_length = G[current_node][n].get("weight", 1)
            is_unused = edge not in used_edges
            reuse_count = edge_usage_count[edge]
            
            # Calculate distance to end node (if specified) for tie-breaking
            distance_to_end = 999999
            if end_node and end_node in G.nodes:
                end_coords = G.nodes[end_node]['coordinates']
                next_coords = G.nodes[n]['coordinates']
                distance_to_end = haversine(next_coords[0], next_coords[1], end_coords[0], end_coords[1])
            
            # Check if this is a U-turn
            is_uturn = False
            is_forced_uturn = False
            if forbid_uturns and prev is not None:
                prev_pt = G.nodes[prev]['coordinates']
                curr_pt = G.nodes[current_node]['coordinates']
                next_pt = G.nodes[n]['coordinates']
                
                b1 = bearing(prev_pt, curr_pt)
                b2 = bearing(curr_pt, next_pt)
                delta = (b2 - b1 + 540) % 360 - 180
                turn_type = _turn_class(delta)
                
                if turn_type == "uturn":
                    is_uturn = True
                    if current_node in intersections:
                        is_forced_uturn = True  # Penalize but don't forbid
            
            coverage_pct = len(used_edges) / len(total_edges)

            unused_edges_from_neighbor = sum(
                1
                for nn in G.neighbors(n)
                if frozenset([n, nn]) not in used_edges
            )
            frontier_ratio = unused_edges_from_neighbor / max(1, G.degree(n))

            is_immediate_backtrack = prev is not None and n == prev

            score = 0.0
            score += reuse_count * profile['reuse_penalty']
            score += edge_length * profile['edge_length_weight']
            score -= frontier_ratio * profile['frontier_bonus']

            if is_unused:
                score -= profile['unused_bonus']
            else:
                score += profile['used_edge_penalty']

            if is_immediate_backtrack:
                score += profile['backtrack_penalty']

            if is_uturn and is_forced_uturn:
                score += profile['uturn_penalty']

            if end_node and coverage_pct >= COVERAGE_THRESHOLD * profile['end_pull_start']:
                score += distance_to_end * profile['end_pull_weight']

            tiebreaker = edge_length
            if best_candidate is None or score < best_score or (
                math.isclose(score, best_score) and tiebreaker < best_tiebreaker
            ):
                best_score = score
                best_tiebreaker = tiebreaker
                best_candidate = n
        
        if best_candidate is None:
            # No more good moves available
            coverage_pct = len(used_edges) / len(total_edges)
            
            # Check if we should exit because we've hit our target coverage
            if end_node and coverage_pct >= COVERAGE_THRESHOLD:
                # We've hit our coverage target - navigate directly to end node if not there
                if current_node != end_node:
                    try:
                        shortest_path = nx.shortest_path(G, current_node, end_node)
                        for node in shortest_path[1:]:  # Skip current node
                            path.append(node)
                        current_node = node
                        reached_end_node = True
                        logger.debug(f"[FIND_ROUTE] Hit coverage target ({coverage_pct*100:.1f}%), navigated to end node")
                    except nx.NetworkXNoPath:
                        logger.debug(f"[FIND_ROUTE] Hit coverage target but no path to end node")
                else:
                    reached_end_node = True
                    logger.debug(f"[FIND_ROUTE] Hit coverage target ({coverage_pct*100:.1f}%) at end node")
                break
            
            # If we haven't hit target, try navigating to end node for more exploration
            if end_node and current_node != end_node:
                try:
                    shortest_path = nx.shortest_path(G, current_node, end_node)
                    for node in shortest_path[1:]:  # Skip current node
                        path.append(node)
                        edge = frozenset([current_node, node])
                        used_edges.add(edge)
                        edge_usage_count[edge] += 1
                        current_node = node
                    reached_end_node = True
                    logger.debug(f"[FIND_ROUTE] Navigated to end node via shortest path")
                except nx.NetworkXNoPath:
                    logger.debug(f"[FIND_ROUTE] No path to end node from {current_node}")
            break
        
        edge = frozenset([current_node, best_candidate])
        path.append(best_candidate)
        used_edges.add(edge)
        edge_usage_count[edge] += 1
        prev = current_node
        current_node = best_candidate
        
        # Check for early exit when we hit coverage threshold
        coverage_pct = len(used_edges) / len(total_edges)
        min_iterations_required = max(10, int(max_iterations * 0.15))  # At least 15% of max_iterations or 10 iterations
        
        if coverage_pct >= COVERAGE_THRESHOLD and iteration > min_iterations_required:
            # Coverage target met - if we have an end node, navigate there; otherwise done
            if end_node:
                if current_node == end_node:
                    # Already at end node
                    logger.debug(f"[FIND_ROUTE] Hit coverage target ({coverage_pct*100:.1f}%) at end node")
                    break
                else:
                    # Navigate to end node (direct jump, not exploration)
                    try:
                        shortest_path = nx.shortest_path(G, current_node, end_node)
                        logger.debug(f"[FIND_ROUTE] Hit coverage target ({coverage_pct*100:.1f}%), jumping to end node")
                        for node in shortest_path[1:]:
                            path.append(node)
                        current_node = end_node
                        reached_end_node = True
                        break
                    except nx.NetworkXNoPath:
                        logger.debug(f"[FIND_ROUTE] Hit coverage target but no path to end node")
                        break
            else:
                # No end node specified - we're done when coverage reached
                break
    
    elapsed = time.time() - start_time
    coverage_by_count = len(used_edges) / len(total_edges)
    
    # Also calculate coverage by edge length (what user sees in result page)
    total_edge_length = sum(G[u][v].get('distance', 0) for u, v in G.edges())
    covered_edge_length = 0
    for edge in used_edges:
        edge_tuple = tuple(edge)
        if len(edge_tuple) == 2:
            u, v = edge_tuple
            if G.has_edge(u, v):
                covered_edge_length += G[u][v].get('distance', 0)
    coverage_by_length = covered_edge_length / total_edge_length if total_edge_length > 0 else 0
    
    if end_node:
        end_status = "reached" if reached_end_node or current_node == end_node else "not reached"
        logger.debug(
            f"[FIND_ROUTE] max_coverage_optimized: {coverage_by_count*100:.1f}% by count ({len(used_edges)}/{len(total_edges)} edges), "
            f"{coverage_by_length*100:.1f}% by length, end node {end_status}, speed_priority={speed_priority}"
        )
    else:
        logger.debug(
            f"[FIND_ROUTE] max_coverage_optimized: {coverage_by_count*100:.1f}% by count ({len(used_edges)}/{len(total_edges)} edges), "
            f"{coverage_by_length*100:.1f}% by length, speed_priority={speed_priority}"
        )
    
    # Expand route with intermediate nodes for better geometry detail
    expanded_route = expand_route_with_geometry(path, G)
    logger.debug(f"[FIND_ROUTE] Final route: {len(path)} simplified points → {len(expanded_route)} detailed points")
    
    # Return route with coverage metadata
    return {
        'route': expanded_route,
        'path': path,  # Simplified node IDs for edge calculations
        'covered_edge_length_m': covered_edge_length,
        'total_edge_length_m': total_edge_length,
        'covered_edge_count': len(used_edges),
        'total_edge_count': len(total_edges)
    }








def find_route_max_coverage(graph, start_node, end_node=None, visualize=False):
    """
    Traverse the graph from start_node, covering as many unique edges as possible,
    minimizing backtracking and revisits. Always starts at the specified node.
    Returns the route as a list of node coordinates, and (optionally) visualization steps.
    Allows up to 2 traversals per edge and early exit at 80% coverage.
    """
    import time
    G = graph.copy()
    path = [start_node]
    used_edges = set()
    edge_usage_count = defaultdict(int)
    current_node = start_node
    steps = []
    visited = set([start_node])
    total_edges = set(frozenset(e) for e in G.edges)
    max_steps = len(total_edges) * 2  # Safety limit
    step_count = 0
    MAX_EDGE_REUSE = 2
    COVERAGE_THRESHOLD = 0.80  # Allow 80% edge coverage before early exit

    memo = {}
    start_time = time.time()
    def dfs(node, path, used_edges, edge_usage_count, visited, steps, step_count):
        state = (node, frozenset(used_edges))
        if state in memo and len(path) >= memo[state]:
            return None, steps  # Already found a shorter or equal path to this state
        memo[state] = len(path)
        # Early exit: allow 80% edge coverage
        if len(used_edges) >= int(COVERAGE_THRESHOLD * len(total_edges)):
            return path, steps
        if step_count > max_steps:
            return None, steps  # Safety break
        neighbors = list(G.neighbors(node))
        # Prefer untraversed edges, but allow used edges with penalty, and max 2 uses per edge
        def edge_priority(n):
            edge = frozenset([node, n])
            return (edge in used_edges, edge_usage_count[edge])
        neighbors.sort(key=edge_priority)
        for neighbor in neighbors:
            edge = frozenset([node, neighbor])
            if edge_usage_count[edge] >= MAX_EDGE_REUSE:
                continue  # Don't allow more than max uses
            edge_reuse = edge in used_edges
            used_edges.add(edge)
            edge_usage_count[edge] += 1
            path.append(neighbor)
            if visualize:
                steps.append({
                    "current_node": node,
                    "chosen": neighbor,
                    "path": list(path),
                    "used_edges": list(used_edges)
                })
            result, s = dfs(neighbor, path, used_edges, edge_usage_count, visited, steps, step_count+1)
            if result:
                return result, s
            # Backtrack
            path.pop()
            edge_usage_count[edge] -= 1
            if edge_usage_count[edge] == 0:
                used_edges.remove(edge)
        return None, steps

    route, steps = dfs(current_node, path, used_edges, edge_usage_count, visited, steps, step_count)
    elapsed = time.time() - start_time
    print(f"[DEBUG] find_route_max_coverage finished in {elapsed:.2f} seconds, steps: {len(steps)}, path length: {len(path)}")
    if not route:
        # Fallback: return the longest path found
        route = path
    coordinates_route = [(G.nodes[n]['coordinates'][0], G.nodes[n]['coordinates'][1]) for n in route]
    # Expand route with intermediate nodes for better geometry detail
    expanded_route = expand_route_with_geometry(route, G)
    # Convert frozenset in steps to lists for JSON serialization
    if visualize:
        for step in steps:
            if 'used_edges' in step:
                step['used_edges'] = [list(e) if isinstance(e, frozenset) else e for e in step['used_edges']]
        return expanded_route, steps
    return expanded_route

def find_route_cpp(graph, start_node=None, max_edge_reuse=2, trim_loops=False, strategy="drive"):
    """
    Route finder with two strategies:
    - "drive": heuristic that prefers efficient driving (right turns, no U-turns at intersections).
    - "cpp": Eulerian route (Chinese Postman) for full coverage.
    Returns list of coordinates (lat, lon).
    """
    if strategy == "drive":
        if start_node is None:
            return []
        return find_route_drive_preferential(graph, start_node, max_edge_reuse=max_edge_reuse)

    # CPP/Eulerian for full coverage
    import networkx as nx
    G = graph.copy()
    G_euler = nx.eulerize(G)
    node_path = None
    if start_node is not None:
        try:
            path_edges = list(nx.eulerian_path(G_euler, source=start_node))
            node_path = [path_edges[0][0]] + [v for u, v in path_edges]
        except Exception:
            path_edges = list(nx.eulerian_circuit(G_euler))
            node_path = [path_edges[0][0]] + [v for u, v in path_edges]
    else:
        path_edges = list(nx.eulerian_circuit(G_euler))
        node_path = [path_edges[0][0]] + [v for u, v in path_edges]
    if not node_path:
        return []
    edge_counts = defaultdict(int)
    filtered_path = [node_path[0]]
    for i in range(1, len(node_path)):
        a, b = node_path[i-1], node_path[i]
        edge = frozenset([a, b])
        edge_counts[edge] += 1
        if edge_counts[edge] <= max_edge_reuse:
            filtered_path.append(b)
        else:
            continue
    coordinates_route = [(G.nodes[n]['coordinates'][0], G.nodes[n]['coordinates'][1]) for n in filtered_path]
    # Expand route with intermediate nodes for better geometry detail
    expanded_route = expand_route_with_geometry(filtered_path, G)
    return expanded_route

def convert_frozensets(obj):
    if isinstance(obj, frozenset):
        return list(obj)
    elif isinstance(obj, list):
        return [convert_frozensets(e) for e in obj]
    elif isinstance(obj, dict):
        return {k: convert_frozensets(v) for k, v in obj.items()}
    else:
        return obj

def bearing(p1, p2):
    """Returns the bearing in degrees from p1 to p2."""
    import math
    lat1, lon1 = map(math.radians, p1)
    lat2, lon2 = map(math.radians, p2)
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    brng = math.atan2(x, y)
    brng = math.degrees(brng)
    return (brng + 360) % 360

def _turn_delta_degrees(prev_pt, curr_pt, next_pt):
    """
    Return signed turn angle in degrees. Positive = right turn, negative = left.
    Range (-180, 180].
    """
    if prev_pt is None:
        return 0
    b1 = bearing(prev_pt, curr_pt)
    b2 = bearing(curr_pt, next_pt)
    delta = (b2 - b1 + 540) % 360 - 180
    return delta

def _turn_class(delta_degrees, straight_thresh=20, uturn_thresh=160):
    abs_delta = abs(delta_degrees)
    if abs_delta <= straight_thresh:
        return "straight"
    if abs_delta >= uturn_thresh:
        return "uturn"
    return "right" if delta_degrees > 0 else "left"

def find_route_drive_preferential(graph, start_node, max_edge_reuse=2, coverage_threshold=0.99):
    """
    Greedy route that prioritizes efficient driving:
    - avoid U-turns at intersections (allowed at cul-de-sacs or if forced)
    - prefer right turns over left turns
    - avoid reusing edges when possible
    - favors shorter edges
    Returns route as list of coordinates (lat, lon).
    """
    G = graph.copy()
    if start_node not in G:
        return []

    def is_culdesac(node_id):
        return G.degree(node_id) == 1

    total_edges = set(frozenset(e) for e in G.edges)
    used_edges = defaultdict(int)
    path = [start_node]
    current = start_node
    prev = None
    max_steps = max(1, len(total_edges) * max_edge_reuse * 4)

    def edge_length(u, v):
        return G[u][v].get("weight", 1)

    edge_lengths = [edge_length(u, v) for u, v in G.edges]
    if edge_lengths:
        edge_lengths_sorted = sorted(edge_lengths)
        long_edge_threshold = edge_lengths_sorted[int(0.75 * (len(edge_lengths_sorted) - 1))]
    else:
        long_edge_threshold = 0

    for _ in range(max_steps):
        neighbors = list(G.neighbors(current))
        if not neighbors:
            break

        unused_neighbors = [n for n in neighbors if used_edges[frozenset([current, n])] == 0]
        candidate_neighbors = unused_neighbors if unused_neighbors else neighbors

        candidates = []
        for n in candidate_neighbors:
            edge = frozenset([current, n])
            if used_edges[edge] >= max_edge_reuse:
                continue
            prev_pt = G.nodes[prev]['coordinates'] if prev is not None else None
            curr_pt = G.nodes[current]['coordinates']
            next_pt = G.nodes[n]['coordinates']
            delta = _turn_delta_degrees(prev_pt, curr_pt, next_pt)
            turn_type = _turn_class(delta)

            # Heavy penalty for U-turns, but don't skip them entirely
            length_value = edge_length(current, n)
            distance_penalty = length_value / 25.0
            reuse_penalty = 120 * used_edges[edge]
            if turn_type == "right":
                turn_penalty = -5
            elif turn_type == "left":
                turn_penalty = 25
            elif turn_type == "uturn":
                turn_penalty = 500 if not is_culdesac(current) else 50
            else:
                turn_penalty = 5

            # Prefer longer edges; penalize very short edges unless convenient
            length_bonus = -min(40, length_value / 20.0) if length_value >= long_edge_threshold else 10

            score = distance_penalty + reuse_penalty + turn_penalty + length_bonus
            candidates.append((score, n, edge))

        if not candidates:
            break

        candidates.sort(key=lambda x: x[0])
        _, next_node, edge = candidates[0]
        used_edges[edge] += 1
        prev = current
        current = next_node
        path.append(current)

    if path and path[-1] != start_node:
        path.append(start_node)

    # Expand route with intermediate nodes for better geometry detail
    expanded_route = expand_route_with_geometry(path, G)
    return expanded_route

def prune_common_sense_nodes(route, way_ids=None, angle_threshold=30, graph=None):
    """
    Aggressively prune: Only keep start, end, and nodes that are true intersections (part of 2+ ways) or street ends (first/last node of a way).
    """
    if len(route) <= 2 or graph is None:
        return route
    # Build a set of intersection and end node coordinates
    node_to_ways = {n: set() for n in graph.nodes}
    for u, v, data in graph.edges(data=True):
        way_id = data.get('way_id', None)
        node_to_ways[u].add(way_id)
        node_to_ways[v].add(way_id)
    intersection_nodes = {n for n, ways in node_to_ways.items() if len(ways) > 1}
    end_nodes = set()
    for way_id in set(w for ways in node_to_ways.values() for w in ways if w is not None):
        way_nodes = [n for n, ways in node_to_ways.items() if way_id in ways]
        if way_nodes:
            end_nodes.add(way_nodes[0])
            end_nodes.add(way_nodes[-1])
    important_nodes = intersection_nodes | end_nodes
    # Convert node IDs to coordinates for fast lookup
    important_coords = set(tuple(graph.nodes[n]['coordinates']) for n in important_nodes if 'coordinates' in graph.nodes[n])
    pruned = [route[0]]
    for pt in route[1:-1]:
        if tuple(pt) in important_coords:
            pruned.append(pt)
    pruned.append(route[-1])
    return pruned
