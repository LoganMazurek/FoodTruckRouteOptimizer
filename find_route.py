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

MIN_STREET_LENGTH_METERS = 200

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

def simplify_graph(nodes, ways):
    node_to_ways = {}

    # Remove patterns for small streets/courts exclusion
    # exclude_patterns = re.compile(r'\b(Court|Ct|Lane|Ln|Place|Pl|Way|Terrace|Ter|Circle|Cir|Alley|Aly)\b', re.IGNORECASE)

    for way in ways:
        way_id = way.get("name")
        node_list = way.get("nodes", [])

        # Filter out unnamed, service, or non-drivable streets, but do NOT exclude courts/lanes/etc.
        highway = way.get("highway")
        if (
            not way_id or
            highway in ["service", "footway", "track", "pedestrian"]
        ):
            continue  # Skip non-drivable or irrelevant roads
        
        # Calculate total length of the way
        total_length = 0
        for i in range(1, len(node_list)):
            coord1 = nodes[node_list[i-1]]
            coord2 = nodes[node_list[i]]
            total_length += geodesic(coord1, coord2).meters

        if total_length < MIN_STREET_LENGTH_METERS:
            continue  # Skip short streets
        
        for node_id in node_list:
            node_to_ways.setdefault(node_id, set()).add(way_id)

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

    # Loop through each way and build the graph
    for way in ways:
        way_id = way.get("name")
        node_list = way.get("nodes", [])

        # Skip irrelevant streets that don't meet the criteria
        highway = way.get("highway")
        if not way_id or highway in ["service", "residential", "footway", "track", "pedestrian"]:
            continue
        
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
                    simplified_graph.add_edge(
                        start_node,
                        node_list[j],
                        distance=total_distance,
                        weight=total_distance,
                        way_id=way_id
                    )
                    break
                j += 1
            i = j

    # Retain only the largest connected component
    largest_cc = max(nx.connected_components(simplified_graph), key=len)
    simplified_graph = simplified_graph.subgraph(largest_cc).copy()

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

        return coordinates_route
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
    # Convert frozenset in steps to lists for JSON serialization
    if visualize:
        for step in steps:
            if 'used_edges' in step:
                step['used_edges'] = [list(e) if isinstance(e, frozenset) else e for e in step['used_edges']]
        return coordinates_route, steps
    return coordinates_route

def find_route_cpp(graph, start_node=None, max_edge_reuse=4, trim_loops=False):
    """
    Find a route that covers every edge at least once (Chinese Postman Problem / Eulerian path).
    Always starts at start_node. End node is ignored.
    Allows more edge reuse (default 4) and disables aggressive loop trimming for better edge coverage.
    If the graph is not Eulerian, it will be eulerized (duplicate edges added as needed).
    Returns the route as a list of coordinates (lat, lon).
    """
    import networkx as nx
    G = graph.copy()
    # Eulerize the graph if needed
    G_euler = nx.eulerize(G)
    # Find Eulerian circuit or path
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
    # --- Allow more edge reuse, do not trim loops ---
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
    # No aggressive loop trimming
    coordinates_route = [(G.nodes[n]['coordinates'][0], G.nodes[n]['coordinates'][1]) for n in filtered_path]
    return coordinates_route

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
