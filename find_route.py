from collections import defaultdict
import itertools
import math
from matplotlib import pyplot as plt
import networkx as nx
import logging
from geopy.distance import geodesic

from visualization import visualize_graph, selected_start_node

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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
    
    # Create a mapping from nodes to ways (streets)
    for way in ways:
        way_id = way.get("name")  # Assuming the street name is used as the way_id
        node_list = way.get("nodes", [])
        
        # Filter out unnamed, service, or non-drivable streets
        highway = way.get("highway")
        if not way_id or highway in ["service", "footway", "track", "pedestrian"]:
            continue  # Skip non-drivable or irrelevant roads
        
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

def find_route(graph):
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

def clean_up_graph(G):
    isolated_nodes = list(nx.isolates(G))
    G.remove_nodes_from(isolated_nodes)
    return G
