import logging
import uuid
from flask import Flask, jsonify, render_template, request, session
import os
import pickle
import networkx as nx

from build_urls import get_google_maps_url
from find_route import clean_up_graph, find_route_cpp, find_route_max_coverage_optimized, simplify_graph, prune_common_sense_nodes
from get_street_data import extract_nodes_and_ways, fetch_overpass_data, get_coordinates
from osrm_client import get_route_with_waypoints

app = Flask(__name__)
app.secret_key = 'z3ByRjbb-tN3VM4X71W2oITQupA='  # Replace with a secure secret key
GRAPH_DIR = os.path.join(os.path.dirname(__file__), "temp")

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

clat = 0
clng = 0


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        logger.debug("POST request received")
        zipcode = request.form.get("zipcode")
        clat, clng = get_coordinates(zipcode)
        
        if clat and clng:
            api_key = os.getenv('GOOGLE_MAPS_API_KEY')
            if not api_key:
                logger.error("GOOGLE_MAPS_API_KEY environment variable not set")
                return "API key not configured", 500
            return render_template("select_boundaries.html", lat=clat, lng=clng, api_key=api_key)
        else:
            logger.error("Failed to get coordinates for ZIP code")
            return "Failed to get coords for ZIP code", 500
    return render_template("index.html")

@app.route("/process_boundaries", methods=["POST"])
def process_boundaries():
    data = request.get_json()
    corners = data.get("corners")
    settings = data.get("settings", {})

    if len(corners) != 4:
        logger.error("Invalid number of corners, expected 4")
        return jsonify({"error": "Please select exactly 4 corners."}), 400

    logger.debug(f"Processing boundaries with corners: {corners}")
    logger.debug(f"Route settings: {settings}")

    latitudes = [corner['lat'] for corner in corners]
    longitudes = [corner['lng'] for corner in corners]

    min_lat = min(latitudes)
    max_lat = max(latitudes)
    min_lng = min(longitudes)
    max_lng = max(longitudes)

    center_lat = (min_lat + max_lat) / 2
    center_lng = (min_lng + max_lng) / 2

    street_data = fetch_overpass_data(min_lat, max_lat, min_lng, max_lng)
    
    if not street_data:
        logger.error("No street or intersection data found")
        return jsonify({"error": "No data found for the selected area"}), 500
    
    logger.debug(f"Street and intersection data: {street_data}")

    boundary_id = str(uuid.uuid4())

    nodes, ways = extract_nodes_and_ways(street_data)

    graph = simplify_graph(nodes, ways, settings=settings)
    graph = clean_up_graph(graph)

    # Save the graph to a file using pickle
    graph_file_path = os.path.join("temp", f"{boundary_id}_graph.pkl")
    with open(graph_file_path, "wb") as f:
        pickle.dump(graph, f)

    return jsonify({"boundary_id": boundary_id})

@app.route("/graph_leaflet")
def graph_leaflet():
    boundary_id = request.args.get("boundary_id")
    return render_template("graph_leaflet.html", boundary_id=boundary_id)

@app.route("/result")
def result():
    boundary_id = request.args.get("boundary_id")
    if not boundary_id:
        return "Missing boundary_id parameter", 400
    start_node = int(request.args.get("start_node"))
    graph_file_path = os.path.join("temp", f"{boundary_id}_graph.pkl")
    if not os.path.exists(graph_file_path):
        return "Graph not found", 404
    with open(graph_file_path, "rb") as graph_file:
        import pickle
        graph = pickle.load(graph_file)
    print("Graph node IDs:", list(graph.nodes))
    print("Optimizing route...")
    # --- Use only the CPP route logic ---
    optimized_route = find_route_max_coverage_optimized(graph, start_node)
    def is_latlon_tuple(x):
        return isinstance(x, (list, tuple)) and len(x) == 2 and all(isinstance(i, (float, int)) for i in x)
    if not optimized_route or not is_latlon_tuple(optimized_route[0]):
        return "No valid route found", 400
    print(f"[DEBUG] Optimized route length: {len(optimized_route)}")
    way_ids = []
    for i in range(len(optimized_route)-1):
        way_ids.append(None)  # No way_ids used
    way_ids.append(way_ids[-1] if way_ids else None)  # Pad to match route length
    pruned_route = prune_common_sense_nodes(optimized_route, way_ids=way_ids, angle_threshold=30, graph=graph)
    print(f"[DEBUG] Pruned route length: {len(pruned_route)}")
    logger.debug(f"[DEBUG] First 3 waypoints in pruned_route: {pruned_route[:3]}")
    logger.debug(f"[DEBUG] Last 3 waypoints in pruned_route: {pruned_route[-3:]}")
    
    # Validate waypoint coordinates
    min_lat = min(pt[0] for pt in pruned_route)
    max_lat = max(pt[0] for pt in pruned_route)
    min_lon = min(pt[1] for pt in pruned_route)
    max_lon = max(pt[1] for pt in pruned_route)
    logger.debug(f"[DEBUG] Route bounds - Lat: {min_lat:.4f} to {max_lat:.4f}, Lon: {min_lon:.4f} to {max_lon:.4f}")
    
    if len(pruned_route) < 2:
        return "Route too short after pruning", 400
    
    # For visualization: use the actual route path directly (no OSRM waypoint routing)
    # This avoids the huge loops from OSRM connecting duplicate waypoints
    route_geometry = pruned_route  # Direct path through graph edges for map display
    
    # For navigation: DON'T use OSRM - it creates massive loops from coordinate snapping issues
    # Instead, use graph edges directly since we already have exact street connections
    logger.debug(f"[NAV] Creating navigation from graph edges for {len(pruned_route)} waypoints")
    
    turn_by_turn_instructions = []
    total_distance_m = 0
    
    # Calculate distance and create basic navigation from graph edges
    for i in range(len(optimized_route) - 1):
        curr_coord = optimized_route[i]
        next_coord = optimized_route[i + 1]
        
        # Find the graph edge
        curr_node = None
        next_node = None
        for node, data in graph.nodes(data=True):
            if data.get('coordinates') == curr_coord:
                curr_node = node
            if data.get('coordinates') == next_coord:
                next_node = node
        
        if curr_node and next_node and graph.has_edge(curr_node, next_node):
            edge_data = graph[curr_node][next_node]
            distance = edge_data.get('distance', 0)
            street_name = edge_data.get('way_id', 'unnamed road')
            total_distance_m += distance
            
            # Create simple instruction
            turn_by_turn_instructions.append({
                'distance': distance,
                'duration': distance / 8.33,  # ~30 km/h = 8.33 m/s
                'instruction': f'Continue on {street_name}',
                'name': street_name
            })
    
    osrm_route_info = {
        'total_distance_km': round(total_distance_m / 1000, 2),
        'total_duration_min': round((total_distance_m / 1000) / 30 * 60, 2)
    }
    
    osrm_geometry = pruned_route  # Use waypoints directly for GPX track
    
    # Convert km to miles
    osrm_route_info['total_distance_miles'] = round(osrm_route_info['total_distance_km'] * 0.621371, 2)
    
    print(f"[DEBUG] Map display: {len(route_geometry)} waypoint markers")
    print(f"[DEBUG] Navigation: {len(turn_by_turn_instructions) if turn_by_turn_instructions else 0} instructions")
    print(f"[DEBUG] Route distance: {osrm_route_info['total_distance_miles']} miles ({osrm_route_info['total_distance_km']} km)")
    
    # Store route data in file for GPX export (avoid session size limits)
    route_data = {
        'waypoints': pruned_route,
        'geometry': osrm_geometry if osrm_geometry else route_geometry,  # Detailed OSRM geometry for GPX navigation
        'instructions': turn_by_turn_instructions,
        'distance_km': osrm_route_info['total_distance_km'],
        'distance_miles': osrm_route_info['total_distance_miles'],
        'duration_min': osrm_route_info['total_duration_min']
    }
    route_file_path = os.path.join("temp", f"{boundary_id}_route.pkl")
    with open(route_file_path, "wb") as f:
        pickle.dump(route_data, f)
    
    google_maps_urls = get_google_maps_url(pruned_route)
    return render_template("result.html", 
                         google_maps_urls=google_maps_urls,
                         boundary_id=boundary_id,
                         turn_by_turn=turn_by_turn_instructions,
                         route_info=osrm_route_info,
                         route_geometry=route_geometry,
                         pruned_route=pruned_route)

@app.route("/delete_nodes", methods=["POST"])
def delete_nodes():
    data = request.get_json()
    boundary_id = data.get("boundary_id")
    nodes_to_delete = data.get("nodes", [])

    if not boundary_id or not nodes_to_delete:
        return jsonify({"success": False, "error": "Missing boundary_id or nodes"}), 400

    graph_file_path = os.path.join("temp", f"{boundary_id}_graph.pkl")
    if not os.path.exists(graph_file_path):
        return jsonify({"success": False, "error": "No graph data found"}), 404

    with open(graph_file_path, "rb") as graph_file:
        G = pickle.load(graph_file)

    for node in nodes_to_delete:
        if node not in G:
            continue
        neighbors = list(G.neighbors(node))
        if len(neighbors) == 2:
            # Add edge between the two neighbors if it doesn't already exist
            if not G.has_edge(neighbors[0], neighbors[1]):
                # Optionally, sum the weights/distances of the two edges being replaced
                weight1 = G.edges[node, neighbors[0]].get('weight', 1)
                weight2 = G.edges[node, neighbors[1]].get('weight', 1)
                total_weight = weight1 + weight2
                G.add_edge(neighbors[0], neighbors[1], weight=total_weight)
        G.remove_node(node)

    if G.number_of_nodes() > 0 and not nx.is_connected(G):
        components = list(nx.connected_components(G))
        if components:
            largest_component = max(components, key=len)
            nodes_to_remove = set(G.nodes()) - set(largest_component)
            G.remove_nodes_from(nodes_to_remove)

    # Save updated graph
    with open(graph_file_path, "wb") as graph_file:
        pickle.dump(G, graph_file)

    return jsonify({"success": True})

@app.route("/graph-data")
def graph_data():
    boundary_id = request.args.get("boundary_id")
    if not boundary_id:
        return jsonify({"error": "No boundary ID provided"}), 400
    # Validate boundary_id format (expect a UUID) to limit allowed characters
    try:
        uuid.UUID(boundary_id)
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid boundary ID"}), 400

    filename = f"{boundary_id}_graph.pkl"
    graph_file_path = os.path.normpath(os.path.join(GRAPH_DIR, filename))
    # Ensure the resolved path is within the expected directory
    if not graph_file_path.startswith(os.path.abspath(GRAPH_DIR) + os.sep):
        return jsonify({"error": "Invalid graph path"}), 400

    if not os.path.exists(graph_file_path):
        return jsonify({"error": "No graph data found"}), 404

    with open(graph_file_path, "rb") as graph_file:
        G = pickle.load(graph_file)

    nodes = [{"id": node, "lat": data['coordinates'][0], "lon": data['coordinates'][1]} for node, data in G.nodes(data=True)]
    links = [{"source": u, "target": v} for u, v in G.edges()]

    return jsonify({"nodes": nodes, "links": links})

@app.route("/visualize_cpp")
def visualize_cpp():
    boundary_id = request.args.get("boundary_id")
    start_node = int(request.args.get("start_node"))
    if not boundary_id:
        return jsonify({"error": "No boundary ID provided"}), 400
    # Validate boundary_id format (expect a UUID) to limit allowed characters
    try:
        uuid.UUID(boundary_id)
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid boundary ID"}), 400

    filename = f"{boundary_id}_graph.pkl"
    graph_file_path = os.path.normpath(os.path.join(GRAPH_DIR, filename))
    # Ensure the resolved path is within the expected directory
    if not graph_file_path.startswith(os.path.abspath(GRAPH_DIR) + os.sep):
        return jsonify({"error": "Invalid graph path"}), 400

    if not os.path.exists(graph_file_path):
        return jsonify({"error": "Graph not found"}), 404
    with open(graph_file_path, "rb") as graph_file:
        import pickle
        graph = pickle.load(graph_file)
    route = find_route_cpp(graph, start_node)
    return jsonify({"route": route})

@app.route("/export_gpx")
def export_gpx():
    """Export the current route as a GPX file for navigation apps"""
    from flask import Response
    from datetime import datetime
    
    boundary_id = request.args.get("boundary_id")
    if not boundary_id:
        return "Missing boundary_id parameter", 400
    
    # Load route data from file instead of session
    route_file_path = os.path.join("temp", f"{boundary_id}_route.pkl")
    if not os.path.exists(route_file_path):
        return "No route data available", 404
    
    with open(route_file_path, "rb") as f:
        route_data = pickle.load(f)
    
    # Generate GPX XML
    distance_miles = route_data.get('distance_miles', 0)
    duration_min = route_data.get('duration_min', 0)
    
    gpx_content = '<?xml version="1.0" encoding="UTF-8"?>\n'
    gpx_content += '<gpx version="1.1" creator="FoodTruckRouteOptimizer" xmlns="http://www.topografix.com/GPX/1/1">\n'
    gpx_content += f'  <metadata>\n'
    gpx_content += f'    <name>Food Truck Coverage Route</name>\n'
    gpx_content += f'    <desc>Route: {distance_miles} miles, {duration_min:.0f} min. Optimized for maximum street coverage.</desc>\n'
    gpx_content += f'    <time>{datetime.utcnow().isoformat()}Z</time>\n'
    gpx_content += f'  </metadata>\n'
    
    # Add waypoints
    waypoints = route_data.get('waypoints', [])
    for i, (lat, lon) in enumerate(waypoints):
        name = 'Start' if i == 0 else ('End' if i == len(waypoints) - 1 else f'Waypoint {i}')
        gpx_content += f'  <wpt lat="{lat}" lon="{lon}">\n'
        gpx_content += f'    <name>{name}</name>\n'
        gpx_content += f'  </wpt>\n'
    
    # Add route track (full geometry from OSRM)
    geometry = route_data.get('geometry')
    if geometry:
        gpx_content += '  <trk>\n'
        gpx_content += '    <name>Food Truck Route</name>\n'
        gpx_content += '    <trkseg>\n'
        for lat, lon in geometry:
            gpx_content += f'      <trkpt lat="{lat}" lon="{lon}"></trkpt>\n'
        gpx_content += '    </trkseg>\n'
        gpx_content += '  </trk>\n'
    
    # Add route with turn-by-turn instructions as route points
    instructions = route_data.get('instructions', [])
    if instructions and waypoints:
        gpx_content += '  <rte>\n'
        gpx_content += '    <name>Turn-by-Turn Route</name>\n'
        for i, (lat, lon) in enumerate(waypoints):
            gpx_content += f'    <rtept lat="{lat}" lon="{lon}">\n'
            if i < len(instructions):
                inst = instructions[i]
                gpx_content += f'      <name>{inst.get("instruction", "Continue")}</name>\n'
                if inst.get('name'):
                    gpx_content += f'      <desc>onto {inst["name"]}</desc>\n'
            gpx_content += f'    </rtept>\n'
        gpx_content += '  </rte>\n'
    
    gpx_content += '</gpx>'
    
    # Return as downloadable file
    return Response(
        gpx_content,
        mimetype='application/gpx+xml',
        headers={
            'Content-Disposition': 'attachment; filename="food_truck_route.gpx"',
            'Content-Type': 'application/gpx+xml; charset=utf-8'
        }
    )

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5001)
