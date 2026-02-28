import logging
import uuid
import warnings
from flask import Flask, jsonify, render_template, request, session
import os
import pickle
import networkx as nx

from build_urls import get_google_maps_url
from find_route import clean_up_graph, find_route_cpp, find_route_max_coverage_optimized, simplify_graph, prune_common_sense_nodes
from get_street_data import extract_nodes_and_ways, fetch_overpass_data, get_coordinates
from osrm_client import get_route_with_waypoints

app = Flask(__name__)
# Use environment variable for secret key
# In production, SECRET_KEY must be set. For development/testing, a random key is generated.
secret_key = os.environ.get('SECRET_KEY')
if not secret_key:
    # For development/testing only - will change on restart
    warnings.warn("SECRET_KEY environment variable not set. Using random key for development.")
    secret_key = os.urandom(24)
app.secret_key = secret_key
GRAPH_DIR = os.path.join(os.path.dirname(__file__), "temp")

# Ensure temp directory exists
os.makedirs(GRAPH_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

clat = 0
clng = 0


def validate_boundary_id(boundary_id):
    """Validate that boundary_id is a valid UUID format."""
    try:
        uuid.UUID(boundary_id)
        return True
    except (ValueError, TypeError):
        return False


def get_safe_file_path(boundary_id, filename_template):
    """
    Get a safe file path with validation to prevent path traversal attacks.
    
    Args:
        boundary_id: The UUID boundary identifier
        filename_template: Template for filename (e.g., "{}_graph.pkl")
    
    Returns:
        Absolute path to the file
        
    Raises:
        ValueError: If the path is invalid or outside the expected directory
    """
    filename = filename_template.format(boundary_id)
    file_path = os.path.normpath(os.path.join(GRAPH_DIR, filename))
    
    # Ensure the resolved path is within the expected directory
    try:
        if os.path.commonpath([os.path.abspath(GRAPH_DIR), os.path.abspath(file_path)]) != os.path.abspath(GRAPH_DIR):
            raise ValueError("Invalid file path")
    except ValueError:
        # Raised when paths are on different drives on Windows or invalid path
        raise ValueError("Invalid file path")
    
    return file_path


def load_deleted_nodes(boundary_id):
    """Load previously deleted node IDs for a boundary."""
    try:
        deleted_nodes_path = get_safe_file_path(boundary_id, "{}_deleted_nodes.pkl")
    except ValueError:
        return set()

    if not os.path.exists(deleted_nodes_path):
        return set()

    with open(deleted_nodes_path, "rb") as f:
        deleted_nodes = pickle.load(f)

    return set(deleted_nodes)


def save_deleted_nodes(boundary_id, deleted_nodes):
    """Persist deleted node IDs for a boundary."""
    deleted_nodes_path = get_safe_file_path(boundary_id, "{}_deleted_nodes.pkl")
    with open(deleted_nodes_path, "wb") as f:
        pickle.dump(sorted(set(deleted_nodes)), f)


def apply_deleted_nodes_to_graph(graph, nodes_to_delete):
    """Apply node deletions with stitching logic, then keep the largest connected component."""
    for node in nodes_to_delete:
        if node not in graph:
            continue

        neighbors = list(graph.neighbors(node))
        if len(neighbors) == 2 and not graph.has_edge(neighbors[0], neighbors[1]):
            weight1 = graph.edges[node, neighbors[0]].get('weight', 1)
            weight2 = graph.edges[node, neighbors[1]].get('weight', 1)
            total_weight = weight1 + weight2
            graph.add_edge(neighbors[0], neighbors[1], weight=total_weight)

        graph.remove_node(node)

    if graph.number_of_nodes() > 0 and not nx.is_connected(graph):
        components = list(nx.connected_components(graph))
        if components:
            largest_component = max(components, key=len)
            nodes_to_remove = set(graph.nodes()) - set(largest_component)
            graph.remove_nodes_from(nodes_to_remove)

    return graph


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
            logger.debug("POST request received")
            zipcode = request.form.get("zipcode")
            logger.debug(f"Geocoding ZIP code: {zipcode}")
            clat, clng = get_coordinates(zipcode)
            
            if clat and clng:
                logger.info(f"Successfully geocoded {zipcode} to ({clat}, {clng})")
                return render_template("select_boundaries.html", lat=clat, lng=clng)
            else:
                logger.error("Failed to get coordinates for ZIP code")
                return "Failed to geocode ZIP code. Please try again or enter a different location.", 500
        except Exception as e:
            logger.error(f"Error processing ZIP code: {e}", exc_info=True)
            return f"Internal server error: {str(e)}", 500
    return render_template("index.html")


@app.route("/select_boundaries")
def select_boundaries():
    return render_template("select_boundaries.html")


@app.route("/geocode", methods=["POST"])
def geocode_location():
    data = request.get_json()
    query = (data.get("query") or "").strip()

    if not query:
        return jsonify({"error": "Please enter a city or ZIP code."}), 400

    try:
        lat, lng = get_coordinates(query)
    except Exception as e:
        logger.error(f"Geocoding failed for '{query}': {e}")
        return jsonify({"error": "Geocoding failed. Please try again."}), 500

    if lat is None or lng is None:
        return jsonify({"error": "Location not found. Try a different city or ZIP code."}), 404

    return jsonify({"lat": lat, "lng": lng})

@app.route("/process_boundaries", methods=["POST"])
def process_boundaries():
    data = request.get_json()
    corners = data.get("corners")

    if len(corners) != 4:
        logger.error("Invalid number of corners, expected 4")
        return jsonify({"error": "Please select exactly 4 corners."}), 400

    logger.debug(f"Processing boundaries with corners: {corners}")

    latitudes = [corner['lat'] for corner in corners]
    longitudes = [corner['lng'] for corner in corners]

    min_lat = min(latitudes)
    max_lat = max(latitudes)
    min_lng = min(longitudes)
    max_lng = max(longitudes)

    center_lat = (min_lat + max_lat) / 2
    center_lng = (min_lng + max_lng) / 2

    try:
        street_data = fetch_overpass_data(min_lat, max_lat, min_lng, max_lng)
    except Exception as e:
        logger.error(f"Failed to fetch street data: {e}")
        return jsonify({
            "error": "Failed to fetch street data from Overpass API. The service may be rate limiting requests. Please try again in a few minutes."
        }), 503
    
    if not street_data:
        logger.error("No street or intersection data found")
        return jsonify({"error": "No data found for the selected area"}), 500
    
    logger.debug(f"Street and intersection data: {street_data}")

    boundary_id = str(uuid.uuid4())

    nodes, ways = extract_nodes_and_ways(street_data)

    graph = simplify_graph(nodes, ways)
    graph = clean_up_graph(graph)

    # Save both the graph and the raw nodes/ways for rebuilding with different settings
    try:
        graph_file_path = get_safe_file_path(boundary_id, "{}_graph.pkl")
        nodes_ways_file_path = get_safe_file_path(boundary_id, "{}_nodes_ways.pkl")
    except ValueError:
        return jsonify({"error": "Invalid graph path"}), 400
    
    os.makedirs(GRAPH_DIR, exist_ok=True)
    with open(graph_file_path, "wb") as f:
        pickle.dump(graph, f)
    
    # Save original nodes and ways for later graph rebuilding with different settings
    with open(nodes_ways_file_path, "wb") as f:
        pickle.dump({"nodes": nodes, "ways": ways}, f)

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
    if not validate_boundary_id(boundary_id):
        return "Invalid boundary_id parameter", 400
    
    start_node = int(request.args.get("start_node"))
    end_node = request.args.get("end_node")
    if end_node:
        end_node = int(end_node)
    
    # Get route settings
    coverage_mode = request.args.get("coverage_mode", "balanced")
    min_street_length = int(request.args.get("min_street_length", 70))
    
    try:
        nodes_ways_file_path = get_safe_file_path(boundary_id, "{}_nodes_ways.pkl")
    except ValueError:
        return "Invalid graph path", 400
    
    if not os.path.exists(nodes_ways_file_path):
        return "Graph data not found", 404
    with open(nodes_ways_file_path, "rb") as data_file:
        import pickle
        data = pickle.load(data_file)
        nodes = data["nodes"]
        ways = data["ways"]

    # Build baseline graph BEFORE route-specific street-length filtering.
    # This is the denominator for coverage calculations.
    baseline_settings = {
        'coverage_mode': coverage_mode,
        'min_street_length': 0,
    }

    graph = simplify_graph(nodes, ways, settings=baseline_settings)
    deleted_nodes = load_deleted_nodes(boundary_id)
    if deleted_nodes:
        graph = apply_deleted_nodes_to_graph(graph, deleted_nodes)
    graph = clean_up_graph(graph)

    if start_node not in graph:
        return "Selected start node is not available with current route settings. Please reselect a start node.", 400
    if end_node and end_node not in graph:
        return "Selected end node is not available with current route settings. Please reselect an end node.", 400
    
    baseline_total_edge_length_m = sum(
        edge_data.get('distance', 0) for _, _, edge_data in graph.edges(data=True)
    )

    logger.info(
        f"[RESULT] Generating 3 route variants with coverage_mode={coverage_mode}, "
        f"baseline_min_street_length=0m, baseline_edge_length_m={baseline_total_edge_length_m:.1f}"
    )
    
    # Generate 3 different route variants with distinct characteristics
    route_variants = []
    speed_priorities = [
        ('fastest', 'Quick Route', 70),
        ('balanced', 'Balanced Route', 100),
        ('thorough', 'Thorough Route', 0)
    ]
    
    for priority, name, min_length_for_route in speed_priorities:
        # Build graph with route-specific min_street_length
        route_graph = simplify_graph(nodes, ways, settings={
            'coverage_mode': coverage_mode,
            'min_street_length': min_length_for_route,
        })
        if deleted_nodes:
            route_graph = apply_deleted_nodes_to_graph(route_graph, deleted_nodes)
        route_graph = clean_up_graph(route_graph)
        
        if start_node not in route_graph:
            logger.warning(f"[RESULT] Start node not in {name} graph, skipping")
            continue
        if end_node and end_node not in route_graph:
            logger.warning(f"[RESULT] End node not in {name} graph, skipping")
            continue
        
        settings_for_route = {
            'coverage_mode': coverage_mode,
            'min_street_length': min_length_for_route,
            'speed_priority': priority
        }
        
        logger.info(f"[RESULT] Generating {name} with speed_priority={priority}, min_street_length={min_length_for_route}m")
        optimized_route = find_route_max_coverage_optimized(route_graph, start_node, end_node, settings=settings_for_route)
        
        def is_latlon_tuple(x):
            return isinstance(x, (list, tuple)) and len(x) == 2 and all(isinstance(i, (float, int)) for i in x)
        
        if not optimized_route or not is_latlon_tuple(optimized_route[0]):
            logger.warning(f"[RESULT] Failed to generate {name}")
            continue
        
        # Prune route
        way_ids = [None] * len(optimized_route)
        pruned_route = prune_common_sense_nodes(optimized_route, way_ids=way_ids, angle_threshold=30, graph=route_graph)
        
        if len(pruned_route) < 2:
            logger.warning(f"[RESULT] {name} too short after pruning")
            continue
        
        # Calculate route statistics and track covered edge length
        total_distance_m = 0
        turn_by_turn_instructions = []
        covered_edges = set()
        covered_edge_length_m = 0
        
        for i in range(len(optimized_route) - 1):
            curr_coord = optimized_route[i]
            next_coord = optimized_route[i + 1]
            
            # Find the graph edge
            curr_node = None
            next_node = None
            for node, data in route_graph.nodes(data=True):
                if data.get('coordinates') == curr_coord:
                    curr_node = node
                if data.get('coordinates') == next_coord:
                    next_node = node
            
            if curr_node and next_node and route_graph.has_edge(curr_node, next_node):
                edge_data = route_graph[curr_node][next_node]
                distance = edge_data.get('distance', 0)
                street_name = edge_data.get('way_id', 'unnamed road')
                total_distance_m += distance
                
                # Track unique covered edge length (undirected edge)
                edge_key = tuple(sorted([curr_node, next_node]))
                if edge_key not in covered_edges:
                    covered_edges.add(edge_key)
                    covered_edge_length_m += distance
                
                turn_by_turn_instructions.append({
                    'distance': distance,
                    'duration': distance / 8.33,  # ~30 km/h = 8.33 m/s
                    'instruction': f'Continue on {street_name}',
                    'name': street_name
                })
        
        # Calculate coverage against baseline graph total edge length
        coverage_percent = (
            round((covered_edge_length_m / baseline_total_edge_length_m) * 100, 1)
            if baseline_total_edge_length_m > 0
            else 0
        )
        
        # Calculate estimated duration in minutes (assuming 30 km/h average speed)
        total_duration_min = (total_distance_m / 1000) / 30 * 60
        
        # Calculate coverage per minute metric (in miles/min)
        coverage_per_minute = (
            round((covered_edge_length_m / 1000 * 0.621371) / total_duration_min, 2)
            if total_duration_min > 0
            else 0
        )
        
        # Build description based on route priority
        if priority == 'fastest':
            description = f"{coverage_per_minute} miles/min coverage"
        else:
            description = f"{coverage_percent}% street coverage"
        
        if min_length_for_route >= 100:
            description += f" (roads ≥{min_length_for_route}m)"
        elif min_length_for_route <= 0:
            description += " (all road lengths)"
        
        route_info = {
            'total_distance_km': round(total_distance_m / 1000, 2),
            'total_distance_miles': round(total_distance_m / 1000 * 0.621371, 2),
            'total_duration_min': round(total_duration_min, 2),
            'coverage_percent': coverage_percent,
            'coverage_per_minute': coverage_per_minute
        }
        
        route_variants.append({
            'priority': priority,
            'name': name,
            'description': description,
            'waypoints': pruned_route,
            'geometry': pruned_route,
            'instructions': turn_by_turn_instructions,
            'route_info': route_info
        })
        
        if priority == 'fastest':
            logger.info(
                f"[RESULT] {name}: {route_info['total_distance_miles']} miles, "
                f"{route_info['total_duration_min']:.0f} min, {len(pruned_route)} waypoints, "
                f"{coverage_per_minute:.2f} km/min coverage, {coverage_percent}% street coverage"
            )
        else:
            logger.info(
                f"[RESULT] {name}: {route_info['total_distance_miles']} miles, "
                f"{route_info['total_duration_min']:.0f} min, {len(pruned_route)} waypoints, "
                f"{coverage_percent}% coverage ({covered_edge_length_m:.1f}m / {baseline_total_edge_length_m:.1f}m)"
            )
    
    if not route_variants:
        return "No valid routes found", 400
    
    # Store all route variants for later GPX export
    try:
        routes_file_path = get_safe_file_path(boundary_id, "{}_routes.pkl")
    except ValueError:
        return "Invalid route path", 400
    
    os.makedirs(GRAPH_DIR, exist_ok=True)
    with open(routes_file_path, "wb") as f:
        pickle.dump(route_variants, f)
    
    return render_template("result.html", 
                         boundary_id=boundary_id,
                         route_variants=route_variants,
                         start_node=start_node,
                         end_node=end_node)

@app.route("/delete_nodes", methods=["POST"])
def delete_nodes():
    data = request.get_json()
    boundary_id = data.get("boundary_id")
    nodes_to_delete = data.get("nodes", [])

    if not boundary_id or not nodes_to_delete:
        return jsonify({"success": False, "error": "Missing boundary_id or nodes"}), 400

    if not validate_boundary_id(boundary_id):
        return jsonify({"success": False, "error": "Invalid boundary_id"}), 400

    try:
        graph_file_path = get_safe_file_path(boundary_id, "{}_graph.pkl")
    except ValueError:
        return jsonify({"success": False, "error": "Invalid graph path"}), 400

    if not os.path.exists(graph_file_path):
        return jsonify({"success": False, "error": "No graph data found"}), 404

    with open(graph_file_path, "rb") as graph_file:
        G = pickle.load(graph_file)

    G = apply_deleted_nodes_to_graph(G, nodes_to_delete)

    deleted_nodes = load_deleted_nodes(boundary_id)
    deleted_nodes.update(nodes_to_delete)
    save_deleted_nodes(boundary_id, deleted_nodes)

    # Save updated graph
    with open(graph_file_path, "wb") as graph_file:
        pickle.dump(G, graph_file)

    return jsonify({"success": True})

@app.route("/graph-data")
def graph_data():
    boundary_id = request.args.get("boundary_id")
    if not boundary_id:
        return jsonify({"error": "No boundary ID provided"}), 400
    if not validate_boundary_id(boundary_id):
        return jsonify({"error": "Invalid boundary ID"}), 400

    try:
        graph_file_path = get_safe_file_path(boundary_id, "{}_graph.pkl")
    except ValueError:
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
    end_node = request.args.get("end_node")
    if end_node:
        end_node = int(end_node)
    
    # Get route settings
    coverage_mode = request.args.get("coverage_mode", "balanced")
    min_street_length = int(request.args.get("min_street_length", 70))
    speed_priority = request.args.get("speed_priority", "balanced")
    
    if not boundary_id:
        return jsonify({"error": "No boundary ID provided"}), 400
    if not validate_boundary_id(boundary_id):
        return jsonify({"error": "Invalid boundary ID"}), 400

    try:
        nodes_ways_file_path = get_safe_file_path(boundary_id, "{}_nodes_ways.pkl")
    except ValueError:
        return jsonify({"error": "Invalid file path"}), 400

    if not os.path.exists(nodes_ways_file_path):
        return jsonify({"error": "Nodes/ways data not found"}), 404
    
    # Load the original nodes and ways
    with open(nodes_ways_file_path, "rb") as f:
        import pickle
        data = pickle.load(f)
        nodes = data["nodes"]
        ways = data["ways"]
    
    # Rebuild graph with user-provided settings
    settings = {
        'coverage_mode': coverage_mode,
        'min_street_length': min_street_length,
        'speed_priority': speed_priority
    }
    logger.info(f"[VISUALIZE_CPP] Received settings from request: coverage_mode={coverage_mode}, min_street_length={min_street_length}m, speed_priority={speed_priority}")
    logger.info(f"[VISUALIZE_CPP] Original data: {len(nodes)} nodes, {len(ways)} ways")
    
    graph = simplify_graph(nodes, ways, settings=settings)
    deleted_nodes = load_deleted_nodes(boundary_id)
    if deleted_nodes:
        graph = apply_deleted_nodes_to_graph(graph, deleted_nodes)
    logger.info(f"[VISUALIZE_CPP] Graph after simplify_graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    graph = clean_up_graph(graph)
    logger.info(f"[VISUALIZE_CPP] Graph after clean_up_graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    if start_node not in graph:
        return jsonify({"error": "Selected start node is not available with current route settings. Please reselect a start node."}), 400
    if end_node and end_node not in graph:
        return jsonify({"error": "Selected end node is not available with current route settings. Please reselect an end node."}), 400
    
    # Use the max coverage optimized algorithm with end_node support and settings
    route = find_route_max_coverage_optimized(graph, start_node, end_node, settings=settings)
    logger.info(f"[VISUALIZE_CPP] Generated route with {len(route)} waypoints")
    
    # Prevent browser caching of dynamic route results
    from flask import make_response
    response = make_response(jsonify({"route": route}))
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route("/export_gpx")
def export_gpx():
    """Export the selected route variant as a GPX file for navigation apps"""
    from flask import Response
    from datetime import datetime
    
    boundary_id = request.args.get("boundary_id")
    route_option = request.args.get("route_option", "balanced")  # fastest, balanced, or thorough
    
    if not boundary_id:
        return "Missing boundary_id parameter", 400

    if not validate_boundary_id(boundary_id):
        return "Invalid boundary_id parameter", 400
    
    # Load route variants from file
    try:
        routes_file_path = get_safe_file_path(boundary_id, "{}_routes.pkl")
    except ValueError:
        return "Invalid route path", 400

    if not os.path.exists(routes_file_path):
        return "No route data available", 404
    
    with open(routes_file_path, "rb") as f:
        route_variants = pickle.load(f)
    
    # Find the selected route variant
    route_data = None
    for variant in route_variants:
        if variant['priority'] == route_option:
            route_data = variant
            break
    
    if not route_data:
        return f"Route option '{route_option}' not found", 404
    
    # Generate GPX XML
    route_info = route_data['route_info']
    distance_miles = route_info.get('total_distance_miles', 0)
    duration_min = route_info.get('total_duration_min', 0)
    route_name = route_data['name']
    
    gpx_content = '<?xml version="1.0" encoding="UTF-8"?>\n'
    gpx_content += '<gpx version="1.1" creator="FoodTruckRouteOptimizer" xmlns="http://www.topografix.com/GPX/1/1">\n'
    gpx_content += f'  <metadata>\n'
    gpx_content += f'    <name>{route_name}</name>\n'
    gpx_content += f'    <desc>Route: {distance_miles} miles, {duration_min:.0f} min. {route_data["description"]}</desc>\n'
    gpx_content += f'    <time>{datetime.utcnow().isoformat()}Z</time>\n'
    gpx_content += f'  </metadata>\n'
    
    # Add waypoints
    waypoints = route_data.get('waypoints', [])
    for i, (lat, lon) in enumerate(waypoints):
        name = 'Start' if i == 0 else ('End' if i == len(waypoints) - 1 else f'Waypoint {i}')
        gpx_content += f'  <wpt lat="{lat}" lon="{lon}">\n'
        gpx_content += f'    <name>{name}</name>\n'
        gpx_content += f'  </wpt>\n'
    
    # Add route track (full geometry)
    geometry = route_data.get('geometry')
    if geometry:
        gpx_content += '  <trk>\n'
        gpx_content += f'    <name>{route_name}</name>\n'
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
    
    # Return as downloadable file with route-specific filename
    filename = f"food_truck_route_{route_option}.gpx"
    return Response(
        gpx_content,
        mimetype='application/gpx+xml',
        headers={
            'Content-Disposition': f'attachment; filename="{filename}"',
            'Content-Type': 'application/gpx+xml; charset=utf-8'
        }
    )

@app.route("/health")
def health():
    """Health check endpoint for monitoring"""
    import sys
    health_status = {
        "status": "healthy",
        "python_version": sys.version,
        "flask_running": True,
        "temp_dir_exists": os.path.exists(GRAPH_DIR),
        "temp_dir_writable": os.access(GRAPH_DIR, os.W_OK) if os.path.exists(GRAPH_DIR) else False
    }
    return jsonify(health_status)

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5001)
