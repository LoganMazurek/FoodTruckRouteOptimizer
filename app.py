import logging
import uuid
from flask import Flask, jsonify, render_template, request, session
import API_KEY
import os
import pickle
import networkx as nx

from build_urls import get_google_maps_url
from find_route import clean_up_graph, find_route_cpp, simplify_graph, prune_common_sense_nodes
from get_street_data import extract_nodes_and_ways, fetch_overpass_data, get_coordinates

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
            return render_template("select_boundaries.html", lat=clat, lng=clng, api_key=API_KEY.API_KEY)
        else:
            logger.error("Failed to get coordinates for ZIP code")
            return "Failed to get coords for ZIP code", 500
    return render_template("index.html")

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

    street_data = fetch_overpass_data(min_lat, max_lat, min_lng, max_lng)
    
    if not street_data:
        logger.error("No street or intersection data found")
        return jsonify({"error": "No data found for the selected area"}), 500
    
    logger.debug(f"Street and intersection data: {street_data}")

    boundary_id = str(uuid.uuid4())

    # Store the reference (boundary_id) in the session
    session['boundary_id'] = boundary_id

    nodes, ways = extract_nodes_and_ways(street_data)

    graph = simplify_graph(nodes, ways)
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
    boundary_id = request.args.get("boundary_id") or session.get("boundary_id")
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
    optimized_route = find_route_cpp(graph, start_node)
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
    if len(pruned_route) < 2:
        return "Route too short after pruning", 400
    google_maps_urls = get_google_maps_url(pruned_route)
    return render_template("result.html", google_maps_urls=google_maps_urls, boundary_id=boundary_id)

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

    if not nx.is_connected(G):
        return jsonify({"success": False, "error": "Deletion would disconnect the graph."}), 400

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

if __name__ == "__main__":
    app.run(debug=True)
