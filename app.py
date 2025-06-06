import logging
import uuid
from flask import Flask, jsonify, render_template, request, session
import API_KEY
import os
import pickle
import networkx as nx

from build_urls import get_google_maps_url
from find_route import clean_up_graph, find_route, simplify_graph
from get_street_data import extract_nodes_and_ways, fetch_overpass_data, get_coordinates
from visualization import animate_cpp_route

app = Flask(__name__)
app.secret_key = 'z3ByRjbb-tN3VM4X71W2oITQupA='  # Replace with a secure secret key

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

clat = 0
clng = 0

# Test data for testing purposes
test_street_data = {
    "Street 1": [(41.8781, -87.6298), (41.8795, -87.6288), (41.8798, -87.6292)],
    "Street 2": [(41.8785, -87.6305), (41.8798, -87.6292)],
    "Street 3": [(41.8770, -87.6270), (41.8781, -87.6298)],
}

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

@app.route("/graph_display")
def graph_display():
    boundary_id = request.args.get("boundary_id")
    if not boundary_id:
        return jsonify({"error": "No boundary ID provided"}), 400
    return render_template("graph_display.html", boundary_id=boundary_id)

streets  = {
        "Lillian Court": [(41.6973526, -88.1879847),(41.6954405, -88.1877877)],
        "Janet Court": [(41.7004725, -88.1904994),(41.700809, -88.1895047)],
        "Andermann Drive": [(41.6979341, -88.1948626),(41.6977249, -88.191288)],
        "Webster Lane": [(41.7020666, -88.192908),(41.7007769, -88.1927203),(41.6995153, -88.191803),(41.6982175, -88.1907354)],
        "Schillinger Drive": [(41.6953958, -88.191604),(41.6967075, -88.1916367),(41.6977249, -88.191288),(41.6982175, -88.1907354),(41.6987062, -88.1892388),(41.6989425, -88.1876563),(41.6989922, -88.1861133)],
        "Walter Lane": [(41.7020185, -88.1911324),(41.7004725, -88.1904994),(41.6987062, -88.1892388)],
        "Whittington Lane": [(41.7020866, -88.1879406),(41.6989425, -88.1876563)],
        "Mark Drive": [(41.6966454, -88.1948191),(41.6967075, -88.1916367)],
        "McGrath Lane": [(41.7020596, -88.195023),(41.7006227, -88.194968),(41.6992083, -88.194917),(41.698538, -88.1948929),(41.6979341, -88.1948626),(41.6966454, -88.1948191),(41.6953537, -88.1947751)],
        "Staffelot Drive": [(41.7006227, -88.194968),(41.7007769, -88.1927203)],
        "Partlow Drive": [(41.6985285, -88.1957236),(41.698538, -88.1948929)],
        "Hartman Drive": [(41.6992083, -88.194917),(41.6995153, -88.191803)],
        "Wagner Road": [(41.7020596, -88.1950230),(41.7020666, -88.1929080),(41.7020185, -88.1911324),(41.7020866, -88.1879406),(41.7021875, -88.1862243)],
        "Richards Drive": [(41.7011011, -88.1861828),(41.7011349, -88.1844148)],
        "103rd Street": [(41.6953537, -88.1947751),(41.6953958, -88.1916040),(41.6954057, -88.1910493),(41.6954355, -88.1884590),(41.6954405, -88.1877877),(41.6954495, -88.1869937),(41.6954586, -88.1859774)],
        "Book Road": [(41.6954586, -88.1859774),(41.6989922, -88.1861133),(41.7011011, -88.1861828),(41.7021875, -88.1862243)]
    }

@app.route("/result")
def result():
    # Retrieve the reference (boundary_id) from the session
    boundary_id = session.get("boundary_id")
    if not boundary_id:
        logger.error("No boundary ID found in session")
        return "No boundary ID found", 500

    # Retrieve the graph from the temporary file
    graph_file_path = os.path.join("temp", f"{boundary_id}_graph.pkl")
    if not os.path.exists(graph_file_path):
        logger.error("No graph data found in temporary file")
        return "No graph data found", 500
    with open(graph_file_path, "rb") as graph_file:
        graph = pickle.load(graph_file)

    # Save the updated graph to a file
    with open(graph_file_path, "wb") as graph_file:
        pickle.dump(graph, graph_file)

    # Find the optimized route using the updated graph
    print("Optimizing route...")
    optimized_route = find_route(graph)

    # Filter to only keep turns
    from find_route import filter_turns
    filtered_route = filter_turns(optimized_route, threshold_degrees=40)

    google_maps_urls = get_google_maps_url(filtered_route)
    
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

    base_path = os.path.abspath("temp")
    graph_file_path = os.path.abspath(os.path.join(base_path, f"{boundary_id}_graph.pkl"))

    # Ensure the file path is within the intended directory
    if not graph_file_path.startswith(base_path):
        return jsonify({"error": "Invalid boundary ID"}), 400

    if not os.path.exists(graph_file_path):
        return jsonify({"error": "No graph data found"}), 404

    with open(graph_file_path, "rb") as graph_file:
        G = pickle.load(graph_file)

    nodes = [{"id": node, "lat": data['coordinates'][0], "lon": data['coordinates'][1]} for node, data in G.nodes(data=True)]
    links = [{"source": u, "target": v} for u, v in G.edges()]

    return jsonify({"nodes": nodes, "links": links})

if __name__ == "__main__":
    app.run(debug=True)
