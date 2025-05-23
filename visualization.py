import os
import requests
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.animation as animation
from utils import cleanup_temp_files  # Import the function from utils.py

selected_start_node = None
selected_nodes = set()

def get_static_map_image(center_lat, center_lng, corners, api_key, zoom=15, size="600x400"):
    """
    Get a static map image of the selected area using Google Static Maps API.
    
    Parameters:
      center_lat: Latitude of the center of the map.
      center_lng: Longitude of the center of the map.
      corners: List of corner coordinates [(lat, lng), ...].
      api_key: Google Maps API key.
      zoom: Zoom level of the map.
      size: Size of the map image (width x height).
    
    Returns:
      The path to the downloaded image file.
    """
    base_url = "https://maps.googleapis.com/maps/api/staticmap"
    markers = "|".join([f"{lat},{lng}" for lat, lng in corners])
    params = {
        "center": f"{center_lat},{center_lng}",
        "zoom": zoom,
        "size": size,
        "maptype": "roadmap",
        "markers": markers,
        "key": api_key
    }
    
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        image_path = os.path.join("temp", "static_map.png")
        os.makedirs("temp", exist_ok=True)
        with open(image_path, "wb") as image_file:
            image_file.write(response.content)
        
        # Cleanup old temp files
        cleanup_temp_files("temp")
        
        return image_path
    else:
        logger.error(f"Failed to get static map image: {response.status_code}")
        return None

def animate_cpp_route(G, route):
    """
    Animate the nearest neighbor route calculation.
    
    Parameters:
        G: The networkx graph containing the nodes and edges with distances.
        route: The optimized route (list of lat, lon tuples in order of visit).
    """
    # Create a figure and axis for plotting
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Get node positions (assuming 'coordinates' attribute is (lat, lon))
    pos = {node: G.nodes[node]['coordinates'] for node in G.nodes}
    
    # Prepare the plot with nodes and edges
    edges = G.edges()
    nx.draw(G, pos, ax=ax, with_labels=True, node_size=500, node_color='skyblue', font_size=10)
    
    # Function to update the plot during animation
    def update(frame):
        ax.clear()  # Clear the previous frame
        nx.draw(G, pos, ax=ax, with_labels=True, node_size=500, node_color='skyblue', font_size=10)
        
        # Highlight the path up to the current frame
        current_path = route[:frame + 1]
        current_edges = [(current_path[i], current_path[i + 1]) for i in range(len(current_path) - 1)]
        
        # Draw edges in the current path in red
        nx.draw_networkx_edges(G, pos, edgelist=current_edges, ax=ax, edge_color='r', width=2)
        
        # Highlight the node being visited at the current frame
        current_node = route[frame]
        nx.draw_networkx_nodes(G, pos, nodelist=[current_node], ax=ax, node_color='orange', node_size=700)
        
        # Set the title showing the current step
        ax.set_title(f"Step {frame + 1} / {len(route)}")
    
    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=len(route), repeat=False, interval=1000)
    
    plt.show()
