import requests
import API_KEY
import logging
import overpy
import time

# Configure logger
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

api = overpy.Overpass()

def extract_nodes_and_ways(street_data):
    # Initialize empty lists for nodes and ways
    nodes = {}
    ways = []

    # Extract nodes from the street data
    for node in street_data.nodes:
        nodes[node.id] = (float(node.lat), float(node.lon))  # Store nodes with coordinates

    # Extract ways (streets)
    for way in street_data.ways:
        street_name = way.tags.get("name")
        if street_name:
            node_refs = [n.id for n in way.nodes]  # Get node ids for the street
            ways.append({"name": street_name, "nodes": node_refs})

    return nodes, ways

def get_coordinates(zipcode):
    """
    Convert a ZIP code to lat/lng using Google Geocoding API
    """
    geocode_url = f"https://maps.googleapis.com/maps/api/geocode/json?address={zipcode}&key={API_KEY.API_KEY}"
    
    try:
        response = requests.get(geocode_url, timeout=10)  # Setting a 10-second timeout
        response.raise_for_status()  # Raise an exception for 4xx/5xx responses
    except requests.exceptions.Timeout:
        print("Request timed out.")
        return None, None
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None, None

    data = response.json()

    if data["status"] == "OK":
        location = data["results"][0]["geometry"]["location"]
        return location["lat"], location["lng"]
    else:
        print(f"Error: {data['status']}")
        return None, None

def make_request_with_retry(query, retries=3, backoff_factor=1):
    """
    Make the API request with retry logic and exponential backoff.
    :param query: The Overpass API query to be executed.
    :param retries: Number of retry attempts.
    :param backoff_factor: Exponential backoff factor to calculate delay between retries.
    :return: The response from the Overpass API if successful, None if failed.
    """
    attempt = 0
    while attempt < retries:
        try:
            # Attempt to make the request
            result = api.query(query)
            if result:
                return result
            else:
                logger.warning(f"Empty result on attempt {attempt + 1}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed on attempt {attempt + 1}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
        
        # Exponential backoff
        attempt += 1
        delay = backoff_factor * (2 ** attempt)  # Exponential backoff
        logger.info(f"Retrying in {delay} seconds...")
        time.sleep(delay)
    
    logger.error("All retry attempts failed.")
    return None

def fetch_overpass_data(min_lat, max_lat, min_lng, max_lng, debug=False):
    
    query = f"""
    (
      way["highway"~"^(primary|secondary|tertiary|residential)"]({min_lat},{min_lng},{max_lat},{max_lng});
    );
    (._;>;);
    out body;
    """
    result = api.query(query)
    if (debug):
        # Print nodes
        print("Nodes:")
        for node in result.nodes:
            print(f"Node ID: {node.id}, Latitude: {node.lat}, Longitude: {node.lon}")

        # Print ways
        print("\nWays:")
        for way in result.ways:
            print(f"Way ID: {way.id}, Name: {way.tags.get('name', 'No name')}")
            for node in way.nodes:
                print(f"  Node ID: {node.id}, Latitude: {node.lat}, Longitude: {node.lon}")

        # Print specific tags (e.g., street names)
        print("\nStreet Names:")
        for way in result.ways:
            street_name = way.tags.get("name", "Unnamed Street")
            print(f"Street Name: {street_name}")
    return result
