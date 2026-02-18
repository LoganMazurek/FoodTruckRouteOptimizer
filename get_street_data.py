import requests
import os
import logging
import overpy
import time

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Overpass API with proper headers to avoid request denied errors
api = overpy.Overpass()
# Set a custom user agent to identify the application properly
api.headers = {
    'User-Agent': 'FoodTruckRouteOptimizer/1.0 (https://github.com/LoganMazurek/FoodTruckRouteOptimizer)'
}

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
            highway_type = way.tags.get("highway", "unclassified")  # Extract highway type for filtering
            ways.append({"name": street_name, "nodes": node_refs, "highway": highway_type})

    return nodes, ways

def get_coordinates(zipcode):
    """
    Convert a ZIP code to lat/lng using Google Geocoding API
    """
    api_key = os.getenv('GOOGLE_MAPS_API_KEY')
    if not api_key:
        raise ValueError("GOOGLE_MAPS_API_KEY environment variable not set")
    geocode_url = f"https://maps.googleapis.com/maps/api/geocode/json?address={zipcode}&key={api_key}"
    
    try:
        response = requests.get(geocode_url, timeout=10)  # Setting a 10-second timeout
        response.raise_for_status()  # Raise an exception for 4xx/5xx responses
    except requests.exceptions.Timeout:
        logger.error("Request timed out.")
        return None, None
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {e}")
        return None, None

    data = response.json()

    if data["status"] == "OK":
        location = data["results"][0]["geometry"]["location"]
        return location["lat"], location["lng"]
    else:
        error_status = data["status"]
        error_message = data.get("error_message", "No error message provided")
        logger.error(f"Google Maps API Error: {error_status} - {error_message}")
        
        # Provide helpful context for common errors
        if error_status == "REQUEST_DENIED":
            logger.error("REQUEST_DENIED typically means:")
            logger.error("  1. The Geocoding API is not enabled in your Google Cloud project")
            logger.error("  2. The API key is invalid or expired")
            logger.error("  3. API key restrictions are blocking the request")
            logger.error("  4. Billing is not enabled on the Google Cloud project")
            logger.error("Check your API key at: https://console.cloud.google.com/apis/credentials")
        elif error_status == "OVER_QUERY_LIMIT":
            logger.error("You have exceeded your daily quota for the Geocoding API")
        
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
            logger.info(f"Attempting Overpass API request (attempt {attempt + 1}/{retries})")
            result = api.query(query)
            if result:
                logger.info(f"Overpass API request successful on attempt {attempt + 1}")
                return result
            else:
                logger.warning(f"Empty result on attempt {attempt + 1}")
        except overpy.exception.OverpassTooManyRequests as e:
            logger.error(f"Rate limited by Overpass API on attempt {attempt + 1}: {e}")
        except overpy.exception.OverpassGatewayTimeout as e:
            logger.error(f"Overpass API timeout on attempt {attempt + 1}: {e}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed on attempt {attempt + 1}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
        
        # Exponential backoff (don't sleep after the last failed attempt)
        attempt += 1
        if attempt < retries:
            delay = backoff_factor * (2 ** attempt)  # Exponential backoff
            logger.info(f"Retrying in {delay} seconds...")
            time.sleep(delay)
    
    logger.error("All retry attempts failed.")
    return None

def fetch_overpass_data(min_lat, max_lat, min_lng, max_lng, debug=False):
    """
    Fetch street data from Overpass API with retry logic and proper error handling.
    """
    query = f"""
    (
      way["highway"~"^(primary|secondary|tertiary|residential)"]({min_lat},{min_lng},{max_lat},{max_lng});
    );
    (._;>;);
    out body;
    """
    
    # Use retry logic to handle temporary API issues
    result = make_request_with_retry(query, retries=3, backoff_factor=2)
    
    if result is None:
        logger.error(f"Failed to fetch Overpass data for bounds: ({min_lat},{min_lng},{max_lat},{max_lng})")
        raise Exception("Overpass API request failed after multiple retries. The API may be rate limiting requests.")
    
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
