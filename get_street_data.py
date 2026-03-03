import requests
import os
import logging
import overpy
import time
from osm_cache import cache_exists, get_cached_data, save_to_cache

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Overpass API with proper headers to avoid request denied errors
api = overpy.Overpass()
# Set a custom user agent to identify the application properly
api.headers = {
    'User-Agent': 'FoodTruckRouteOptimizer/1.0 (https://github.com/LoganMazurek/FoodTruckRouteOptimizer)'
}

def _node_within_bbox(lat, lon, min_lat, max_lat, min_lng, max_lng):
    return min_lat <= lat <= max_lat and min_lng <= lon <= max_lng


def _split_contiguous_segments(node_refs, valid_node_ids):
    segments = []
    current = []

    for node_id in node_refs:
        if node_id in valid_node_ids:
            current.append(node_id)
        else:
            if len(current) >= 2:
                segments.append(current)
            current = []

    if len(current) >= 2:
        segments.append(current)

    return segments


def extract_nodes_and_ways(street_data, min_lat=None, max_lat=None, min_lng=None, max_lng=None):
    """
    Extract nodes and ways from Overpass results.

    When bounds are provided, nodes are strictly clipped to the bounding box and
    ways are split into contiguous in-bounds segments to prevent out-of-bounds
    endpoints from leaking into the graph.
    """
    nodes = {}
    ways = []

    use_bbox = all(v is not None for v in [min_lat, max_lat, min_lng, max_lng])

    for node in street_data.nodes:
        lat = float(node.lat)
        lon = float(node.lon)
        if use_bbox and not _node_within_bbox(lat, lon, min_lat, max_lat, min_lng, max_lng):
            continue
        nodes[node.id] = (lat, lon)

    valid_node_ids = set(nodes.keys())

    for way in street_data.ways:
        street_name = way.tags.get("name")
        if not street_name:
            continue

        highway_type = way.tags.get("highway", "unclassified")
        node_refs = [n.id for n in way.nodes]

        if use_bbox:
            segments = _split_contiguous_segments(node_refs, valid_node_ids)
            for segment in segments:
                ways.append({"name": street_name, "nodes": segment, "highway": highway_type})
        elif len(node_refs) >= 2:
            ways.append({"name": street_name, "nodes": node_refs, "highway": highway_type})

    return nodes, ways

def get_coordinates(zipcode):
    """
    Convert a ZIP code to lat/lng using Nominatim (OpenStreetMap's free geocoding service)
    """
    # Use Nominatim API (free, no API key required)
    # Important: Nominatim requires a User-Agent header
    geocode_url = f"https://nominatim.openstreetmap.org/search"
    params = {
        'q': zipcode,
        'format': 'json',
        'limit': 1,
        'countrycodes': 'us'  # Adjust if you need other countries
    }
    headers = {
        'User-Agent': 'FoodTruckRouteOptimizer/1.0 (https://github.com/LoganMazurek/FoodTruckRouteOptimizer)'
    }
    
    try:
        response = requests.get(geocode_url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.exceptions.Timeout:
        logger.error("Nominatim request timed out.")
        return None, None
    except requests.exceptions.RequestException as e:
        logger.error(f"Nominatim request failed: {e}")
        return None, None

    data = response.json()

    if data and len(data) > 0:
        location = data[0]
        lat = float(location['lat'])
        lon = float(location['lon'])
        logger.info(f"Geocoded '{zipcode}' to ({lat}, {lon})")
        return lat, lon
    else:
        logger.error(f"Nominatim could not geocode: {zipcode}")
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

def fetch_overpass_data(min_lat, max_lat, min_lng, max_lng, debug=False, use_cache=True):
    """
    Fetch street data from Overpass API with caching and retry logic.
    
    Args:
        min_lat, max_lat, min_lng, max_lng: Bounding box coordinates
        debug: Print detailed information
        use_cache: Check/store cache (default True)
    
    Returns:
        Overpass API result object with nodes and ways
    """
    # Check cache first if enabled
    if use_cache and cache_exists(min_lat, max_lat, min_lng, max_lng):
        logger.info(f"Retrieving cached OSM data for bounds: ({min_lat},{min_lng},{max_lat},{max_lng})")
        nodes, ways = get_cached_data(min_lat, max_lat, min_lng, max_lng)
        if nodes is not None and ways is not None:
            # Convert cached data back to Overpass result format
            return _create_overpass_result_from_cache(nodes, ways)
    
    # Query not in cache, fetch from API
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
    
    # Cache the result for future queries
    if use_cache:
        nodes_dict = {node.id: (float(node.lat), float(node.lon)) for node in result.nodes}
        ways_list = []
        for way in result.ways:
            street_name = way.tags.get("name")
            if street_name:
                highway_type = way.tags.get("highway", "unclassified")
                node_refs = [n.id for n in way.nodes]
                ways_list.append({
                    "name": street_name,
                    "nodes": node_refs,
                    "highway": highway_type
                })
        save_to_cache(min_lat, max_lat, min_lng, max_lng, nodes_dict, ways_list)
    
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


def _create_overpass_result_from_cache(nodes_dict, ways_list):
    """
    Convert cached nodes and ways back to Overpass result format.
    
    Args:
        nodes_dict: Dictionary of {node_id: (lat, lon)}
        ways_list: List of way dictionaries
    
    Returns:
        SimpleNamespace object mimicking overpy.Result structure
    """
    from types import SimpleNamespace
    from collections import namedtuple
    
    # Create node objects
    CachedNode = namedtuple('CachedNode', ['id', 'lat', 'lon'])
    cached_nodes = [
        CachedNode(id=int(node_id), lat=lat, lon=lon)
        for node_id, (lat, lon) in nodes_dict.items()
    ]
    
    # Create way objects with minimal attributes
    class CachedWay:
        def __init__(self, name, nodes, highway):
            self.id = None
            self.tags = {'name': name, 'highway': highway}
            self.nodes = [SimpleNamespace(id=n) for n in nodes]
    
    cached_ways = [
        CachedWay(way['name'], way['nodes'], way['highway'])
        for way in ways_list
    ]
    
    # Return result object
    return SimpleNamespace(nodes=cached_nodes, ways=cached_ways)
