import requests
import logging

logger = logging.getLogger(__name__)

# OSRM server configuration
# Local OSRM servers by state, fall back to public OSRM for other regions
ILLINOIS_OSRM_URL = "http://localhost:5000"
WISCONSIN_OSRM_URL = "http://localhost:5002"
PUBLIC_OSRM_URL = "http://router.project-osrm.org"

# State bounding boxes for local OSRM
ILLINOIS_BOUNDS = {
    "min_lat": 36.97,
    "max_lat": 42.51,
    "min_lon": -91.51,
    "max_lon": -87.02
}

WISCONSIN_BOUNDS = {
    "min_lat": 42.49,
    "max_lat": 47.31,
    "min_lon": -92.89,
    "max_lon": -86.25
}


def is_in_illinois(lat, lon):
    """Check if coordinates are within Illinois."""
    return (ILLINOIS_BOUNDS["min_lat"] <= lat <= ILLINOIS_BOUNDS["max_lat"] and
            ILLINOIS_BOUNDS["min_lon"] <= lon <= ILLINOIS_BOUNDS["max_lon"])


def is_in_wisconsin(lat, lon):
    """Check if coordinates are within Wisconsin."""
    return (WISCONSIN_BOUNDS["min_lat"] <= lat <= WISCONSIN_BOUNDS["max_lat"] and
            WISCONSIN_BOUNDS["min_lon"] <= lon <= WISCONSIN_BOUNDS["max_lon"])


def get_osrm_url(waypoints):
    """
    Determine which OSRM server to use based on waypoints.
    Returns local server for Illinois or Wisconsin, otherwise public.
    """
    # Check if all waypoints are in Illinois
    all_in_illinois = all(is_in_illinois(lat, lon) for lat, lon in waypoints)
    if all_in_illinois:
        logger.info("All coordinates in Illinois, using Illinois OSRM")
        return ILLINOIS_OSRM_URL
    
    # Check if all waypoints are in Wisconsin
    all_in_wisconsin = all(is_in_wisconsin(lat, lon) for lat, lon in waypoints)
    if all_in_wisconsin:
        logger.info("All coordinates in Wisconsin, using Wisconsin OSRM")
        return WISCONSIN_OSRM_URL
    
    # Otherwise use public OSRM
    logger.info("Coordinates span multiple states or unknown region, using public OSRM")
    return PUBLIC_OSRM_URL

def get_route_between_points(start_coords, end_coords, overview="full"):
    """
    Get a route between two points using the local OSRM server.
    
    Args:
        start_coords: Tuple of (lat, lon) for start point
        end_coords: Tuple of (lat, lon) for end point
        overview: Level of detail - "full", "simplified", or "false"
    
    Returns:
        dict with route info including geometry, distance, duration
    """
    # OSRM expects lon,lat format
    start_lon, start_lat = start_coords[1], start_coords[0]
    end_lon, end_lat = end_coords[1], end_coords[0]

    base_url = get_osrm_url([start_coords, end_coords])
    url = f"{base_url}/route/v1/driving/{start_lon},{start_lat};{end_lon},{end_lat}"
    params = {
        "overview": overview,
        "geometries": "geojson",
        "steps": "true"
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data.get("code") == "Ok":
            return data["routes"][0]
        else:
            logger.error(f"OSRM error: {data.get('message', 'Unknown error')}")
            return None
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to get route from OSRM: {e}")
        return None


def get_route_with_waypoints(waypoints, overview="simplified"):
    """
    Get a route through multiple waypoints using the local OSRM server.
    
    Args:
        waypoints: List of (lat, lon) tuples
        overview: Level of detail - "full", "simplified", or "false"
    
    Returns:
        dict with route info including geometry, distance, duration
    """
    if len(waypoints) < 2:
        logger.error("Need at least 2 waypoints for routing")
        return None
    
    base_url = get_osrm_url(waypoints)

    # OSRM expects lon,lat format
    coords_str = ";".join([f"{lon},{lat}" for lat, lon in waypoints])

    url = f"{base_url}/route/v1/driving/{coords_str}"
    params = {
        "overview": overview,
        "geometries": "geojson",
        "steps": "true",
        "continue_straight": "false"
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data.get("code") == "Ok":
            return data["routes"][0]
        else:
            logger.error(f"OSRM error: {data.get('message', 'Unknown error')}")
            return None
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to get route from OSRM: {e}")
        return None


def get_table(sources, destinations=None):
    """
    Get distance/duration matrix between points using OSRM Table service.
    
    Args:
        sources: List of (lat, lon) tuples for source points
        destinations: Optional list of (lat, lon) tuples for destinations
                     If None, uses sources as destinations
    
    Returns:
        dict with "durations" and "distances" matrices
    """
    # Determine routing server based on all coordinates
    all_points = sources if destinations is None else sources + destinations
    base_url = get_osrm_url(all_points)

    # OSRM expects lon,lat format
    coords = sources if destinations is None else sources + destinations
    coords_str = ";".join([f"{lon},{lat}" for lat, lon in coords])

    url = f"{base_url}/table/v1/driving/{coords_str}"
    params = {}
    
    if destinations is not None:
        # Specify source and destination indices
        params["sources"] = ";".join(str(i) for i in range(len(sources)))
        params["destinations"] = ";".join(str(i) for i in range(len(sources), len(sources) + len(destinations)))
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data.get("code") == "Ok":
            return {
                "durations": data.get("durations"),
                "distances": data.get("distances")
            }
        else:
            logger.error(f"OSRM error: {data.get('message', 'Unknown error')}")
            return None
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to get table from OSRM: {e}")
        return None


def decode_polyline(encoded):
    """
    Decode a polyline string into a list of (lat, lon) coordinates.
    Note: OSRM with geojson geometry returns coordinates directly, so this is for polyline format.
    """
    coords = []
    index = 0
    lat = 0
    lon = 0
    
    while index < len(encoded):
        # Decode latitude
        result = 0
        shift = 0
        while True:
            b = ord(encoded[index]) - 63
            index += 1
            result |= (b & 0x1f) << shift
            shift += 5
            if b < 0x20:
                break
        dlat = ~(result >> 1) if result & 1 else result >> 1
        lat += dlat
        
        # Decode longitude
        result = 0
        shift = 0
        while True:
            b = ord(encoded[index]) - 63
            index += 1
            result |= (b & 0x1f) << shift
            shift += 5
            if b < 0x20:
                break
        dlon = ~(result >> 1) if result & 1 else result >> 1
        lon += dlon
        
        coords.append((lat / 1e5, lon / 1e5))
    
    return coords
