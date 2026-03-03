import requests
import logging

logger = logging.getLogger(__name__)

# OSRM server configuration
# Use local OSRM for Illinois, fall back to public OSRM for other regions
ILLINOIS_OSRM_URL = "http://localhost:5000"
PUBLIC_OSRM_URL = "http://router.project-osrm.org"

# Illinois bounding box for local OSRM
ILLINOIS_BOUNDS = {
    "min_lat": 36.97,
    "max_lat": 42.51,
    "min_lon": -91.51,
    "max_lon": -87.02
}


def is_in_illinois(lat, lon):
    """Check if coordinates are within Illinois."""
    return (ILLINOIS_BOUNDS["min_lat"] <= lat <= ILLINOIS_BOUNDS["max_lat"] and
            ILLINOIS_BOUNDS["min_lon"] <= lon <= ILLINOIS_BOUNDS["max_lon"])


def get_osrm_url(waypoints):
    """
    Determine which OSRM server to use based on waypoints.
    Returns local server if all waypoints are in Illinois, otherwise public.
    """
    # Check if all waypoints are in Illinois
    all_in_illinois = all(is_in_illinois(lat, lon) for lat, lon in waypoints)
    if all_in_illinois:
        logger.info("All coordinates in Illinois, using local OSRM")
        return ILLINOIS_OSRM_URL
    
    # Otherwise use public OSRM
    logger.info("Coordinates outside Illinois, using public OSRM")
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


def extract_turn_by_turn_instructions(osrm_route):
    """
    Extract professional turn-by-turn instructions from OSRM route response.
    
    Args:
        osrm_route: Route dict from get_route_with_waypoints or similar OSRM API call
    
    Returns:
        List of instruction dicts with keys: instruction, name, distance, duration
        Returns empty list if osrm_route is None or invalid
    """
    if not osrm_route:
        return []
    
    instructions = []
    
    # Extract legs (each leg is between two waypoints)
    legs = osrm_route.get('legs', [])
    for leg in legs:
        steps = leg.get('steps', [])
        
        for step in steps:
            # Each step has a maneuver with detailed turn information
            maneuver = step.get('maneuver', {})
            modifier = maneuver.get('modifier', '')
            step_type = maneuver.get('type', 'turn')
            
            name = step.get('name', 'unnamed road')
            distance = step.get('distance', 0)
            duration = step.get('duration', 0)
            
            # Build professional instruction text from maneuver type
            instruction_text = _build_instruction_text(step_type, modifier, name)
            
            instructions.append({
                'instruction': instruction_text,
                'name': name,
                'distance': round(distance, 1),
                'duration': round(duration, 1),
                'maneuver_type': step_type,
                'modifier': modifier
            })
    
    logger.info(f"Extracted {len(instructions)} turn-by-turn instructions from OSRM")
    return instructions


def _build_instruction_text(maneuver_type, modifier, name):
    """
    Build human-readable instruction text from OSRM maneuver data.
    
    Args:
        maneuver_type: Type of maneuver (e.g., 'turn', 'merge', 'exit', 'roundabout')
        modifier: Direction modifier (e.g., 'left', 'right', 'sharp right', 'uturn')
        name: Street or road name
    
    Returns:
        Human-readable instruction string
    """
    if maneuver_type == 'depart':
        return f"Start on {name}"
    
    elif maneuver_type == 'arrive':
        return f"Arrive at {name}"
    
    elif maneuver_type == 'turn':
        # Basic turn with direction
        if modifier == 'sharp left':
            return f"Make a sharp left turn onto {name}"
        elif modifier == 'left':
            return f"Turn left onto {name}"
        elif modifier == 'slight left':
            return f"Keep left onto {name}"
        elif modifier == 'sharp right':
            return f"Make a sharp right turn onto {name}"
        elif modifier == 'right':
            return f"Turn right onto {name}"
        elif modifier == 'slight right':
            return f"Keep right onto {name}"
        elif modifier == 'uturn':
            return f"Make a U-turn onto {name}"
        else:
            return f"Turn onto {name}"
    
    elif maneuver_type == 'merge':
        if modifier:
            return f"Merge {modifier} onto {name}"
        return f"Merge onto {name}"
    
    elif maneuver_type == 'enter roundabout' or maneuver_type == 'roundabout':
        if modifier:
            return f"Enter roundabout and take the {modifier} exit onto {name}"
        return f"Enter roundabout onto {name}"
    
    elif maneuver_type == 'exit roundabout':
        return f"Exit roundabout onto {name}"
    
    elif maneuver_type == 'on ramp':
        if modifier:
            return f"Take the {modifier} ramp onto {name}"
        return f"Take the ramp onto {name}"
    
    elif maneuver_type == 'off ramp':
        if modifier:
            return f"Take the {modifier} exit onto {name}"
        return f"Take the exit onto {name}"
    
    elif maneuver_type == 'fork':
        if modifier:
            return f"At the fork, keep {modifier} onto {name}"
        return f"Choose the correct fork onto {name}"
    
    elif maneuver_type == 'end of road':
        if modifier:
            return f"At the end of the road, turn {modifier} onto {name}"
        return f"Continue onto {name}"
    
    elif maneuver_type == 'new name':
        return f"Continue on {name}"
    
    elif maneuver_type == 'continue':
        return f"Continue on {name}"
    
    elif maneuver_type == 'waypoint':
        return f"Pass through waypoint on {name}"
    
    else:
        return f"Continue on {name}"
