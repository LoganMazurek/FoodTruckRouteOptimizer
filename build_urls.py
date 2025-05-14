import requests
import json

def get_google_maps_url(route):
    MAX_WAYPOINTS = 10  # Google Maps URL limit for waypoints

    if len(route) < 2:
        print("Route must contain at least two points")
        return None

    origin = route[0]
    destination = route[-1]
    waypoints = route[1:-1]
    
    # Split into chunks (Google Maps limits waypoints to 10 per link)
    waypoint_chunks = [waypoints[i:i + MAX_WAYPOINTS] for i in range(0, len(waypoints), MAX_WAYPOINTS)]
    
    urls = []
    for i, chunk in enumerate(waypoint_chunks):
        if i == 0:
            origin_str = f"{origin[0]},{origin[1]}"
        else:
            # Use the last waypoint of the previous chunk as the origin for the next chunk
            origin_str = f"{waypoint_chunks[i-1][-1][0]},{waypoint_chunks[i-1][-1][1]}"
        
        if i == len(waypoint_chunks) - 1:
            destination_str = f"{destination[0]},{destination[1]}"
        else:
            # Use the first waypoint of the next chunk as the destination for the current chunk
            destination_str = f"{chunk[-1][0]},{chunk[-1][1]}"
        
        waypoints_str = "|".join([f"{lat},{lon}" for lat, lon in chunk])
        
        url = f"https://www.google.com/maps/dir/?api=1&origin={origin_str}&destination={destination_str}&waypoints={waypoints_str}&travelmode=driving"
        urls.append(url)
    
    return urls