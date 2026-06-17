"""
Drive the REAL routing pipeline on a small, realistic neighborhood and emit a
GPX file using the exact logic from app.export_gpx, so the output can be
inspected the way OsmAnd would consume it.

Run:  python tests/gen_real_gpx.py
"""
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from find_route import (
    simplify_graph,
    find_route_max_coverage_optimized,
    prune_common_sense_nodes,
)
from app import consolidate_turn_by_turn


# --- Build a small neighborhood: a 4x3 grid + one curved street -------------
# Coordinates near Palo Alto, spacing ~ a city block (~0.0009 deg ~ 100m).
LAT0, LON0 = 37.4400, -122.1500
DLAT = 0.0009
DLON = 0.0011

nodes = {}
ROWS, COLS = 3, 4
def nid(r, c):
    return r * 100 + c

for r in range(ROWS):
    for c in range(COLS):
        nodes[nid(r, c)] = (LAT0 + r * DLAT, LON0 + c * DLON)

# Add a couple of mid-block geometry nodes to give one street real curvature.
nodes[9001] = (LAT0 + 0.5 * DLAT + 0.0003, LON0 + 1.5 * DLON)  # bulge on a curve

ways = []
# Horizontal streets (named like real residential roads).
hnames = ["Maple Street", "Oak Avenue", "Elm Street"]
for r in range(ROWS):
    ways.append({
        "name": hnames[r],
        "highway": "residential",
        "nodes": [nid(r, c) for c in range(COLS)],
    })
# Vertical streets.
vnames = ["First Street", "Second Street", "Third Street", "Fourth Street"]
for c in range(COLS):
    ways.append({
        "name": vnames[c],
        "highway": "residential",
        "nodes": [nid(r, c) for r in range(ROWS)],
    })
# One curved connector that passes through a mid-block geometry node.
ways.append({
    "name": "Willow Bend",
    "highway": "residential",
    "nodes": [nid(0, 1), 9001, nid(1, 2)],
})

settings = {
    "coverage_mode": "balanced",
    "min_street_length": 0,
    "speed_priority": "balanced",
    "node_snap_distance_m": 18,
    "filter_dead_ends": False,
}

start_node = nid(0, 0)
end_node = nid(2, 3)

graph = simplify_graph(nodes, ways, settings=settings,
                       start_node=start_node, end_node=end_node)
print(f"graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

result = find_route_max_coverage_optimized(graph, start_node, end_node, settings=settings)
optimized_route = result["route"]
print(f"optimized_route (expanded): {len(optimized_route)} coords")

way_ids = [None] * len(optimized_route)
pruned_route = prune_common_sense_nodes(optimized_route, way_ids=way_ids,
                                        angle_threshold=30, graph=graph)
print(f"pruned_route (waypoints):   {len(pruned_route)} coords")

# Build turn-by-turn exactly like app.py /result does.
turn_by_turn_instructions = []
for i in range(len(optimized_route) - 1):
    curr_coord = optimized_route[i]
    next_coord = optimized_route[i + 1]
    curr_node = next_node = None
    for node, data in graph.nodes(data=True):
        if data.get("coordinates") == curr_coord:
            curr_node = node
        if data.get("coordinates") == next_coord:
            next_node = node
    if curr_node and next_node and graph.has_edge(curr_node, next_node):
        edge_data = graph[curr_node][next_node]
        distance = edge_data.get("distance", 0)
        street_name = edge_data.get("way_id", "unnamed road")
        turn_by_turn_instructions.append({
            "instruction": f"Continue on {street_name}",
            "name": street_name,
            "distance": distance,
            "duration": distance / 8.33,
        })

instructions = consolidate_turn_by_turn(turn_by_turn_instructions)
print(f"raw instruction segments:   {len(turn_by_turn_instructions)}")
print(f"consolidated instructions:  {len(instructions)}")

route_data = {
    "priority": "balanced",
    "name": "Balanced Route",
    "description": "test",
    "waypoints": pruned_route,
    "geometry": pruned_route,
    "track": optimized_route,  # dense road-following geometry for GPX export
    "instructions": instructions,
    "route_info": {"total_distance_miles": 1.23, "total_duration_min": 9.0},
}

# --- Emit GPX with the EXACT logic from app.export_gpx ----------------------
distance_miles = route_data["route_info"]["total_distance_miles"]
duration_min = route_data["route_info"]["total_duration_min"]
route_name = route_data["name"]

track = route_data.get("track") or route_data.get("geometry") or route_data.get("waypoints") or []

gpx = '<?xml version="1.0" encoding="UTF-8"?>\n'
gpx += '<gpx version="1.1" creator="FoodTruckRouteOptimizer" xmlns="http://www.topografix.com/GPX/1/1">\n'
gpx += '  <metadata>\n'
gpx += f'    <name>{route_name}</name>\n'
gpx += f'    <desc>Route: {distance_miles} miles, {duration_min:.0f} min. {route_data["description"]}</desc>\n'
gpx += f'    <time>{datetime.utcnow().isoformat()}Z</time>\n'
gpx += '  </metadata>\n'

gpx += '  <trk>\n'
gpx += f'    <name>{route_name}</name>\n'
gpx += '    <trkseg>\n'
for lat, lon in track:
    gpx += f'      <trkpt lat="{lat}" lon="{lon}"></trkpt>\n'
gpx += '    </trkseg>\n  </trk>\n'

gpx += '</gpx>'

out = os.path.join(os.path.dirname(__file__), "sample_route.gpx")
with open(out, "w") as f:
    f.write(gpx)
print(f"\nwrote {out}")
print(f"counts -> dense trkpt:{len(track)}  (vs pruned waypoints:{len(pruned_route)})")
