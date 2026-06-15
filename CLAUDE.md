# CLAUDE.md

Guidance for AI assistants (Claude Code and similar) working in this repository.

## Project Overview

Food Truck Route Optimizer is a Flask web app that helps food trucks (and similar
mobile services) plan driving routes that maximize street coverage within a
user-drawn boundary. The user picks a location, draws a rectangular boundary on a
map, the app fetches the street network from OpenStreetMap, builds a graph with
NetworkX, runs a coverage-maximizing routing algorithm, and returns up to three
route variants (Quick / Balanced / Thorough) with turn-by-turn directions and GPX
export.

No API keys are required. Geocoding uses Nominatim, street data comes from the
Overpass API (with a local SQLite cache), and routing runs locally via NetworkX.

## Tech Stack

- **Backend**: Python 3 / Flask 2.3 (served by Waitress in production)
- **Graph/routing**: NetworkX 3.1, geopy (geodesic distances)
- **Street data**: OpenStreetMap via Overpass API, cached in SQLite
- **Geocoding**: Nominatim (OpenStreetMap)
- **Frontend**: Server-rendered Jinja templates + Leaflet.js (no JS build step)
- **Testing**: pytest + pytest-mock
- **Deploy target**: DigitalOcean droplet via nginx + supervisor + certbot

## Request / Page Flow

1. `GET /` → `templates/index.html` — user enters a ZIP code / location.
2. `POST /` → geocodes via `get_coordinates()`, renders `select_boundaries.html`
   (or `GET /select_boundaries`).
3. User draws a 4-corner rectangle on the map. `POST /process_boundaries`:
   - Fetches OSM data for the bbox via `fetch_overpass_data()` (cache-aware).
   - Extracts nodes/ways (`extract_nodes_and_ways`), builds and simplifies the
     graph (`simplify_graph`, `clean_up_graph`).
   - Persists `{boundary_id}_graph.pkl` and `{boundary_id}_nodes_ways.pkl` to
     `temp/`, returns a new `boundary_id` (UUID4).
4. `GET /graph_leaflet?boundary_id=...` → `graph_leaflet.html`, an interactive
   editor for the street graph:
   - `GET /graph-data` returns nodes/links as JSON for rendering.
   - `POST /delete_nodes` removes selected nodes (with stitching), persists the
     updated graph and a `{boundary_id}_deleted_nodes.pkl` set.
   - `GET /visualize_cpp` rebuilds the graph from raw nodes/ways with the
     user's current settings and returns a single preview route.
   - User picks a start (and optional end) node, then navigates to `/result`.
5. `GET /result?boundary_id=...&start_node=...&end_node=...` → builds **three**
   route variants (fastest/balanced/thorough), each via
   `find_route_max_coverage_optimized`, prunes them with
   `prune_common_sense_nodes`, computes coverage/distance/duration stats, saves
   `{boundary_id}_routes.pkl`, and renders `result.html`.
6. `GET /export_gpx?boundary_id=...&route_option=fastest|balanced|thorough`
   streams a GPX file built from the saved route variant.

All per-session graph/route state is file-based pickle storage under `temp/`,
keyed by a UUID `boundary_id` — there is no database.

## Key Files

| File | Purpose |
|---|---|
| `app.py` | Flask routes, request validation, file persistence, GPX export, turn-by-turn consolidation |
| `find_route.py` | Graph simplification (`simplify_graph`), cleanup, node merging, dead-end filtering, and all route-optimization algorithms |
| `get_street_data.py` | Overpass API queries (with retry across multiple mirrors), Nominatim geocoding, OSM element extraction |
| `osm_cache.py` | SQLite-backed cache for Overpass responses (exact-bbox and spatial-containment lookups) |
| `predownload_cache.py` | CLI to pre-warm the OSM cache for ZIP codes / bboxes / a JSON area list (see `example_areas.json`) |
| `build_urls.py` | Builds chunked Google Maps navigation URLs from a route (no API key needed) |
| `visualization.py` | Matplotlib-based debug animation of route construction (dev/debug only) |
| `utils.py` | Small helpers (e.g., temp file cleanup) |
| `templates/*.html` | Jinja + Leaflet UI: `index`, `select_boundaries`, `graph_leaflet`, `result` |
| `tests/` | pytest suite (see Testing below) |

## Core Algorithm Notes (`find_route.py`)

- **`simplify_graph(nodes, ways, settings, start_node, end_node)`**: Builds a
  NetworkX graph from raw OSM nodes/ways.
  - `coverage_mode` (`balanced` | `maximum` | `major-roads`) controls which
    `highway` types are excluded.
  - `min_street_length` filters out streets shorter than N meters, measured by
    **aggregated street length** (same name+highway across segments), not
    per-segment length — OSM frequently splits one street into many ways.
  - Intersections and way endpoints become graph nodes; intermediate geometry is
    stored as coordinates on edges (`intermediate_nodes`) so it survives later
    filtering and can be re-expanded for the final route geometry.
  - `merge_nearby_nodes` collapses near-duplicate nodes within
    `node_snap_distance_m` (default 18m), protecting start/end nodes.
  - The graph is reduced to the connected component containing `start_node`
    (and `end_node` if given).
  - Optional `filter_dead_ends` removes degree-1 nodes and cul-de-sac-like
    streets (matched by name keywords: court, circle, loop, place, etc.),
    protecting start/end nodes.

- **`find_route_max_coverage_optimized(graph, start_node, end_node, settings)`**:
  Greedy edge-scoring traversal driven by a `speed_priority` profile
  (`fastest` | `balanced` | `thorough`), each with its own coverage target,
  edge-reuse limits, and penalty weights (reuse, U-turn, backtrack, frontier
  exploration, distance-to-end pull). `fastest` instead delegates to
  `find_route_drive_efficient` (a snaking through-route with no edge reuse).
  Returns a dict: `{route, path, covered_edge_length_m, total_edge_length_m,
  covered_edge_count, total_edge_count}`. Tuning these profiles is sensitive —
  see git history (many commits tune coverage targets) and
  `tests/quick_coverage_test.py` / `tests/test_real_graph.py` for how coverage
  is validated against synthetic and real graphs.

- **`prune_common_sense_nodes` / `filter_turns`**: Reduce a route to only
  intersections and street-end nodes for cleaner waypoints/turn-by-turn output.

- **`find_route_cpp`**: Alternate strategies — `"drive"` (preferential
  right-turn heuristic) or `"cpp"` (Eulerian circuit/path via
  `nx.eulerize` for true Chinese-Postman-style full coverage).

## Settings Reference

These settings flow from the UI through `/visualize_cpp` and `/result` into
`simplify_graph` and `find_route_max_coverage_optimized`:

- `coverage_mode`: `balanced` (default) | `maximum` | `major-roads`
- `min_street_length`: meters, default 50 in `find_route.py`
  (`MIN_STREET_LENGTH_METERS`), default 70 in `app.py` request parsing
- `speed_priority`: `fastest` | `balanced` | `thorough`
- `filter_dead_ends`: bool, default `true` from the request
- `node_snap_distance_m`: fixed at `NODE_SNAP_DISTANCE_M = 18` in `app.py`

The `/result` route always builds a **baseline graph** with
`min_street_length: 0` to compute the coverage-percentage denominator, then
builds a separate graph per route variant.

## Caching (`osm_cache.py`, `CACHING.md`)

- SQLite DB at `data/cache/osm_cache.db` (directory is gitignored).
- `fetch_overpass_data()` checks for an exact bbox match, then for a larger
  cached bbox that *contains* the request (returns spatially filtered data),
  before calling the Overpass API.
- Successful API results are cached automatically.
- `predownload_cache.py` is a standalone CLI for pre-warming the cache
  (`--zipcode`, `--bbox`, `--file areas.json`, `--stats`, `--list`, `--clear`,
  `--cleanup N`). `deploy.sh` pre-caches a fixed list of ZIP codes on deploy.

## Security Notes

- `validate_boundary_id` requires `boundary_id` to be a valid UUID before any
  file lookup.
- `get_safe_file_path` normalizes and confirms the resolved path stays inside
  `GRAPH_DIR` (`temp/`) — always use this helper for any new pickle file paths,
  never build paths from request data directly.
- `SECRET_KEY` must be set via environment variable in production; a random key
  is generated (with a warning) if absent — fine for dev, not for prod restarts
  (sessions won't survive a restart without a fixed key).
- Don't hardcode secrets/API keys. This project intentionally needs none.

## Development Workflow

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python app.py            # dev server on http://localhost:5001 (debug=False)
```

Health check: `GET /health` reports Python version and whether `temp/` exists
and is writable.

## Testing

```bash
pytest                          # run all tests
pytest -v tests/test_app.py     # single file
pytest tests/test_app.py::test_home_get
```

- `tests/test_app.py`: Flask route tests using the `client` fixture
  (`app.config['TESTING'] = True`), mocking external calls with `mocker`
  (pytest-mock). External services (Overpass, Nominatim) and graph functions
  are always mocked — never hit real network in tests.
- `tests/test_graph_filters.py`: Unit tests for `extract_nodes_and_ways` (bbox
  clipping/segment splitting) and `simplify_graph` (node merging).
- `tests/test_real_graph.py` and `tests/quick_coverage_test.py`: Manual/CLI
  diagnostic scripts (not pytest-discovered as `test_*` functions in the usual
  sense for the latter — they're run directly, e.g.
  `python tests/test_real_graph.py <boundary_id>`) that validate coverage
  percentages of the speed profiles against real or synthetic graphs.
- `tests/run_coverage_tests.py` imports `tests.test_coverage_targets`, which
  **does not currently exist** in the repo — that script is currently broken;
  don't rely on it without first checking whether `test_coverage_targets.py`
  needs to be (re)created.

When adding routing/graph features, prefer adding pytest-style tests in
`tests/test_*.py` following the existing `client`/`mocker` pattern in
`test_app.py`, and mock `app.simplify_graph`, `app.find_route_*`,
`app.fetch_overpass_data`, `app.get_coordinates`, etc. as needed.

## Code Conventions

- PEP 8, 4-space indents, ~100 char line length, snake_case.
- Use the `logging` module (never `print`) for anything outside one-off debug
  scripts. Tag log messages with a bracketed context prefix matching existing
  style, e.g. `logger.info(f"[SIMPLIFY_GRAPH] ...")`, `[FIND_ROUTE]`,
  `[RESULT]`, `[VISUALIZE_CPP]`, `[MERGE_NODES]`, `[FILTER_DEAD_ENDS]`.
- Import order: stdlib, third-party, local — separated by blank lines.
- New dependencies must be added to `requirements.txt`.
- Don't commit anything under `temp/` or `data/` (both gitignored except
  `.gitkeep`).
- Note: `requirements.txt` lists `overpy`, but `get_street_data.py` talks to
  Overpass directly via `requests` + `SimpleNamespace` — `overpy` is currently
  unused. Don't assume `overpy`'s API is in play when reading this module.

## Deployment

See `DEPLOYMENT.md` for the full runbook. Summary: `deploy.sh` provisions a
DigitalOcean droplet (nginx reverse proxy + supervisor running
`waitress ... app:app` on port 5001 + certbot for
`nroute.loganmazurek.com`), preserving existing SSL certs on redeploy.
`check_deployment.sh`, `debug_droplet.sh`, `check_ssl.sh`, `setup_ssl.sh`, and
`quick_fix.sh` are operational helper scripts for the live server, not part of
the app itself.

## Other Docs in This Repo

- `MULTI_ROUTE_FEATURE.md` — design notes for the 3-route-variant result page.
- `SETTINGS_FIX_SUMMARY.md` — history of how `coverage_mode` /
  `min_street_length` / `speed_priority` get threaded through the pipeline.
- `CACHING.md` — full OSM cache design/usage docs.
- `.github/copilot-instructions.md` — overlapping guidance for GitHub Copilot;
  keep in sync if conventions change here.
- `.github/agents/*.md` — role-specific agent prompts (code review, docs,
  tests) with the same project conventions, useful as additional context for
  what "good" looks like in this repo.
