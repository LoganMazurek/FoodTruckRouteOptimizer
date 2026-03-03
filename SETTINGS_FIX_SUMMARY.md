# Route Settings Implementation - Summary of Changes

## Problem
Route settings (coverage_mode, min_street_length, speed_priority) were not affecting the final route calculation when users hit the "Optimize Route" button.

## Root Causes Fixed

### 1. Settings Not Passed Through to Route Calculation
**Issue**: The `/visualize_cpp` endpoint was receiving settings but not rebuilding the graph with them or passing them to the routing algorithm.

**Fix**: 
- Updated `simplify_graph()` function in `find_route.py` to properly use coverage_mode and min_street_length
- Modified `find_route_max_coverage_optimized()` to accept and use speed_priority setting
- Updated both `/visualize_cpp` and `/result` endpoints to pass settings through the entire pipeline

### 2. Missing Speed Priority Implementation
**Issue**: The `speed_priority` setting was accepted but never used in the routing algorithm.

**Fix in `find_route_max_coverage_optimized()`**:
- **'fastest'**: Uses 75% coverage threshold (exits route earlier for quicker completion)
- **'balanced'** (default): Uses 80% coverage threshold (standard behavior)
- **'thorough'**: Uses 95% coverage threshold (tries to visit more streets)

### 3. Coverage Mode Not Applied During Graph Rebuild
**Issue**: Initial graph was created in `process_boundaries()` without user settings, then user changes weren't reflected.

**Fix**:
- App now saves original nodes/ways data for rebuilding graphs
- Each time user changes coverage_mode, the graph is rebuilt with the specified excluded road types
- Coverage modes impact which streets are included:
  - **'maximum'**: Excludes only [footway, path, cycleway, steps]
  - **'balanced'**: Excludes [footway, track, pedestrian, path, cycleway]
  - **'major-roads'**: Excludes residential and service roads

### 4. Browser Caching
**Issue**: Browser may cache route responses, causing old routes to display even when settings change.

**Fix**:
- Added cache prevention headers to `/visualize_cpp` response:
  - `Cache-Control: no-store, no-cache, must-revalidate, max-age=0`
  - `Pragma: no-cache`
  - `Expires: 0`

## Data Flow After Fix

```
User Changes Settings (Frontend)
    ↓
  Sends: /visualize_cpp?boundary_id=xxx&coverage_mode=X&min_street_length=Y&speed_priority=Z
    ↓
  App receives settings and logs them [VISUALIZE_CPP]
    ↓
  Load original nodes_ways.pkl
    ↓
  Rebuild graph with coverage settings via simplify_graph()
    ↓
  Graph reflects user's coverage_mode and min_street_length choices
    ↓
  Call find_route_max_coverage_optimized() with settings
    ↓
  Route algorithm uses speed_priority to determine coverage threshold
    ↓
  Return route to frontend with cache prevention headers
    ↓
  Frontend displays new route reflecting all user settings
```

## Logging Added

### App Level (`app.py`)
```
[VISUALIZE_CPP] Received settings from request: coverage_mode=X, min_street_length=Ym, speed_priority=Z
[VISUALIZE_CPP] Original data: N nodes, M ways
[VISUALIZE_CPP] Graph after simplify_graph: X nodes, Y edges
[VISUALIZE_CPP] Graph after clean_up_graph: X nodes, Y edges
[VISUALIZE_CPP] Generated route with N waypoints
[RESULT] Optimizing with settings: coverage_mode=X, min_street_length=Ym, speed_priority=Z
```

### Graph Building Level (`find_route.py`)
```
[SIMPLIFY_GRAPH] Building graph with: coverage=X, min_length=Ym, speed=Z
[SIMPLIFY_GRAPH] Coverage=X: excluded types = [...]
[SIMPLIFY_GRAPH] After filtering: N relevant nodes from M ways
[SIMPLIFY_GRAPH] Final graph: X nodes, Y edges

[FIND_ROUTE] Speed priority=X: using Y% coverage threshold
[FIND_ROUTE] max_coverage_optimized: Z% coverage (A/B edges), speed_priority=X
```

## Testing Added

Created `tests/test_route_settings.py` with tests for:
1. Coverage mode affects graph size (maximum > balanced > major-roads)
2. Min street length affects graph size (shorter min = more edges)
3. Speed priority affects coverage threshold (fastest < balanced < thorough)
4. All settings work together
5. Backward compatibility (settings dict optional)

## Expected Behavior After Fix

1. **Change Coverage Mode** → Graph rebuilds to include/exclude different road types → Route covers more/fewer streets

2. **Change Min Street Length** → Graph filters out streets shorter than threshold → Route may include more small streets or only major roads

3. **Change Speed Priority** → Route algorithm adjusts when to stop optimal coverage
   - Fastest: Stops earlier (75% coverage)
   - Balanced: Standard (80% coverage)
   - Thorough: Tries harder (95% coverage)

4. **Multiple Optimizations** → Each time user changes settings and clicks Optimize, a completely new route is calculated reflecting those settings

## Files Modified

1. **app.py**
   - Updated `/visualize_cpp` to pass settings through entire pipeline
   - Updated `/result` to pass settings to routing function
   - Added cache prevention headers

2. **find_route.py**
   - Enhanced `simplify_graph()` with logging for coverage mode application
   - Updated `find_route_max_coverage_optimized()` to accept and use settings dict
   - Implemented speed_priority -> coverage threshold mapping
   - Added comprehensive logging of algorithm behavior

3. **tests/test_route_settings.py** (new)
   - Tests for setting impact on graph generation
   - Tests for setting impact on route generation
   - Backward compatibility tests

## How to Verify

1. Set up a route with some boundaries
2. Select start node and optimize route
3. Change coverage_mode to "maximum", click optimize again
   - New route should show different street coverage
4. Change min_street_length to 200, click optimize again
   - New route should mostly show major roads
5. Change speed_priority to "fastest", click optimize again
   - Route should exit earlier (potential shorter coverage %)
6. Change to "thorough", click optimize again
   - Route should visit more streets

4. Check logs for `[VISUALIZE_CPP]`, `[SIMPLIFY_GRAPH]`, and `[FIND_ROUTE]` entries confirming settings are being applied
