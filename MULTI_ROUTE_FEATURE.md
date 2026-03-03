# Multi-Route Option Feature

## Overview

The Food Truck Route Optimizer now generates **3 distinct route options** for users to compare and choose from, instead of showing just one route. This addresses the concern that the algorithm's routes don't always match human intuition by giving users choice and control.

## What Changed

### Backend Changes (app.py)

#### 1. Modified `/result` Route
- **Before**: Generated 1 route using the user-selected `speed_priority` setting
- **After**: Generates 3 route variants simultaneously using all three speed priorities:
  - **Quick Route** (fastest): ~65% street coverage, fastest completion time
  - **Balanced Route** (balanced): ~88% street coverage, good balance
  - **Thorough Route** (thorough): ~97% street coverage, maximum coverage

#### 2. Modified `/export_gpx` Route
- **Before**: Exported the single generated route as GPX
- **After**: Accepts a `route_option` parameter (fastest, balanced, or thorough) to export the specific route variant the user selected

#### 3. Data Storage
- **Before**: Stored single route in `{boundary_id}_route.pkl`
- **After**: Stores all 3 route variants in `{boundary_id}_routes.pkl`

### Frontend Changes (result.html)

#### Complete UI Redesign
The result page now features:

1. **Side-by-Side Comparison Grid**
   - 3 cards displaying route options
   - Responsive grid layout (stacks on mobile)
   - Each card shows:
     - Route name and description
     - Priority badge (fastest/balanced/thorough)
     - Preview map with the route
     - Key statistics (distance, time, waypoints)
     - Download GPX button
     - View Details button

2. **Interactive Preview Maps**
   - Each route card has its own Leaflet map
   - Maps are non-interactive (no zooming/panning) for clean preview
   - Shows the full route with a start marker

3. **Detailed View Modals**
   - Click "View Details" to see full information for a route
   - Large interactive map
   - Collapsible turn-by-turn directions
   - Full statistics
   - Direct download button

4. **Route Selection**
   - Users can directly download any route's GPX file
   - Each download button includes the specific route variant

## Algorithm Differences

The three route variants use the same graph and edges but with different optimization parameters:

| Setting | Quick Route | Balanced Route | Thorough Route |
|---------|-------------|----------------|----------------|
| Coverage Target | 65% | 88% | 97% |
| Max Edge Reuse | 1 | 2 | 3 |
| Reuse Penalty | 45 | 60 | 70 |
| Unused Bonus | 5 | 22 | 38 |
| U-turn Penalty | 12 | 28 | 35 |

These parameters create visibly different routes:
- **Quick Route**: Takes shortcuts, skips minor streets, prioritizes speed
- **Balanced Route**: Good compromise between coverage and efficiency
- **Thorough Route**: Covers almost all streets, accepts some backtracking

## User Benefits

1. **Choice & Control**: Users aren't stuck with a single algorithmic output
2. **Visual Comparison**: Easy to see differences between routes at a glance
3. **Flexibility**: Can choose based on their specific needs for the day
4. **Better UX**: If one route doesn't look right, they have 2 other options

## Technical Details

### Route Generation
All 3 routes are generated sequentially in the `/result` endpoint:
```python
for priority, name, description in speed_priorities:
    settings_for_route = {
        'coverage_mode': coverage_mode,
        'min_street_length': min_street_length,
        'speed_priority': priority
    }
    optimized_route = find_route_max_coverage_optimized(
        graph, start_node, end_node, 
        settings=settings_for_route
    )
    # ... process and store route variant
```

### GPX Export
The export now accepts a route option parameter:
```
/export_gpx?boundary_id={id}&route_option=balanced
```

Valid route options: `fastest`, `balanced`, `thorough`

### File Storage
Route variants are stored with all necessary data:
```python
{
    'priority': 'balanced',
    'name': 'Balanced Route',
    'description': 'Good balance of speed and coverage (~88%)',
    'waypoints': [...],
    'geometry': [...],
    'instructions': [...],
    'route_info': {
        'total_distance_km': 12.5,
        'total_distance_miles': 7.77,
        'total_duration_min': 25.0
    }
}
```

## Performance Considerations

- **Generation Time**: Generating 3 routes takes ~3x longer than 1 route
- **Storage**: Route storage is ~3x larger but still minimal (< 1MB typically)
- **Browser Performance**: Preview maps use simpler rendering (non-interactive) to stay performant

## Future Enhancements

Possible improvements:
1. Add more route variants (4-5 options)
2. Allow custom parameter tuning per variant
3. Add route comparison overlay (all 3 on one map)
4. Show route quality metrics (coverage %, backtracking %)
5. Remember user's preferred route type across sessions

## Migration Notes

- No database migration needed (file-based storage)
- Old `{boundary_id}_route.pkl` files are no longer used
- The `/graph_leaflet` visualization endpoint still works unchanged
- The `/visualize_cpp` endpoint still generates single routes (for the graph editor view)

## Testing

To test the new feature:
1. Go through the normal flow (select boundaries, choose start point)
2. On the result page, you should see 3 route cards side by side
3. Compare the maps and statistics
4. Click "View Details" on any route to see the full map and directions
5. Download the GPX file for your preferred route
6. Import the GPX into a navigation app to verify it works

## Backward Compatibility

- Graph editor still works the same way
- The `/visualize_cpp` endpoint generates routes with user-selected speed priority (unchanged)
- Only the final result page is affected by this change
