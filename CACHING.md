# OSM Data Caching System

## Overview

The Food Truck Route Optimizer now includes a SQLite-based caching system that stores street network data locally, eliminating Overpass API rate limiting and server outage issues.

**Benefits:**
- ✅ No more "rate limited" errors from Overpass API
- ✅ Faster queries (no API latency)
- ✅ Works offline for cached areas
- ✅ Reduces load on public Overpass infrastructure
- ✅ Pre-cache common service areas before deployment

## How It Works

1. **First query for an area**: Street data is fetched from Overpass API and cached locally
2. **Subsequent queries**: 
   - Exact match: Direct cache hit for identical bbox
   - Spatial containment: If request is within a larger cached area, filtered data is returned
3. **Automatic**: Caching happens transparently—no code changes needed
4. **SQLite storage**: Data stored in `data/cache/osm_cache.db`

## Usage

### Automatic Caching (Default)

Caching is enabled by default. The first time you request street data for a new area, it's fetched from Overpass API and automatically cached:

```python
# This caches automatically
result = fetch_overpass_data(41.8, 41.9, -87.7, -87.6)
# "Retrieving cached OSM data..." on subsequent calls ✓
```

To disable caching for a specific query:

```python
result = fetch_overpass_data(41.8, 41.9, -87.7, -87.6, use_cache=False)
```

### Pre-Download Areas

Pre-download street data for common service areas to eliminate any API dependency:

#### By ZIP Code

```bash
# Single ZIP code
python predownload_cache.py --zipcode 60601

# Multiple ZIP codes
python predownload_cache.py --zipcodes 60601 60602 60603

# With custom radius (default 5 miles)
python predownload_cache.py --zipcode 60601 --radius 10
```

#### By Bounding Box

```bash
# Single bounding box (min_lat, max_lat, min_lng, max_lng)
python predownload_cache.py --bbox 41.8 41.9 -87.7 -87.6
```

#### From JSON File

Create `areas.json`:
```json
{
  "areas": [
    {"name": "Chicago", "zipcode": "60601"},
    {"name": "Downtown Chicago", "bbox": [41.8, 41.85, -87.65, -87.6]},
    {"name": "Atlanta", "zipcode": "30303"}
  ]
}
```

Then pre-download:
```bash
python predownload_cache.py --file areas.json
```

An example file is provided as `example_areas.json`.

### Cache Management

#### View Statistics

```bash
python predownload_cache.py --stats
```

Output:
```
============================================================
CACHE STATISTICS
============================================================
Cached Regions: 15
Total Cache Size: 245.67 MB
Cache Location: data/cache/osm_cache.db

Cached Bounding Boxes:
------------------------------------------------------------
  Bbox: (41.70, -87.80, 41.90, -87.60)
    Size: 12.34 MB | Cached: 2026-03-03 14:22:15
  ...
============================================================
```

#### List All Cached Areas

```bash
python predownload_cache.py --list
```

#### Clear All Cache

```bash
python predownload_cache.py --clear
```

#### Remove Old Entries

Remove cached data older than N days:
```bash
python predownload_cache.py --cleanup 30
```

## Deployment Integration

The deployment script (`deploy.sh`) automatically:

1. Creates the `data/cache` directory
2. Initializes the SQLite database
3. Sets proper file permissions

After deployment, you can pre-cache areas on the production server:

```bash
cd /home/ubuntu/FoodTruckRouteOptimizer
venv/bin/python predownload_cache.py --file example_areas.json
```

## Storage Requirements

- **Per 5-mile radius area**: ~1-5 MB
- **Single city (e.g., Chicago)**: ~15-30 MB
- **Entire state**: ~50-100 MB
- **Multiple states**: Proportionally larger

### Example Sizes

| Area | Size | Nodes | Ways |
|------|------|-------|------|
| Chicago ZIP code | ~12 MB | ~45,000 | ~8,500 |
| Downtown Chicago | ~2 MB | ~8,000 | ~1,500 |
| Atlanta | ~18 MB | ~65,000 | ~12,000 |

## Technical Details

### Cache Schema

```sql
-- Stores cached OSM data for specific bounding boxes
CREATE TABLE cached_regions (
    id INTEGER PRIMARY KEY,
    min_lat REAL NOT NULL,
    max_lat REAL NOT NULL,
    min_lng REAL NOT NULL,
    max_lng REAL NOT NULL,
    data_json TEXT NOT NULL,
    cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(min_lat, max_lat, min_lng, max_lng)
);
```

### Bounding Box Matching

The cache supports intelligent bounding box matching at multiple levels:

1. **Exact Match**: Queries are matched against cached regions using 4 decimal place precision (approximately 11 meters)
2. **Spatial Containment**: If a requested bbox is fully contained within a larger cached bbox, the cached data is used (no API call needed)
3. **Filtered Results**: When serving from a larger cached region, only the relevant nodes and ways within the requested bbox are returned

**Example:**
- Cached: `(41.70, -87.80, 41.90, -87.60)` - 10 mile × 10 mile area (~30 MB)
- Request 1: `(41.75, -87.75, 41.85, -87.65)` - 5 mile × 5 mile area inside cached region ✓ **HIT**
- Request 2: `(41.8, -87.7, 41.85, -87.65)` - 3 mile × 3 mile area inside cached region ✓ **HIT**

This dramatically improves cache hit rates for overlapping queries within the same general area.

### Data Format

Cached data includes:
- **Nodes**: Dictionary of node IDs to (latitude, longitude) tuples
- **Ways**: List of street segments with name, nodes, and highway type

## Troubleshooting

### Cache not working

Verify database is initialized:
```bash
python -c "from osm_cache import get_cache_stats; print(get_cache_stats())"
```

### Pre-download failing

Check Overpass API status at https://overpass-api.de/
Try with smaller areas or fewer simultaneous downloads.

### Clear corrupted cache

```bash
python predownload_cache.py --clear
```

The cache will automatically rebuild as you make queries.

## Best Practices

1. **Pre-download with generous coverage**: Cache larger areas (e.g., 10-mile radius) around your service zones to catch all queries within those areas
   - A single 10-mile radius cache will serve dozens of smaller queries
   - Much more efficient than caching many small 5-mile areas

2. **Use ZIP codes for user concentrations**: Pre-cache ZIP codes where you have concentrated users, with larger radius
   ```bash
   # Cache a 10-mile radius around user ZIP codes
   python predownload_cache.py --zipcodes 60565 60564 60504 --radius 10
   ```

3. **Regular cleanup**: Remove old cache entries periodically
   ```bash
   python predownload_cache.py --cleanup 30
   ```

4. **Monitor size**: Check cache statistics occasionally to manage disk space
   ```bash
   python predownload_cache.py --stats
   ```

5. **Use graph-based routing**: Rely on local graph optimization + cached Overpass data for stable performance

## Future Enhancements

- Automatic cache warming for configured service areas
- Cache statistics endpoint in Flask API
- Incremental updates instead of full re-download
- Support for other data sources (Mapbox, Google Maps)
