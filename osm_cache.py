"""
OSM Data Caching Module

Provides SQLite-based caching for Overpass API queries to eliminate rate limiting
and API failures. Stores complete street network data for regions locally.
"""

import sqlite3
import json
import logging
import os
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

CACHE_DIR = Path("data/cache")
CACHE_DB = CACHE_DIR / "osm_cache.db"


def ensure_cache_directory():
    """Create cache directory if it doesn't exist."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def initialize_cache_db():
    """Initialize SQLite database schema for caching."""
    ensure_cache_directory()
    
    conn = sqlite3.connect(CACHE_DB)
    cursor = conn.cursor()
    
    # Table for cached bounding boxes and their metadata
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cached_regions (
            id INTEGER PRIMARY KEY,
            min_lat REAL NOT NULL,
            max_lat REAL NOT NULL,
            min_lng REAL NOT NULL,
            max_lng REAL NOT NULL,
            data_json TEXT NOT NULL,
            cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(min_lat, max_lat, min_lng, max_lng)
        )
    """)
    
    # Table for tracking which areas have been cached
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cache_metadata (
            key TEXT PRIMARY KEY,
            value TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.commit()
    conn.close()
    logger.info(f"Cache database initialized at {CACHE_DB}")


def _normalize_bbox(min_lat, max_lat, min_lng, max_lng, precision=4):
    """
    Normalize bounding box values to match cached entries.
    Uses 4 decimal places (approximately 11 meters precision).
    """
    return (
        round(min_lat, precision),
        round(max_lat, precision),
        round(min_lng, precision),
        round(max_lng, precision)
    )


def cache_exists(min_lat, max_lat, min_lng, max_lng):
    """Check if data for this bounding box is cached."""
    initialize_cache_db()
    
    min_lat, max_lat, min_lng, max_lng = _normalize_bbox(min_lat, max_lat, min_lng, max_lng)
    
    conn = sqlite3.connect(CACHE_DB)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id FROM cached_regions 
        WHERE min_lat = ? AND max_lat = ? AND min_lng = ? AND max_lng = ?
    """, (min_lat, max_lat, min_lng, max_lng))
    
    result = cursor.fetchone()
    conn.close()
    
    return result is not None


def get_cached_data(min_lat, max_lat, min_lng, max_lng):
    """
    Retrieve cached OSM data for a bounding box.
    
    Returns: (nodes, ways) or (None, None) if not cached
    """
    initialize_cache_db()
    
    min_lat, max_lat, min_lng, max_lng = _normalize_bbox(min_lat, max_lat, min_lng, max_lng)
    
    conn = sqlite3.connect(CACHE_DB)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT data_json FROM cached_regions 
        WHERE min_lat = ? AND max_lat = ? AND min_lng = ? AND max_lng = ?
    """, (min_lat, max_lat, min_lng, max_lng))
    
    result = cursor.fetchone()
    conn.close()
    
    if result:
        try:
            data = json.loads(result[0])
            logger.info(f"Cache hit for bbox ({min_lat},{min_lng},{max_lat},{max_lng})")
            return data['nodes'], data['ways']
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to decode cached data: {e}")
            return None, None
    
    return None, None


def _bbox_contains_point(cached_min_lat, cached_max_lat, cached_min_lng, cached_max_lng,
                        point_lat, point_lon):
    """Check if a point is within a bounding box."""
    return (cached_min_lat <= point_lat <= cached_max_lat and
            cached_min_lng <= point_lon <= cached_max_lng)


def _bbox_contains_bbox(cached_min_lat, cached_max_lat, cached_min_lng, cached_max_lng,
                       req_min_lat, req_max_lat, req_min_lng, req_max_lng):
    """Check if a cached bbox fully contains a requested bbox."""
    return (cached_min_lat <= req_min_lat and
            cached_max_lat >= req_max_lat and
            cached_min_lng <= req_min_lng and
            cached_max_lng >= req_max_lng)


def find_containing_cached_region(min_lat, max_lat, min_lng, max_lng):
    """
    Find a cached region that fully contains the requested bbox.
    
    If found, returns (nodes, ways) extracted from that region.
    If not found, returns (None, None).
    
    This allows serving requests that fall within a larger cached area
    without making a new API call.
    """
    initialize_cache_db()
    
    min_lat, max_lat, min_lng, max_lng = _normalize_bbox(min_lat, max_lat, min_lng, max_lng)
    
    conn = sqlite3.connect(CACHE_DB)
    cursor = conn.cursor()
    
    # Find all cached regions
    cursor.execute("""
        SELECT min_lat, max_lat, min_lng, max_lng, data_json 
        FROM cached_regions
        ORDER BY (max_lat - min_lat) * (max_lng - min_lng) ASC
    """)
    
    regions = cursor.fetchall()
    conn.close()
    
    # Find the smallest cached region that contains our bbox
    for cached_min_lat, cached_max_lat, cached_min_lng, cached_max_lng, data_json in regions:
        if _bbox_contains_bbox(cached_min_lat, cached_max_lat, cached_min_lng, cached_max_lng,
                               min_lat, max_lat, min_lng, max_lng):
            try:
                data = json.loads(data_json)
                nodes = data['nodes']
                ways = data['ways']
                
                # Filter nodes and ways to only those within the requested bbox
                filtered_nodes = _filter_nodes_by_bbox(nodes, min_lat, max_lat, min_lng, max_lng)
                filtered_ways = _filter_ways_by_bbox(ways, filtered_nodes)
                
                logger.info(
                    f"Spatial cache hit: Found cached region "
                    f"({cached_min_lat},{cached_min_lng},{cached_max_lat},{cached_max_lng}) "
                    f"containing requested bbox ({min_lat},{min_lng},{max_lat},{max_lng})"
                )
                
                return filtered_nodes, filtered_ways
            except (json.JSONDecodeError, KeyError) as e:
                logger.error(f"Failed to process cached data: {e}")
                continue
    
    return None, None


def _filter_nodes_by_bbox(nodes_dict, min_lat, max_lat, min_lng, max_lng):
    """
    Filter nodes to only include those within a bounding box.
    
    Args:
        nodes_dict: Dictionary of {node_id: (lat, lon)}
        min_lat, max_lat, min_lng, max_lng: Bounding box
    
    Returns:
        Filtered dictionary of nodes within bbox
    """
    filtered = {}
    for node_id, (lat, lon) in nodes_dict.items():
        if min_lat <= lat <= max_lat and min_lng <= lon <= max_lng:
            filtered[node_id] = (lat, lon)
    return filtered


def _filter_ways_by_bbox(ways_list, valid_nodes_dict):
    """
    Filter ways to only include those that reference filtered nodes.
    
    Removes ways where node references don't exist in filtered nodes.
    Also splits ways into contiguous segments of valid nodes.
    
    Args:
        ways_list: List of way dictionaries
        valid_nodes_dict: Dictionary of filtered nodes
    
    Returns:
        Filtered list of way dictionaries
    """
    valid_node_ids = set(valid_nodes_dict.keys())
    filtered_ways = []
    
    for way in ways_list:
        node_refs = way['nodes']
        
        # Split into contiguous segments of valid nodes
        current_segment = []
        for node_id in node_refs:
            # Convert to string for comparison (nodes_dict uses string keys)
            node_id_str = str(node_id)
            if node_id_str in valid_node_ids or node_id in valid_node_ids:
                current_segment.append(node_id)
            else:
                if len(current_segment) >= 2:
                    filtered_ways.append({
                        'name': way['name'],
                        'nodes': current_segment,
                        'highway': way['highway']
                    })
                current_segment = []
        
        # Add final segment if it has at least 2 nodes
        if len(current_segment) >= 2:
            filtered_ways.append({
                'name': way['name'],
                'nodes': current_segment,
                'highway': way['highway']
            })
    
    return filtered_ways


def save_to_cache(min_lat, max_lat, min_lng, max_lng, nodes, ways):
    """
    Save OSM data to cache for a bounding box.
    
    Args:
        min_lat, max_lat, min_lng, max_lng: Bounding box coordinates
        nodes: Dictionary of {node_id: (lat, lon)}
        ways: List of way dictionaries
    """
    initialize_cache_db()
    
    min_lat, max_lat, min_lng, max_lng = _normalize_bbox(min_lat, max_lat, min_lng, max_lng)
    
    # Convert node IDs to strings for JSON serialization
    nodes_serializable = {str(k): v for k, v in nodes.items()}
    data = {
        'nodes': nodes_serializable,
        'ways': ways
    }
    data_json = json.dumps(data)
    
    conn = sqlite3.connect(CACHE_DB)
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            INSERT OR REPLACE INTO cached_regions 
            (min_lat, max_lat, min_lng, max_lng, data_json)
            VALUES (?, ?, ?, ?, ?)
        """, (min_lat, max_lat, min_lng, max_lng, data_json))
        
        conn.commit()
        logger.info(
            f"Cached OSM data for bbox ({min_lat},{min_lng},{max_lat},{max_lng}) - "
            f"{len(nodes)} nodes, {len(ways)} ways"
        )
    except sqlite3.Error as e:
        logger.error(f"Failed to cache data: {e}")
    finally:
        conn.close()


def get_cache_stats():
    """Get statistics about the cache."""
    initialize_cache_db()
    
    conn = sqlite3.connect(CACHE_DB)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM cached_regions")
    region_count = cursor.fetchone()[0]
    
    cursor.execute("""
        SELECT 
            COUNT(*),
            SUM(LENGTH(data_json)) as total_size
        FROM cached_regions
    """)
    
    count, total_bytes = cursor.fetchone()
    conn.close()
    
    total_mb = total_bytes / (1024 * 1024) if total_bytes else 0
    
    return {
        'cached_regions': region_count,
        'total_cache_size_mb': round(total_mb, 2),
        'cache_file': str(CACHE_DB)
    }


def clear_cache():
    """Clear all cached data."""
    initialize_cache_db()
    
    conn = sqlite3.connect(CACHE_DB)
    cursor = conn.cursor()
    
    cursor.execute("DELETE FROM cached_regions")
    conn.commit()
    conn.close()
    
    logger.info("Cache cleared")


def list_cached_regions():
    """List all cached bounding boxes."""
    initialize_cache_db()
    
    conn = sqlite3.connect(CACHE_DB)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT min_lat, max_lat, min_lng, max_lng, 
               LENGTH(data_json) as size_bytes, cached_at
        FROM cached_regions
        ORDER BY cached_at DESC
    """)
    
    regions = cursor.fetchall()
    conn.close()
    
    return regions


def cache_cleanup(max_age_days=30):
    """
    Remove cached regions older than max_age_days.
    
    Args:
        max_age_days: Remove entries older than this many days
    """
    initialize_cache_db()
    
    conn = sqlite3.connect(CACHE_DB)
    cursor = conn.cursor()
    
    cursor.execute("""
        DELETE FROM cached_regions 
        WHERE cached_at < datetime('now', ? || ' days')
    """, (f"-{max_age_days}",))
    
    deleted = cursor.rowcount
    conn.commit()
    conn.close()
    
    logger.info(f"Cleaned up {deleted} cached regions older than {max_age_days} days")
    
    return deleted
