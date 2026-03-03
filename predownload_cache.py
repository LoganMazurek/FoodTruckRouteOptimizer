#!/usr/bin/env python3
"""
Pre-download utility for caching OSM street data.

This script allows you to pre-download and cache street data for specific
areas without waiting for real-time API calls. Useful for:
- Caching common service areas before deployment
- Batch downloading multiple regions
- Reducing reliance on Overpass API

Usage:
    python predownload_cache.py --zipcode 60601
    python predownload_cache.py --bbox 41.8 41.9 -87.7 -87.6
    python predownload_cache.py --file areas.json
    python predownload_cache.py --stats
    python predownload_cache.py --clear

Example areas.json:
{
    "areas": [
        {"name": "Chicago", "zipcode": "60601"},
        {"name": "Downtown Chicago", "bbox": [41.8, 41.8, -87.7, -87.6]},
        {"name": "Atlanta", "zipcode": "30303"}
    ]
}
"""

import argparse
import json
import sys
import logging
from typing import List, Tuple, Dict, Optional
from get_street_data import fetch_overpass_data, get_coordinates
from osm_cache import get_cache_stats, list_cached_regions, clear_cache, cache_cleanup

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_areas_from_file(filepath: str) -> List[Dict]:
    """Load area definitions from JSON file."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
            return data.get('areas', [])
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        sys.exit(1)
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in {filepath}")
        sys.exit(1)


def zipcode_to_bbox(zipcode: str, radius_miles: float = 5) -> Tuple[float, float, float, float]:
    """
    Convert a ZIP code to a bounding box.
    
    Args:
        zipcode: US ZIP code
        radius_miles: Radius around the ZIP code center
    
    Returns:
        (min_lat, max_lat, min_lng, max_lng) bounding box
    """
    lat, lng = get_coordinates(zipcode)
    if lat is None or lng is None:
        raise ValueError(f"Could not geocode ZIP code: {zipcode}")
    
    # Approximate degrees per mile (varies by latitude, using ~69 miles per degree)
    degrees_per_mile = 1 / 69
    offset = radius_miles * degrees_per_mile
    
    return (lat - offset, lat + offset, lng - offset, lng + offset)


def download_and_cache_area(min_lat: float, max_lat: float, min_lng: float, max_lng: float, 
                           area_name: Optional[str] = None) -> bool:
    """
    Download and cache street data for a specific area.
    
    Args:
        min_lat, max_lat, min_lng, max_lng: Bounding box
        area_name: Human-readable name for logging
    
    Returns:
        True if successful, False otherwise
    """
    name_str = f" ({area_name})" if area_name else ""
    try:
        logger.info(f"Downloading{name_str} bbox ({min_lat:.2f}, {min_lng:.2f}, {max_lat:.2f}, {max_lng:.2f})")
        result = fetch_overpass_data(min_lat, max_lat, min_lng, max_lng, use_cache=True)
        
        # Count the results
        node_count = len(result.nodes) if hasattr(result.nodes, '__len__') else 0
        way_count = len(result.ways) if hasattr(result.ways, '__len__') else 0
        
        logger.info(f"✓ Cached {way_count} streets with {node_count} nodes{name_str}")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to cache{name_str}: {e}")
        return False


def download_from_zipcodes(zipcodes: List[str], radius_miles: float = 5) -> None:
    """Download and cache data for multiple ZIP codes."""
    total = len(zipcodes)
    successful = 0
    
    for i, zipcode in enumerate(zipcodes, 1):
        logger.info(f"[{i}/{total}] Processing ZIP code {zipcode}")
        try:
            min_lat, max_lat, min_lng, max_lng = zipcode_to_bbox(zipcode, radius_miles)
            if download_and_cache_area(min_lat, max_lat, min_lng, max_lng, f"ZIP {zipcode}"):
                successful += 1
        except Exception as e:
            logger.error(f"Error processing {zipcode}: {e}")
    
    logger.info(f"\n✓ Successfully cached {successful}/{total} ZIP code areas")


def download_from_bboxes(bboxes: List[Tuple[float, float, float, float]]) -> None:
    """Download and cache data for multiple bounding boxes."""
    total = len(bboxes)
    successful = 0
    
    for i, (min_lat, max_lat, min_lng, max_lng) in enumerate(bboxes, 1):
        logger.info(f"[{i}/{total}] Processing bounding box")
        if download_and_cache_area(min_lat, max_lat, min_lng, max_lng):
            successful += 1
    
    logger.info(f"\n✓ Successfully cached {successful}/{total} areas")


def display_cache_stats() -> None:
    """Display cache statistics."""
    stats = get_cache_stats()
    regions = list_cached_regions()
    
    print("\n" + "="*60)
    print("CACHE STATISTICS")
    print("="*60)
    print(f"Cached Regions: {stats['cached_regions']}")
    print(f"Total Cache Size: {stats['total_cache_size_mb']} MB")
    print(f"Cache Location: {stats['cache_file']}")
    
    if regions:
        print("\nCached Bounding Boxes:")
        print("-"*60)
        for region in regions:
            min_lat, max_lat, min_lng, max_lng, size_bytes, cached_at = region
            size_mb = size_bytes / (1024 * 1024)
            print(f"  Bbox: ({min_lat:.2f}, {min_lng:.2f}, {max_lat:.2f}, {max_lng:.2f})")
            print(f"    Size: {size_mb:.2f} MB | Cached: {cached_at}")
    
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Pre-download and cache OSM street data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--zipcode', '-z', type=str, 
                       help='ZIP code to cache')
    parser.add_argument('--zipcodes', '-Z', type=str, nargs='+',
                       help='Multiple ZIP codes to cache')
    parser.add_argument('--bbox', '-b', type=float, nargs=4, metavar=('MIN_LAT', 'MAX_LAT', 'MIN_LNG', 'MAX_LNG'),
                       help='Bounding box (min_lat, max_lat, min_lng, max_lng)')
    parser.add_argument('--file', '-f', type=str,
                       help='JSON file with area definitions')
    parser.add_argument('--radius', '-r', type=float, default=5,
                       help='Radius in miles around ZIP code (default: 5)')
    parser.add_argument('--stats', '-s', action='store_true',
                       help='Display cache statistics')
    parser.add_argument('--list', '-l', action='store_true',
                       help='List all cached regions')
    parser.add_argument('--clear', action='store_true',
                       help='Clear all cached data')
    parser.add_argument('--cleanup', type=int, metavar='DAYS',
                       help='Remove cached data older than N days')
    
    args = parser.parse_args()
    
    # Handle display-only commands
    if args.stats:
        display_cache_stats()
        return
    
    if args.list:
        regions = list_cached_regions()
        if regions:
            print("\nCached Regions:")
            for region in regions:
                min_lat, max_lat, min_lng, max_lng, size_bytes, cached_at = region
                size_mb = size_bytes / (1024 * 1024)
                print(f"  ({min_lat:.2f}, {min_lng:.2f}, {max_lat:.2f}, {max_lng:.2f}) "
                      f"- {size_mb:.2f}MB - {cached_at}")
        else:
            print("No cached regions found.")
        return
    
    if args.clear:
        if input("Are you sure you want to clear all cached data? (yes/no): ").lower() == 'yes':
            clear_cache()
            print("Cache cleared.")
        return
    
    if args.cleanup:
        deleted = cache_cleanup(args.cleanup)
        print(f"Deleted {deleted} cached regions older than {args.cleanup} days.")
        return
    
    # Handle download commands
    if args.zipcode:
        logger.info(f"Downloading area for ZIP code {args.zipcode}")
        try:
            min_lat, max_lat, min_lng, max_lng = zipcode_to_bbox(args.zipcode, args.radius)
            download_and_cache_area(min_lat, max_lat, min_lng, max_lng, f"ZIP {args.zipcode}")
            display_cache_stats()
        except Exception as e:
            logger.error(f"Error: {e}")
            sys.exit(1)
    
    elif args.zipcodes:
        logger.info(f"Downloading areas for {len(args.zipcodes)} ZIP codes")
        download_from_zipcodes(args.zipcodes, args.radius)
        display_cache_stats()
    
    elif args.bbox:
        logger.info("Downloading area for bounding box")
        download_and_cache_area(args.bbox[0], args.bbox[1], args.bbox[2], args.bbox[3])
        display_cache_stats()
    
    elif args.file:
        logger.info(f"Loading areas from {args.file}")
        areas = load_areas_from_file(args.file)
        
        total = len(areas)
        successful = 0
        
        for i, area in enumerate(areas, 1):
            name = area.get('name', f"Area {i}")
            logger.info(f"[{i}/{total}] Processing {name}")
            
            try:
                if 'bbox' in area:
                    bbox = area['bbox']
                    if download_and_cache_area(bbox[0], bbox[1], bbox[2], bbox[3], name):
                        successful += 1
                elif 'zipcode' in area:
                    min_lat, max_lat, min_lng, max_lng = zipcode_to_bbox(area['zipcode'], args.radius)
                    if download_and_cache_area(min_lat, max_lat, min_lng, max_lng, name):
                        successful += 1
            except Exception as e:
                logger.error(f"Error processing {name}: {e}")
        
        logger.info(f"\n✓ Successfully cached {successful}/{total} areas")
        display_cache_stats()
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
