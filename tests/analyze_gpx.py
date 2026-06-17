"""
Diagnose a route GPX exported by the app: count geometry artifacts that cause
navigation apps (OsmAnd etc.) to insert spurious U-turns when recalculating a
route between the track points.

Usage:
    python tests/analyze_gpx.py path/to/route.gpx

Reports:
  - consecutive duplicate points (degenerate via-points)
  - exact revisited coordinates (intended coverage re-drives + artifacts)
  - out-and-back "spikes" (go far out, return next to where you started)
  - near-U-turn vertices (sharp >150 deg reversals)

High spike / near-U-turn counts that DON'T correspond to intentional coverage
re-drives point at geometry bugs (e.g. mis-oriented edge geometry). Some U-turns
are inherent to a coverage route and unavoidable under "Calculate route between
points"; use this to separate the two.
"""
import math
import re
import sys
from collections import Counter


def _meters(a, b):
    dlat = (a[0] - b[0]) * 111000
    dlon = (a[1] - b[1]) * 111000 * math.cos(math.radians(a[0]))
    return math.hypot(dlat, dlon)


def _angle(p0, p1, p2):
    v1 = (p1[0] - p0[0], p1[1] - p0[1])
    v2 = (p2[0] - p1[0], p2[1] - p1[1])
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 == 0 or n2 == 0:
        return None
    dot = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
    return math.degrees(math.acos(dot))


def analyze(path):
    text = open(path).read()
    pts = [(float(la), float(lo))
           for la, lo in re.findall(r'<trkpt lat="([-\d.]+)" lon="([-\d.]+)"', text)]
    n = len(pts)
    dups = sum(1 for i in range(1, n) if pts[i] == pts[i - 1])
    spikes = sum(1 for i in range(1, n - 1)
                 if _meters(pts[i - 1], pts[i + 1]) < 15 and _meters(pts[i - 1], pts[i]) > 40)
    counts = Counter(pts)
    revisits = sum(v - 1 for v in counts.values() if v > 1)
    uturns = sum(1 for i in range(1, n - 1)
                 if (_angle(pts[i - 1], pts[i], pts[i + 1]) or 0) > 150
                 and _meters(pts[i - 1], pts[i]) > 20)
    print(f"file: {path}")
    print(f"  trkpt total ............. {n}  (unique {len(counts)})")
    print(f"  consecutive duplicates .. {dups}")
    print(f"  revisited coords (extra)  {revisits}")
    print(f"  out-and-back spikes ..... {spikes}")
    print(f"  near-U-turn vertices .... {uturns}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    analyze(sys.argv[1])
