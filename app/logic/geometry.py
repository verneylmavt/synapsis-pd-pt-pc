from __future__ import annotations

from typing import Iterable, Tuple

# Small epsilon value used as numerical tolerance.
# This helps to avoid floating-point rounding errors when comparing equality of real numbers.
EPS = 1e-12


def _point_on_segment(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> bool:
    """
    Check whether a given point P(px, py) lies exactly on the line segment A(ax, ay) -> B(bx, by).
    Returns True if the point is on the segment (within tolerance).
    
    Important:
    - For this application, points on the boundary are treated as OUTSIDE of polygons.
    Therefore, this function helps detect such boundary points so that the main
    point_in_polygon() function can exclude them.
    """

    # ---------------------------------------------------------------------
    # Step 1: Check if the three points A, B, and P are colinear.
    # ---------------------------------------------------------------------
    # The cross product of vectors AB and AP gives twice the area of the triangle ABP.
    # If the absolute value of this area is approximately zero, the points are colinear.
    #
    # cross = (bx - ax)*(py - ay) - (by - ay)*(px - ax)
    # When cross ≈ 0, P lies on the line passing through A and B.
    cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
    if abs(cross) > EPS:
        # If cross product is not small enough, P is not on the line AB.
        return False

    # ---------------------------------------------------------------------
    # Step 2: Check if P lies within the bounding box of A and B.
    # ---------------------------------------------------------------------
    # Even if P is on the same line, it might be beyond the endpoints.
    # We therefore check if P’s coordinates are within the min/max range of A and B.
    #
    # A small tolerance (EPS) is used to allow for minor floating-point variations.
    minx, maxx = (ax, bx) if ax <= bx else (bx, ax)
    miny, maxy = (ay, by) if ay <= by else (by, ay)
    return (
        (px >= minx - EPS)
        and (px <= maxx + EPS)
        and (py >= miny - EPS)
        and (py <= maxy + EPS)
    )


def point_in_polygon(pt: Tuple[float, float], polygon: Iterable[dict]) -> bool:
    """
    Determine whether a 2D point lies inside a polygon.

    Algorithm:
        - Uses the "ray casting" or "even-odd rule" method.
        - A horizontal ray is projected to the right from the test point.
        - The number of times this ray intersects the polygon’s edges is counted.
        * If the count is odd → the point is inside.
        * If the count is even → the point is outside.

    Special handling:
        - If the point lies exactly on any polygon edge or vertex, it is treated as OUTSIDE.
        (This avoids ambiguity in counting applications such as people detection zones.)
    
    Input format:
        - pt: a tuple (x, y) representing the point to test.
        - polygon: an iterable of dictionaries like {"x": float, "y": float}.
        Coordinates can be normalized (0–1) or in any consistent scale.
    """

    # Extract the test point coordinates
    x, y = pt

    # Convert polygon vertex dictionaries into a list of (x, y) tuples.
    coords = [(p["x"], p["y"]) for p in polygon]
    n = len(coords)

    # Polygons must have at least 3 vertices to be valid.
    if n < 3:
        return False

    # ---------------------------------------------------------------------
    # Step 1: Check for boundary cases.
    # ---------------------------------------------------------------------
    # If the point lies exactly on any edge of the polygon, treat it as OUTSIDE.
    for i in range(n):
        # Get the coordinates of two consecutive vertices forming an edge.
        x1, y1 = coords[i]
        x2, y2 = coords[(i + 1) % n]  # wrap around to close the polygon

        # If the point lies on this segment, return False (on boundary)
        if _point_on_segment(x, y, x1, y1, x2, y2):
            return False

    # ---------------------------------------------------------------------
    # Step 2: Apply the ray-casting method.
    # ---------------------------------------------------------------------
    # We cast a horizontal ray from the point (x, y) towards +∞ in the x direction.
    # For each polygon edge, we check if the ray crosses that edge.
    # The parity (odd/even) of the number of crossings determines inside/outside.
    inside = False
    j = n - 1  # j trails i (previous vertex)

    for i in range(n):
        xi, yi = coords[i]   # current vertex
        xj, yj = coords[j]   # previous vertex

        # Check whether the y-coordinate of the point lies between yi and yj.
        # Using strict inequality on one side avoids counting vertices twice.
        yi_gt = yi > y
        yj_gt = yj > y

        # Edge crosses the horizontal ray if:
        #   (a) y is between yi and yj (one above, one below the ray)
        #   (b) The x-coordinate of the intersection is to the right of the point.
        intersects = (yi_gt != yj_gt) and (
            x < (xj - xi) * (y - yi) / (yj - yi + EPS) + xi
        )

        # Every time a crossing is found, toggle the 'inside' flag.
        if intersects:
            inside = not inside

        # Move to the next edge
        j = i

    # After checking all edges:
    # - inside = True  → point is inside polygon
    # - inside = False → point is outside polygon
    return inside