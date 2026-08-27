"""Abstract geometric primitives -- the analytic Point / Line / Segment / Plane
and their operations (distances, projections, intersections, angles)."""

from utils.spatial.primitives.point import Point, centroid
from utils.spatial.primitives.line import Line
from utils.spatial.primitives.segment import Segment
from utils.spatial.primitives.plane import Plane

__all__ = ["Point", "Line", "Segment", "Plane", "centroid"]
