# default
from __future__ import annotations

# third-party
import numpy as np

# ours
from utils.spatial.primitives.point import Point

class Line:
    """An infinite line in 3-D, stored as a base ``point`` and a unit ``direction``."""

    __slots__ = ("point", "direction")

    def __init__(self, point, direction):
        self.point = Point(point)
        d = Point(direction)
        if d.norm < 1e-12:
            raise ValueError("Line direction must be non-zero")
        self.direction = d.unit()

    # --- constructors -------------------------------------------------- --
    @classmethod
    def from_points(cls, a, b) -> "Line":
        a, b = Point(a), Point(b)
        return cls(a, b - a)

    @classmethod
    def vertical_through(cls, x: float, y: float) -> "Line":
        """The vertical (global +Z) line passing through (x, y)."""
        return cls(Point(x, y, 0.0), Point(0.0, 0.0, 1.0))

    # --- queries ------------------------------------------------------ ---
    def point_at(self, t: float) -> Point:
        return Point(self.point.array + self.direction.array * float(t))

    def parameter_of(self, p) -> float:
        """The ``t`` such that ``point_at(t)`` is the foot of the perpendicular
        from ``p`` onto the line."""
        return (Point(p) - self.point).dot(self.direction)

    def closest_point(self, p) -> Point:
        return self.point_at(self.parameter_of(p))

    def distance_to_point(self, p) -> float:
        p = Point(p)
        return p.distance_to(self.closest_point(p))

    def contains(self, p, tol: float = 1e-6) -> bool:
        return self.distance_to_point(p) < tol

    def intersect_plane(self, plane) -> Point | None:
        return plane.intersect_line(self)

    def angle_to(self, other: "Line", degrees: bool = True) -> float:
        """Acute angle (0-90) between the two line directions."""
        d = abs(float(np.clip(self.direction.dot(other.direction), -1.0, 1.0)))
        ang = float(np.arccos(d))
        return float(np.degrees(ang)) if degrees else ang

    def segment(self, length_mm: float, center=None) -> tuple[Point, Point]:
        """The two endpoints of a segment of ``length_mm`` centered on ``center``
        (default: the line's base point), lying on the line."""
        c = self.closest_point(center) if center is not None else self.point
        half = self.direction * (float(length_mm) / 2.0)
        return c - half, c + half

    def to_3d(self, scene, length_mm: float = 1000.0, center=None,
              radius_mm: float = 2.0, color: str = "#888888", opacity: float = 1.0,
              scale: float = 1.0 / 1000.0):
        """Default 3-D representation: a thin cylinder of ``length_mm`` centered on
        ``center`` running along the line."""
        from utils.spatial.shapes.cylinder import Cylinder
        a, b = self.segment(length_mm, center)
        return Cylinder(a, b, radius_mm, color, opacity).to_3d(scene, scale)

    def __repr__(self) -> str:
        return f"Line(point={self.point!r}, direction={self.direction!r})"
