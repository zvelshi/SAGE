# default
from __future__ import annotations

# third-party
import numpy as np

# ours
from utils.spatial.primitives.point import Point
from utils.spatial.primitives.line import Line

class Segment:
    """A finite line segment between two points (mm, body frame)."""

    __slots__ = ("start", "end")

    def __init__(self, start, end):
        self.start = Point(start)
        self.end = Point(end)

    # --- accessors ------------------------------------------------------- -
    @property
    def vector(self) -> Point:
        return self.end - self.start

    @property
    def length(self) -> float:
        return self.vector.norm

    @property
    def midpoint(self) -> Point:
        return self.start.midpoint_to(self.end)

    @property
    def direction(self) -> Point:
        v = self.vector
        return v.unit() if v.norm > 1e-12 else Point(0.0, 1.0, 0.0)

    def as_line(self) -> Line:
        return Line(self.start, self.direction)

    def point_at(self, frac: float) -> Point:
        """Point a fraction ``frac`` (0 -> start, 1 -> end) along the segment."""
        return Point(self.start.array + self.vector.array * float(frac))

    # --- distances --------------------------------------------------- -----
    def closest_point(self, p) -> Point:
        p = Point(p)
        v = self.vector
        L2 = v.dot(v)
        if L2 < 1e-18:
            return self.start
        t = float(np.clip((p - self.start).dot(v) / L2, 0.0, 1.0))
        return self.point_at(t)

    def distance_to_point(self, p) -> float:
        return Point(p).distance_to(self.closest_point(p))

    def distance_to_segment(self, other: "Segment") -> float:
        """Closest distance between this segment and ``other`` (3-D)."""
        p1, q1 = self.start.array, self.end.array
        p2, q2 = other.start.array, other.end.array
        d1, d2, r = q1 - p1, q2 - p2, p1 - p2
        a, e, f = float(d1 @ d1), float(d2 @ d2), float(d2 @ r)
        if a <= 1e-12 and e <= 1e-12:
            return float(np.linalg.norm(r))
        if a <= 1e-12:
            s, t = 0.0, np.clip(f / e, 0.0, 1.0)
        else:
            c = float(d1 @ r)
            if e <= 1e-12:
                t, s = 0.0, np.clip(-c / a, 0.0, 1.0)
            else:
                b = float(d1 @ d2)
                denom = a * e - b * b
                s = np.clip((b * f - c * e) / denom, 0.0, 1.0) if denom > 1e-12 else 0.0
                t = (b * s + f) / e
                if t < 0.0:
                    t, s = 0.0, np.clip(-c / a, 0.0, 1.0)
                elif t > 1.0:
                    t, s = 1.0, np.clip((b - c) / a, 0.0, 1.0)
        return float(np.linalg.norm((p1 + s * d1) - (p2 + t * d2)))

    def __repr__(self) -> str:
        return f"Segment({self.start!r} -> {self.end!r})"
