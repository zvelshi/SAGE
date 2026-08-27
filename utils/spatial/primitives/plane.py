# default
from __future__ import annotations

# third-party
import numpy as np

# ours
from utils.spatial.primitives.point import Point
from utils.spatial.primitives.line import Line

class Plane:
    """An infinite plane in 3-D, stored as a base ``point`` and a unit ``normal``.

    Implicit form: ``normal . X == offset`` where ``offset = normal . point``.
    """

    __slots__ = ("point", "normal")

    def __init__(self, point, normal):
        self.point = Point(point)
        n = Point(normal)
        if n.norm < 1e-12:
            raise ValueError("Plane normal must be non-zero")
        self.normal = n.unit()

    # --- constructors -------------------------------------------------- --
    @classmethod
    def from_points(cls, a, b, c) -> "Plane":
        a, b, c = Point(a), Point(b), Point(c)
        normal = (b - a).cross(c - a)
        if normal.norm < 1e-12:
            raise ValueError("the three points are collinear")
        return cls(a, normal)

    @classmethod
    def from_points_and_direction(cls, a, b, direction) -> "Plane":
        """Plane containing points ``a`` and ``b`` and parallel to ``direction``
        -- i.e. spanned by ``(b - a)`` and ``direction``. Anchored at ``a``."""
        a, b = Point(a), Point(b)
        normal = (b - a).cross(direction)
        if normal.norm < 1e-12:
            raise ValueError("(b - a) is parallel to direction; plane is undefined")
        return cls(a, normal)

    @classmethod
    def horizontal_through(cls, point) -> "Plane":
        """Plane parallel to the global X-Y plane passing through ``point``."""
        return cls(Point(point), Point(0.0, 0.0, 1.0))

    @classmethod
    def best_fit(cls, points) -> "Plane":
        """Least-squares best-fit plane through >= 3 finite points. The normal is
        oriented +Z-up so ``z_at`` / signed distances are consistent."""
        pts = np.array([Point(p).array for p in points], dtype=float)
        pts = pts[np.all(np.isfinite(pts), axis=1)]
        if len(pts) < 3:
            raise ValueError("best_fit needs at least 3 finite points")
        c = pts.mean(axis=0)
        _, _, vh = np.linalg.svd(pts - c)
        normal = vh[-1]
        if normal[2] < 0:
            normal = -normal
        return cls(Point(c), Point(normal))

    # --- implicit form ----------------------------------------------- ----
    @property
    def offset(self) -> float:
        return self.normal.dot(self.point)

    def signed_distance(self, p) -> float:
        """Distance from ``p`` to the plane, positive on the side the normal points to."""
        return self.normal.dot(Point(p)) - self.offset

    def distance_to_point(self, p) -> float:
        return abs(self.signed_distance(p))

    def contains(self, p, tol: float = 1e-6) -> bool:
        return abs(self.signed_distance(p)) < tol

    def project_point(self, p) -> Point:
        p = Point(p)
        return Point(p.array - self.signed_distance(p) * self.normal.array)

    # --- height query (non-vertical planes) --------------------------- ---
    def z_at(self, x: float, y: float) -> float:
        """Z of the plane directly above/below (x, y). Raises for a vertical plane."""
        nz = self.normal.z
        if abs(nz) < 1e-12:
            raise ValueError("plane is vertical; Z is undefined over (x, y)")
        return self.point.z - (self.normal.x * (x - self.point.x)
                               + self.normal.y * (y - self.point.y)) / nz

    # --- constructions --------------------------------------------- ------
    def offset_by(self, distance: float) -> "Plane":
        """A parallel plane shifted ``distance`` along this plane's own normal."""
        return Plane(Point(self.point.array + self.normal.array * float(distance)),
                     self.normal)

    def translated(self, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> "Plane":
        return Plane(self.point.translated(dx, dy, dz), self.normal)

    def intersect_line(self, line: Line) -> Point | None:
        denom = self.normal.dot(line.direction)
        if abs(denom) < 1e-12:
            return None
        t = (self.offset - self.normal.dot(line.point)) / denom
        return line.point_at(t)

    def intersect_plane(self, other: "Plane") -> Line | None:
        d = self.normal.cross(other.normal)
        if d.norm < 1e-12:
            return None
        A = np.array([self.normal.array, other.normal.array, d.array])
        b = np.array([self.offset, other.offset, 0.0])
        try:
            p = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return None
        return Line(Point(p), d)

    # --- angles ---------------------------------------------------- ------
    def angle_to(self, other: "Plane", degrees: bool = True) -> float:
        """Dihedral angle (0-90) between the two planes."""
        d = abs(float(np.clip(self.normal.dot(other.normal), -1.0, 1.0)))
        ang = float(np.arccos(d))
        return float(np.degrees(ang)) if degrees else ang

    def sagittal_angle_to(self, other: "Plane", degrees: bool = True) -> float:
        """Signed angle between the two planes' traces in the sagittal (X-Z)
        plane (Y = 0). Positive when ``other`` tilts nose-down relative to
        ``self`` -- its +X edge sits lower. This is the rake between the planes
        as seen from the side of the car."""
        def slope(plane: "Plane") -> float:
            n = plane.normal
            if abs(n.z) < 1e-12:
                return 0.0
            return float(np.arctan2(-n.x, n.z))
        ang = slope(other) - slope(self)
        return float(np.degrees(ang)) if degrees else ang

    def __repr__(self) -> str:
        return f"Plane(point={self.point!r}, normal={self.normal!r})"
