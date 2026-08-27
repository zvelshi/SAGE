# default
from __future__ import annotations

# third-party
import numpy as np

class Point:
    """A point (or free vector) in 3-D space.

    A thin wrapper around a length-3 numpy array that carries the spatial-geometry
    helpers used across SAGE (distances, projections, simple vector algebra). All
    coordinates are in millimetres, body frame (X longitudinal, Y lateral, Z up),
    matching the rest of the codebase.
    """

    __slots__ = ("_a",)

    def __init__(self, x, y=None, z=None):
        if isinstance(x, Point):
            self._a = x._a.copy()
        elif y is None and z is None:
            self._a = np.asarray(x, dtype=float).reshape(3)
        else:
            self._a = np.array([x, y, z], dtype=float)

    # --- constructors -----------------------------------------------------
    @classmethod
    def origin(cls) -> "Point":
        return cls(0.0, 0.0, 0.0)

    @classmethod
    def from_array(cls, arr) -> "Point":
        return cls(arr)

    # --- accessors ------------------------------------------------------- -
    @property
    def x(self) -> float:
        return float(self._a[0])

    @property
    def y(self) -> float:
        return float(self._a[1])

    @property
    def z(self) -> float:
        return float(self._a[2])

    @property
    def array(self) -> np.ndarray:
        """A copy of the underlying (3,) array."""
        return self._a.copy()

    def to_np(self) -> np.ndarray:
        """A copy of the coordinates as a length-3 numpy array."""
        return self._a.copy()

    to_array = to_np

    def to_list(self) -> list[float]:
        return [self.x, self.y, self.z]

    def to_tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def to_3d(self, scene, radius_mm: float = 10.0, color: str = "#4466bb",
              opacity: float = 1.0, scale: float = 1.0 / 1000.0):
        """Default 3-D representation: a small sphere at this point."""
        from utils.spatial.shapes.sphere import Sphere
        return Sphere(self, radius_mm, color, opacity).to_3d(scene, scale)

    # --- dunder ----------------------------------------------------------
    def __iter__(self):
        return iter(self._a)

    def __getitem__(self, i):
        v = self._a[i]
        return float(v) if np.ndim(v) == 0 else v

    def __len__(self) -> int:
        return 3

    def __repr__(self) -> str:
        return f"Point({self.x:.4g}, {self.y:.4g}, {self.z:.4g})"

    def __eq__(self, other) -> bool:
        try:
            return bool(np.allclose(self._a, Point(other)._a))
        except Exception:
            return NotImplemented

    __hash__ = None

    def __add__(self, other) -> "Point":
        return Point(self._a + Point(other)._a)

    def __sub__(self, other) -> "Point":
        return Point(self._a - Point(other)._a)

    def __mul__(self, s: float) -> "Point":
        return Point(self._a * float(s))

    __rmul__ = __mul__

    def __truediv__(self, s: float) -> "Point":
        return Point(self._a / float(s))

    def __neg__(self) -> "Point":
        return Point(-self._a)

    # --- vector algebra (Point treated as a vector from the origin) ------
    @property
    def norm(self) -> float:
        return float(np.linalg.norm(self._a))

    def unit(self) -> "Point":
        n = self.norm
        if n < 1e-12:
            raise ValueError("cannot normalise a zero-length vector")
        return Point(self._a / n)

    def dot(self, other) -> float:
        return float(np.dot(self._a, Point(other)._a))

    def cross(self, other) -> "Point":
        return Point(np.cross(self._a, Point(other)._a))

    # --- spatial helpers -----------------------------------------------
    def distance_to(self, other) -> float:
        """Euclidean distance to another Point, or perpendicular distance to a
        Line / Plane (anything exposing ``distance_to_point``)."""
        if isinstance(other, Point):
            return float(np.linalg.norm(self._a - other._a))
        return float(other.distance_to_point(self))

    def midpoint_to(self, other) -> "Point":
        return Point((self._a + Point(other)._a) / 2.0)

    def translated(self, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> "Point":
        return Point(self.x + dx, self.y + dy, self.z + dz)

    def project_onto_plane(self, plane) -> "Point":
        return plane.project_point(self)

    def project_onto_line(self, line) -> "Point":
        return line.closest_point(self)

    def is_finite(self) -> bool:
        return bool(np.all(np.isfinite(self._a)))


def centroid(points) -> Point:
    """Arithmetic mean of an iterable of points."""
    arr = np.array([Point(p).array for p in points], dtype=float)
    if not len(arr):
        raise ValueError("centroid of an empty point set")
    return Point(arr.mean(axis=0))
