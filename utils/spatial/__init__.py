"""3-D spatial geometry for SAGE (millimetres, body frame: X longitudinal,
Y lateral, Z up).

- ``primitives``: analytic geometry -- ``Point`` / ``Line`` / ``Plane``
- ``shapes``: drawable 3-D shapes -- ``Sphere`` / ``Cylinder`` / ``Cuboid`` / ``DashedLine``
"""

from utils.spatial.primitives import Point, Line, Plane, centroid
from utils.spatial.shapes import (
    SCENE_SCALE, align_y_to_direction, Sphere, Cylinder, Cuboid, DashedLine,
)

__all__ = [
    "Point", "Line", "Plane", "centroid",
    "SCENE_SCALE", "align_y_to_direction", "Sphere", "Cylinder", "Cuboid", "DashedLine",
]
