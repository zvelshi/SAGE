"""3-D spatial geometry for SAGE (millimetres, body frame: X longitudinal,
Y lateral, Z up).

- ``primitives``: analytic geometry -- ``Point`` / ``Line`` / ``Segment`` / ``Plane``
- ``shapes``: drawable 3-D shapes -- primitive ``Sphere`` / ``Cylinder`` /
  ``Cuboid`` / ``DashedLine`` and the suspension components built from them
  (``Shock``, ``Axle``, ``Wheel``, ``AArm``, corners, ...)

Everything drawable shares one contract: ``.to_3d(scene)`` to create,
``.place(obj)`` to re-position for a new frame.
"""

from utils.spatial.primitives import Point, Line, Segment, Plane, centroid
from utils.spatial.shapes import (
    SCENE_SCALE, align_y_to_direction,
    Sphere, Cylinder, Cuboid, DashedLine, Composite,
    Link, AArm, TrailingLink, Shock, Axle, Wheel,
    DoubleAArmCorner, SemiTrailingLinkCorner, corner_shape,
)

__all__ = [
    "Point", "Line", "Segment", "Plane", "centroid",
    "SCENE_SCALE", "align_y_to_direction",
    "Sphere", "Cylinder", "Cuboid", "DashedLine", "Composite",
    "Link", "AArm", "TrailingLink", "Shock", "Axle", "Wheel",
    "DoubleAArmCorner", "SemiTrailingLinkCorner", "corner_shape",
]
