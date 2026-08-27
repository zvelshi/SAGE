"""Drawable 3-D shapes.

- primitives: ``Sphere`` / ``Cylinder`` / ``Cuboid`` / ``DashedLine``
- ``composites``: suspension components built from primitives (``Link``, ``AArm``,
  ``TrailingLink``, ``Shock``, ``Axle``, ``Wheel``, corners), with ``Composite``
  the base for any shape made of shapes.

Every drawable shares one contract: ``.to_3d(scene)`` to create, ``.place(obj)``
to re-position for a new frame.
"""

from utils.spatial.shapes._common import SCENE_SCALE, align_y_to_direction
from utils.spatial.shapes.sphere import Sphere
from utils.spatial.shapes.cylinder import Cylinder
from utils.spatial.shapes.cuboid import Cuboid
from utils.spatial.shapes.dashed_line import DashedLine
from utils.spatial.shapes.composites import (
    Composite, Link, AArm, TrailingLink, Shock, Axle, Wheel,
    DoubleAArmCorner, SemiTrailingLinkCorner, corner_shape,
)

__all__ = [
    "SCENE_SCALE", "align_y_to_direction",
    "Sphere", "Cylinder", "Cuboid", "DashedLine", "Composite",
    "Link", "AArm", "TrailingLink", "Shock", "Axle", "Wheel",
    "DoubleAArmCorner", "SemiTrailingLinkCorner", "corner_shape",
]
