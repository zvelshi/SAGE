"""Drawable 3-D shape primitives -- ``Sphere`` / ``Cylinder`` / ``Cuboid`` /
``DashedLine``. Every drawable shares one contract: ``.to_3d(scene)`` to create,
``.place(obj)`` to re-position for a new frame."""

from utils.spatial.shapes._common import SCENE_SCALE, align_y_to_direction
from utils.spatial.shapes.sphere import Sphere
from utils.spatial.shapes.cylinder import Cylinder
from utils.spatial.shapes.cuboid import Cuboid
from utils.spatial.shapes.dashed_line import DashedLine

__all__ = ["SCENE_SCALE", "align_y_to_direction",
           "Sphere", "Cylinder", "Cuboid", "DashedLine"]
