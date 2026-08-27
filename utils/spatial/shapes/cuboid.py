# default
from __future__ import annotations
from dataclasses import dataclass

# ours
from utils.spatial.primitives import Point
from utils.spatial.shapes._common import SCENE_SCALE, align_y_to_direction, xyz


@dataclass
class Cuboid:
    """A box, most often used as a thin plate standing in for a bounded patch of a
    plane. Fundamental values: center + normal (local +Y is aligned to the
    normal), plus the in-plane width/depth and the thickness along the normal."""
    center: Point
    normal: Point
    width_mm: float = 1000.0
    depth_mm: float = 1000.0
    thickness_mm: float = 3.0
    color: str = "#3a7bd5"
    opacity: float = 0.28

    def to_3d(self, scene, scale: float = SCENE_SCALE):
        obj = scene.box(self.width_mm * scale, self.thickness_mm * scale, self.depth_mm * scale)
        obj.material(self.color, opacity=self.opacity, side="both")
        return self.place(obj, scale)

    def place(self, obj, scale: float = SCENE_SCALE, restyle: bool = False):
        if not getattr(obj, "visible_", True):
            return obj  # hidden -> skip the transform RPCs entirely
        obj.rotate(*align_y_to_direction(Point(self.normal).unit().array))
        obj.move(*xyz(self.center, scale))
        if restyle:
            obj.material(self.color, opacity=self.opacity, side="both")
        return obj
