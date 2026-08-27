# default
from __future__ import annotations
from dataclasses import dataclass

# ours
from utils.spatial.primitives import Point
from utils.spatial.shapes._common import SCENE_SCALE, align_y_to_direction, xyz


@dataclass
class Cylinder:
    """A right circular cylinder spanning two points. Fundamental values: the two
    end points + a radius. Length is free (geometry is unit-height, Y-scaled), so
    the same object can follow moving end points frame to frame."""
    start: Point
    end: Point
    radius_mm: float = 4.0
    color: str = "#1e1e1e"
    opacity: float = 1.0

    @property
    def length_mm(self) -> float:
        return (Point(self.end) - Point(self.start)).norm

    @property
    def midpoint(self) -> Point:
        return Point(self.start).midpoint_to(self.end)

    @property
    def direction(self) -> Point:
        d = Point(self.end) - Point(self.start)
        return d.unit() if d.norm > 1e-9 else Point(0.0, 1.0, 0.0)

    def to_3d(self, scene, scale: float = SCENE_SCALE):
        r = self.radius_mm * scale
        obj = scene.cylinder(top_radius=r, bottom_radius=r, height=1.0)
        obj.material(self.color, opacity=self.opacity)
        return self.place(obj, scale)

    def place(self, obj, scale: float = SCENE_SCALE, restyle: bool = False):
        obj.scale(1.0, max(self.length_mm * scale, 1e-9), 1.0)
        obj.move(*xyz(self.midpoint, scale))
        obj.rotate(*align_y_to_direction(self.direction.array))
        if restyle:
            obj.material(self.color, opacity=self.opacity)
        return obj
