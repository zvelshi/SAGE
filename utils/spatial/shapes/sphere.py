# default
from __future__ import annotations
from dataclasses import dataclass

# ours
from utils.spatial.primitives import Point
from utils.spatial.shapes._common import SCENE_SCALE, xyz


@dataclass
class Sphere:
    """A point, drawn. Fundamental value: a center; render props: radius + color."""
    center: Point
    radius_mm: float = 10.0
    color: str = "#4466bb"
    opacity: float = 1.0

    def to_3d(self, scene, scale: float = SCENE_SCALE):
        obj = scene.sphere(radius=self.radius_mm * scale)
        obj.material(self.color, opacity=self.opacity)
        return self.place(obj, scale)

    def place(self, obj, scale: float = SCENE_SCALE, restyle: bool = False):
        obj.move(*xyz(self.center, scale))
        if restyle:
            obj.material(self.color, opacity=self.opacity)
        return obj
