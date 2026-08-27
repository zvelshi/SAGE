# default
from __future__ import annotations
from dataclasses import dataclass

# ours
from utils.spatial.primitives import Line, Point
from utils.spatial.shapes._common import SCENE_SCALE
from utils.spatial.shapes.cylinder import Cylinder


@dataclass
class DashedLine:
    """A dashed reference line -- a composite of short cylinders. Fundamental
    values: a center, a direction and a total length."""
    center: Point
    direction: Point
    length_mm: float
    n_dashes: int = 14
    radius_mm: float = 1.8
    color: str = "#888888"
    opacity: float = 0.55

    def _dashes(self) -> list[Cylinder]:
        line = Line(self.center, self.direction)
        half = self.length_mm / 2.0
        dash = self.length_mm / (2 * self.n_dashes - 1)
        out = []
        for i in range(self.n_dashes):
            t0 = -half + i * 2 * dash
            out.append(Cylinder(line.point_at(t0), line.point_at(t0 + dash),
                                self.radius_mm, self.color, self.opacity))
        return out

    def to_3d(self, scene, scale: float = SCENE_SCALE) -> list:
        return [d.to_3d(scene, scale) for d in self._dashes()]

    def place(self, objs: list, scale: float = SCENE_SCALE, restyle: bool = False):
        for d, obj in zip(self._dashes(), objs):
            d.place(obj, scale, restyle)
        return objs
