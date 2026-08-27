# default
from __future__ import annotations

# ours
from utils.spatial.primitives import Point
from utils.spatial.shapes._common import AX_DEFAULT
from utils.spatial.shapes.composites.composite import Composite
from utils.spatial.shapes.cylinder import Cylinder
from utils.spatial.shapes.sphere import Sphere


class Wheel(Composite):
    """The tire/wheel: a short fat cylinder about the spin axis, with a hub marker."""

    def __init__(self, center, axis, radius_mm: float, width_mm: float, color: str = "#888888"):
        self.center = Point(center)
        axis = Point(axis)
        self.axis = axis.unit() if axis.norm > 1e-9 else AX_DEFAULT
        self.radius_mm = float(radius_mm)
        self.width_mm = float(width_mm)
        self.color = color

    def parts(self):
        half = self.axis * (self.width_mm / 2.0)
        return {
            "tire": Cylinder(self.center - half, self.center + half, self.radius_mm,
                             self.color, opacity=0.35),
            "hub": Sphere(self.center, 14.0, "#4466bb"),
        }
