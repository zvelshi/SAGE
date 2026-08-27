# default
from __future__ import annotations

# ours
from utils.spatial.primitives import Point
from utils.spatial.shapes._common import POINT_COLOR
from utils.spatial.shapes.composites.composite import Composite
from utils.spatial.shapes.cylinder import Cylinder
from utils.spatial.shapes.sphere import Sphere


class Shock(Composite):
    """A coilover: a fixed-length body sleeve from the inboard mount, and the
    piston rod that telescopes out of it to the outboard mount."""

    def __init__(self, inboard, outboard, body_length_mm: float, color: str = "#6e6e82"):
        self.inboard = Point(inboard)
        self.outboard = Point(outboard)
        self.body_length_mm = float(body_length_mm)
        self.color = color

    @property
    def body_end(self) -> Point:
        """Outboard end of the fixed-length body sleeve, along the mount axis."""
        d = self.outboard - self.inboard
        if d.norm < 1e-9:
            return self.inboard.translated(dy=self.body_length_mm)
        return self.inboard + d.unit() * self.body_length_mm

    def parts(self):
        be = self.body_end
        return {
            "body": Cylinder(self.inboard, be, 12.0, self.color),
            "piston": Cylinder(be, self.outboard, 3.0, self.color),
            "ob": Sphere(self.outboard, 10.0, POINT_COLOR),
        }
