# default
from __future__ import annotations

# ours
from utils.spatial.primitives import Point
from utils.spatial.shapes._common import POINT_COLOR
from utils.spatial.shapes.composites.composite import Composite
from utils.spatial.shapes.cylinder import Cylinder
from utils.spatial.shapes.sphere import Sphere


class Link(Composite):
    """A single rigid link between two points, with a joint marker at each end.
    Used for uprights, tie rods, camber links -- any two-point member."""

    def __init__(self, inboard, outboard, color=POINT_COLOR, link_color="#1e1e1e",
                 radius_mm: float = 4.0, point_radius_mm: float = 10.0):
        self.inboard = Point(inboard)
        self.outboard = Point(outboard)
        self.color = color
        self.link_color = link_color
        self.radius_mm = radius_mm
        self.point_radius_mm = point_radius_mm

    def parts(self):
        return {
            "link": Cylinder(self.inboard, self.outboard, self.radius_mm, self.link_color),
            "ib": Sphere(self.inboard, self.point_radius_mm, self.color),
            "ob": Sphere(self.outboard, self.point_radius_mm, self.color),
        }
