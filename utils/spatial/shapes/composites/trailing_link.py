# default
from __future__ import annotations

# ours
from utils.spatial.primitives import Point
from utils.spatial.shapes._common import POINT_COLOR
from utils.spatial.shapes.composites.composite import Composite
from utils.spatial.shapes.cylinder import Cylinder
from utils.spatial.shapes.sphere import Sphere


class TrailingLink(Composite):
    """The rear semi-trailing link: a front pivot tied to the two camber-link
    outboard points, with a marker at the pivot."""

    def __init__(self, pivot, upper_outboard, lower_outboard, link_color="#1e1e1e",
                 radius_mm: float = 4.0):
        self.pivot = Point(pivot)
        self.upper_outboard = Point(upper_outboard)
        self.lower_outboard = Point(lower_outboard)
        self.link_color = link_color
        self.radius_mm = radius_mm

    def parts(self):
        return {
            "upper": Cylinder(self.pivot, self.upper_outboard, self.radius_mm, self.link_color),
            "lower": Cylinder(self.pivot, self.lower_outboard, self.radius_mm, self.link_color),
            "sp_pivot": Sphere(self.pivot, 10.0, POINT_COLOR),
        }
