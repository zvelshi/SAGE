# default
from __future__ import annotations

# ours
from utils.spatial.primitives import Point
from utils.spatial.shapes._common import POINT_COLOR
from utils.spatial.shapes.composites.composite import Composite
from utils.spatial.shapes.cylinder import Cylinder
from utils.spatial.shapes.sphere import Sphere


class AArm(Composite):
    """An A-arm (wishbone): two links from a pair of chassis pickups to a shared
    apex (ball joint). The apex marker is left to whatever else connects there
    (typically the upright)."""

    def __init__(self, pickup_front, pickup_rear, apex, link_color="#1e1e1e",
                 radius_mm: float = 4.0):
        self.pickup_front = Point(pickup_front)
        self.pickup_rear = Point(pickup_rear)
        self.apex = Point(apex)
        self.link_color = link_color
        self.radius_mm = radius_mm

    def parts(self):
        return {
            "front": Cylinder(self.pickup_front, self.apex, self.radius_mm, self.link_color),
            "rear": Cylinder(self.pickup_rear, self.apex, self.radius_mm, self.link_color),
            "sp_front": Sphere(self.pickup_front, 10.0, POINT_COLOR),
            "sp_rear": Sphere(self.pickup_rear, 10.0, POINT_COLOR),
        }
