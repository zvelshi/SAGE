# default
from __future__ import annotations

# ours
from utils.spatial.primitives import Point
from utils.spatial.shapes.composites.composite import Composite
from utils.spatial.shapes.cylinder import Cylinder
from utils.spatial.shapes.sphere import Sphere


class Axle(Composite):
    """The axle / driveshaft: an inboard (plunging) segment to the outboard CV,
    then the stub to the wheel center, with a marker at each CV."""

    def __init__(self, pivot_inboard, pivot_outboard, wheel_center, plunge_mm: float = 0.0,
                 color: str = "#cc2828"):
        self.pivot_inboard = Point(pivot_inboard)
        self.pivot_outboard = Point(pivot_outboard)
        self.wheel_center = Point(wheel_center)
        self.plunge_mm = float(plunge_mm)
        self.color = color

    @property
    def inboard_dyn(self) -> Point:
        """Inboard CV point slid along the shaft axis by the current plunge."""
        d = self.pivot_outboard - self.pivot_inboard
        if d.norm < 1e-9:
            return self.pivot_inboard
        return self.pivot_inboard + d.unit() * self.plunge_mm

    def parts(self):
        ib = self.inboard_dyn
        return {
            "shaft_inboard": Cylinder(ib, self.pivot_outboard, 4.0, self.color),
            "shaft_outboard": Cylinder(self.pivot_outboard, self.wheel_center, 4.0, self.color),
            "sp_ib": Sphere(ib, 10.0, "#000000"),
            "sp_ob": Sphere(self.pivot_outboard, 10.0, "#000000"),
        }
