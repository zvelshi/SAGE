from __future__ import annotations

# default
from functools import cached_property
from typing import Tuple

# third-party
import numpy as np

# ours
from models.hardpoints import DoubleAArm, SemiTrailingLink
from models.vehicle_config import VehicleConfig
from models.corners.double_a_arm import DoubleAArmNumeric
from models.corners.semi_trailing_link import SemiTrailingLinkNumeric
from models.components.axle import Axle
from models.components.cv_joint import CVJoint, PlungingCVJoint
from models.components.shock import Shock
from models.wheel import Wheel
from utils.spatial import Point, Plane
from utils.logging_setup import get_logger

log = get_logger(__name__)

class Vehicle:
    nickname: str

    def __init__(self, config: VehicleConfig):
        self.config = config
        self.nickname = config.nickname

        sp = config.mass_properties
        self.total_sprung_mass = sp.sprung_mass
        self.cog = tuple(sp.cog)
        self.inertia_matrix = np.array(sp.inertia)

        # Calculate front bias mathematically from exact CoG X-position
        f_x = config.front.wheel_center[0]
        r_x = config.rear.wheel_center[0]
        wb = abs(f_x - r_x)
        dist_from_rear = abs(self.cog[0] - r_x)
        self.sprung_bias_f = dist_from_rear / wb

        u_mass = sp.unsprung_mass

        self.front_left  = Corner(config, (0, 0), u_mass['fl'])
        self.front_right = Corner(config, (1, 0), u_mass['fr'])
        self.rear_left   = Corner(config, (0, 1), u_mass['rl'])
        self.rear_right  = Corner(config, (1, 1), u_mass['rr'])

        self.cog_point = Point(self.cog)
        self.chassis_bottom_plane = self._build_chassis_bottom_plane()

        log.debug("initialized vehicle '%s'", self.nickname)
        log.debug("CoG at (%.2f, %.2f, %.2f)", *self.cog)
        log.debug("sprung mass %.2f kg, front bias %.1f%%", self.total_sprung_mass, self.sprung_bias_f * 100)

    def _build_chassis_bottom_plane(self):
        """Chassis-bottom reference plane for ground-clearance analysis: horizontal
        (parallel to the global X-Y / ground plane), placed one inch (25.4 mm) below
        the lowest inboard front lower-A-arm pickup (``lower_a_arm_front`` /
        ``lower_a_arm_rear``). Returns None if the front corner has no lower A-arm."""
        hp = self.front_left.hardpoints
        if not (hasattr(hp, "lf") and hasattr(hp, "lr")):
            log.debug("chassis bottom plane: front corner has no lower A-arm; skipped")
            return None
        lowest_inboard = min((Point(hp.lf), Point(hp.lr)), key=lambda p: p.z)
        plane = Plane.horizontal_through(lowest_inboard.translated(dz=-25.4))
        log.debug("chassis bottom plane at Z=%.2fmm (1in below %r)", plane.point.z, lowest_inboard)
        return plane

    @cached_property
    def bump_z_limits(self) -> dict:
        """Wheel-bump (bump_z) envelope per axle, from each side's shock stroke.
        Front and rear have different motion ratios, so the same shock travel
        moves their wheels by different amounts -- the heave/roll scenarios use
        this to keep their wheel-bump input inside every corner's shock limits.
        Computed lazily: only heave/roll need it, and it costs a few extra solves."""
        limits = {
            "front": self.front_left._bump_z_range(),
            "rear":  self.rear_left._bump_z_range(),
        }
        log.debug("bump_z envelope front %.1f..%.1f mm, rear %.1f..%.1f mm",
                  *limits["front"], *limits["rear"])
        return limits

    def get_corner_from_id(self, id) -> 'Corner':
        if id == [0, 0]: return self.front_left
        if id == [1, 0]: return self.front_right
        if id == [0, 1]: return self.rear_left
        if id == [1, 1]: return self.rear_right
        raise ValueError(f"Invalid corner_id: {id}")

class Corner:
    """
         (0, 0) _________ (1, 0)
                |       |
                |       |
                |       |
                |       |
                |       |
         (0, 1) |_______| (1, 1)
    """
    def __init__(self, config: VehicleConfig, id: Tuple[int, int], unsprung_mass: float):
        self.id = id
        self.unsprung_mass = unsprung_mass

        if self.id[1] == 0:
            corner_cfg = config.front
            hp = DoubleAArm.from_config(corner_cfg)
        else:
            corner_cfg = config.rear
            hp = SemiTrailingLink.from_config(corner_cfg)

        if self.id[0] == 0:  # left side -> mirror across y-axis
            hp = type(hp).mirror_points(hp)

        hp._fill_vehicle_properties(config)

        self.hardpoints = hp

        axle = Axle(
            joint1=PlungingCVJoint(max_angle=30, plunge_limit=30.0), # Inboard slider
            joint2=CVJoint(max_angle=30), # Outboard fixed
            length=float(np.linalg.norm(hp.piv_ob - hp.piv_ib)),
        )

        self.solver = DoubleAArmNumeric(hp, axle) if isinstance(hp, DoubleAArm) else SemiTrailingLinkNumeric(hp, axle)
        self.shock = Shock.from_config(corner_cfg.shock_setup, config.shock_max, config.shock_min)
        self.wheel = Wheel.from_config(config)

    def _bump_z_range(self) -> Tuple[float, float]:
        """Wheel-center Z offsets (droop-negative) this corner reaches at its two
        shock-travel limits -- the mechanically reachable wheel-bump envelope for
        this axle. The solver seed is reset afterwards so later sims start clean."""
        solver = self.solver
        hp = self.hardpoints
        shock_static = solver.len["shock_static"]
        wc_z0 = float(hp.wc[2])
        offsets = []
        for target_len in (hp.shock_max - 1e-4, hp.shock_min + 1e-4):
            step = solver.solve(travel_mm=shock_static - target_len)
            if step is not None:
                offsets.append(float(step["wc"][2]) - wc_z0)
            else:
                log.warning("corner %s cannot solve at shock length %.2fmm; "
                            "heave/roll range may be degraded", self.id, target_len)
        solver.reset()
        if len(offsets) < 2:
            return (0.0, 0.0)
        return (min(offsets), max(offsets))