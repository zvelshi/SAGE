from __future__ import annotations

# default
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
from utils.misc import log_to_file

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

        log_to_file(f"Initialized Vehicle '{self.nickname}'")
        log_to_file(f"Calculated COG at (x={self.cog[0]:.2f}, y={self.cog[1]:.2f}, z={self.cog[2]:.2f})")
        log_to_file(f"Total Sprung Mass: {self.total_sprung_mass:.2f} kg | Front Bias: {self.sprung_bias_f*100:.1f}%")

    def _build_chassis_bottom_plane(self):
        """Chassis-bottom reference plane for ground-clearance analysis: horizontal
        (parallel to the global X-Y / ground plane), placed one inch (25.4 mm) below
        the lowest inboard front lower-A-arm pickup (``lower_a_arm_front`` /
        ``lower_a_arm_rear``). Returns None if the front corner has no lower A-arm."""
        hp = self.front_left.hardpoints
        if not (hasattr(hp, "lf") and hasattr(hp, "lr")):
            log_to_file("Chassis bottom plane: front corner has no lower A-arm; skipped.")
            return None
        lowest_inboard = min((Point(hp.lf), Point(hp.lr)), key=lambda p: p.z)
        plane = Plane.horizontal_through(lowest_inboard.translated(dz=-25.4))
        log_to_file(f"Chassis bottom plane at Z={plane.point.z:.2f}mm "
                    f"(1in below inboard lower-A-arm point {lowest_inboard!r})")
        return plane

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