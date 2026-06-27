from __future__ import annotations

# default
from typing import Dict, Tuple

# third-party
import numpy as np

# ours
from models.hardpoints import DoubleAArm, SemiTrailingLink
from models.corners.double_a_arm_analytical import DoubleAArmAnalytical
from models.corners.semi_trailing_link_analytical import SemiTrailingLinkAnalytical
from models.components.shock import Shock
from models.wheel import Wheel
from utils.misc import log_to_file

class Vehicle:
    nickname: str

    def __init__(self, data: Dict = {}):
        self.nickname = list(data.keys())[0]
        vehicle_data = data[self.nickname]
        
        self.config = vehicle_data 
        
        sp = vehicle_data['mass_properties']
        self.total_sprung_mass = sp['sprung_mass']
        self.cog = tuple(sp['cog'])
        self.inertia_matrix = np.array(vehicle_data['mass_properties']['inertia'])
        
        # Calculate front bias mathematically from exact CoG X-position
        f_x = vehicle_data['front']['wheel_center'][0]
        r_x = vehicle_data['rear']['wheel_center'][0]
        wb = abs(f_x - r_x)
        dist_from_rear = abs(self.cog[0] - r_x)
        self.sprung_bias_f = dist_from_rear / wb

        u_mass = vehicle_data['mass_properties']['unsprung_mass']
        
        self.front_left  = Corner(vehicle_data, (0, 0), u_mass['fl'])
        self.front_right = Corner(vehicle_data, (1, 0), u_mass['fr'])
        self.rear_left   = Corner(vehicle_data, (0, 1), u_mass['rl'])
        self.rear_right  = Corner(vehicle_data, (1, 1), u_mass['rr'])

        log_to_file(f"Initialized Vehicle '{self.nickname}'")
        log_to_file(f"Calculated COG at (x={self.cog[0]:.2f}, y={self.cog[1]:.2f}, z={self.cog[2]:.2f})")
        log_to_file(f"Total Sprung Mass: {self.total_sprung_mass:.2f} kg | Front Bias: {self.sprung_bias_f*100:.1f}%")

    def run_simulation(self, simulation_class, **kwargs):
        simulation = simulation_class(self, kwargs.get("config", {}))
        return simulation.run()

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
    def __init__(self, data: Dict, id: Tuple[int, int], unsprung_mass: float):
        self.id = id
        self.unsprung_mass = unsprung_mass

        if self.id[1] == 0:
            corner_data = data['front']
            hp = DoubleAArm.from_data(data=data['front'])
        else:
            corner_data = data['rear']
            hp = SemiTrailingLink.from_data(data=data['rear'])

        if self.id[0] == 0:  # left side -> mirror across y-axis
            hp = type(hp).mirror_points(hp)

        hp._fill_vehicle_properties(data=data)

        self.hardpoints = hp
        self.solver = DoubleAArmAnalytical(hp) if isinstance(hp, DoubleAArm) else SemiTrailingLinkAnalytical(hp)
        self.shock = Shock.from_config(corner_data, data['shock_max'], data['shock_min'])
        self.wheel = Wheel.from_config(data)