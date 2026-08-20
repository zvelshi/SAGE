# default
from typing import List, Dict, Any

# third-party
import numpy as np

# ours
from simulations.scenarios.base import Scenario
from simulations.solvers import SingleCornerSolver
from utils.geometry import calculate_ackermann_percentage, get_toe_angle, get_mechanical_trail
from utils.misc import log_to_file

class FrontSteerScenario(Scenario):
    """
    Simulates both front wheels steering to calculate Ackermann percentage.
    """

    def __init__(self, vehicle, config):
        self.vehicle = vehicle
        self.config = config
        self.l_solver = SingleCornerSolver(vehicle, corner_id=[0, 0])
        self.r_solver = SingleCornerSolver(vehicle, corner_id=[1, 0])
        self.wr_l = vehicle.front_left.hardpoints.wr
        self.wr_r = vehicle.front_right.hardpoints.wr

    def run(self) -> List[Dict]:
        results = []
        log_to_file("Starting Ackermann Analysis...")

        wb = abs(self.vehicle.front_left.hardpoints.wc[0] - self.vehicle.rear_left.hardpoints.wc[0])
        tw = abs(self.vehicle.front_left.hardpoints.wc[1] - self.vehicle.front_right.hardpoints.wc[1])

        steer_steps = np.linspace(
            self.config['STEER']['MIN'],
            self.config['STEER']['MAX'],
            self.config['SIM_STEPS']
        )

        for steer in steer_steps:
            input = steer if steer != 0 else 0.001

            left = self.l_solver.solve(steer_mm=input, bump_z=0.0)
            right = self.r_solver.solve(steer_mm=input, bump_z=0.0)

            if left and right:
                toe_l = get_toe_angle(left)
                toe_r = get_toe_angle(right)

                if steer < 0:
                    ack_pct = calculate_ackermann_percentage(toe_r, toe_l, tw, wb)
                else:
                    ack_pct = calculate_ackermann_percentage(toe_l, toe_r, tw, wb)

                track_mm = abs(left['wc'][1] - right['wc'][1])

                trail_l = get_mechanical_trail(left, self.wr_l) if 'lbj' in left and 'ubj' in left else None
                trail_r = get_mechanical_trail(right, self.wr_r) if 'lbj' in right and 'ubj' in right else None

                results.append({
                    "input": input,
                    "left": left,
                    "right": right,
                    "ackermann_pct": ack_pct,
                    "toe_l_deg": toe_l,
                    "toe_r_deg": toe_r,
                    "track_change_mm": track_mm - tw,
                    "trail_l_mm": trail_l,
                    "trail_r_mm": trail_r,
                })
            else:
                log_to_file(f"[WARN] Ackermann step failed at input {input:.2f}. Left={bool(left)}, Right={bool(right)}")
                results.append({
                    "input": input,
                    "left": left,
                    "right": right,
                    "ackermann_pct": np.nan,
                    "toe_l_deg": np.nan,
                    "toe_r_deg": np.nan,
                    "track_change_mm": np.nan,
                    "trail_l_mm": None,
                    "trail_r_mm": None,
                })

        return results
