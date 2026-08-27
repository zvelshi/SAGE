# default
from typing import List, Dict, Any

# third-party
import numpy as np

# ours
from simulations.scenarios.base import Scenario
from simulations.solvers import SingleCornerSolver
from utils.config import SweepConfig
from utils.misc import log_to_file

class SuspensionSweep(Scenario):
    """
    Sweeps a single corner through Travel AND/OR Steer.
    """

    def __init__(self, vehicle, config: SweepConfig):
        self.config = config

        self.corner_id = [0, 0]
        if config.half == 'rear':
            self.corner_id[1] = 1
        elif config.side == 'right':
            self.corner_id[0] = 1

        self.solver = SingleCornerSolver(vehicle, self.corner_id)

    def run(self) -> List[Dict]:
        steps = []
        count = self.config.sim_steps

        # Helper to generate ranges ('TRAVEL' / 'STEER')
        def get_range(key):
            rng = self.config.travel if key == "TRAVEL" else self.config.steer
            return np.linspace(rng.min, rng.max, count)

        sim_type = self.config.simulation
        log_to_file(f"Starting SuspensionSweep: {sim_type} on corner {self.corner_id}")

        if sim_type == "steer":
            steer_vals = get_range('STEER')
            for s in steer_vals:
                res = self.solver.solve(steer_mm=s, bump_z=0.0)
                if res:
                    res['x_val'] = s
                    res['x_label'] = "Rack Travel [mm]"
                    steps.append(res)
                else:
                    log_to_file(f"[WARN] Steer sweep step failed at {s:.2f}mm")

        elif sim_type == "travel":
            travel_vals = get_range('TRAVEL')
            for t in travel_vals:
                res = self.solver.solve(steer_mm=0.0, travel_mm=t)
                if res:
                    res['x_val'] = t
                    res['x_label'] = "Shock Travel [mm]"
                    steps.append(res)
                else:
                    log_to_file(f"[WARN] Travel sweep step failed at {t:.2f}mm")

        elif sim_type == "droop_steer":
            steer_vals = get_range('STEER')
            for s in steer_vals:
                res = self.solver.solve(steer_mm=s, travel_mm=self.config.travel.min)
                if res:
                    res['x_val'] = s
                    res['x_label'] = "Rack Travel [mm]"
                    steps.append(res)
                else:
                    log_to_file(f"[WARN] Steer sweep step failed at {s:.2f}mm")

        elif sim_type == "jounce_steer":
            steer_vals = get_range('STEER')
            for s in steer_vals:
                res = self.solver.solve(steer_mm=s, travel_mm=self.config.travel.max)
                if res:
                    res['x_val'] = s
                    res['x_label'] = "Rack Travel [mm]"
                    steps.append(res)
                else:
                    log_to_file(f"[WARN] Steer sweep step failed at {s:.2f}mm")

        elif sim_type == "left_travel":
            travel_vals = get_range('TRAVEL')
            for t in travel_vals:
                res = self.solver.solve(steer_mm=self.config.steer.min, travel_mm=t)
                if res:
                    res['x_val'] = t
                    res['x_label'] = "Shock Travel [mm]"
                    steps.append(res)
                else:
                    log_to_file(f"[WARN] Travel sweep step failed at {t:.2f}mm")

        elif sim_type == "right_travel":
            travel_vals = get_range('TRAVEL')
            for t in travel_vals:
                res = self.solver.solve(steer_mm=self.config.steer.max, travel_mm=t)
                if res:
                    res['x_val'] = t
                    res['x_label'] = "Shock Travel [mm]"
                    steps.append(res)
                else:
                    log_to_file(f"[WARN] Travel sweep step failed at {t:.2f}mm")
                  
        elif sim_type == "sweep_space":
            travel_vals = get_range('TRAVEL')
            steer_vals = get_range('STEER')
            for t in travel_vals:
                for s in steer_vals:
                    res = self.solver.solve(steer_mm=s, travel_mm=t)
                    if res:
                        res['x_val'] = t
                        res['x_label'] = "Shock Travel [mm]"
                        steps.append(res)
                    else:
                        log_to_file(f"[WARN] Travel sweep step failed at {t:.2f}mm")
        return steps