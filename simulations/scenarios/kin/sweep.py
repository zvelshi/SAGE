# default
from typing import List, Dict, Any

# third-party
import numpy as np

# ours
from simulations.scenarios.base import Scenario
from simulations.solvers import SingleCornerSolver
from utils.misc import log_to_file

class SuspensionSweep(Scenario):
    """
    Sweeps a single corner through Travel AND/OR Steer.
    """

    def __init__(self, vehicle, config):
        self.config = config

        self.corner_id = [0, 0]
        if config["HALF"] == 'rear':
            self.corner_id[1] = 1
        elif config["SIDE"] == 'right':
            self.corner_id[0] = 1

        self.solver = SingleCornerSolver(vehicle, self.corner_id)

    def run(self) -> List[Dict]:
        steps = []
        count = self.config['SIM_STEPS']
        
        # Helper to generate ranges
        def get_range(key):
            return np.linspace(self.config[key]['MIN'], self.config[key]['MAX'], count)

        sim_type = self.config["SIMULATION"]
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

        elif sim_type == "steer_travel":
             s_vals = get_range('STEER')
             t_vals = get_range('TRAVEL')
             for s, t in zip(s_vals, t_vals):
                res = self.solver.solve(steer_mm=s, travel_mm=t)
                if res: 
                    res['x_val'] = t # Default X-axis to travel for combined sweeps
                    res['x_label'] = "Shock Travel [mm] (with Steer)"
                    steps.append(res)
                else:
                    log_to_file(f"[WARN] Combined sweep step failed at steer={s:.2f}, travel={t:.2f}")

        elif sim_type == "droop_steer":
            steer_vals = get_range('STEER')
            for s in steer_vals:
                res = self.solver.solve(steer_mm=s, travel_mm=self.config["TRAVEL"]["MIN"])
                if res: 
                    res['x_val'] = s
                    res['x_label'] = "Rack Travel [mm]"
                    steps.append(res)
                else:
                    log_to_file(f"[WARN] Steer sweep step failed at {s:.2f}mm")

        elif sim_type == "jounce_steer":
            steer_vals = get_range('STEER')
            for s in steer_vals:
                res = self.solver.solve(steer_mm=s, travel_mm=self.config["TRAVEL"]["MAX"])
                if res: 
                    res['x_val'] = s
                    res['x_label'] = "Rack Travel [mm]"
                    steps.append(res)
                else:
                    log_to_file(f"[WARN] Steer sweep step failed at {s:.2f}mm")
                  
        return steps