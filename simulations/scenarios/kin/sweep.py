# default
from typing import List, Dict, Any

# third-party
import numpy as np

# ours
from simulations.scenarios.base import Scenario
from simulations.solvers import SingleCornerSolver
from utils.config import SweepConfig
from utils.logging_setup import get_logger

log = get_logger(__name__)

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

        # droop_steer / jounce_steer hold the corner at one travel extreme while
        # sweeping steer. Anchor those extremes to the shock's mechanical stroke
        # (full extension / full compression), tightened by TRAVEL only if the
        # config asks for less -- feeding a raw TRAVEL value straight in silently
        # yields zero steps whenever it exceeds the shock's limits.
        corner = vehicle.get_corner_from_id(self.corner_id)
        shock_static = corner.solver.len["shock_static"]
        eps = 1e-3
        mech_droop  = shock_static - corner.hardpoints.shock_max + eps   # travel_mm < 0
        mech_jounce = shock_static - corner.hardpoints.shock_min - eps   # travel_mm > 0
        self._droop_travel  = max(mech_droop, config.travel.min)
        self._jounce_travel = min(mech_jounce, config.travel.max)
        log.debug("steer-at-extreme travel: droop %.1fmm, jounce %.1fmm",
                  self._droop_travel, self._jounce_travel)

    def run(self) -> List[Dict]:
        steps = []
        count = self.config.sim_steps

        # Helper to generate ranges ('TRAVEL' / 'STEER')
        def get_range(key):
            rng = self.config.travel if key == "TRAVEL" else self.config.steer
            return np.linspace(rng.min, rng.max, count)

        sim_type = self.config.simulation
        log.debug("sweep %s on corner %s", sim_type, self.corner_id)

        if sim_type == "steer":
            steer_vals = get_range('STEER')
            for s in steer_vals:
                res = self.solver.solve(steer_mm=s, bump_z=0.0)
                if res:
                    res['x_val'] = s
                    res['x_label'] = "Rack Travel [mm]"
                    steps.append(res)
                else:
                    log.debug("steer sweep step failed at %.2fmm", s)

        elif sim_type == "travel":
            travel_vals = get_range('TRAVEL')
            for t in travel_vals:
                res = self.solver.solve(steer_mm=0.0, travel_mm=t)
                if res:
                    res['x_val'] = t
                    res['x_label'] = "Shock Travel [mm]"
                    steps.append(res)
                else:
                    log.debug("travel sweep step failed at %.2fmm", t)

        elif sim_type == "droop_steer":
            steer_vals = get_range('STEER')
            for s in steer_vals:
                res = self.solver.solve(steer_mm=s, travel_mm=self._droop_travel)
                if res:
                    res['x_val'] = s
                    res['x_label'] = "Rack Travel [mm]"
                    steps.append(res)
                else:
                    log.debug("steer sweep step failed at %.2fmm", s)

        elif sim_type == "jounce_steer":
            steer_vals = get_range('STEER')
            for s in steer_vals:
                res = self.solver.solve(steer_mm=s, travel_mm=self._jounce_travel)
                if res:
                    res['x_val'] = s
                    res['x_label'] = "Rack Travel [mm]"
                    steps.append(res)
                else:
                    log.debug("steer sweep step failed at %.2fmm", s)

        elif sim_type == "left_travel":
            travel_vals = get_range('TRAVEL')
            for t in travel_vals:
                res = self.solver.solve(steer_mm=self.config.steer.min, travel_mm=t)
                if res:
                    res['x_val'] = t
                    res['x_label'] = "Shock Travel [mm]"
                    steps.append(res)
                else:
                    log.debug("travel sweep step failed at %.2fmm", t)

        elif sim_type == "right_travel":
            travel_vals = get_range('TRAVEL')
            for t in travel_vals:
                res = self.solver.solve(steer_mm=self.config.steer.max, travel_mm=t)
                if res:
                    res['x_val'] = t
                    res['x_label'] = "Shock Travel [mm]"
                    steps.append(res)
                else:
                    log.debug("travel sweep step failed at %.2fmm", t)
                  
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
                        log.debug("travel sweep step failed at %.2fmm", t)
        return steps