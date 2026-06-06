# default
from typing import List, Dict, Any

# third-party
import numpy as np
from scipy.optimize import root_scalar

# ours
from .solvers import SingleCornerSolver
from utils.geometry import calculate_ackermann_percentage, get_toe_angle
from utils.misc import log_to_file, pack_points_nicely

class Scenario:
    def run(self) -> List[Any]:
        raise NotImplementedError

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

class ExtremePoints(Scenario):
    def __init__(self, vehicle, config):
        self.config = config
        self.vehicle = vehicle

    def run(self) -> List[Dict]:
        log_to_file(f"Starting ExtremePoints Simulation for all corners...")

        raw_pkg = {}
        corners = [[0, 0], [1, 0], [0, 1], [1, 1]]
        
        steer_min = self.config.get("STEER", {}).get("MIN", -40)
        steer_max = self.config.get("STEER", {}).get("MAX", 40)
        
        steers = {
            "neutral": 0,
            f"{steer_min}": steer_min,
            f"{steer_max}": steer_max
        }

        for id in corners:
            log_to_file(f"Finding extreme points for corner {id}...")
            sol = SingleCornerSolver(self.vehicle, id)
            corner = self.vehicle.get_corner_from_id(id)
            hp = corner.hardpoints

            # Solve static first to get the baseline shock length
            static_step = sol.solve(travel_mm=0, steer_mm=0)
            if static_step is None:
                print(f"[WARN] Failed to find static position for corner {id}")
                log_to_file(f"[WARN] Failed to find static position for corner {id}")
                continue
                
            shock_static = static_step["shock_length"]

            # Compute exact travel required to hit physical shock limits
            travels = {
                "static": 0,
                "jounce": shock_static - hp.shock_min,
                "droop":  shock_static - hp.shock_max
            }

            for t_name, t_val in travels.items():
                for s_name, s_val in steers.items():
                    # Rear suspensions do not steer
                    if id[1] == 1 and s_val != 0:
                        continue

                    step = sol.solve(travel_mm=t_val, steer_mm=s_val)
                    
                    key = f"side{id[0]}_half{id[1]}_{t_name}_{s_name}"
                    
                    if step is None:
                        print(f"[WARN] No solution found for {key} at travel={t_val:.2f}mm, steer={s_val}mm")
                        log_to_file(f"[WARN] No solution found for {key} at travel={t_val:.2f}mm, steer={s_val}mm")
                    else:
                        pack = pack_points_nicely(self.vehicle, id, step)
                        raw_pkg[key] = pack if pack else {"error": "No solution found"}

        # Group and structure the results into [half][side][condition][steer]
        pack = {}
        for key, data in raw_pkg.items():
            parts = key.split("_")
            side_str = parts[0]   # "side0" or "side1"
            half_str = parts[1]   # "half0" or "half1"
            condition = parts[2]  # "static", "jounce", etc.
            s_name = parts[3]     # "neutral" or "-40"
            
            side = 'left' if side_str == "side0" else 'right'
            half = 'front' if half_str == "half0" else 'rear'
            steer_part = "0_steer" if s_name == "neutral" else f"{s_name}_steer"

            if half not in pack:
                pack[half] = {}
            if side not in pack[half]:
                pack[half][side] = {}
            if condition not in pack[half][side]:
                pack[half][side][condition] = {}
            
            pack[half][side][condition][steer_part] = data

        for half, sides in pack.items():
            for side, conditions in sides.items():
                for condition, steer_data in conditions.items():
                    for steer, data in steer_data.items():
                        print(f"\n--- {half.upper()} - {side.upper()} - {condition.upper()} - {steer.upper()} ---")

        return pack

class AckermannScenario(Scenario):
    """
    Simulates both front wheels steering to calculate Ackermann percentage.
    """

    def __init__(self, vehicle, config):
        self.vehicle = vehicle
        self.config = config
        self.l_solver = SingleCornerSolver(vehicle, corner_id=[0, 0])
        self.r_solver = SingleCornerSolver(vehicle, corner_id=[1, 0])

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
                    ack_pct = calculate_ackermann_percentage(toe_l, toe_r, tw, wb)
                else:
                    ack_pct = calculate_ackermann_percentage(toe_r, toe_l, tw, wb)

                results.append({
                    "input": input,
                    "left": left,
                    "right": right,
                    "ackermann_pct": ack_pct
                })
            else:
                log_to_file(f"[WARN] Ackermann step failed at input {input:.2f}. Left={bool(left)}, Right={bool(right)}")
                results.append({
                    "input": input,
                    "left": left,
                    "right": right,
                    "ackermann_pct": np.nan,
                })
                
        return results

class DynamicScenario(Scenario):
    """
    Simulates the vehicle driving over a single sinusoidal terrain profile.
    CoG height is fixed at the design height, but moves longitudinally.
    """
    def __init__(self, vehicle, config):
        self.vehicle = vehicle
        self.config = config

        self.solvers = {
            'fl': SingleCornerSolver(vehicle, [0, 0]),
            'fr': SingleCornerSolver(vehicle, [1, 0]),
            'rl': SingleCornerSolver(vehicle, [0, 1]),
            'rr': SingleCornerSolver(vehicle, [1, 1]),
        }

    def _get_terrain_height_mm(self, x_mm: float) -> float:
        """Calculates terrain Z using a single sine wave."""
        t_cfg = self.config['TERRAIN']
        x_m = x_mm / 1000.0
        z_mm = t_cfg['AMPLITUDE'] * np.sin(2 * np.pi * t_cfg['FREQUENCY'] * x_m)
        return z_mm

    def run(self) -> List[Dict]:
        results = []
        sol_dt = self.config['SOL_DT']
        viz_dt = self.config['VIZ_DT']

        total_time = self.config['DURATION']
        velocity = -self.config['VELOCITY'] * 1000.0 # [mm/s]

        sub_steps = int(viz_dt / sol_dt)
        total_render_frames = int(total_time / viz_dt)
        print(f"--- Running Kinematic Sim ---")

        for i in range(total_render_frames):
            t_render = i * viz_dt
            car_x = velocity * t_render

            for _ in range(sub_steps):
                # Physics 
                pass
            
            step_data = {
                'time': t_render,
                'x_pos': car_x,
                'cog_z': self.vehicle.cog[2],
                'cog_x': self.vehicle.front_left.hardpoints.wc[0] + (self.vehicle.cog[0] * self.vehicle.sprung_bias_f),
                'corners': {}
            }
            
            for name, solver in self.solvers.items():
                corner = self.vehicle.get_corner_from_id(solver.corner_id)
                wx = car_x + corner.hardpoints.wc[0]
                ground_z = self._get_terrain_height_mm(wx)
                bump_z = ground_z - (self.vehicle.cog[2] - corner.hardpoints.wr) 

                res = solver.solve(steer_mm=0.0, bump_z=bump_z)
                if res:
                    step_data['corners'][name] = res
            
            results.append(step_data)
        return results