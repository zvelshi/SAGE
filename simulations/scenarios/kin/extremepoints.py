# default
from typing import List, Dict, Any

# ours
from simulations.scenarios.base import Scenario
from simulations.solvers import SingleCornerSolver
from utils.misc import log_to_file, pack_points_nicely

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
