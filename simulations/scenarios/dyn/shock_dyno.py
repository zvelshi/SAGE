# default
from typing import Dict, List, Callable

# third-party
import numpy as np

# ours
from simulations.scenarios.base import Scenario
from utils.misc import log_to_file

class ShockDyno(Scenario):
    """
    Simulates a Shock Dynamometer test.
    Applies a sinusoidal displacement to an isolated shock absorber to generate Force-Velocity data.
    """
    
    def __init__(self, vehicle, config: Dict, on_progress: Callable | None = None):
        self.vehicle = vehicle
        self.config = config
        
        # Pull parameters from dyn_config
        self.stroke_mm = float(config["DYNO_STROKE"])
        self.freq_hz = float(config["DYNO_FREQUENCY"])
        
        self._on_progress = on_progress if on_progress else lambda *args: None

    def _progress(self, fraction: float, message: str) -> None:
        self._on_progress(float(np.clip(fraction, 0.0, 1.0)), message)

    def run(self) -> List[Dict]:
        print(f"ShockDyno | Frequency: {self.freq_hz} Hz | Stroke: {self.stroke_mm} mm")
        log_to_file(f"ShockDyno: freq={self.freq_hz}Hz, stroke={self.stroke_mm}mm")
        
        # Get the targeted shock from the vehicle based on KIN config
        # Default to rear right if not specified for some reason
        half = self.config["HALF"]
        side = self.config["SIDE"]
        corner_attr = f"{half}_{side}"
        corner = getattr(self.vehicle, corner_attr)
        shock = corner.shock
        
        # Ensure stroke doesn't exceed total shock travel
        total_travel = shock.shock_max - shock.shock_min
        if self.stroke_mm > total_travel:
            print(f"  Warning: requested stroke ({self.stroke_mm}mm) exceeds total travel. Capping to {total_travel}mm.")
            self.stroke_mm = total_travel

        amplitude = self.stroke_mm / 2.0
        
        # Simulate exactly 1 full cycle
        period = 1.0 / self.freq_hz
        num_steps = 200 # Fixed resolution for the dyno plot
        t_array = np.linspace(0, period, num_steps)
        omega = 2.0 * np.pi * self.freq_hz
        
        steps = []
        for i, t in enumerate(t_array):
            # Start at fully extended, compress down by stroke_mm, and return
            # displacement goes from 0 -> stroke_mm -> 0
            displacement = amplitude - amplitude * np.cos(omega * t)
            velocity = amplitude * omega * np.sin(omega * t)
            
            # Shock dyno convention: compression is positive velocity, rebound is negative.
            current_len = shock.shock_max - displacement 
            
            # Shock model velocity: positive = extending (rebound), negative = compressing
            # Since displacement is positive when compressing, dx/dt is positive when compressing.
            # Thus shock model velocity = -dx/dt (compressing = negative shock vel)
            shock_vel = -velocity
            
            f_total = shock.get_total_force(current_len, shock_vel)
            f_spring = shock.spring.force(shock.shock_max - current_len)
            f_damper = shock.damper.force(shock_vel)
            
            steps.append({
                "t": float(t),
                "displacement": float(displacement),
                "velocity": float(velocity), # Dyno velocity (positive = compression)
                "shock_len": float(current_len),
                "shock_max": float(shock.shock_max),
                "shock_min": float(shock.shock_min),
                "force_total": float(f_total),
                "force_spring": float(f_spring),
                "force_damper": float(f_damper),
            })
            
            if i % 20 == 0:
                self._progress(i / num_steps, f"Dyno stroking... ({i}/{num_steps})")
                
        self._progress(1.0, "Dyno cycle complete")
        return steps
