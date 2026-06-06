# default
from dataclasses import dataclass, field

@dataclass
class Spring:
    stiffness: float # [N/mm]
    preload: float # [mm]

    def force(self, shock_travel: float) -> float:
        total_compression = self.preload + shock_travel
        return max(0.0, total_compression * self.stiffness)

@dataclass
class Damper:
    c_comp: float # [N*s/mm]
    c_rebound: float # [N*s/mm]
    
    def force(self, velocity: float) -> float:
        c = self.c_comp if velocity > 0 else self.c_rebound
        return c * velocity

@dataclass
class Shock:
    spring: Spring = field(default_factory=Spring)
    damper: Damper = field(default_factory=Damper)
    shock_max: float = 500.0 # [mm]
    shock_min: float = 300.0 # [mm]
    bump_stop_k: float = 500.0 # extreme stiffness when bottomed out

    @classmethod
    def from_config(cls, config: dict, shock_max_ref: float, shock_min_ref: float):
        s_data = config.get('shock_setup', {})        
        return cls(
            spring=Spring(stiffness=s_data['spring_rate'], preload=s_data['preload']),
            damper=Damper(c_comp=s_data['compression_damping'], c_rebound=s_data['rebound_damping']),
            shock_max=shock_max_ref,
            shock_min=shock_min_ref
        )

    def get_total_force(self, current_length: float, velocity: float) -> float:
        # If shock is extended beyond max, it can't pull, so total force is zero
        if current_length >= self.shock_max:
            return 0.0 

        travel = self.shock_max - current_length
        
        # Calculate forces from spring and damper
        f_s = self.spring.force(travel)
        f_d = self.damper.force(velocity)

        # Bump stop force only applies when shock is compressed beyond its minimum length
        f_bump = 0.0
        if current_length < self.shock_min:
            f_bump = (self.shock_min - current_length) * self.bump_stop_k

        # Total force is the sum of spring, damper, and bump stop forces
        return f_s + f_d + f_bump