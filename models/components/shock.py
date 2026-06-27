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
    ls_comp: float    # [N*s/mm] Low-speed compression
    hs_comp: float    # [N*s/mm] High-speed compression
    ls_rebound: float # [N*s/mm] Low-speed rebound
    hs_rebound: float # [N*s/mm] High-speed rebound
    split_vel: float = 50.0 # [mm/s] velocity where LS transitions to HS
    
    def force(self, velocity: float) -> float:
        v = abs(velocity)
        if velocity > 0: # Rebound
            if v <= self.split_vel:
                return -self.ls_rebound * velocity
            else:
                return -(self.ls_rebound * self.split_vel + self.hs_rebound * (v - self.split_vel))
        else: # Compression (velocity <= 0)
            if v <= self.split_vel:
                return -self.ls_comp * velocity
            else:
                return -(-self.ls_comp * self.split_vel - self.hs_comp * (v - self.split_vel))

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
        # Backwards compatibility: fallback to c_comp/c_rebound if hs/ls aren't specified
        c_comp = s_data.get('compression_damping', 0.0)
        c_rebound = s_data.get('rebound_damping', 0.0)
        return cls(
            spring=Spring(stiffness=s_data.get('spring_rate', 0.0), preload=s_data.get('preload', 0.0)),
            damper=Damper(
                ls_comp=s_data.get('ls_comp', c_comp),
                hs_comp=s_data.get('hs_comp', c_comp),
                ls_rebound=s_data.get('ls_rebound', c_rebound),
                hs_rebound=s_data.get('hs_rebound', c_rebound),
                split_vel=s_data.get('split_vel', 50.0)
            ),
            shock_max=shock_max_ref,
            shock_min=shock_min_ref
        )

    def get_total_force(self, current_length: float, velocity: float) -> float:
        travel = self.shock_max - current_length
        
        # Calculate forces from spring and damper
        f_s = self.spring.force(travel)
        f_d = self.damper.force(velocity)

        # Bump stop force applies when shock is compressed beyond minimum or extended beyond max
        f_bump = 0.0
        if current_length < self.shock_min:
            f_bump = (self.shock_min - current_length) * self.bump_stop_k
        elif current_length > self.shock_max:
            f_bump = (self.shock_max - current_length) * self.bump_stop_k

        # Total force is the sum of spring, damper, and bump stop forces
        return f_s + f_d + f_bump