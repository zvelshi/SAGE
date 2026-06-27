from __future__ import annotations

# default
from dataclasses import dataclass

@dataclass
class Wheel:
    radius:    float   # [mm]
    width:     float   # [mm]
    stiffness: float   # k_tire [N/mm]
    damping:   float   # c_tire [N·s/mm]

    def contact_force(self, z_hub: float, dz_hub: float, ground_z: float = 0.0) -> float:
        """Vertical tire contact force [N], positive = upward.
        Only acts when the wheel is in contact with the ground."""
        penetration = ground_z - (z_hub - self.radius)
        if penetration <= 0.0:
            return 0.0
        return max(0.0, self.stiffness * penetration - self.damping * dz_hub)

    @classmethod
    def from_config(cls, data: dict) -> "Wheel":
        wp = data["wheel_properties"]
        return cls(
            radius=wp["radius"],
            width=wp["width"],
            stiffness=wp["stiffness"],
            damping=wp["damping"],
        )
