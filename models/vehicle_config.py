"""Typed, validated vehicle (hardpoint) configuration.

The ``config/hardpoints/*.yml`` files describe one car: its mass properties,
tyre, and the corner pickup points. :func:`load_vehicle_config` validates one --
a missing point, a two-element coordinate, an unknown ``front:`` key -- and fails
with a message that names the field, instead of a bare ``KeyError`` deep inside
``Vehicle.__init__``.

The single top-level nickname key (``baja_2026:``) is unwrapped by the loader
and carried on :attr:`VehicleConfig.nickname`.
"""
from __future__ import annotations

# default
from pathlib import Path
from typing import Any, Literal

# third-party
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

# ours
from utils.config import ConfigError, _format

Xyz = tuple[float, float, float]  # a point / vector in millimetres


class _Model(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class ShockSetup(_Model):
    # Tolerate the odd extra key (e.g. legacy compression_damping); Shock reads
    # what it needs by name.
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    spring_rate: float = 0.0
    ls_comp: float = 0.0
    hs_comp: float = 0.0
    ls_rebound: float = 0.0
    hs_rebound: float = 0.0
    split_vel: float = 50.0
    preload: float = 0.0
    free_length: float | None = None


class WheelProps(_Model):
    radius: float = Field(gt=0.0)     # mm
    width: float = Field(gt=0.0)      # mm
    stiffness: float = Field(gt=0.0)  # N/mm
    damping: float = Field(ge=0.0)    # N*s/mm


class MassProps(_Model):
    sprung_mass: float = Field(gt=0.0)                                  # kg
    unsprung_mass: dict[Literal["fl", "fr", "rl", "rr"], float]         # kg
    cog: Xyz                                                            # mm
    inertia: tuple[Xyz, Xyz, Xyz]                                       # kg*m^2

    @model_validator(mode="after")
    def _all_corners(self):
        missing = {"fl", "fr", "rl", "rr"} - set(self.unsprung_mass)
        if missing:
            raise ValueError(f"unsprung_mass missing corner(s): {sorted(missing)}")
        return self


class DoubleAArmCorner(_Model):
    type_: Literal["DoubleAArm"] = Field(alias="_type", default="DoubleAArm")
    shock_setup: ShockSetup = Field(default_factory=ShockSetup)
    shock_location: Literal["upper", "lower"]

    upper_a_arm_front: Xyz
    upper_a_arm_rear: Xyz
    lower_a_arm_front: Xyz
    lower_a_arm_rear: Xyz
    upper_ball_joint: Xyz
    lower_ball_joint: Xyz
    tie_rod_inboard: Xyz
    tie_rod_outboard: Xyz
    shock_inboard: Xyz
    shock_outboard: Xyz
    pivot_inboard: Xyz
    pivot_outboard: Xyz
    wheel_center: Xyz


class SemiTrailingLinkCorner(_Model):
    type_: Literal["SemiTrailingLink"] = Field(alias="_type", default="SemiTrailingLink")
    shock_setup: ShockSetup = Field(default_factory=ShockSetup)
    shock_location: Literal["upper", "lower"] = "lower"

    trailing_link_front: Xyz
    upper_camber_link_inboard: Xyz
    upper_camber_link_outboard: Xyz
    lower_camber_link_inboard: Xyz
    lower_camber_link_outboard: Xyz
    shock_inboard: Xyz
    shock_outboard: Xyz
    pivot_inboard: Xyz
    pivot_outboard: Xyz
    wheel_center: Xyz


class VehicleConfig(_Model):
    nickname: str
    shock_min: float = Field(gt=0.0)   # mm
    shock_max: float = Field(gt=0.0)   # mm
    wheel_properties: WheelProps
    mass_properties: MassProps
    front: DoubleAArmCorner
    rear: SemiTrailingLinkCorner

    @model_validator(mode="after")
    def _shock_range(self):
        if self.shock_min >= self.shock_max:
            raise ValueError(f"shock_min ({self.shock_min}) must be below shock_max ({self.shock_max})")
        return self


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def parse_vehicle_config(raw: Any, source: str = "hardpoints") -> VehicleConfig:
    if not isinstance(raw, dict) or len(raw) != 1:
        raise ConfigError(
            f"{source}: a hardpoint file must have exactly one top-level nickname key "
            f"(got {list(raw) if isinstance(raw, dict) else type(raw).__name__})"
        )
    nickname, inner = next(iter(raw.items()))
    if not isinstance(inner, dict):
        raise ConfigError(f"{source}: '{nickname}' must map to a block, not {type(inner).__name__}")
    try:
        return VehicleConfig.model_validate({"nickname": nickname, **inner})
    except ValidationError as exc:
        raise ConfigError(_format(exc, f"{source} ({nickname})")) from None


def load_vehicle_config(path: str | Path) -> VehicleConfig:
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"hardpoints file not found: {p}")
    with p.open(encoding="utf-8") as f:
        return parse_vehicle_config(yaml.safe_load(f), str(p))
