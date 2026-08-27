"""Typed, validated run configuration.

The YAML files under ``config/`` stay the on-disk format; these models are the
in-memory contract. Load a file with :func:`load_sweep_config` /
:func:`load_opt_config` / :func:`load_dyn_config` -- a bad key, a misspelled
objective type, ``TRAVEL.MIN > MAX`` etc. now fail immediately with a message
that names the offending field, instead of silently doing nothing downstream.

Nothing else in the codebase has to change at once: every model exposes
``.legacy_dict()`` returning exactly the ``UPPERCASE``-keyed dict the current
scenario / engine / objective code already consumes, so call sites can be
migrated one at a time.
"""
from __future__ import annotations

# default
from pathlib import Path
from typing import Annotated, Any, Literal, Union

# third-party
import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

# ---------------------------------------------------------------------------
# Vocabularies (kept in sync with the runtime by tests/test_config.py)
# ---------------------------------------------------------------------------

CORNER_SIM_TYPES = (
    "travel", "steer", "droop_steer", "jounce_steer",
    "left_travel", "right_travel", "sweep_space",
)
HALF_SIM_TYPES = ("front_steer",)
FULL_SIM_TYPES = ("heave", "roll")
KIN_SIM_TYPES = CORNER_SIM_TYPES + HALF_SIM_TYPES + FULL_SIM_TYPES + ("extreme",)
DYN_SIM_TYPES = ("static", "shock_dyno")

OBJECTIVE_SCENARIOS = CORNER_SIM_TYPES + HALF_SIM_TYPES + FULL_SIM_TYPES

AGGREGATES = ("rmse", "mean_abs", "max_abs", "max_abs_plus_range")
LIMIT_STATS = ("value", "max", "min", "mean", "range", "abs_max")

# Named metric helpers (objectives also accept any scalar / dotted step key, so
# this is advisory only -- an unknown name warns, it does not fail).
KNOWN_METRICS = (
    "toe_deg", "camber_deg", "caster_deg", "kingpin_angle_deg",
    "caster_trail_mm", "kingpin_offset_wc_mm",
    "axle_plunge_mm", "axle_angle_deg", "ground_clearance_mm",
    "ackermann_pct", "track_change_mm",
)


class ConfigError(ValueError):
    """A run config failed schema validation."""


def _format(exc: ValidationError, source: str) -> str:
    lines = [f"{source}: {exc.error_count()} problem(s)"]
    for e in exc.errors():
        loc = ".".join(str(p) for p in e["loc"]) or "(root)"
        lines.append(f"  - {loc}: {e['msg']}")
    return "\n".join(lines)


class _Model(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")


# ---------------------------------------------------------------------------
# Shared value types
# ---------------------------------------------------------------------------

class Range(_Model):
    min: float = Field(alias="MIN")
    max: float = Field(alias="MAX")

    @model_validator(mode="after")
    def _ordered(self):
        if self.min > self.max:
            raise ValueError(f"MIN ({self.min}) must not exceed MAX ({self.max})")
        return self

    def legacy_dict(self) -> dict:
        return {"MIN": self.min, "MAX": self.max}


class AxisBox(_Model):
    """Per-axis search offsets for one free hardpoint: ``[lo, hi]`` millimetres
    relative to the point's current position. ``lo == hi`` freezes that axis."""
    x: tuple[float, float] | None = None
    y: tuple[float, float] | None = None
    z: tuple[float, float] | None = None

    @model_validator(mode="after")
    def _nonempty_and_ordered(self):
        axes = {"x": self.x, "y": self.y, "z": self.z}
        if all(v is None for v in axes.values()):
            raise ValueError("free point has no x/y/z range")
        for name, v in axes.items():
            if v is not None and v[0] > v[1]:
                raise ValueError(f"{name}: lo ({v[0]}) must not exceed hi ({v[1]})")
        return self

    def legacy_dict(self) -> dict:
        return {k: list(v) for k, v in
                (("x", self.x), ("y", self.y), ("z", self.z)) if v is not None}


class KeepoutZone(_Model):
    name: str
    point_a: str
    point_b: str
    shape: Literal["cylinder", "box"] = "cylinder"
    dim1: float = Field(gt=0.0)
    dim2: float | None = Field(default=None, gt=0.0)

    @model_validator(mode="after")
    def _box_needs_dim2(self):
        if self.shape == "box" and self.dim2 is None:
            raise ValueError("shape 'box' requires dim2")
        return self

    def legacy_dict(self) -> dict:
        return self.model_dump(exclude_none=True)


# ---------------------------------------------------------------------------
# Objectives (discriminated on `type`)
# ---------------------------------------------------------------------------

class _ObjBase(_Model):
    name: str | None = None
    scenario: str
    cost_scale: float = Field(default=1.0, gt=0.0)

    @field_validator("scenario")
    @classmethod
    def _known_scenario(cls, v: str) -> str:
        if v not in OBJECTIVE_SCENARIOS:
            raise ValueError(f"unknown scenario '{v}' (options: {', '.join(OBJECTIVE_SCENARIOS)})")
        return v

    def legacy_dict(self) -> dict:
        return self.model_dump(exclude_none=True)


class _MetricObjBase(_ObjBase):
    metric: str
    aggregate: Literal[AGGREGATES] = "rmse"  # type: ignore[valid-type]

    @model_validator(mode="after")
    def _warn_unknown_metric(self):
        if "." not in self.metric and self.metric not in KNOWN_METRICS:
            import warnings
            warnings.warn(
                f"objective metric '{self.metric}' is not a known helper; it must "
                f"be a scalar key present on every {self.scenario} step",
                stacklevel=2,
            )
        return self


class TargetCurveSpec(_MetricObjBase):
    type: Literal["target_curve"]
    points: list[tuple[float, float]] = Field(min_length=2)


class TargetRangeSpec(_MetricObjBase):
    type: Literal["target_range"]
    min: float
    max: float


class TargetConstSpec(_MetricObjBase):
    type: Literal["target_const"]
    const: float


class TargetZeroSpec(_MetricObjBase):
    type: Literal["target_zero"]


class _Band(_Model):
    min: float | None = None
    max: float | None = None

    @model_validator(mode="after")
    def _has_a_bound(self):
        if self.min is None and self.max is None:
            raise ValueError("band needs a 'min' and/or 'max'")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError(f"min ({self.min}) exceeds max ({self.max})")
        return self


class LimitSpec(_MetricObjBase):
    type: Literal["limit"]
    aggregate: Literal[AGGREGATES] = "max_abs"  # type: ignore[valid-type]
    bounds: dict[Literal[LIMIT_STATS], _Band] = Field(min_length=1)  # type: ignore[valid-type]


class CollisionSpec(_ObjBase):
    type: Literal["collision"]
    scenario: str = "droop_steer"


ObjectiveSpec = Annotated[
    Union[
        TargetCurveSpec, TargetRangeSpec, TargetConstSpec, TargetZeroSpec,
        LimitSpec, CollisionSpec,
    ],
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Top-level files
# ---------------------------------------------------------------------------

class SweepConfig(_Model):
    """``config/kin_config.yml``"""
    hardpoints: str = Field(alias="HARDPOINTS")
    sim_steps: int = Field(alias="SIM_STEPS", ge=2)
    simulation: str = Field(alias="SIMULATION", default="travel")
    half: Literal["front", "rear"] = Field(alias="HALF", default="front")
    side: Literal["left", "right"] = Field(alias="SIDE", default="right")
    travel: Range = Field(alias="TRAVEL")
    steer: Range = Field(alias="STEER")

    @field_validator("simulation")
    @classmethod
    def _known_sim(cls, v: str) -> str:
        if v not in KIN_SIM_TYPES:
            raise ValueError(f"unknown SIMULATION '{v}' (options: {', '.join(KIN_SIM_TYPES)})")
        return v

    def legacy_dict(self) -> dict:
        return {
            "HARDPOINTS": self.hardpoints,
            "SIM_STEPS": self.sim_steps,
            "SIMULATION": self.simulation,
            "HALF": self.half,
            "SIDE": self.side,
            "TRAVEL": self.travel.legacy_dict(),
            "STEER": self.steer.legacy_dict(),
        }


class OptConfig(_Model):
    """``config/opt_config.yml``"""
    pop_size: int = Field(alias="POP_SIZE", default=40, gt=0)
    n_offsprings: int = Field(alias="N_OFFSPRINGS", default=10, gt=0)
    max_gen: int = Field(alias="MAX_GEN", default=50, gt=0)
    m_prob: float = Field(alias="M_PROB", default=1.0, ge=0.0, le=1.0)
    m_eta: float = Field(alias="M_ETA", default=15.0, gt=0.0)
    objectives: list[ObjectiveSpec] = Field(alias="OBJECTIVES", min_length=1)
    free_points: dict[str, AxisBox] = Field(alias="FREE_POINTS", default_factory=dict)
    keepout_zones: list[KeepoutZone] = Field(alias="KEEPOUT_ZONES", default_factory=list)
    collision_groups: dict[str, list[str]] | None = Field(alias="COLLISION_GROUPS", default=None)

    _MAX_GROUP_SIZE = 10

    @model_validator(mode="after")
    def _cross_checks(self):
        if self.n_offsprings > self.pop_size:
            import warnings
            warnings.warn(
                f"N_OFFSPRINGS ({self.n_offsprings}) > POP_SIZE ({self.pop_size}); "
                f"NSGA-II ranks pop+offspring each generation, so the extra offspring "
                f"just cost evaluations",
                stacklevel=2,
            )
        if self.collision_groups:
            zone_names = {z.name for z in self.keepout_zones}
            for group, members in self.collision_groups.items():
                for m in members:
                    if m not in zone_names:
                        raise ValueError(
                            f"COLLISION_GROUPS.{group} references unknown zone '{m}'"
                        )
                if len(members) > self._MAX_GROUP_SIZE:
                    raise ValueError(
                        f"COLLISION_GROUPS.{group} has {len(members)} zones "
                        f"(max {self._MAX_GROUP_SIZE})"
                    )
        if any(o.type == "collision" for o in self.objectives) and len(self.keepout_zones) < 2:
            raise ValueError("a 'collision' objective needs at least 2 KEEPOUT_ZONES")
        return self

    def legacy_dict(self) -> dict:
        d: dict[str, Any] = {
            "POP_SIZE": self.pop_size,
            "N_OFFSPRINGS": self.n_offsprings,
            "MAX_GEN": self.max_gen,
            "M_PROB": self.m_prob,
            "M_ETA": self.m_eta,
            "OBJECTIVES": [o.legacy_dict() for o in self.objectives],
            "FREE_POINTS": {k: v.legacy_dict() for k, v in self.free_points.items()},
            "KEEPOUT_ZONES": [z.legacy_dict() for z in self.keepout_zones],
        }
        if self.collision_groups is not None:
            d["COLLISION_GROUPS"] = self.collision_groups
        return d


class DynConfig(_Model):
    """``config/dyn_config.yml``"""
    simulation: Literal[DYN_SIM_TYPES] = Field(alias="SIMULATION", default="shock_dyno")  # type: ignore[valid-type]
    sol_dt: float = Field(alias="SOL_DT", default=0.001, gt=0.0)
    viz_dt: float = Field(alias="VIZ_DT", default=0.01, gt=0.0)
    hoist_duration: float = Field(alias="HOIST_DURATION", default=0.5, ge=0.0)
    hoist_height: float = Field(alias="HOIST_HEIGHT", default=0.5)  # metres
    max_sim_time: float = Field(alias="MAX_SIM_TIME", default=60.0, gt=0.0)
    dyno_stroke: float = Field(alias="DYNO_STROKE", default=50.0, gt=0.0)   # mm
    dyno_frequency: float = Field(alias="DYNO_FREQUENCY", default=1.63, gt=0.0)  # Hz

    def legacy_dict(self) -> dict:
        return {
            "SIMULATION": self.simulation,
            "SOL_DT": self.sol_dt, "VIZ_DT": self.viz_dt,
            "HOIST_DURATION": self.hoist_duration, "HOIST_HEIGHT": self.hoist_height,
            "MAX_SIM_TIME": self.max_sim_time,
            "DYNO_STROKE": self.dyno_stroke, "DYNO_FREQUENCY": self.dyno_frequency,
        }


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load(model: type[BaseModel], data: Any, source: str):
    if not isinstance(data, dict):
        raise ConfigError(f"{source}: expected a YAML mapping, got {type(data).__name__}")
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(_format(exc, source)) from None


def parse_sweep_config(data: Any, source: str = "kin config") -> SweepConfig:
    return _load(SweepConfig, data, source)


def parse_opt_config(data: Any, source: str = "opt config") -> OptConfig:
    return _load(OptConfig, data, source)


def parse_dyn_config(data: Any, source: str = "dyn config") -> DynConfig:
    return _load(DynConfig, data, source)


def _read_yaml(path: str | Path) -> Any:
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"config file not found: {p}")
    with p.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_sweep_config(path: str | Path) -> SweepConfig:
    return parse_sweep_config(_read_yaml(path), str(path))


def load_opt_config(path: str | Path) -> OptConfig:
    return parse_opt_config(_read_yaml(path), str(path))


def load_dyn_config(path: str | Path) -> DynConfig:
    return parse_dyn_config(_read_yaml(path), str(path))
