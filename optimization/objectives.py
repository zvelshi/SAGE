# default
from abc import ABC, abstractmethod
from itertools import combinations

# third-party
import numpy as np

# ours
from utils.config import (
    CollisionSpec,
    LimitSpec,
    ObjectiveSpecT,
    OptConfig,
    TargetConstSpec,
    TargetCurveSpec,
    TargetRangeSpec,
    TargetZeroSpec,
)
from utils.geometry import (
    get_toe_angle,
    get_camber_angle,
    get_caster_angle,
    get_kingpin_angle,
    get_caster_trail,
    get_kingpin_offset_wheel,
)
from utils.logging_setup import get_logger
from utils.spatial import Segment

log = get_logger(__name__)

# ===========================================================================
# Optimizer objectives
# ===========================================================================
#
# Objectives are built from validated `*Spec` models (see utils.config) -- one
# per entry in opt_config.yml's OBJECTIVES list. The spec already guarantees the
# fields are present and well-formed, so these classes just do the maths.
#
#   type: target_curve   track a metric to an arbitrary piecewise-linear curve
#   type: target_range   ... to a line between the two sweep endpoints
#   type: target_const   ... to a flat line at a constant
#   type: target_zero    ... to a flat line at 0
#   type: limit          keep sweep statistics of a metric inside bands
#   type: collision      penalize keepout-zone interference across a sweep


class OptimizationObjective(ABC):
    """`spec` is the validated model from the OBJECTIVES list; `opt` is the full
    OptConfig (only `collision` needs it, for the shared zone/group sections)."""

    def __init__(self, spec, opt: OptConfig | None = None):
        self.spec = spec
        self.opt = opt
        self.scenario: str = spec.scenario
        self.name: str = spec.name or self._default_name()

    def _default_name(self) -> str:
        return getattr(self.spec, "metric", None) or self.__class__.__name__

    def get_scenario_type(self) -> str:
        return self.scenario

    @abstractmethod
    def calculate_cost(self, results: list) -> float:
        """Scalar cost for one simulation run (lower is better)."""


# ---------------------------------------------------------------------------
# Metric resolution
# ---------------------------------------------------------------------------

# Scenario-step keys that hold the sweep's independent variable, in priority order.
SCENARIO_X_KEYS = ("x_val", "input")


def _axle_plunge_mm(step: dict) -> float:
    v = (step.get("axle_data") or {}).get("plunge_mm")
    return float("nan") if v is None else float(v)


def _axle_angle_deg(step: dict) -> float:
    a = step.get("axle_data") or {}
    ib, ob = a.get("angle_ib_deg"), a.get("angle_ob_deg")
    return float("nan") if ib is None or ob is None else float(max(ib, ob))


def _ground_clearance_mm(step: dict) -> float:
    """The binding (smaller) of the front / rear ground clearances."""
    vals = [step.get("front_ground_clearance_mm"), step.get("rear_ground_clearance_mm")]
    vals = [v for v in vals if v is not None]
    return float(min(vals)) if vals else float("nan")


# Named metrics that need a helper. Anything not listed here is read as a plain
# scalar key off the step dict, with dotted names ('axle_data.plunge_mm') walking
# into nested dicts.
METRIC_RESOLVERS = {
    "toe_deg": get_toe_angle,
    "camber_deg": get_camber_angle,
    "caster_deg": get_caster_angle,
    "kingpin_angle_deg": get_kingpin_angle,
    "caster_trail_mm": get_caster_trail,
    "kingpin_offset_wc_mm": get_kingpin_offset_wheel,
    "axle_plunge_mm": _axle_plunge_mm,
    "axle_angle_deg": _axle_angle_deg,
    "ground_clearance_mm": _ground_clearance_mm,
}

AGGREGATES = {
    "rmse": lambda e: float(np.sqrt(np.mean(e ** 2))),
    "mean_abs": lambda e: float(np.mean(np.abs(e))),
    "max_abs": lambda e: float(np.max(np.abs(e))),
    "max_abs_plus_range": lambda e: float(np.max(np.abs(e)) + (np.max(e) - np.min(e))),
}


def _resolve_scenario_x(step: dict) -> float:
    for key in SCENARIO_X_KEYS:
        if step.get(key) is not None:
            return float(step[key])
    raise KeyError(f"scenario step has no sweep-variable key (looked for {SCENARIO_X_KEYS})")


def _resolve_metric(metric: str, step: dict) -> float:
    fn = METRIC_RESOLVERS.get(metric)
    if fn is not None:
        return float(fn(step))
    node = step
    for part in metric.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(
                f"unknown metric '{metric}': not in METRIC_RESOLVERS ({sorted(METRIC_RESOLVERS)}) "
                f"and not a (dotted) key on the scenario step"
            )
        node = node[part]
    if node is None:
        return float("nan")
    if isinstance(node, dict):
        raise KeyError(f"metric '{metric}' resolves to a dict, not a scalar")
    return float(node)


# ---------------------------------------------------------------------------
# Target-tracking objectives
# ---------------------------------------------------------------------------
#
# Run `scenario`, read `metric` at every swept step, cost how far the metric
# drifts from a target curve:  aggregate(metric(step) - target(x)) / cost_scale
# where `x` is the scenario's own sweep variable. A failed/NaN step -> 1e2.

class TargetCurve(OptimizationObjective):
    """`type: target_curve` -- track `metric` to the piecewise-linear curve
    through `points` (x = scenario sweep variable)."""

    def __init__(self, spec: TargetCurveSpec, opt: OptConfig | None = None):
        super().__init__(spec, opt)
        self.metric = spec.metric
        self.cost_scale = spec.cost_scale
        self.aggregate = spec.aggregate
        pts = getattr(spec, "points", None)
        self._static_points = self._parse_points(pts) if pts else None

    @staticmethod
    def _parse_points(points):
        arr = np.array([[float(x), float(y)] for x, y in points], dtype=float)
        return arr[np.argsort(arr[:, 0])]

    def _target_curve(self, xs: np.ndarray):
        """(x_knots, y_knots) for np.interp. Subclasses that derive the knots
        from the swept x-range override this."""
        return self._static_points[:, 0], self._static_points[:, 1]

    def calculate_cost(self, results):
        xs = np.array([_resolve_scenario_x(s) for s in results], dtype=float)
        vals = np.array([_resolve_metric(self.metric, s) for s in results], dtype=float)
        kx, ky = self._target_curve(xs)
        err = vals - np.interp(xs, kx, ky)
        if not np.all(np.isfinite(err)):
            return 1e2
        return AGGREGATES[self.aggregate](err) / self.cost_scale


class TargetRange(TargetCurve):
    """`type: target_range` -- `min` / `max` are the desired metric values at the
    scenario's minimum and maximum swept input."""

    def __init__(self, spec: TargetRangeSpec, opt: OptConfig | None = None):
        super().__init__(spec, opt)
        self.y_lo = spec.min
        self.y_hi = spec.max

    def _target_curve(self, xs):
        return np.array([np.min(xs), np.max(xs)]), np.array([self.y_lo, self.y_hi])


class TargetConstant(TargetCurve):
    """`type: target_const` -- hold `metric` at a single `const` across the sweep."""

    def __init__(self, spec: TargetConstSpec, opt: OptConfig | None = None):
        super().__init__(spec, opt)
        self.const = getattr(spec, "const", 0.0)

    def _target_curve(self, xs):
        return np.array([0.0, 1.0]), np.array([self.const, self.const])


class TargetZero(TargetConstant):
    """`type: target_zero` -- hold `metric` at 0 across the sweep. The common
    case: bump steer, Ackermann error, track change, scrub, etc."""


# ---------------------------------------------------------------------------
# Limit objective
# ---------------------------------------------------------------------------
#
# Keeps chosen statistics of a metric across the sweep inside bands, costing
# only the spill-out:
#
#     violation = sum over (stat, band) of
#                   max(0, band.min - stat(series)) + max(0, stat(series) - band.max)
#     cost      = aggregate(all violations) / cost_scale
#
# `bounds` maps a stat -> {min?, max?}. Stats: value (per step) | max | min |
# mean | range | abs_max. NaN/failed step -> 1e2.

STAT_REDUCERS = {
    "value": lambda a: a,
    "max": lambda a: np.array([np.max(a)]),
    "min": lambda a: np.array([np.min(a)]),
    "mean": lambda a: np.array([np.mean(a)]),
    "range": lambda a: np.array([np.max(a) - np.min(a)]),
    "abs_max": lambda a: np.array([np.max(np.abs(a))]),
}


class MetricLimit(OptimizationObjective):
    """`type: limit` -- keep the `bounds` statistics of `metric` inside their
    bands, costing only the excursion outside."""

    def __init__(self, spec: LimitSpec, opt: OptConfig | None = None):
        super().__init__(spec, opt)
        self.metric = spec.metric
        self.cost_scale = spec.cost_scale
        self.aggregate = spec.aggregate
        self.constraints = [(stat, band.min, band.max) for stat, band in spec.bounds.items()]

    def calculate_cost(self, results):
        vals = np.array([_resolve_metric(self.metric, s) for s in results], dtype=float)
        finite = np.isfinite(vals)
        if not finite.all():
            # heave/roll can drop a few extreme steps when a candidate's motion
            # ratio puts a shock past its limit before the swept range ends --
            # score on what solved, but a mostly-failed sweep is infeasible.
            if finite.sum() < max(2, int(0.6 * len(vals))):
                return 1e2
            vals = vals[finite]
        parts = []
        for stat, lo, hi in self.constraints:
            v = STAT_REDUCERS[stat](vals)
            viol = np.zeros_like(v, dtype=float)
            if lo is not None:
                viol = viol + np.clip(lo - v, 0.0, None)
            if hi is not None:
                viol = viol + np.clip(v - hi, 0.0, None)
            parts.append(viol)
        return AGGREGATES[self.aggregate](np.concatenate(parts)) / self.cost_scale


# ---------------------------------------------------------------------------
# Collision objective
# ---------------------------------------------------------------------------

class CollisionObjective(OptimizationObjective):
    """
    `type: collision` -- keepout-zone interference penalty. Each zone in the
    top-level KEEPOUT_ZONES is a shape (cylinder or box) extruded along the axis
    between two named hardpoints; cost is the mean penetration between every
    checked pair of zones across the sweep.

    Zones in the same COLLISION_GROUPS group are allowed to overlap (that pair is
    exempt); every other pair is checked. The schema guarantees groups only
    reference real zones and stay within the size cap.
    """

    def _default_name(self) -> str:
        return "collision"

    def __init__(self, spec: CollisionSpec, opt: OptConfig | None = None):
        super().__init__(spec, opt)
        self.zones = list(opt.keepout_zones) if opt else []
        self.groups = (opt.collision_groups if opt else None)
        self._pairs = self._build_pairs()
        if not self._pairs:
            log.warning("collision objective '%s' has no collidable zone pairs "
                        "(zones=%d, groups=%s); cost will always be 0",
                        self.name, len(self.zones), "set" if self.groups else "unset")

    def _build_pairs(self):
        if not self.groups:
            return list(combinations(self.zones, 2))
        member_group = {m: g for g, members in self.groups.items() for m in members}
        return [
            (za, zb) for za, zb in combinations(self.zones, 2)
            if member_group.get(za.name) is None
            or member_group.get(za.name) != member_group.get(zb.name)
        ]

    @staticmethod
    def _zone_radius(zone) -> float:
        if zone.shape == "box":
            return 0.5 * float(np.hypot(zone.dim1, zone.dim2 or zone.dim1))
        return float(zone.dim1)

    @staticmethod
    def _resolve_point(name: str, step: dict) -> np.ndarray:
        return np.asarray(step[name], dtype=float)

    def calculate_cost(self, results):
        if not self._pairs:
            return 0.0

        total_violation = 0.0
        for step in results:
            try:
                endpoints = {
                    id(z): (self._resolve_point(z.point_a, step),
                            self._resolve_point(z.point_b, step))
                    for z in self.zones
                }
            except KeyError as e:
                raise KeyError(f"KEEPOUT_ZONES point {e} not found in scenario step output") from e

            for za, zb in self._pairs:
                pa1, pa2 = endpoints[id(za)]
                pb1, pb2 = endpoints[id(zb)]
                d = Segment(pa1, pa2).distance_to_segment(Segment(pb1, pb2))
                r_sum = self._zone_radius(za) + self._zone_radius(zb)
                total_violation += max(0.0, r_sum - d)

        return total_violation / len(results)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_OBJECTIVE_FOR_SPEC = {
    TargetCurveSpec: TargetCurve,
    TargetRangeSpec: TargetRange,
    TargetConstSpec: TargetConstant,
    TargetZeroSpec: TargetZero,
    LimitSpec: MetricLimit,
    CollisionSpec: CollisionObjective,
}


def objective_from_spec(spec: ObjectiveSpecT, opt: OptConfig | None = None) -> OptimizationObjective:
    return _OBJECTIVE_FOR_SPEC[type(spec)](spec, opt)


def build_objectives(opt: OptConfig) -> list[OptimizationObjective]:
    """Build every objective from a validated OptConfig's OBJECTIVES list."""
    return [objective_from_spec(spec, opt) for spec in opt.objectives]
