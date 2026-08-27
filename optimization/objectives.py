# default
from abc import ABC, abstractmethod
from itertools import combinations

# third-party
import numpy as np

# ours
from utils.geometry import (
    get_toe_angle,
    get_camber_angle,
    get_caster_angle,
    get_kingpin_angle,
    get_caster_trail,
    get_kingpin_offset_wheel,
)
from utils.spatial import Segment

# ===========================================================================
# Optimizer objectives
# ===========================================================================
#
# Every entry in opt_config.yml's OBJECTIVES list is a mapping with a `type:`
# selecting one of the classes below and carrying that objective's parameters.
# There are no bare class names and no hard-coded objectives -- everything the
# optimizer minimizes is spelled out in the config.
#
#   type: target_curve   track a metric to an arbitrary piecewise-linear curve
#   type: target_range   ... to a line between the two sweep endpoints
#   type: target_const   ... to a flat line at a constant
#   type: target_zero    ... to a flat line at 0
#   type: collision      penalize keepout-zone interference across a sweep
#
# The four target_* classes form an inheritance chain (each a narrower special
# case of the one before): TargetCurve <- TargetRange / TargetConstant <- TargetZero.

class OptimizationObjective(ABC):
    """Base class. `spec` is the raw mapping from the OBJECTIVES list; `config`
    is the full merged run config (only needed by objectives that read shared
    top-level sections, e.g. `collision`)."""

    def __init__(self, spec: dict, config: dict = None):
        self.spec = spec or {}
        self.config = config or {}
        self.scenario = self.spec.get("scenario", self.default_scenario())
        if not self.scenario:
            raise ValueError(f"objective {self.spec!r} is missing required field 'scenario'")
        self.name = self.spec.get("name", self.default_name())

    def default_scenario(self) -> "str | None":
        return None

    def default_name(self) -> str:
        return self.__class__.__name__

    def get_scenario_type(self) -> str:
        return self.scenario

    @abstractmethod
    def calculate_cost(self, results: list) -> float:
        """Scalar cost for one simulation run (lower is better)."""


# ---------------------------------------------------------------------------
# Generic target-tracking objectives
# ---------------------------------------------------------------------------
#
# All of them run `scenario`, read `metric` at every swept step, and cost how
# far the metric drifts from a target curve:
#
#     cost = aggregate(metric(step) - target(x)) / cost_scale
#
# where `x` is the scenario's own sweep variable. A failed/NaN step yields the
# large 1e2 infeasibility penalty.
#
# Shared spec fields:
#   name        label for logs / Pareto plots            (default: the metric)
#   metric      see METRIC_RESOLVERS, or any scalar key present on a step
#   scenario    scenario type to run ('travel', 'front_steer', ...)
#   cost_scale  divisor on the aggregated error -- the error magnitude that
#               should read as a cost of 1.0             (default: 1.0)
#   aggregate   rmse | mean_abs | max_abs | max_abs_plus_range  (default: rmse)

# Scenario-step keys that hold the sweep's independent variable, in priority order.
SCENARIO_X_KEYS = ("x_val", "input")

# Named metrics that need a helper to compute. Anything not listed here is read
# as a plain scalar key off the step dict (e.g. 'ackermann_pct', 'track_change_mm').
METRIC_RESOLVERS = {
    "toe_deg": get_toe_angle,
    "camber_deg": get_camber_angle,
    "caster_deg": get_caster_angle,
    "kingpin_angle_deg": get_kingpin_angle,
    "caster_trail_mm": get_caster_trail,
    "kingpin_offset_wc_mm": get_kingpin_offset_wheel,
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
    if metric in step:
        val = step[metric]
        return float("nan") if val is None else float(val)
    raise KeyError(
        f"unknown metric '{metric}': not in METRIC_RESOLVERS ({sorted(METRIC_RESOLVERS)}) "
        f"and not a key on the scenario step"
    )


class TargetCurve(OptimizationObjective):
    """`type: target_curve` -- track `metric` to an arbitrary piecewise-linear
    curve given as `points: [[x, y], ...]` (x = scenario sweep variable)."""

    def __init__(self, spec: dict, config: dict = None):
        super().__init__(spec, config)
        try:
            self.metric = spec["metric"]
        except KeyError as e:
            raise ValueError(f"objective '{self.name}' is missing required field {e}") from e
        self.cost_scale = float(spec.get("cost_scale", 1.0))
        self.aggregate = spec.get("aggregate", "rmse")
        if self.aggregate not in AGGREGATES:
            raise ValueError(
                f"objective '{self.name}': unknown aggregate '{self.aggregate}' "
                f"(options: {sorted(AGGREGATES)})"
            )
        self._static_points = self._parse_points(spec.get("points"))

    def default_name(self):
        return self.spec.get("metric", self.__class__.__name__)

    @staticmethod
    def _parse_points(points):
        if not points:
            return None
        arr = np.array([[float(x), float(y)] for x, y in points], dtype=float)
        return arr[np.argsort(arr[:, 0])]

    def _target_curve(self, xs: np.ndarray):
        """(x_knots, y_knots) for np.interp. Subclasses that derive the knots
        from the swept x-range override this."""
        if self._static_points is None:
            raise ValueError(f"objective '{self.name}': type target_curve requires 'points'")
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
    """`type: target_range` -- two-knot case: `min` / `max` are the desired
    metric values at the scenario's minimum and maximum swept input."""

    def __init__(self, spec: dict, config: dict = None):
        super().__init__(spec, config)
        try:
            self.y_lo = float(spec["min"])
            self.y_hi = float(spec["max"])
        except KeyError as e:
            raise ValueError(f"objective '{self.name}': type target_range requires {e}") from e

    def _target_curve(self, xs):
        return np.array([np.min(xs), np.max(xs)]), np.array([self.y_lo, self.y_hi])


class TargetConstant(TargetCurve):
    """`type: target_const` -- hold `metric` at a single `const` across the sweep."""

    def __init__(self, spec: dict, config: dict = None):
        super().__init__(spec, config)
        try:
            self.const = float(spec["const"])
        except KeyError as e:
            raise ValueError(f"objective '{self.name}': type target_const requires {e}") from e

    def _target_curve(self, xs):
        return np.array([0.0, 1.0]), np.array([self.const, self.const])


class TargetZero(TargetConstant):
    """`type: target_zero` -- hold `metric` at 0 across the sweep. The common
    case: bump steer, Ackermann error, track change, scrub, etc."""

    def __init__(self, spec: dict, config: dict = None):
        super().__init__({**(spec or {}), "const": 0.0}, config)


# ---------------------------------------------------------------------------
# Collision objective
# ---------------------------------------------------------------------------

class CollisionObjective(OptimizationObjective):
    """
    `type: collision` -- keepout-zone interference penalty. Each zone in the
    top-level KEEPOUT_ZONES defines a shape (cylinder or box) extruded along the
    axis between two named hardpoints. Cost is the mean penetration between every
    checked pair of zones across the sweep.

    If the top-level COLLISION_GROUPS is set (group name -> list of zone names),
    zones in the SAME group are allowed to overlap each other (their pair is
    exempt). Every other pair is checked. Without COLLISION_GROUPS every pair is
    checked (needs >= 2 zones). A group may hold at most MAX_GROUP_SIZE zones;
    members past that cap lose the exemption.

    Spec fields:
      name       label                              (default: "collision")
      scenario   sweep to check over                (default: "droop_steer")
    """
    MAX_GROUP_SIZE = 10

    def default_scenario(self):
        return "droop_steer"

    def default_name(self):
        return "collision"

    def __init__(self, spec: dict, config: dict = None):
        super().__init__(spec, config)
        self.zones = self.config.get("KEEPOUT_ZONES", []) or []
        self.groups = self.config.get("COLLISION_GROUPS", None)
        self._pairs = self._build_pairs()
        if not self._pairs:
            print(f"WARNING: collision objective '{self.name}' has no collidable zone pairs "
                  f"(zones={len(self.zones)}, groups={'set' if self.groups else 'unset'}). "
                  f"Cost will always be 0.")

    def _build_pairs(self):
        if not self.groups:
            return list(combinations(self.zones, 2))

        zones_by_name = {z.get("name"): z for z in self.zones}
        exempt_group = {}  # zone name -> group name, only for zones within the size cap
        for group_name, members in self.groups.items():
            valid_members = []
            for member in members:
                if member not in zones_by_name:
                    print(f"WARNING: COLLISION_GROUPS['{group_name}'] references "
                          f"unknown zone '{member}'.")
                    continue
                valid_members.append(member)
            if len(valid_members) > self.MAX_GROUP_SIZE:
                print(f"WARNING: COLLISION_GROUPS['{group_name}'] has {len(valid_members)} zones, "
                      f"exceeding the max of {self.MAX_GROUP_SIZE}. Only the first "
                      f"{self.MAX_GROUP_SIZE} are exempt from collision with each other; "
                      f"the rest will be checked normally.")
                valid_members = valid_members[:self.MAX_GROUP_SIZE]
            for member in valid_members:
                exempt_group[member] = group_name

        pairs = []
        for za, zb in combinations(self.zones, 2):
            ga = exempt_group.get(za.get("name"))
            gb = exempt_group.get(zb.get("name"))
            if ga is not None and ga == gb:
                continue  # same group -> allowed to overlap, skip
            pairs.append((za, zb))
        return pairs

    @staticmethod
    def _zone_radius(zone: dict) -> float:
        if zone.get("shape") == "box":
            dim1 = float(zone.get("dim1", 0.0))
            dim2 = float(zone.get("dim2", 0.0))
            return 0.5 * float(np.hypot(dim1, dim2))
        return float(zone.get("dim1", 0.0))

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
                    id(z): (self._resolve_point(z["point_a"], step),
                            self._resolve_point(z["point_b"], step))
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

OBJECTIVE_TYPES = {
    "target_curve": TargetCurve,
    "target_range": TargetRange,
    "target_const": TargetConstant,
    "target_zero": TargetZero,
    "collision": CollisionObjective,
}


def build_objective(spec: dict, config: dict = None) -> OptimizationObjective:
    if not isinstance(spec, dict):
        raise TypeError(
            f"each OBJECTIVES entry must be a mapping with a 'type:', got {type(spec).__name__}"
        )
    kind = spec.get("type")
    if kind not in OBJECTIVE_TYPES:
        raise ValueError(
            f"OBJECTIVES entry {spec!r} needs a 'type' in {sorted(OBJECTIVE_TYPES)}"
        )
    return OBJECTIVE_TYPES[kind](spec, config)


def build_objectives(config: dict) -> list:
    """Build every objective from the merged run config's OBJECTIVES list."""
    return [build_objective(spec, config) for spec in config.get("OBJECTIVES", [])]
