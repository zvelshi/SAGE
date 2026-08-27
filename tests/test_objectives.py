"""Objective construction from validated specs, and cost maths."""
import warnings

import numpy as np
import pytest

from optimization.objectives import (
    CollisionObjective,
    MetricLimit,
    TargetZero,
    build_objectives,
    objective_from_spec,
)
from utils.config import parse_opt_config


def _opt(objectives, **extra):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return parse_opt_config({"OBJECTIVES": objectives, **extra})


def _axis_step(deg_x, x):
    return {"wheel_axis": np.array([np.sin(np.deg2rad(deg_x)), 0.0, 0.0]), "x_val": float(x)}


def test_build_objectives_dispatch():
    opt = _opt([
        {"type": "target_zero", "metric": "toe_deg", "scenario": "travel"},
        {"type": "limit", "metric": "axle_plunge_mm", "scenario": "sweep_space",
         "bounds": {"value": {"min": -15, "max": 15}}},
    ])
    objs = build_objectives(opt)
    assert [type(o).__name__ for o in objs] == ["TargetZero", "MetricLimit"]
    assert objs[0].name == "toe_deg" and objs[0].scenario == "travel"


def test_named_objective_keeps_its_name():
    (obj,) = build_objectives(_opt([
        {"name": "BumpSteer", "type": "target_zero", "metric": "toe_deg", "scenario": "travel"}]))
    assert obj.name == "BumpSteer"


def test_target_zero_cost_matches_hand_calc():
    spec = _opt([{"type": "target_zero", "metric": "toe_deg", "scenario": "travel",
                  "aggregate": "max_abs_plus_range", "cost_scale": 150.0}]).objectives[0]
    obj = objective_from_spec(spec)
    steps = [_axis_step(t * 0.1, t) for t in (-50, -25, 0, 25, 50)]
    from utils.geometry import get_toe_angle
    toes = np.array([get_toe_angle(s) for s in steps])
    expected = (np.max(np.abs(toes)) + (np.max(toes) - np.min(toes))) / 150.0
    assert obj.calculate_cost(steps) == pytest.approx(expected)


def test_limit_window_penalizes_only_excursion():
    spec = _opt([{"type": "limit", "metric": "axle_plunge_mm", "scenario": "sweep_space",
                  "bounds": {"value": {"min": -15, "max": 15}}, "cost_scale": 5.0}]).objectives[0]
    obj = MetricLimit(spec)
    mk = lambda vs: [{"axle_data": {"plunge_mm": v}, "x_val": 0.0} for v in vs]
    assert obj.calculate_cost(mk([-10, 0, 12])) == 0.0
    assert obj.calculate_cost(mk([-20, 0, 18])) == pytest.approx(max(5.0, 3.0) / 5.0)


def test_limit_two_stat_bounds():
    spec = _opt([{"type": "limit", "metric": "ground_clearance_mm", "scenario": "heave",
                  "bounds": {"max": {"min": 406.4}, "min": {"min": 76.2}}, "cost_scale": 50.0}]).objectives[0]
    obj = MetricLimit(spec)
    mk = lambda vs: [{"front_ground_clearance_mm": v, "rear_ground_clearance_mm": v + 10} for v in vs]
    assert obj.calculate_cost(mk([80, 200, 410])) == 0.0
    assert obj.calculate_cost(mk([50, 390])) == pytest.approx(max(406.4 - 390, 76.2 - 50) / 50.0)


def test_nan_step_gives_infeasible_penalty():
    spec = _opt([{"type": "target_zero", "metric": "ackermann_pct", "scenario": "front_steer"}]).objectives[0]
    obj = objective_from_spec(spec)
    assert obj.calculate_cost([{"ackermann_pct": None, "input": 0.0},
                               {"ackermann_pct": 1.0, "input": 10.0}]) == 1e2


def test_collision_reads_zones_from_opt_config():
    opt = _opt(
        [{"type": "collision", "scenario": "sweep_space"}],
        KEEPOUT_ZONES=[
            {"name": "a", "point_a": "p", "point_b": "q", "shape": "cylinder", "dim1": 10},
            {"name": "b", "point_a": "r", "point_b": "s", "shape": "cylinder", "dim1": 10},
        ],
    )
    (obj,) = build_objectives(opt)
    assert isinstance(obj, CollisionObjective)
    assert len(obj._pairs) == 1  # a-b


def test_collision_group_exempts_pair():
    opt = _opt(
        [{"type": "collision", "scenario": "sweep_space"}],
        KEEPOUT_ZONES=[
            {"name": "a", "point_a": "p", "point_b": "q", "dim1": 10},
            {"name": "b", "point_a": "r", "point_b": "s", "dim1": 10},
        ],
        COLLISION_GROUPS={"g": ["a", "b"]},
    )
    (obj,) = build_objectives(opt)
    assert obj._pairs == []
