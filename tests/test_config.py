"""Schema validation for the run configs (utils.config)."""
import warnings

import pytest

from utils.config import (
    AGGREGATES,
    ConfigError,
    DYN_SIM_TYPES,
    KIN_SIM_TYPES,
    LIMIT_STATS,
    OBJECTIVE_SCENARIOS,
    load_dyn_config,
    load_opt_config,
    load_sweep_config,
    parse_dyn_config,
    parse_opt_config,
    parse_sweep_config,
)

MIN_SWEEP = {
    "HARDPOINTS": "2026",
    "SIM_STEPS": 20,
    "TRAVEL": {"MIN": -30, "MAX": 100},
    "STEER": {"MIN": -40, "MAX": 40},
}
MIN_OBJ = {"type": "target_zero", "metric": "toe_deg", "scenario": "travel"}
MIN_OPT = {"OBJECTIVES": [MIN_OBJ]}


def _q(fn, data):
    """parse, swallowing advisory warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(data)


# --- the shipped configs must always validate ------------------------------

def test_repo_configs_parse():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        load_sweep_config("config/kin_config.yml")
        load_opt_config("config/opt_config.yml")
        load_dyn_config("config/dyn_config.yml")


def test_legacy_dict_roundtrips_keys():
    s = _q(parse_sweep_config, MIN_SWEEP).legacy_dict()
    assert set(s) == {"HARDPOINTS", "SIM_STEPS", "SIMULATION", "HALF", "SIDE", "TRAVEL", "STEER"}
    assert s["TRAVEL"] == {"MIN": -30.0, "MAX": 100.0}

    o = _q(parse_opt_config, MIN_OPT).legacy_dict()
    assert o["OBJECTIVES"][0]["type"] == "target_zero"
    assert o["POP_SIZE"] == 40 and o["N_OFFSPRINGS"] == 10  # defaults


# --- sweep config ----------------------------------------------------------

def test_travel_min_max_ordered():
    with pytest.raises(ConfigError, match="MIN.*MAX"):
        parse_sweep_config({**MIN_SWEEP, "TRAVEL": {"MIN": 10, "MAX": 1}})


def test_unknown_simulation():
    with pytest.raises(ConfigError, match="SIMULATION"):
        parse_sweep_config({**MIN_SWEEP, "SIMULATION": "wobble"})


def test_typo_key_is_rejected():
    bad = {k: v for k, v in MIN_SWEEP.items() if k != "SIM_STEPS"}
    bad["SIM_STPES"] = 20
    with pytest.raises(ConfigError, match="SIM_STPES"):
        parse_sweep_config(bad)


def test_sim_steps_minimum():
    with pytest.raises(ConfigError):
        parse_sweep_config({**MIN_SWEEP, "SIM_STEPS": 1})


# --- opt config ----------------------------------------------------------

def test_unknown_objective_type():
    with pytest.raises(ConfigError, match="target_zero"):
        parse_opt_config({"OBJECTIVES": [{"type": "nope", "scenario": "travel"}]})


def test_limit_needs_a_band():
    with pytest.raises(ConfigError, match="min.*max|band"):
        parse_opt_config({"OBJECTIVES": [{
            "type": "limit", "metric": "axle_plunge_mm",
            "scenario": "sweep_space", "bounds": {"value": {}}}]})


def test_limit_bad_stat():
    with pytest.raises(ConfigError):
        parse_opt_config({"OBJECTIVES": [{
            "type": "limit", "metric": "x", "scenario": "sweep_space",
            "bounds": {"nonsense": {"min": 1}}}]})


def test_collision_needs_two_zones():
    with pytest.raises(ConfigError, match="KEEPOUT_ZONES"):
        parse_opt_config({
            "OBJECTIVES": [{"type": "collision", "scenario": "sweep_space"}],
            "KEEPOUT_ZONES": [{"name": "a", "point_a": "p", "point_b": "q", "dim1": 5}],
        })


def test_collision_group_references_known_zone():
    with pytest.raises(ConfigError, match="unknown zone"):
        parse_opt_config({
            **MIN_OPT,
            "KEEPOUT_ZONES": [{"name": "a", "point_a": "p", "point_b": "q", "dim1": 5}],
            "COLLISION_GROUPS": {"g": ["a", "ghost"]},
        })


def test_free_point_axis_order():
    with pytest.raises(ConfigError, match="lo.*hi"):
        parse_opt_config({**MIN_OPT, "FREE_POINTS": {"tie_rod_inboard": {"x": [10, -10]}}})


def test_free_point_needs_an_axis():
    with pytest.raises(ConfigError):
        parse_opt_config({**MIN_OPT, "FREE_POINTS": {"tie_rod_inboard": {}}})


def test_bad_aggregate():
    with pytest.raises(ConfigError):
        parse_opt_config({"OBJECTIVES": [{**MIN_OBJ, "aggregate": "bogus"}]})


def test_offspring_over_pop_warns():
    with pytest.warns(UserWarning, match="N_OFFSPRINGS"):
        parse_opt_config({**MIN_OPT, "POP_SIZE": 10, "N_OFFSPRINGS": 40})


def test_unknown_metric_warns_but_parses():
    with pytest.warns(UserWarning, match="metric"):
        cfg = parse_opt_config({"OBJECTIVES": [{
            "type": "target_zero", "metric": "made_up_field", "scenario": "travel"}]})
    assert cfg.objectives[0].metric == "made_up_field"


def test_dotted_metric_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        parse_opt_config({"OBJECTIVES": [{
            "type": "target_zero", "metric": "axle_data.plunge_mm", "scenario": "sweep_space"}]})


# --- dyn config ----------------------------------------------------------

def test_dyn_bad_simulation():
    with pytest.raises(ConfigError):
        parse_dyn_config({"SIMULATION": "terrain"})


# --- vocabularies stay in sync with the runtime ---------------------------

def test_aggregates_match_runtime():
    from optimization.objectives import AGGREGATES as RT
    assert set(AGGREGATES) == set(RT)


def test_limit_stats_match_runtime():
    from optimization.objectives import STAT_REDUCERS
    assert set(LIMIT_STATS) == set(STAT_REDUCERS)


def test_objective_scenarios_are_buildable():
    from optimization.engine import SuspensionOptimizer
    # every scenario an objective may name must be one build_scenario knows
    known = {"steer", "travel", "droop_steer", "jounce_steer", "left_travel",
             "right_travel", "sweep_space", "front_steer", "heave", "roll"}
    assert set(OBJECTIVE_SCENARIOS) <= known


def test_kin_and_dyn_sim_type_lists_nonempty():
    assert "sweep_space" in KIN_SIM_TYPES and "extreme" in KIN_SIM_TYPES
    assert set(DYN_SIM_TYPES) == {"static", "shock_dyno"}
