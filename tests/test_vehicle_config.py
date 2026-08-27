"""Hardpoint-file schema (models.vehicle_config) and Vehicle build."""
import copy

import pytest
import yaml

from models.vehicle import Vehicle
from models.vehicle_config import (
    VehicleConfig,
    load_vehicle_config,
    parse_vehicle_config,
)
from utils.config import ConfigError

REAL = "config/hardpoints/2026.yml"


@pytest.fixture
def raw():
    with open(REAL) as f:
        return yaml.safe_load(f)


def test_real_file_loads():
    vc = load_vehicle_config(REAL)
    assert vc.nickname == "baja_2026"
    assert len(vc.front.wheel_center) == 3
    assert vc.mass_properties.unsprung_mass["fl"] == 18
    assert vc.rear.shock_location in ("upper", "lower")


def test_build_vehicle_and_solve():
    v = Vehicle(load_vehicle_config(REAL))
    step = v.get_corner_from_id([1, 0]).solver.solve(travel_mm=10.0, steer_mm=5.0)
    assert step is not None
    assert abs(step["wheel_axis"][1]) > 0.9  # roughly lateral


def test_single_nickname_key_required(raw):
    with pytest.raises(ConfigError, match="one top-level"):
        parse_vehicle_config({**raw, "extra_car": {}})


def test_missing_point_is_named(raw):
    bad = copy.deepcopy(raw)
    del bad["baja_2026"]["front"]["tie_rod_inboard"]
    with pytest.raises(ConfigError, match="tie_rod_inboard"):
        parse_vehicle_config(bad)


def test_two_element_coordinate_rejected(raw):
    bad = copy.deepcopy(raw)
    bad["baja_2026"]["front"]["wheel_center"] = [1.0, 2.0]
    with pytest.raises(ConfigError, match="wheel_center"):
        parse_vehicle_config(bad)


def test_unknown_corner_key_rejected(raw):
    bad = copy.deepcopy(raw)
    bad["baja_2026"]["front"]["upper_a_arm_frnot"] = [1, 2, 3]
    with pytest.raises(ConfigError, match="upper_a_arm_frnot"):
        parse_vehicle_config(bad)


def test_shock_range_ordered(raw):
    bad = copy.deepcopy(raw)
    bad["baja_2026"]["shock_min"] = 999.0
    with pytest.raises(ConfigError, match="shock_min"):
        parse_vehicle_config(bad)


def test_unsprung_mass_needs_all_corners(raw):
    bad = copy.deepcopy(raw)
    del bad["baja_2026"]["mass_properties"]["unsprung_mass"]["rl"]
    with pytest.raises(ConfigError, match="rl"):
        parse_vehicle_config(bad)


def test_legacy_flat_wheel_files_fail_clearly():
    # 2021/2024 use the old wheel_radius/wheel_width layout, no mass_properties
    with pytest.raises(ConfigError, match="wheel_properties|mass_properties"):
        load_vehicle_config("config/hardpoints/2024.yml")


def test_model_copy_patch_roundtrips():
    vc = load_vehicle_config(REAL)
    patched = vc.model_copy(deep=True)
    pt = list(patched.front.tie_rod_inboard)
    pt[2] += 5.0
    patched.front.tie_rod_inboard = tuple(pt)
    assert patched.front.tie_rod_inboard[2] == vc.front.tie_rod_inboard[2] + 5.0
    assert vc.front.tie_rod_inboard[2] != patched.front.tie_rod_inboard[2]  # original untouched
    Vehicle(patched)  # still builds


def test_dump_revalidates():
    vc = load_vehicle_config(REAL)
    assert VehicleConfig.model_validate(vc.model_dump(by_alias=True)) == vc
