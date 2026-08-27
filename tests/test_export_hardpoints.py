"""write_optimized_hardpoints: patch only the free-vars, keep the rest verbatim."""
import yaml

from models.vehicle_config import load_vehicle_config
from utils.misc import write_optimized_hardpoints

BASE = "2026"


def _flat(d, p=""):
    if isinstance(d, dict):
        for k in d:
            yield from _flat(d[k], f"{p}/{k}")
    elif isinstance(d, list):
        for i, v in enumerate(d):
            yield from _flat(v, f"{p}[{i}]")
    else:
        yield p, d


def test_patches_only_free_vars(tmp_path):
    points_map = [("front", "tie_rod_inboard", 0),
                  ("front", "tie_rod_inboard", 2),
                  ("front", "shock_outboard", 1)]
    x = [111.0, 222.0, 333.0]

    dest = write_optimized_hardpoints(BASE, points_map, x, "my run! 2", out_dir=str(tmp_path))
    assert dest.endswith("my_run_2.yml")

    base = dict(_flat(yaml.safe_load(open(f"config/hardpoints/{BASE}.yml"))))
    new = dict(_flat(yaml.safe_load(open(dest))))
    changed = {k: (base[k], new[k]) for k in base if base.get(k) != new.get(k)}
    assert set(changed) == {
        "/baja_2026/front/tie_rod_inboard[0]",
        "/baja_2026/front/tie_rod_inboard[2]",
        "/baja_2026/front/shock_outboard[1]",
    }
    assert new["/baja_2026/front/tie_rod_inboard[0]"] == 111.0
    assert new["/baja_2026/front/shock_outboard[1]"] == 333.0


def test_output_is_a_valid_vehicle(tmp_path):
    dest = write_optimized_hardpoints(BASE, [("front", "wheel_center", 2)], [305.5],
                                      "check", out_dir=str(tmp_path))
    vc = load_vehicle_config(dest)
    assert vc.nickname == "baja_2026"
    assert vc.front.wheel_center[2] == 305.5


def test_blank_name_falls_back(tmp_path):
    dest = write_optimized_hardpoints(BASE, [], [], "  !!  ", out_dir=str(tmp_path))
    assert dest.endswith(f"{BASE}_opt.yml")
