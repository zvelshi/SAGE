# default
import csv
import json
import math
import os

# third-party
import numpy as np

# ours
from utils.geometry import get_wheel_attitude, motion_ratio_series
from utils.misc import log_to_file
from utils.plot2d import _build_kin_stats, _build_dyn_stats

_KIN_DATA_JSON = "kin_data.json"
_KIN_DATA_CSV  = "kin_data.csv"
_DYN_DATA_JSON = "dyn_data.json"
_DYN_DATA_CSV  = "dyn_data.csv"
_OPT_DATA_JSON = "opt_data.json"
_RUN_META_JSON = "run_meta.json"

NO_COMPARE_SIM_TYPES = {"extreme", "shock_dyno"}

def _json_safe(obj):
    """Recursively convert numpy types (arrays/scalars) to plain Python so json.dump works."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj

def _restore_arrays(obj):
    if isinstance(obj, dict):
        return {k: _restore_arrays(v) for k, v in obj.items()}
    if isinstance(obj, list):
        if obj and all(isinstance(v, (int, float)) for v in obj):
            return np.array(obj, dtype=float)
        return [_restore_arrays(v) for v in obj]
    return obj

def build_kin_static_values(steps: list, sim_type: str, hp, half_label: str) -> list:
    if not steps or sim_type in NO_COMPARE_SIM_TYPES:
        return []

    if sim_type == "ackermann":
        pct = steps[-1].get("ackermann_pct")
        if pct is None or (isinstance(pct, float) and math.isnan(pct)):
            return []
        return [("Ackermann % (Full Steer)", f"{pct:.2f}")]

    stats = []
    axle_steps = [s["axle_data"] for s in steps if s.get("axle_data")]
    if axle_steps:
        min_plunge = min(a["plunge_mm"] for a in axle_steps)
        max_plunge = max(a["plunge_mm"] for a in axle_steps)
        abs_angle  = max(max(a["angle_ib_deg"], a["angle_ob_deg"]) for a in axle_steps)
        stats.append((f"{half_label} Plunge Range [mm]", f"{min_plunge:.2f} to {max_plunge:.2f}"))
        stats.append((f"{half_label} Max Joint Angle [deg]", f"{abs_angle:.2f}"))

    stats.extend(_build_kin_stats(steps, wr=hp.wr))
    return stats

def build_kin_series(steps: list, sim_type: str, wr: float = 0.0) -> dict:
    """One column per plotted metric -- the same data the 2D/3D charts are built from."""
    if not steps:
        return {}

    if sim_type == "ackermann":
        return {
            "rack_travel_mm": [s["input"] for s in steps],
            "ackermann_pct":  [s["ackermann_pct"] for s in steps],
        }

    if "x_val" in steps[0]:
        x_key, xs = steps[0].get("x_label", "input"), [s["x_val"] for s in steps]
    elif "travel_mm" in steps[0]:
        x_key, xs = "Shock Travel [mm]", [s["travel_mm"] for s in steps]
    elif "steer_mm" in steps[0]:
        x_key, xs = "Rack Travel [mm]", [s["steer_mm"] for s in steps]
    else:
        x_key, xs = "step", list(range(len(steps)))

    series = {x_key: xs}
    if sim_type == "sweep_space":
        series["travel_mm"] = [s.get("travel_mm") for s in steps]
        series["steer_mm"]  = [s.get("steer_mm") for s in steps]

    atts = [get_wheel_attitude(s) for s in steps]
    series.update({
        "camber_deg":   [a["camber"] for a in atts],
        "caster_deg":   [a["caster"] for a in atts],
        "toe_deg":      [a["toe"] for a in atts],
        "motion_ratio": motion_ratio_series(steps, wr=wr).tolist(),
    })

    plunge, a_ib, a_ob = [], [], []
    for s in steps:
        d = s.get("axle_data") or {}
        plunge.append(d.get("plunge_mm"))
        a_ib.append(d.get("angle_ib_deg"))
        a_ob.append(d.get("angle_ob_deg"))
    if any(v is not None for v in plunge):
        series["axle_plunge_mm"]        = plunge
        series["cv_angle_inboard_deg"]  = a_ib
        series["cv_angle_outboard_deg"] = a_ob

    return series

def export_kin_run_data(run_dir: str, sim_type: str, steps, vehicle, corner_id: list, hardpoints_name: str) -> None:
    """Writes kin_data.json (full raw steps, for reload/3D reconstruction), kin_data.csv
    (flattened per-metric series, for Excel/manual inspection), and run_meta.json (for the
    Web UI's 'Load Past Run' listing)."""
    hp = vehicle.get_corner_from_id(corner_id).hardpoints
    half_label = "Rear" if corner_id[1] == 1 else "Front"

    if sim_type == "extreme":
        static_values, series = [], {}
    else:
        static_values = build_kin_static_values(steps, sim_type, hp, half_label)
        series = build_kin_series(steps, sim_type, wr=hp.wr)

    payload = {
        "mode": "kin",
        "sim_type": sim_type,
        "hardpoints_name": hardpoints_name,
        "corner_id": corner_id,
        "static_values": static_values,
        "steps": _json_safe(steps),
    }
    with open(os.path.join(run_dir, _KIN_DATA_JSON), "w") as f:
        json.dump(payload, f, indent=2)

    if series:
        with open(os.path.join(run_dir, _KIN_DATA_CSV), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(list(series.keys()))
            for row in zip(*series.values()):
                writer.writerow(row)

    with open(os.path.join(run_dir, _RUN_META_JSON), "w") as f:
        json.dump({"mode": "kin", "sim_type": sim_type, "hardpoints_name": hardpoints_name}, f)

    log_to_file(f"Exported run data: {_KIN_DATA_JSON}" + (f", {_KIN_DATA_CSV}" if series else ""))

def load_kin_run_data(run_dir: str) -> dict:
    with open(os.path.join(run_dir, _KIN_DATA_JSON)) as f:
        payload = json.load(f)
    payload["steps"] = _restore_arrays(payload["steps"])
    return payload

def build_dyn_static_values(steps: list, sim_type: str, vehicle) -> list:
    """Mirrors the 'after settle' value cards shown in the Web UI for a dyn run. Delegates to
    plot2d._build_dyn_stats() -- see build_kin_static_values() for why that matters."""
    if not steps or sim_type in NO_COMPARE_SIM_TYPES:
        return []
    corner_wr = {
        "fl": vehicle.front_left.hardpoints.wr,
        "fr": vehicle.front_right.hardpoints.wr,
        "rl": vehicle.rear_left.hardpoints.wr,
        "rr": vehicle.rear_right.hardpoints.wr,
    }
    return _build_dyn_stats(steps, corner_wr=corner_wr)

def build_dyn_series(steps: list, sim_type: str) -> dict:
    if not steps:
        return {}

    if sim_type == "shock_dyno":
        keys = ["t", "displacement", "velocity", "shock_len", "shock_max", "shock_min",
                "force_total", "force_spring", "force_damper"]
        return {k: [s.get(k) for s in steps] for k in keys}

    series = {
        "t":         [s["t"] for s in steps],
        "cog_z_mm":  [s["cog_pos"][2] for s in steps],
        "roll_deg":  [float(np.degrees(s["phi"])) for s in steps],
        "pitch_deg": [float(np.degrees(s["theta"])) for s in steps],
    }
    for key in ("fl", "fr", "rl", "rr"):
        series[f"{key}_shock_length_mm"] = [s[key]["shock_length"] if s.get(key) else None for s in steps]
    for key in ("fl", "fr", "rl", "rr"):
        atts = [get_wheel_attitude(s[key]) if s.get(key) else {"camber": None, "caster": None, "toe": None}
                for s in steps]
        series[f"{key}_camber_deg"] = [a["camber"] for a in atts]
        series[f"{key}_caster_deg"] = [a["caster"] for a in atts]
        series[f"{key}_toe_deg"]    = [a["toe"] for a in atts]
    return series

def export_dyn_run_data(run_dir: str, sim_type: str, steps, vehicle, hardpoints_name: str) -> None:
    """Writes dyn_data.json/csv + run_meta.json, same idea as export_kin_run_data."""
    static_values = build_dyn_static_values(steps, sim_type, vehicle)
    series = build_dyn_series(steps, sim_type)

    payload = {
        "mode": "dyn",
        "sim_type": sim_type,
        "hardpoints_name": hardpoints_name,
        "static_values": static_values,
        "steps": _json_safe(steps),
    }
    with open(os.path.join(run_dir, _DYN_DATA_JSON), "w") as f:
        json.dump(payload, f, indent=2)

    if series:
        with open(os.path.join(run_dir, _DYN_DATA_CSV), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(list(series.keys()))
            for row in zip(*series.values()):
                writer.writerow(row)

    with open(os.path.join(run_dir, _RUN_META_JSON), "w") as f:
        json.dump({"mode": "dyn", "sim_type": sim_type, "hardpoints_name": hardpoints_name}, f)

    log_to_file(f"Exported run data: {_DYN_DATA_JSON}" + (f", {_DYN_DATA_CSV}" if series else ""))

def load_dyn_run_data(run_dir: str) -> dict:
    with open(os.path.join(run_dir, _DYN_DATA_JSON)) as f:
        payload = json.load(f)
    payload["steps"] = _restore_arrays(payload["steps"])
    return payload

def export_opt_run_data(run_dir: str, res, optimizer, cfg: dict, hardpoints_name: str) -> None:
    payload = {
        "mode": "opt",
        "sim_type": "run",
        "hardpoints_name": hardpoints_name,
        "cfg": _json_safe(cfg),
        "obj_names": [o.name for o in optimizer.objectives],
        "res_X": _json_safe(res.X) if res.X is not None else None,
        "res_F": _json_safe(res.F) if res.F is not None else None,
        "all_X": _json_safe(optimizer.all_X),
        "all_F": _json_safe(optimizer.all_F),
    }
    with open(os.path.join(run_dir, _OPT_DATA_JSON), "w") as f:
        json.dump(payload, f, indent=2)

    with open(os.path.join(run_dir, _RUN_META_JSON), "w") as f:
        json.dump({"mode": "opt", "sim_type": "run", "hardpoints_name": hardpoints_name}, f)

    log_to_file(f"Exported run data: {_OPT_DATA_JSON}")

def load_opt_run_data(run_dir: str) -> dict:
    with open(os.path.join(run_dir, _OPT_DATA_JSON)) as f:
        return json.load(f)

def list_available_runs() -> list:
    """Scans out/kin_sim, out/dyn_sim and out/opt for run directories that have a
    run_meta.json (i.e. were produced after this export feature was added), newest first."""
    runs = []
    for mode, subdir in (("kin", "kin_sim"), ("dyn", "dyn_sim"), ("opt", "opt")):
        base = os.path.join("out", subdir)
        if not os.path.isdir(base):
            continue
        for ts in sorted(os.listdir(base), reverse=True):
            run_dir  = os.path.join(base, ts)
            meta_path = os.path.join(run_dir, _RUN_META_JSON)
            if not os.path.isfile(meta_path):
                continue
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
            except Exception:
                continue
            sim_type = meta.get("sim_type", "?")
            hp_name  = meta.get("hardpoints_name", "?")
            runs.append({
                "mode": mode,
                "run_dir": run_dir,
                "timestamp": ts,
                "sim_type": sim_type,
                "label": f"[{mode}] {ts} - {sim_type} ({hp_name})",
            })
    return runs
