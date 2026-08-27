# default
import logging
import os
import csv
import types

# third-party
import yaml
import numpy as np

# ours
import optimization.objectives as opt_objs
from utils.config import (
    parse_sweep_config, parse_opt_config, parse_dyn_config,
    SweepConfig, DynConfig,
)
from models.vehicle import Vehicle
from models.vehicle_config import load_vehicle_config, VehicleConfig
from simulations.scenarios.kin.front_steer import FrontSteerScenario
from simulations.scenarios.kin.full_vehicle import FullVehicleScenario, FULL_VEHICLE_TYPES
from simulations.scenarios.kin.extremepoints import ExtremePoints
from simulations.scenarios.kin.sweep import SuspensionSweep
from simulations.scenarios.dyn.shock_dyno import ShockDyno
from simulations.scenarios.dyn.static import StaticDrop
from optimization.engine import SuspensionOptimizer
from utils.logging_setup import get_logger, run_log_file
from utils.misc import new_run_dir, save_configs, export_extreme_points_to_xlsx, export_static_hardpoints
from utils.export import (export_kin_run_data, export_dyn_run_data, export_opt_run_data,
                           load_kin_run_data, load_dyn_run_data, load_opt_run_data, list_available_runs)

log = get_logger(__name__)


def _corner_id(sweep: SweepConfig) -> list[int]:
    return [1 if sweep.side == "right" else 0, 1 if sweep.half == "rear" else 0]


def _vehicle_config(hp_name: str, run_dir: str | None = None) -> VehicleConfig:
    """Load a hardpoint file, preferring a copy saved inside `run_dir`."""
    if run_dir:
        saved = os.path.join(run_dir, f"{hp_name}.yml")
        if os.path.exists(saved):
            return load_vehicle_config(saved)
    return load_vehicle_config(f"config/hardpoints/{hp_name}.yml")


def _run_kin(kin_text: str, sim_type: str):
    run_dir = new_run_dir("kin_sim")
    with run_log_file(run_dir):
        log.info("kin run '%s' -> %s", sim_type, run_dir)
        with open(os.path.join(run_dir, "kin_config.yml"), "w") as f:
            f.write(kin_text)

        sweep = parse_sweep_config(yaml.safe_load(kin_text) or {}, "kin config").model_copy(
            update={"simulation": sim_type})

        save_configs(run_dir, [], sweep.hardpoints)
        vehicle = Vehicle(_vehicle_config(sweep.hardpoints))

        if sim_type == "front_steer":
            corner_id = [0, 0]
            steps = FrontSteerScenario(vehicle, sweep).run()
        elif sim_type in FULL_VEHICLE_TYPES:
            corner_id = [0, 0]
            steps = FullVehicleScenario(vehicle, sweep, mode=sim_type).run()
        elif sim_type == "extreme":
            corner_id = _corner_id(sweep)
            steps = ExtremePoints(vehicle, sweep).run()
            export_extreme_points_to_xlsx(steps, run_dir, sweep,
                                          template_path="utils/HARDPOINTS_TEMPLATE.xlsx")
        else:
            corner_id = _corner_id(sweep)
            steps = SuspensionSweep(vehicle, sweep).run()

        export_kin_run_data(run_dir, sim_type, steps, vehicle, corner_id, sweep.hardpoints)
        log.info("kin run done: %d steps", len(steps))
        return sim_type, steps, vehicle, sweep, corner_id, run_dir


def _load_kin_run(run_dir: str):
    """Reconstructs a (sim_type, steps, vehicle, sweep, corner_id, run_dir) tuple from a
    previously-saved run, so it can be handed straight to _render_kin() -- no re-simulation."""
    payload = load_kin_run_data(run_dir)
    hp_name = payload.get("hardpoints_name", "unknown")
    vehicle = Vehicle(_vehicle_config(hp_name, run_dir))

    sim_type = payload["sim_type"]
    kin_cfg_path = os.path.join(run_dir, "kin_config.yml")
    if os.path.exists(kin_cfg_path):
        with open(kin_cfg_path) as f:
            raw = yaml.safe_load(f) or {}
    else:
        raw = {"HARDPOINTS": hp_name, "SIM_STEPS": 2,
               "TRAVEL": {"MIN": -1, "MAX": 1}, "STEER": {"MIN": -1, "MAX": 1}}
    sweep = parse_sweep_config(raw, kin_cfg_path).model_copy(update={"simulation": sim_type})

    corner_id = payload.get("corner_id") or [0, 0]
    steps = payload["steps"]

    return sim_type, steps, vehicle, sweep, corner_id, run_dir


def _run_dyn(kin_text: str, dyn_text: str, sim_type: str, progress_store: dict | None = None):
    run_dir = new_run_dir("dyn_sim")
    with run_log_file(run_dir):
        log.info("dyn run '%s' -> %s", sim_type, run_dir)
        with open(os.path.join(run_dir, "kin_config.yml"), "w") as f:
            f.write(kin_text)
        with open(os.path.join(run_dir, "dyn_config.yml"), "w") as f:
            f.write(dyn_text)

        sweep = parse_sweep_config(yaml.safe_load(kin_text) or {}, "kin config")
        dyn = parse_dyn_config(yaml.safe_load(dyn_text) or {}, "dyn config").model_copy(
            update={"simulation": sim_type})

        save_configs(run_dir, [], sweep.hardpoints)
        vehicle = Vehicle(_vehicle_config(sweep.hardpoints))

        def on_progress(fraction: float, message: str) -> None:
            if progress_store is not None:
                progress_store["fraction"] = fraction
                progress_store["message"] = message

        if sim_type == "shock_dyno":
            steps = ShockDyno(vehicle, dyn, sweep.half, sweep.side, on_progress=on_progress).run()
            csv_path = os.path.join(run_dir, "shock_dyno_results.csv")
            if steps:
                keys = list(steps[0].keys())
                with open(csv_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=keys)
                    writer.writeheader()
                    writer.writerows(steps)
        else:
            steps = StaticDrop(vehicle, dyn, on_progress=on_progress).run()
            if steps:
                export_static_hardpoints(vehicle, steps[-1], sweep.hardpoints, run_dir)

        export_dyn_run_data(run_dir, sim_type, steps, vehicle, sweep.hardpoints)
        return steps, vehicle, run_dir, dyn


def _load_dyn_run(run_dir: str):
    """Same idea as _load_kin_run(), for a saved dyn run -- feeds straight into _render_dyn()."""
    payload = load_dyn_run_data(run_dir)
    hp_name = payload.get("hardpoints_name", "unknown")
    vehicle = Vehicle(_vehicle_config(hp_name, run_dir))

    sim_type = payload["sim_type"]
    dyn_cfg_path = os.path.join(run_dir, "dyn_config.yml")
    if os.path.exists(dyn_cfg_path):
        with open(dyn_cfg_path) as f:
            raw = yaml.safe_load(f) or {}
    else:
        raw = {}
    dyn = parse_dyn_config(raw, dyn_cfg_path).model_copy(update={"simulation": sim_type})

    steps = payload["steps"]
    return steps, vehicle, run_dir, dyn


def _run_opt(kin_text: str, opt_text: str):
    run_dir = new_run_dir("opt")
    # opt runs are millions of solves -- keep the file at INFO unless SAGE_DEBUG.
    file_level = logging.DEBUG if os.environ.get("SAGE_DEBUG") else logging.INFO
    with run_log_file(run_dir, file_level):
        log.info("opt run -> %s", run_dir)
        with open(os.path.join(run_dir, "kin_config.yml"), "w") as f:
            f.write(kin_text)
        with open(os.path.join(run_dir, "opt_config.yml"), "w") as f:
            f.write(opt_text)

        sweep = parse_sweep_config(yaml.safe_load(kin_text) or {}, "kin config")
        opt = parse_opt_config(yaml.safe_load(opt_text) or {}, "opt config")

        save_configs(run_dir, [], sweep.hardpoints)

        objectives = opt_objs.build_objectives(opt)
        optimizer = SuspensionOptimizer(_vehicle_config(sweep.hardpoints), sweep, opt, objectives)
        res = optimizer.run()

        export_opt_run_data(run_dir, res, optimizer, sweep.hardpoints)
        return res, optimizer, run_dir


def _load_opt_run(run_dir: str):
    """Reconstructs a (res, optimizer, run_dir) tuple from a saved opt run, for
    _render_opt() -- no re-optimization."""
    payload = load_opt_run_data(run_dir)
    hp_name = payload.get("hardpoints_name", "unknown")

    sweep = parse_sweep_config(payload["sweep"], f"{run_dir} (saved sweep)")
    opt = parse_opt_config(payload["opt"], f"{run_dir} (saved opt)")
    objectives = opt_objs.build_objectives(opt)
    optimizer = SuspensionOptimizer(_vehicle_config(hp_name, run_dir), sweep, opt, objectives)
    optimizer.all_X = [np.array(x, dtype=float) for x in (payload.get("all_X") or [])]
    optimizer.all_F = [np.array(f, dtype=float) for f in (payload.get("all_F") or [])]

    res_X = np.array(payload["res_X"], dtype=float) if payload.get("res_X") is not None else None
    res_F = np.array(payload["res_F"], dtype=float) if payload.get("res_F") is not None else None
    res = types.SimpleNamespace(X=res_X, F=res_F)

    return res, optimizer, run_dir
