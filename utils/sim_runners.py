# default
import os
import csv
import types

# third-party
import yaml
import numpy as np

# ours
import optimization.objectives as opt_objs
from models.vehicle import Vehicle
from simulations.scenarios.kin.front_steer import FrontSteerScenario
from simulations.scenarios.kin.full_vehicle import FullVehicleScenario, FULL_VEHICLE_TYPES
from simulations.scenarios.kin.extremepoints import ExtremePoints
from simulations.scenarios.kin.sweep import SuspensionSweep
from simulations.scenarios.dyn.shock_dyno import ShockDyno
from simulations.scenarios.dyn.static import StaticDrop
from optimization.engine import SuspensionOptimizer
from utils.misc import setup_logging, save_configs, export_extreme_points_to_xlsx, export_static_hardpoints
from utils.export import (export_kin_run_data, export_dyn_run_data, export_opt_run_data,
                           load_kin_run_data, load_dyn_run_data, load_opt_run_data, list_available_runs)

def _run_kin(kin_text: str, sim_type: str):
    run_dir = setup_logging("kin_sim")
    with open(os.path.join(run_dir, "kin_config.yml"), "w") as f:
        f.write(kin_text)
        
    cfg = yaml.safe_load(kin_text)
    cfg["SIMULATION"] = sim_type
    
    save_configs(run_dir, [], cfg.get('HARDPOINTS'))
    
    with open(f"config/hardpoints/{cfg['HARDPOINTS']}.yml") as f:
        hp_data = yaml.safe_load(f)
    vehicle = Vehicle(hp_data)

    if sim_type == "front_steer":
        corner_id = [0, 0]
        steps = FrontSteerScenario(vehicle, cfg).run()
    elif sim_type in FULL_VEHICLE_TYPES:
        corner_id = [0, 0]
        steps = FullVehicleScenario(vehicle, cfg, mode=sim_type).run()
    elif sim_type == "extreme":
        corner_id = [1 if cfg.get("SIDE") == "right" else 0, 1 if cfg.get("HALF") == "rear" else 0]
        steps = ExtremePoints(vehicle, cfg).run()
        export_extreme_points_to_xlsx(steps, run_dir, cfg, template_path="utils/HARDPOINTS_TEMPLATE.xlsx")
    else:
        corner_id = [1 if cfg.get("SIDE") == "right" else 0, 1 if cfg.get("HALF") == "rear" else 0]
        steps = SuspensionSweep(vehicle, cfg).run()

    export_kin_run_data(run_dir, sim_type, steps, vehicle, corner_id, cfg.get('HARDPOINTS'))

    return sim_type, steps, vehicle, cfg, corner_id, run_dir

def _load_kin_run(run_dir: str):
    """Reconstructs a (sim_type, steps, vehicle, cfg, corner_id, run_dir) tuple from a
    previously-saved run, so it can be handed straight to _render_kin() -- no re-simulation,
    no touching config/kin_config.yml or the live editors."""
    payload = load_kin_run_data(run_dir)
    hp_name = payload.get("hardpoints_name", "unknown")

    hp_path = os.path.join(run_dir, f"{hp_name}.yml")
    if not os.path.exists(hp_path):
        hp_path = f"config/hardpoints/{hp_name}.yml"
    with open(hp_path) as f:
        hp_data = yaml.safe_load(f)
    vehicle = Vehicle(hp_data)

    kin_cfg_path = os.path.join(run_dir, "kin_config.yml")
    if os.path.exists(kin_cfg_path):
        with open(kin_cfg_path) as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = {"HARDPOINTS": hp_name}

    sim_type = payload["sim_type"]
    cfg["SIMULATION"] = sim_type
    corner_id = payload.get("corner_id") or [0, 0]
    steps = payload["steps"]

    return sim_type, steps, vehicle, cfg, corner_id, run_dir

def _run_dyn(kin_text: str, dyn_text: str, sim_type: str, progress_store: dict | None = None):
    run_dir = setup_logging("dyn_sim")
    with open(os.path.join(run_dir, "kin_config.yml"), "w") as f:
        f.write(kin_text)
    with open(os.path.join(run_dir, "dyn_config.yml"), "w") as f:
        f.write(dyn_text)

    kin_cfg = yaml.safe_load(kin_text)
    dyn_cfg = yaml.safe_load(dyn_text)

    save_configs(run_dir, [], kin_cfg.get('HARDPOINTS'))

    with open(f"config/hardpoints/{kin_cfg['HARDPOINTS']}.yml") as f:
        hp_data = yaml.safe_load(f)
    vehicle = Vehicle(hp_data)

    def on_progress(fraction: float, message: str) -> None:
        if progress_store is not None:
            progress_store["fraction"] = fraction
            progress_store["message"]  = message

    dyn_cfg["SIMULATION"] = sim_type

    if sim_type == "shock_dyno":

        # Combine kin and dyn configs so it can pull HALF/SIDE
        full_cfg = {**kin_cfg, **dyn_cfg}
        steps = ShockDyno(vehicle, full_cfg, on_progress=on_progress).run()
        
        # Export to CSV
        csv_path = os.path.join(run_dir, "shock_dyno_results.csv")
        if steps:
            keys = list(steps[0].keys())
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(steps)
    else:
        steps = StaticDrop(vehicle, dyn_cfg, on_progress=on_progress).run()

        if steps:
            export_static_hardpoints(vehicle, steps[-1], kin_cfg['HARDPOINTS'], run_dir)

    export_dyn_run_data(run_dir, sim_type, steps, vehicle, kin_cfg.get('HARDPOINTS'))

    return steps, vehicle, run_dir, dyn_cfg

def _load_dyn_run(run_dir: str):
    """Same idea as _load_kin_run(), for a saved dyn run -- feeds straight into _render_dyn()."""
    payload = load_dyn_run_data(run_dir)
    hp_name = payload.get("hardpoints_name", "unknown")

    hp_path = os.path.join(run_dir, f"{hp_name}.yml")
    if not os.path.exists(hp_path):
        hp_path = f"config/hardpoints/{hp_name}.yml"
    with open(hp_path) as f:
        hp_data = yaml.safe_load(f)
    vehicle = Vehicle(hp_data)

    dyn_cfg_path = os.path.join(run_dir, "dyn_config.yml")
    if os.path.exists(dyn_cfg_path):
        with open(dyn_cfg_path) as f:
            dyn_cfg = yaml.safe_load(f) or {}
    else:
        dyn_cfg = {}

    sim_type = payload["sim_type"]
    dyn_cfg["SIMULATION"] = sim_type
    steps = payload["steps"]

    return steps, vehicle, run_dir, dyn_cfg

def _run_opt(kin_text: str, opt_text: str):
    run_dir = setup_logging("opt")
    with open(os.path.join(run_dir, "kin_config.yml"), "w") as f:
        f.write(kin_text)
    with open(os.path.join(run_dir, "opt_config.yml"), "w") as f:
        f.write(opt_text)
        
    kin_cfg = yaml.safe_load(kin_text)
    opt_cfg = yaml.safe_load(opt_text)
    
    save_configs(run_dir, [], kin_cfg.get('HARDPOINTS'))
    
    with open(f"config/hardpoints/{kin_cfg['HARDPOINTS']}.yml") as f:
        hp_data = yaml.safe_load(f)
    cfg = {**kin_cfg, **opt_cfg}
    objectives = opt_objs.build_objectives(cfg)
    optimizer = SuspensionOptimizer(hp_data, cfg, objectives)
    res = optimizer.run()

    export_opt_run_data(run_dir, res, optimizer, cfg, kin_cfg.get('HARDPOINTS'))

    return res, optimizer, cfg, run_dir

def _load_opt_run(run_dir: str):
    """Reconstructs a (res, optimizer, cfg, run_dir) tuple from a saved opt run, for
    _render_opt() -- no re-optimization. SuspensionOptimizer.__init__() only parses
    FREE_POINTS into bounds/points_map (see optimization/engine.py); it doesn't run anything,
    so re-instantiating it from the saved hardpoints + cfg is enough to make
    create_vehicle_from_ref() work again for the saved solutions."""
    payload = load_opt_run_data(run_dir)
    hp_name = payload.get("hardpoints_name", "unknown")

    hp_path = os.path.join(run_dir, f"{hp_name}.yml")
    if not os.path.exists(hp_path):
        hp_path = f"config/hardpoints/{hp_name}.yml"
    with open(hp_path) as f:
        hp_data = yaml.safe_load(f)

    cfg = payload["cfg"]
    objectives = opt_objs.build_objectives(cfg)
    optimizer = SuspensionOptimizer(hp_data, cfg, objectives)
    optimizer.all_X = [np.array(x, dtype=float) for x in (payload.get("all_X") or [])]
    optimizer.all_F = [np.array(f, dtype=float) for f in (payload.get("all_F") or [])]

    res_X = np.array(payload["res_X"], dtype=float) if payload.get("res_X") is not None else None
    res_F = np.array(payload["res_F"], dtype=float) if payload.get("res_F") is not None else None
    res = types.SimpleNamespace(X=res_X, F=res_F)

    return res, optimizer, cfg, run_dir
