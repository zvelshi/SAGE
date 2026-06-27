# default
import os
import csv

# third-party
import yaml

# ours
import optimization.objectives as opt_objs
from models.vehicle import Vehicle
from simulations.scenarios.kin.ackermann import AckermannScenario
from simulations.scenarios.kin.extremepoints import ExtremePoints
from simulations.scenarios.kin.sweep import SuspensionSweep
from simulations.scenarios.dyn.shock_dyno import ShockDyno
from simulations.scenarios.dyn.static import StaticDrop
from optimization.engine import SuspensionOptimizer
from utils.misc import setup_logging, save_configs, export_extreme_points_to_xlsx

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
    corner_id = [
        1 if cfg.get("SIDE") == "right" else 0,
        1 if cfg.get("HALF") == "rear"  else 0,
    ]
    if sim_type == "ackermann":
        steps = AckermannScenario(vehicle, cfg).run()
    elif sim_type == "extreme":
        steps = ExtremePoints(vehicle, cfg).run()
        export_extreme_points_to_xlsx(steps, run_dir, cfg, template_path="utils/HARDPOINTS_TEMPLATE.xlsx")
    else:
        steps = SuspensionSweep(vehicle, cfg).run()
        
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
    objectives = [getattr(opt_objs, n)() for n in cfg.get("OBJECTIVES", [])]
    optimizer = SuspensionOptimizer(hp_data, cfg, objectives)
    return optimizer.run(), optimizer, cfg, run_dir
