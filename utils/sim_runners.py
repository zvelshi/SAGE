import yaml
import os
from models.vehicle import Vehicle
from simulations.scenarios.kin.ackermann import AckermannScenario
from simulations.scenarios.kin.extremepoints import ExtremePoints
from simulations.scenarios.kin.sweep import SuspensionSweep
from optimization.engine import SuspensionOptimizer
import optimization.objectives as opt_objs
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
