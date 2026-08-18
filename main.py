# default
import argparse
import sys

# third-party
import yaml

# ours
import optimization.objectives as opt_objs
from models.vehicle import Vehicle
from simulations.scenarios.kin.ackermann import AckermannScenario
from simulations.scenarios.kin.extremepoints import ExtremePoints
from simulations.scenarios.kin.sweep import SuspensionSweep
from optimization.engine import SuspensionOptimizer
from utils.plotting import Plotter, ParetoPlotter, DynamicPlotter
from utils.misc import setup_logging, log_to_file, save_configs, export_extreme_points_to_xlsx

def load_data_only(kin_config_path: str):
    """Loads configuration dicts without instantiating the Vehicle."""
    with open(kin_config_path, "r") as f:
        kin_config = yaml.safe_load(f)
    hp_file = f"config/hardpoints/{kin_config['HARDPOINTS']}.yml"
    with open(hp_file, 'r') as f:
        hp_data = yaml.safe_load(f)
    return hp_data, kin_config

def load_vehicle(kin_config_path: str):
    """Helper to load config and vehicle objects."""
    hp_data, config = load_data_only(kin_config_path)    
    vehicle = Vehicle(hp_data)
    return vehicle, config

def handle_kinsim(args):
    run_dir = setup_logging("kin_sim")
    
    print(f"--- Simulation Mode: {args.config} ---")
    log_to_file(f"Loaded configuration arguments: {args}")

    vehicle, config = load_vehicle(args.config)
    save_configs(run_dir, [args.config], config.get('HARDPOINTS'))
    
    sim_type = config.get("SIMULATION", "travel")
    scenario = None
    
    if sim_type == "ackermann":
        print("-> Running Ackermann Steering Geometry Analysis...")
        scenario = AckermannScenario(vehicle, config)
    
    elif sim_type in ["steer", "travel", "droop_steer", "jounce_steer", "left_travel", "right_travel", ]:
        corner_id = [0, 0]
        if config["HALF"] == 'rear':
            corner_id[1] = 1
        elif config["SIDE"] == 'right':
            corner_id[0] = 1

        print(f"-> Running Single Corner Sweep: {corner_id} [{sim_type}]...")
        scenario = SuspensionSweep(vehicle, config)

    elif sim_type == "extreme":
        print("-> Running Extreme Points Analysis...")
        scenario = ExtremePoints(vehicle, config)
        results = scenario.run()
        for name, data in results.items():
            print(f"\n{name.upper()}:")
            for key, value in data.items():
                print(f"  {key}: {value}")
        export_extreme_points_to_xlsx(results, run_dir, config, template_path="utils/HARDPOINTS_TEMPLATE.xlsx")
        return

    else:
        print(f"Error: Unknown simulation type '{sim_type}'")
        return

    results = scenario.run()
    if not results:
        print("No valid solution steps returned. Check geometry constraints.")
        return

    print(f"-> Generating Plots ({len(results)} steps)...")
    plotter = Plotter(title=f"Sim: {sim_type}", save_dir=run_dir)

    if sim_type == "ackermann":
        plotter.plot_3d_ackermann(results, vehicle)
        plotter.plot_ackermann_curve(results)
    else:
        id = [0, 0]
        if config["HALF"] == 'rear':
            id = [id[0], 1]
        if config["SIDE"] == 'right':
            id = [1, id[1]]
        corner = vehicle.get_corner_from_id(id)
        plotter.plot_3d_corner(results, corner.hardpoints)
        plotter.plot_kinematics_curves(results)
    plotter.show()

def handle_optimization(args):
    run_dir = setup_logging("opt")
    print(f"--- Optimization Mode ---")

    with open(args.opt_config, "r") as f:
        opt_config = yaml.safe_load(f)
    base_hp_data, kin_config = load_data_only(args.kin_config)

    config = {**kin_config, **opt_config}
    save_configs(run_dir, [args.kin_config, args.opt_config], kin_config.get('HARDPOINTS'))
    obj_names = config.get("OBJECTIVES", [])
    objectives = []
    print(f"-> Loading Objectives: {obj_names}")
    
    for name in obj_names:
        try:
            obj_cls = getattr(opt_objs, name)
            objectives.append(obj_cls())
        except AttributeError:
            print(f"FATAL ERROR: Objective class '{name}' not found.")
            return

    if not objectives:
        print("No objectives defined. Exiting.")
        return

    optimizer = SuspensionOptimizer(base_hp_data, config, objectives)
    res = optimizer.run()
    
    if res.X is None:
        print("Optimization failed to find feasible solutions.")
        return

    print("\n" + "="*50)
    print(f"Pareto Set Found: {len(res.X)} solutions")
    print("="*50)
    
    if res.F.ndim == 1:
        F_safe = res.F.reshape(-1, 1)
    else:
        F_safe = res.F

    for i, (f_vec, x) in enumerate(zip(F_safe, res.X)):
        f_str = ", ".join([f"{val:.4f}" for val in f_vec])
        log_to_file(f"Detailed Solution {i}: Vars={x}") 
        print(f"Sol {i}: Costs = [{f_str}]")

    print(f"\n-> Plotting Results...")
    plotter = ParetoPlotter(optimizer, save_dir=run_dir)
    plotter.plot_front(res, history=res.history)

def handle_dynsim(args):
    run_dir = setup_logging("dyn_sim")
    print(f"--- Dynamic Simulation Mode ---")
    
    with open(args.dyn_config, "r") as f:
        dyn_config = yaml.safe_load(f)
    vehicle, kin_config = load_vehicle(args.kin_config)
    
    # Merge configs (dyn_config overrides kin_config where they overlap)
    config = {**kin_config, **dyn_config}
    save_configs(run_dir, [args.kin_config, args.dyn_config], config.get('HARDPOINTS'))

    # Determine which dynamic scenario to run (default to 'terrain')
    sim_type = config["SIMULATION"]
    
    if sim_type == "terrain":
        print("-> Running Dynamic Terrain Simulation...")
        scenario = DynamicScenario(vehicle, config)
        plot_title = "Dynamic Terrain Run"
    else:
        print(f"Error: Unknown dynamic simulation type '{sim_type}' in dyn_config.yml")
        return
    results = scenario.run()

    if not results:
        print(f"[{sim_type.upper()}] run produced no results.")
        return

    print(f"-> Starting Animation ({len(results)} frames)...")
    plotter = DynamicPlotter(title=plot_title, save_dir=run_dir)
    plotter.animate(results, vehicle, config)

def main():
    parser = argparse.ArgumentParser(description="Suspension Simulation & Optimization Tool")
    subparsers = parser.add_subparsers(dest="command", help="Mode of operation")
    
    sim_parser = subparsers.add_parser("kin", help="Run a kinematic simulation")
    sim_parser.add_argument("--config", type=str, default="config/kin_config.yml", help="Path to sim config")

    opt_parser = subparsers.add_parser("opt", help="Run the optimizer")
    opt_parser.add_argument("--kin_config", type=str, default="config/kin_config.yml", help="Base sim config")
    opt_parser.add_argument("--opt_config", type=str, default="config/opt_config.yml", help="Optimization config")

    dyn_parser = subparsers.add_parser("dyn", help="Run dynamic terrain simulation")
    dyn_parser.add_argument("--kin_config", type=str, default="config/kin_config.yml", help="Base config")
    dyn_parser.add_argument("--dyn_config", type=str, default="config/dyn_config.yml", help="Dynamic parameters")

    args = parser.parse_args()

    if args.command == "kin":
        handle_kinsim(args)
    elif args.command == "opt":
        handle_optimization(args)
    elif args.command == "dyn":
        handle_dynsim(args)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()