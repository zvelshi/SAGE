# default
import time
import copy
from typing import List, Dict, Any

# third-party
import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.operators.mutation.pm import PolynomialMutation

# ours
from models.vehicle import Vehicle
from simulations.scenarios.kin.ackermann import AckermannScenario
from simulations.scenarios.kin.sweep import SuspensionSweep
from utils.misc import log_to_file

class SuspensionProblem(ElementwiseProblem):
    def __init__(self, optimizer):
        self.opt = optimizer
        super().__init__(
            n_var=len(optimizer.x0),
            n_obj=len(optimizer.objectives),
            xl=np.array([b[0] for b in optimizer.bounds]),
            xu=np.array([b[1] for b in optimizer.bounds])
        )

    def _evaluate(self, x, out, *args, **kwargs):
        """
        For a given design vector x, this evaluates all objectives by running the corresponding scenarios and calculating costs.
        """
        vehicle = self.opt.create_vehicle_from_ref(x)

        x_str = ", ".join([f"{v:.4f}" for v in x])
        log_to_file(f"[EVAL] Testing Design: [{x_str}]")
        costs = []

        for obj in self.opt.objectives:
            s_type = obj.get_scenario_type()

            run_config = self.opt.config.copy()
            run_config["SIMULATION"] = s_type

            scenario_cls = self.opt.get_scenario_class(s_type)
            scenario = scenario_cls(vehicle, run_config)

            try:
                results = scenario.run()
            except Exception as e:
                log_to_file(f"  [CRASH] Sim '{s_type}' failed: {e}")
                results = None

            if not results:
                costs.append(1e2)
            else:
                try:
                    val = obj.calculate_cost(results)
                    costs.append(val)
                except Exception as e:
                    log_to_file(f"  [ERROR] {obj.name} cost calc failed: {e}")
                    costs.append(1e2)

        out["F"] = np.array(costs)

        self.opt.all_X.append(np.array(x, dtype=float))
        self.opt.all_F.append(np.array(costs, dtype=float))

        c_str = ", ".join([f"{c:.6f}" for c in costs])
        log_to_file(f"  -> Result Costs: [{c_str}]")

class SuspensionOptimizer:
    def __init__(
        self, 
        base_hp_data: Dict[str, Any],
        config: Dict,
        objectives: List
    ):
        self.base_hp_data = base_hp_data
        self.config = config
        self.nickname = list(base_hp_data.keys())[0]
        self.objectives = objectives

        self.bounds = []
        self.x0 = []
        self.points_map = [] 

        self.pareto_front = None
        self.pareto_set = None

        # every design ever evaluated during the run (not just the final non-dominated front)
        self.all_X: List[np.ndarray] = []
        self.all_F: List[np.ndarray] = []

        self._parse_config_bounds()

    def _parse_config_bounds(self):
        """
        Reads opt_config to find which points to optimize and sets up the mapping.
        """
        if "FREE_POINTS" not in self.config:
            return

        half = 'front'
        if self.config.get("HALF") == 'rear':
            half = 'rear'

        section_data = self.base_hp_data[self.nickname].get(half, {})
        for pt_name, axes_limits in self.config["FREE_POINTS"].items():
            if pt_name not in section_data:
                print(f"WARNING: Point '{pt_name}' not found in '{half}' hardpoints. Skipping.")
                continue

            current_xyz = section_data[pt_name]
            axis_map = {'x': 0, 'y': 1, 'z': 2}

            for axis_char, axis_idx in axis_map.items():
                if axis_char in axes_limits:
                    limits = axes_limits[axis_char]
                    if limits[0] != limits[1]:
                        current_val = float(current_xyz[axis_idx])
                        self.x0.append(current_val)
                        lower_bound = current_val + limits[0]
                        upper_bound = current_val + limits[1]
                        self.bounds.append((lower_bound, upper_bound))
                        self.points_map.append((half, pt_name, axis_idx))

    def create_vehicle_from_ref(self, x: np.ndarray) -> Vehicle:
        """
        Creates a new Vehicle instance by patching the base dictionary.
        """
        new_hp_data = copy.deepcopy(self.base_hp_data)
        for val, (section, pt_name, axis_idx) in zip(x, self.points_map):
            new_hp_data[self.nickname][section][pt_name][axis_idx] = float(val)
        return Vehicle(new_hp_data)

    def get_scenario_class(self, key: str):
        """
         Maps scenario keys to their corresponding classes.
        """
        if key in ['steer', 'travel', 'droop_steer', 'jounce_steer', "left_travel", "right_travel", "sweep_space"]: 
            return SuspensionSweep
        if key == 'ackermann': 
            return AckermannScenario
        raise ValueError(f"Unknown scenario type: {key}")

    def run(self):
        """
        Main optimization routine.
        """
        print(f"--- Starting MOO ---")
        num_vars = len(self.x0)
        num_objs = len(self.objectives)
        
        pop_size = self.config.get("POP_SIZE", 40)
        n_offsprings = self.config.get("N_OFFSPRINGS", 10)
        prob = self.config.get("M_PROB", 1.0)
        eta = self.config.get("M_ETA", 15)
        
        print(f"Optimizing {num_vars} variables for {num_objs} objectives.")
        print(f"Population: {pop_size} | Offspring/Gen: {n_offsprings}")
        log_to_file(f"Setup: Vars={num_vars}, Objs={num_objs}, Pop={pop_size}, Offspring={n_offsprings}")
        log_to_file(f"Bounds: {self.bounds}")

        problem = SuspensionProblem(self)
        xl = np.array([b[0] for b in self.bounds])
        xu = np.array([b[1] for b in self.bounds])
        initial_pop = np.random.random((pop_size, num_vars)) * (xu - xl) + xl
        if len(self.x0) > 0:
            initial_pop[0, :] = np.array(self.x0)
            log_to_file(f"Seeding Initial Design: {self.x0}")

        algorithm = NSGA2(
            pop_size=pop_size,
            n_offsprings=n_offsprings,
            sampling=initial_pop,
            mutation=PolynomialMutation(prob=prob, eta=eta),
            eliminate_duplicates=True
        )

        termination = get_termination("n_gen", self.config.get("MAX_GEN", 50))

        t0 = time.time()
        res = minimize(
            problem,
            algorithm,
            termination,
            seed=1,
            save_history=True,
            verbose=True
        )

        self.pareto_front = res.F
        self.pareto_set = res.X

        duration = time.time() - t0
        print(f"\nOptimization Complete in {duration:.2f}s.")
        print(f"Found {len(res.F)} non-dominated solutions (Pareto Front).")

        log_to_file("\n" + "="*50)
        log_to_file(f"OPTIMIZATION RESULTS (Time: {duration:.2f}s)")
        log_to_file(f"Pareto Front Size: {len(res.F)}")
        log_to_file("="*50)

        return res