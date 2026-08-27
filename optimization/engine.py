# default
import logging
import logging.handlers
import multiprocessing as mp
import time
from typing import Any, Dict, List

# third-party
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
from pymoo.core.problem import ElementwiseProblem
from pymoo.indicators.hv import HV
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.optimize import minimize
from pymoo.parallelization import StarmapParallelization
from pymoo.termination import get_termination

# ours
from models.vehicle import Vehicle
from models.vehicle_config import VehicleConfig
from simulations.scenarios.kin.front_steer import FrontSteerScenario
from simulations.scenarios.kin.sweep import SuspensionSweep
from simulations.scenarios.kin.full_vehicle import FullVehicleScenario, FULL_VEHICLE_TYPES
from utils.config import OptConfig, SweepConfig
from utils.logging_setup import get_logger

log = get_logger(__name__)

INFEASIBLE = 1e2  # per-objective cost assigned to a failed scenario / cost calc


def _worker_logging_init(queue) -> None:
    """Route a pool worker's ``sage`` logging back to the parent via a queue."""
    root = logging.getLogger("sage")
    root.handlers[:] = [logging.handlers.QueueHandler(queue)]
    root.setLevel(logging.DEBUG)
    root.propagate = False


class SuspensionProblem(ElementwiseProblem):
    def __init__(self, optimizer, **kwargs):
        self.opt = optimizer
        super().__init__(
            n_var=len(optimizer.x0),
            n_obj=len(optimizer.objectives),
            xl=np.array([b[0] for b in optimizer.bounds]),
            xu=np.array([b[1] for b in optimizer.bounds]),
            **kwargs,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        """Pure: design vector -> objective costs. No shared-state writes, so this
        is safe to run in a worker process."""
        vehicle = self.opt.create_vehicle_from_ref(x)
        costs = []
        results_by_scenario: Dict[str, Any] = {}

        for obj in self.opt.objectives:
            s_type = obj.get_scenario_type()
            if s_type not in results_by_scenario:
                try:
                    results_by_scenario[s_type] = self.opt.build_scenario(s_type, vehicle).run()
                except Exception as e:
                    log.warning("scenario '%s' crashed: %s", s_type, e)
                    results_by_scenario[s_type] = None

            results = results_by_scenario[s_type]
            if not results:
                costs.append(INFEASIBLE)
            else:
                try:
                    costs.append(obj.calculate_cost(results))
                except Exception as e:
                    log.warning("%s cost calc failed: %s", obj.name, e)
                    costs.append(INFEASIBLE)

        out["F"] = np.array(costs)


class _ProgressCallback(Callback):
    """After each generation: fold the newly-evaluated designs into the
    optimizer's ``all_X``/``all_F`` (parent-side, so it survives parallel eval),
    record per-generation health metrics, and push a live snapshot to
    ``progress_store`` for the UI."""

    def __init__(self, optimizer, progress_store: dict | None, total_evals: int):
        super().__init__()
        self.opt = optimizer
        self.store = progress_store
        self.total = total_evals
        self.t0 = time.perf_counter()
        self._ref_point: np.ndarray | None = None
        self._hv: HV | None = None
        self._last_t = self.t0

    def _hypervolume(self, front_F: np.ndarray) -> float:
        if not len(front_F):
            return 0.0
        if self._hv is None:
            self._ref_point = front_F.max(axis=0) * 1.1 + 1e-9
            self._hv = HV(ref_point=self._ref_point)
        assert self._ref_point is not None
        return float(self._hv(np.minimum(front_F, self._ref_point)))

    def notify(self, algorithm):
        now = time.perf_counter()
        off = algorithm.off if algorithm.off is not None else algorithm.pop
        X, F = off.get("X"), off.get("F")
        self.opt.all_X.extend(np.asarray(x, float) for x in X)
        self.opt.all_F.extend(np.asarray(f, float) for f in F)

        gen_feasible = np.all(F < INFEASIBLE, axis=1)
        opt_F = np.atleast_2d(algorithm.opt.get("F"))
        front_feasible = opt_F[np.all(opt_F < INFEASIBLE, axis=1)]

        rec = {
            "gen": int(algorithm.n_gen),
            "n_eval": int(algorithm.evaluator.n_eval),
            "n_nds": int(len(front_feasible)),
            "hv": self._hypervolume(front_feasible),
            "feasible_frac": float(gen_feasible.mean()) if len(gen_feasible) else 0.0,
            "front_best": front_feasible.min(axis=0).tolist() if len(front_feasible)
                          else [float("nan")] * self.opt.n_obj,
            "t": now - self.t0,
            "dt": now - self._last_t,
        }
        self._last_t = now
        self.opt.history.append(rec)

        log.info("gen %d | %d designs | %d on front | hv %.4g | %.0f%% feasible",
                 rec["gen"], rec["n_eval"], rec["n_nds"], rec["hv"],
                 rec["feasible_frac"] * 100)

        if self.store is not None:
            rate = rec["n_eval"] / max(rec["t"], 1e-6)
            self.store.update(
                fraction=min(rec["n_eval"] / max(self.total, 1), 0.99),
                gen=rec["gen"], max_gen=self.opt.max_gen,
                n_eval=rec["n_eval"], rate=rate,
                eta=max(self.total - rec["n_eval"], 0) / max(rate, 1e-6),
                feasible_frac=rec["feasible_frac"],
                history=list(self.opt.history),
                front_F=front_feasible.tolist(),
            )


class SuspensionOptimizer:
    def __init__(
        self,
        base_vehicle: VehicleConfig,
        sweep: SweepConfig,
        opt: OptConfig,
        objectives: List,
    ):
        self.base_vehicle = base_vehicle
        self.sweep = sweep
        self.opt = opt
        self.nickname = base_vehicle.nickname
        self.objectives = objectives
        self.n_obj = len(objectives)
        self.max_gen = opt.max_gen

        self.bounds: list = []
        self.x0: list = []
        self.points_map: list = []

        self.pareto_front = None
        self.pareto_set = None

        # every design ever evaluated (parent-side; see _ProgressCallback)
        self.all_X: List[np.ndarray] = []
        self.all_F: List[np.ndarray] = []
        # per-generation health metrics
        self.history: List[dict] = []
        self.wall_s: float = 0.0
        self.serial_design_s: float = 0.0
        self.n_workers: int = 1

        self._parse_config_bounds()

    def __getstate__(self):
        """Ship a lean copy to pool workers -- they never touch the run history."""
        state = self.__dict__.copy()
        state["all_X"] = state["all_F"] = state["history"] = []
        return state

    def _parse_config_bounds(self):
        if not self.opt.free_points:
            return
        half = "rear" if self.sweep.half == "rear" else "front"
        corner_cfg = getattr(self.base_vehicle, half)
        for pt_name, box in self.opt.free_points.items():
            if not hasattr(corner_cfg, pt_name):
                log.warning("FREE_POINTS point '%s' not in '%s' hardpoints; skipping", pt_name, half)
                continue
            current_xyz = getattr(corner_cfg, pt_name)
            for axis_char, axis_idx in (("x", 0), ("y", 1), ("z", 2)):
                limits = getattr(box, axis_char)
                if limits is not None and limits[0] != limits[1]:
                    current_val = float(current_xyz[axis_idx])
                    self.x0.append(current_val)
                    self.bounds.append((current_val + limits[0], current_val + limits[1]))
                    self.points_map.append((half, pt_name, axis_idx))

    def create_vehicle_from_ref(self, x: np.ndarray) -> Vehicle:
        """New Vehicle with the free-point coords patched onto a deep copy of the
        base config (once per design, not per solve)."""
        vc = self.base_vehicle.model_copy(deep=True)
        for val, (section, pt_name, axis_idx) in zip(x, self.points_map):
            corner_cfg = getattr(vc, section)
            pt = list(getattr(corner_cfg, pt_name))
            pt[axis_idx] = float(val)
            setattr(corner_cfg, pt_name, tuple(pt))
        return Vehicle(vc)

    def build_scenario(self, key: str, vehicle: Vehicle):
        sweep = self.sweep.model_copy(update={"simulation": key})
        if key in ['steer', 'travel', 'droop_steer', 'jounce_steer',
                   'left_travel', 'right_travel', 'sweep_space']:
            return SuspensionSweep(vehicle, sweep)
        if key == 'front_steer':
            return FrontSteerScenario(vehicle, sweep)
        if key in FULL_VEHICLE_TYPES:
            need_rc = any("roll_center" in getattr(o, "metric", "") for o in self.objectives)
            return FullVehicleScenario(vehicle, sweep, mode=key, roll_center=need_rc)
        raise ValueError(f"Unknown scenario type: {key}")

    # ------------------------------------------------------------------ run ---

    def run(self, progress_store: dict | None = None):
        num_vars, num_objs = len(self.x0), self.n_obj
        pop_size, n_offsprings = self.opt.pop_size, self.opt.n_offsprings
        n_workers = max(1, min(self.opt.n_workers, mp.cpu_count()))
        self.n_workers = n_workers
        total_evals = pop_size + self.max_gen * n_offsprings

        log.info("starting MOO: %d vars, %d objectives, pop %d, %d offspring/gen, "
                 "%d gens, %d worker(s)",
                 num_vars, num_objs, pop_size, n_offsprings, self.max_gen, n_workers)

        xl = np.array([b[0] for b in self.bounds])
        xu = np.array([b[1] for b in self.bounds])
        initial_pop = np.random.random((pop_size, num_vars)) * (xu - xl) + xl
        if self.x0:
            initial_pop[0, :] = np.array(self.x0)

        # one timed serial evaluation -> a baseline for the "Nx faster" readout
        t = time.perf_counter()
        SuspensionProblem(self)._evaluate(np.array(self.x0), {})
        self.serial_design_s = time.perf_counter() - t

        algorithm = NSGA2(
            pop_size=pop_size, n_offsprings=n_offsprings, sampling=initial_pop,
            mutation=PolynomialMutation(prob=self.opt.m_prob, eta=self.opt.m_eta),
            eliminate_duplicates=True,
        )
        callback = _ProgressCallback(self, progress_store, total_evals)

        pool = listener = None
        try:
            if n_workers > 1:
                queue = mp.Manager().Queue()
                listener = logging.handlers.QueueListener(
                    queue, *logging.getLogger("sage").handlers, respect_handler_level=True)
                listener.start()
                pool = mp.Pool(n_workers, initializer=_worker_logging_init, initargs=(queue,))
                problem = SuspensionProblem(
                    self, elementwise_runner=StarmapParallelization(pool.starmap))
            else:
                problem = SuspensionProblem(self)

            t0 = time.time()
            res = minimize(problem, algorithm, get_termination("n_gen", self.max_gen),
                           seed=1, verbose=False, callback=callback)
        finally:
            if pool is not None:
                pool.terminate(); pool.join()
            if listener is not None:
                listener.stop()

        self.wall_s = time.time() - t0
        self.pareto_front, self.pareto_set = res.F, res.X

        feasible = int(np.sum([np.all(np.asarray(f) < INFEASIBLE) for f in self.all_F])) if self.all_F else 0
        n_eval = len(self.all_F)
        est_serial = n_eval * self.serial_design_s
        log.info("optimization complete: %d designs (%d feasible), %d on front, %.1fs "
                 "(%.3fs/design, ~%.1fx vs 1 core)",
                 n_eval, feasible, len(res.F) if res.F is not None else 0, self.wall_s,
                 self.wall_s / max(n_eval, 1), est_serial / max(self.wall_s, 1e-6))
        if progress_store is not None:
            progress_store.update(fraction=1.0, done=True)
        return res
