"""Parallel design evaluation and per-generation health metrics."""
import pickle
import types
import warnings

import numpy as np
import pytest

import optimization.objectives as O
from models.vehicle_config import load_vehicle_config
from optimization.engine import INFEASIBLE, SuspensionOptimizer, SuspensionProblem, _ProgressCallback
from utils.config import load_opt_config, load_sweep_config


@pytest.fixture(scope="module")
def tiny():
    sw = load_sweep_config("config/kin_config.yml").model_copy(update={"sim_steps": 6})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        op = load_opt_config("config/opt_config.yml").model_copy(
            update={"pop_size": 4, "n_offsprings": 4, "max_gen": 2})
    vc = load_vehicle_config("config/hardpoints/2026.yml")
    return vc, sw, op


def _opt(tiny, **over):
    vc, sw, op = tiny
    op = op.model_copy(update=over)
    return SuspensionOptimizer(vc, sw, op, O.build_objectives(op))


# --- pickling -------------------------------------------------------------

def test_optimizer_pickles_lean(tiny):
    o = _opt(tiny)
    o.all_X.append(np.zeros(3)); o.all_F.append(np.zeros(5)); o.history.append({"gen": 1})
    o2 = pickle.loads(pickle.dumps(o))
    assert o2.all_X == [] and o2.all_F == [] and o2.history == []      # __getstate__
    assert o2.nickname == o.nickname and len(o2.objectives) == len(o.objectives)


def test_problem_pickles(tiny):
    pickle.loads(pickle.dumps(SuspensionProblem(_opt(tiny))))


# --- purity of _evaluate ------------------------------------------------

def test_evaluate_has_no_side_effects(tiny):
    o = _opt(tiny)
    prob = SuspensionProblem(o)
    out = {}
    prob._evaluate(np.array(o.x0), out)
    assert out["F"].shape == (o.n_obj,)
    assert o.all_F == [] and o.history == []


# --- serial run + health metrics --------------------------------------

def test_serial_run_records_history(tiny):
    o = _opt(tiny, n_workers=1)
    np.random.seed(0)
    res = o.run()
    assert res.F is not None
    assert len(o.all_F) == o.history[-1]["n_eval"]
    gens = [h["gen"] for h in o.history]
    assert gens == sorted(gens) and gens[0] == 1
    hv = [h["hv"] for h in o.history]
    assert all(b >= a - 1e-9 for a, b in zip(hv, hv[1:]))     # hypervolume non-decreasing
    for h in o.history:
        assert 0.0 <= h["feasible_frac"] <= 1.0
        assert len(h["front_best"]) == o.n_obj
    assert o.serial_design_s > 0 and o.wall_s > 0


def test_progress_store_populated(tiny):
    o = _opt(tiny, n_workers=1)
    store: dict = {}
    np.random.seed(0)
    o.run(store)
    assert store["done"] is True and store["fraction"] == 1.0
    assert store["max_gen"] == o.max_gen
    assert len(store["history"]) == len(o.history)


# --- parallel == serial ----------------------------------------------

@pytest.mark.slow
def test_parallel_matches_serial(tiny):
    np.random.seed(0)
    serial = _opt(tiny, n_workers=1).run()
    np.random.seed(0)
    par = _opt(tiny, n_workers=2).run()
    assert np.allclose(np.sort(serial.F.ravel()), np.sort(par.F.ravel()))
    assert np.allclose(np.sort(serial.X.ravel()), np.sort(par.X.ravel()))


# --- callback unit ---------------------------------------------------

def test_progress_callback_accumulates():
    opt = types.SimpleNamespace(all_X=[], all_F=[], history=[], n_obj=2, max_gen=5)
    cb = _ProgressCallback(opt, {}, total_evals=20)

    def gen(n, F):
        F = np.array(F, float)
        pop = types.SimpleNamespace(get=lambda k: {"X": np.zeros((len(F), 2)), "F": F}[k])
        return types.SimpleNamespace(
            n_gen=n, off=pop, pop=pop,
            evaluator=types.SimpleNamespace(n_eval=n * len(F)),
            opt=types.SimpleNamespace(get=lambda k: F),
        )

    cb.notify(gen(1, [[1.0, 4.0], [3.0, 1.0], [INFEASIBLE, INFEASIBLE]]))
    cb.notify(gen(2, [[0.5, 3.0], [2.0, 0.5]]))
    assert len(opt.all_F) == 5
    assert [h["gen"] for h in opt.history] == [1, 2]
    assert opt.history[0]["feasible_frac"] == pytest.approx(2 / 3)
    assert opt.history[1]["hv"] >= opt.history[0]["hv"]
