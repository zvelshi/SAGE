from __future__ import annotations

# default
from collections import deque
from typing import List, Dict, Callable

# third-party
import numpy as np

# ours
from simulations.scenarios.base import Scenario
from utils.config import DynConfig
from utils.logging_setup import get_logger

log = get_logger(__name__)
from utils.dynamics import (
    derivatives, euler_step, solve_all_corners, build_step_dict, initial_shock_lengths,
    CORNERS_ATTR, IDX_Z_COG, IDX_PHI, IDX_THETA, IDX_DZ_COG, IDX_DPHI, IDX_DTHETA,
    IDX_Z_WU, IDX_DZ_WU,
)

def _noop_progress(*_): pass

def _is_settled(state: np.ndarray, z_cog_history: deque, z_cog_static: float, tol_frac: float = 0.02) -> bool:
    """Check if the vehicle CoG has settled to within `tol_frac` (2% by default)
    of the static ride height, sustained over the trailing history window."""
    z_cog_history.append(state[IDX_Z_COG])
    if len(z_cog_history) < z_cog_history.maxlen:
        return False
    band = max(z_cog_history) - min(z_cog_history)
    tol = tol_frac * abs(z_cog_static)
    return band <= tol

class StaticDrop(Scenario):
    """Full-vehicle drop from a hoisted height.

    Phase 1 (hoist): CoG pinned at HOIST_HEIGHT for HOIST_DURATION seconds. Suspension sags to droop under gravity.
    Phase 2 (drop):  Body released; car falls, contacts ground, and settles.
    """

    def __init__(self, vehicle, config: DynConfig, on_progress: Callable | None = None):
        self.vehicle = vehicle
        self.sol_dt = config.sol_dt
        self.viz_dt = config.viz_dt
        self.hoist_height_mm = config.hoist_height * 1000.0
        self.hoist_duration = config.hoist_duration
        self.max_sim_time = config.max_sim_time
        self._on_progress = on_progress if on_progress is not None else _noop_progress

    def _progress(self, fraction: float, message: str) -> None:
        """Update progress."""
        self._on_progress(float(np.clip(fraction, 0.0, 1.0)), message)

    def run(self) -> List[Dict]:
        """Run the static drop simulation."""
        vehicle = self.vehicle
        dt = self.sol_dt
        viz_dt = self.viz_dt

        z_cog_static = float(vehicle.cog[2])
        cog_static = np.array(vehicle.cog)
        z_cog_hoist = z_cog_static + self.hoist_height_mm

        hoist_steps = max(1, round(self.hoist_duration / dt))
        drop_steps = max(1, round(self.max_sim_time / dt))
        log_10pct_h = max(1, round(0.10 * hoist_steps))
        log_10pct_d = max(1, round(0.10 * drop_steps))

        log.info("StaticDrop %s: hoist CoG +%.0fmm for %.2fs (%d steps), drop <= %.2fs "
                 "(%d steps), sol_dt=%.1fms viz_dt=%.0fms",
                 vehicle.nickname, self.hoist_height_mm, self.hoist_duration, hoist_steps,
                 self.max_sim_time, drop_steps, dt * 1000, viz_dt * 1000)

        # Initialize state
        self._progress(0.05, "Initializing state...")
        state = np.zeros(14)
        state[IDX_Z_COG] = z_cog_hoist
        for i, attr in enumerate(CORNERS_ATTR):
            corner  = getattr(vehicle, attr)
            wc_stat = np.array(corner.hardpoints.wc, float)
            state[6 + i] = z_cog_hoist + (wc_stat[2] - z_cog_static)

        shock_lengths = initial_shock_lengths(vehicle)

        steps: List[Dict] = []
        t = 0.0
        viz_accum = 0.0

        # Phase 1: Hoist
        self._progress(0.08, "Phase 1/2: hoisting — suspension drooping...")
        log.info("hoist phase: %d steps", hoist_steps)

        log_pct_next = log_10pct_h
        for step_i in range(hoist_steps):

            # Euler: body stays pinned, only advance wheel DOFs
            deriv, shock_lengths = derivatives(state, vehicle, shock_lengths, dt)
            state[IDX_Z_WU] = state[IDX_Z_WU] + dt * deriv[IDX_Z_WU]
            state[IDX_DZ_WU] = state[IDX_DZ_WU] + dt * deriv[IDX_DZ_WU]

            # Body pinned
            state[IDX_Z_COG] = z_cog_hoist
            state[IDX_PHI] = 0.0
            state[IDX_THETA] = 0.0
            state[IDX_DZ_COG] = 0.0
            state[IDX_DPHI] = 0.0
            state[IDX_DTHETA] = 0.0

            t += dt
            viz_accum += dt

            if viz_accum >= viz_dt:
                viz_accum = 0.0
                cs = solve_all_corners(state, vehicle, cog_static)
                steps.append(build_step_dict(state, cs, vehicle, t, "hoist"))

            if step_i + 1 >= log_pct_next:
                pct = (step_i + 1) / hoist_steps
                z_wu = state[IDX_Z_WU]
                log.debug("hoist %.0f%% t=%.3fs wheel z FL=%.0f FR=%.0f RL=%.0f RR=%.0f",
                          pct * 100, t, z_wu[0], z_wu[1], z_wu[2], z_wu[3])
                self._progress(0.08 + pct * 0.42, f"Phase 1/2: hoisting ({pct*100:.0f}%)…")
                log_pct_next += log_10pct_h

        hoist_n = len(steps)
        log.info("hoist done at t=%.3fs, %d viz frames, avg hub %.1fmm",
                 t, hoist_n, state[IDX_Z_WU].mean())

        # Phase 2: Drop
        self._progress(0.50, "Phase 2/2: dropping — vehicle in free fall...")
        log.info("drop phase: releasing body at t=%.3fs (<= %d steps)", t, drop_steps)

        settle_window_s = 0.5
        settle_window_steps = max(1, round(settle_window_s / dt))
        z_cog_history: deque = deque(maxlen=settle_window_steps)
        contact_logged = False
        log_pct_next = log_10pct_d

        for step_i in range(drop_steps):
            state, shock_lengths = euler_step(state, dt, vehicle, shock_lengths)
            t += dt
            viz_accum += dt

            if viz_accum >= viz_dt:
                viz_accum = 0.0
                cs = solve_all_corners(state, vehicle, cog_static)
                steps.append(build_step_dict(state, cs, vehicle, t, "drop"))

            if not contact_logged:
                wheel_r = vehicle.front_left.wheel.radius
                z_wu = state[IDX_Z_WU]
                labels = ["FL", "FR", "RL", "RR"]
                in_contact = [labels[i] for i in range(4) if z_wu[i] - wheel_r <= 0.0]
                if in_contact:
                    log.info("first ground contact at t=%.3fs: %s", t, ", ".join(in_contact))
                    contact_logged = True

            if step_i + 1 >= log_pct_next:
                pct = (step_i + 1) / drop_steps
                z_cog_now = state[IDX_Z_COG]
                phi_deg = float(np.degrees(state[IDX_PHI]))
                theta_deg = float(np.degrees(state[IDX_THETA]))
                max_vel = float(np.max(np.abs(
                    np.concatenate([state[3:6], state[10:14]]))))
                log.debug("drop %.0f%% t=%.3fs CoG z=%.1f roll=%.2f pitch=%.2f max|v|=%.0f",
                          pct * 100, t, z_cog_now, phi_deg, theta_deg, max_vel)
                self._progress(0.50 + pct * 0.45,
                               f"Phase 2/2: drop ({pct*100:.0f}%) — "
                               f"CoG z={z_cog_now:.0f} mm  max|v|={max_vel:.0f} mm/s")
                log_pct_next += log_10pct_d

            if _is_settled(state, z_cog_history, z_cog_static):
                z_final = state[IDX_Z_COG]
                log.info("SETTLED at t=%.3fs: CoG z=%.1fmm (delta from static %+.1fmm)",
                         t, z_final, z_final - z_cog_static)
                cs = solve_all_corners(state, vehicle, cog_static)
                steps.append(build_step_dict(state, cs, vehicle, t, "settled"))
                break
        else:
            log.warning("reached MAX_SIM_TIME=%.1fs without settling", self.max_sim_time)

        total = len(steps)
        log.info("StaticDrop complete: %d viz frames (%d hoist + %d drop)",
                 total, hoist_n, total - hoist_n)
        self._progress(1.0, f"Done — {total} frames ready")
        return steps
