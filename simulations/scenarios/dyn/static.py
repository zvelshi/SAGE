from __future__ import annotations

# default
from collections import deque
from typing import List, Dict, Callable

# third-party
import numpy as np

# ours
from simulations.scenarios.base import Scenario
from utils.misc import log_to_file

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

    def __init__(self, vehicle, config: Dict, on_progress: Callable | None = None):
        self.vehicle = vehicle
        self.sol_dt = float(config["SOL_DT"])
        self.viz_dt = float(config["VIZ_DT"])
        self.hoist_height_mm = float(config["HOIST_HEIGHT"]) * 1000.0
        self.hoist_duration  = float(config["HOIST_DURATION"])
        self.max_sim_time = float(config.get("MAX_SIM_TIME", 60.0))
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

        print(f"StaticDrop | {vehicle.nickname}")
        print(f"  Hoist: CoG {z_cog_hoist:.0f} mm  (+{self.hoist_height_mm:.0f} mm)  "
              f"for {self.hoist_duration:.2f} s  ({hoist_steps:,} steps)")
        print(f"  Drop:  max {self.max_sim_time:.2f} s  ({drop_steps:,} steps) | "
              f"SOL_DT={dt*1000:.1f} ms  VIZ_DT={viz_dt*1000:.0f} ms")

        log_to_file(f"StaticDrop: hoist_height={self.hoist_height_mm:.0f} mm, "
                    f"hoist_duration={self.hoist_duration:.2f} s, "
                    f"sol_dt={dt:.4f} s, viz_dt={viz_dt:.4f} s, "
                    f"max_sim_time={self.max_sim_time:.1f} s")

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
        print(f"  [HOIST] t=0.000 s — starting ({hoist_steps:,} steps)")

        log_pct_next = log_10pct_h
        for step_i in range(hoist_steps):

            # Euler: body stays pinned, only advance wheel DOFs
            deriv, shock_lengths, _ = derivatives(state, vehicle, shock_lengths, dt)
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
                print(f"  [HOIST {pct*100:3.0f}%] t={t:.3f} s  "
                      f"wheel z: FL={z_wu[0]:.0f} FR={z_wu[1]:.0f} "
                      f"RL={z_wu[2]:.0f} RR={z_wu[3]:.0f} mm")
                log_to_file(f"  [HOIST {pct*100:.0f}%] t={t:.3f}s  z_wu={z_wu.tolist()}")
                self._progress(0.08 + pct * 0.42, f"Phase 1/2: hoisting ({pct*100:.0f}%)…")
                log_pct_next += log_10pct_h

        hoist_n = len(steps)
        print(f"  [HOIST] done — {hoist_n} viz frames, "
              f"avg wheel hub = {state[IDX_Z_WU].mean():.1f} mm")
        log_to_file(f"StaticDrop: hoist done at t={t:.3f} s, {hoist_n} viz frames")

        # Phase 2: Drop
        self._progress(0.50, "Phase 2/2: dropping — vehicle in free fall...")
        print(f"  [DROP]  t={t:.3f} s — releasing body ({drop_steps:,} steps max)")

        settle_window_s = 0.5
        settle_window_steps = max(1, round(settle_window_s / dt))
        z_cog_history: deque = deque(maxlen=settle_window_steps)
        contact_logged = [False]
        log_pct_next = log_10pct_d

        for step_i in range(drop_steps):
            state, shock_lengths, _ = euler_step(state, dt, vehicle, shock_lengths)
            t += dt
            viz_accum += dt

            if viz_accum >= viz_dt:
                viz_accum = 0.0
                cs = solve_all_corners(state, vehicle, cog_static)
                steps.append(build_step_dict(state, cs, vehicle, t, "drop"))

            if not contact_logged[0]:
                wheel_r = vehicle.front_left.wheel.radius
                z_wu = state[IDX_Z_WU]
                labels = ["FL", "FR", "RL", "RR"]
                in_contact = [labels[i] for i in range(4) if z_wu[i] - wheel_r <= 0.0]
                if in_contact:
                    print(f"  [DROP]  t={t:.3f} s — first ground contact: "
                          f"{', '.join(in_contact)}")
                    log_to_file(f"StaticDrop: first contact t={t:.3f} s: {in_contact}")
                    contact_logged[0] = True

            if step_i + 1 >= log_pct_next:
                pct = (step_i + 1) / drop_steps
                z_cog_now = state[IDX_Z_COG]
                phi_deg = float(np.degrees(state[IDX_PHI]))
                theta_deg = float(np.degrees(state[IDX_THETA]))
                max_vel = float(np.max(np.abs(
                    np.concatenate([state[3:6], state[10:14]]))))
                print(f"  [DROP  {pct*100:3.0f}%] t={t:.3f} s  "
                      f"CoG z={z_cog_now:.1f} mm  roll={phi_deg:.2f}°  "
                      f"pitch={theta_deg:.2f}deg  max|v|={max_vel:.0f} mm/s")
                log_to_file(f"  [DROP {pct*100:.0f}%] t={t:.3f}s  "
                            f"z_cog={z_cog_now:.1f}  phi={phi_deg:.3f}°  "
                            f"theta={theta_deg:.3f}°  max_vel={max_vel:.1f}")
                self._progress(0.50 + pct * 0.45,
                               f"Phase 2/2: drop ({pct*100:.0f}%) — "
                               f"CoG z={z_cog_now:.0f} mm  max|v|={max_vel:.0f} mm/s")
                log_pct_next += log_10pct_d

            if _is_settled(state, z_cog_history, z_cog_static):
                z_final = state[IDX_Z_COG]
                print(f"  [DROP]  t={t:.3f} s — SETTLED  "
                      f"CoG z={z_final:.1f} mm  "
                      f"(Δ from static = {z_final - z_cog_static:+.1f} mm)")
                log_to_file(f"StaticDrop: settled t={t:.3f}s  "
                            f"z_cog={z_final:.2f}  delta={z_final - z_cog_static:+.2f} mm")
                cs = solve_all_corners(state, vehicle, cog_static)
                steps.append(build_step_dict(state, cs, vehicle, t, "settled"))
                break
        else:
            print(f"  [DROP]  reached MAX_SIM_TIME={self.max_sim_time:.1f} s without settling")
            log_to_file(f"StaticDrop: hit MAX_SIM_TIME without settling")

        total = len(steps)
        print(f"  StaticDrop done — {total} viz frames "
              f"({hoist_n} hoist + {total - hoist_n} drop)")
        log_to_file(f"StaticDrop: complete — {total} total viz frames")
        self._progress(1.0, f"Done — {total} frames ready")
        return steps
