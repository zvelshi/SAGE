from __future__ import annotations

# default
from collections import deque
from typing import List, Dict, Any, Callable

# third-party
import numpy as np

# ours
from simulations.scenarios.base import Scenario
from utils.misc import log_to_file
from utils.geometry import _bump_z_for_corner, _body_rotation, _apply_body_transform

# constants
_G_M = 9.80665 # [m/s^2]

def _noop_progress(*_): pass

# ---------------------------------------------------------------------------
# State layout (14 scalars):
#   [0]  z_cog   – CoG vertical position [mm]
#   [1]  phi     – roll  angle [rad]
#   [2]  theta   – pitch angle [rad]
#   [3]  dz_cog  – CoG vertical velocity [mm/s]
#   [4]  dphi    – roll  rate [rad/s]
#   [5]  dtheta  – pitch rate [rad/s]
#   [6..9]  z_wu  [FL, FR, RL, RR]  – wheel hub vertical position [mm]
#   [10..13] dz_wu [FL, FR, RL, RR] – wheel hub vertical velocity [mm/s]
# ---------------------------------------------------------------------------
_IDX_Z_COG  = 0
_IDX_PHI    = 1
_IDX_THETA  = 2
_IDX_DZ_COG = 3
_IDX_DPHI   = 4
_IDX_DTHETA = 5
_IDX_Z_WU   = slice(6, 10)
_IDX_DZ_WU  = slice(10, 14)

_CORNERS_ATTR = ["front_left", "front_right", "rear_left", "rear_right"]

def _derivatives(state: np.ndarray, vehicle, shock_lengths_prev: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Full EOM derivatives. Calls analytical kinematic solver for each corner."""
    deriv = np.zeros(14)

    z_cog  = state[_IDX_Z_COG]
    phi    = state[_IDX_PHI]
    theta  = state[_IDX_THETA]
    dz_cog = state[_IDX_DZ_COG]
    dphi   = state[_IDX_DPHI]
    dtheta = state[_IDX_DTHETA]
    z_wu   = state[_IDX_Z_WU]
    dz_wu  = state[_IDX_DZ_WU]

    cog_static = np.array(vehicle.cog)
    m_s  = vehicle.total_sprung_mass
    Ixx  = vehicle.inertia_matrix[0, 0]
    Iyy  = vehicle.inertia_matrix[1, 1]

    deriv[_IDX_Z_COG]  = dz_cog
    deriv[_IDX_PHI]    = dphi
    deriv[_IDX_THETA]  = dtheta
    deriv[_IDX_Z_WU]   = dz_wu

    Fz_body  = -m_s * _G_M
    Mx_roll  = 0.0
    My_pitch = 0.0
    new_sl = shock_lengths_prev.copy()

    for i, attr in enumerate(_CORNERS_ATTR):
        corner = getattr(vehicle, attr)
        hp = corner.hardpoints
        m_u = corner.unsprung_mass
        wc_stat = np.array(hp.wc, float)

        bump_z = _bump_z_for_corner(z_wu[i], z_cog, phi, theta, wc_stat, cog_static)
        step = corner.solver.solve(bump_z=bump_z)

        if step is None:
            Fz_shock = 0.0
        else:
            s_ib = np.asarray(step["s_ib"], float)
            s_ob = np.asarray(step["s_ob"], float)
            axis = s_ib - s_ob
            alen = float(np.linalg.norm(axis))
            cos_z = float(axis[2] / alen) if alen > 1e-9 else 0.0
            sl = float(step["shock_length"])
            new_sl[i] = sl
            v_sh = (sl - shock_lengths_prev[i]) / dt
            F_sh = corner.shock.get_total_force(sl, v_sh)
            Fz_shock = F_sh * cos_z

        Fz_tire = corner.wheel.contact_force(z_wu[i], dz_wu[i], ground_z=0.0)

        dy_m = (wc_stat[1] - cog_static[1]) * 1e-3
        dx_m = (wc_stat[0] - cog_static[0]) * 1e-3

        Fz_body += Fz_shock
        Mx_roll += Fz_shock * dy_m
        My_pitch -= Fz_shock * dx_m
        deriv[10 + i] = ((-m_u * _G_M - Fz_shock + Fz_tire) / m_u) * 1000.0

    deriv[_IDX_DZ_COG] = (Fz_body / m_s) * 1000.0
    deriv[_IDX_DPHI] = Mx_roll / Ixx
    deriv[_IDX_DTHETA] = My_pitch / Iyy

    return deriv, new_sl

def _euler_step(state: np.ndarray, dt: float, vehicle, shock_lengths_prev: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Euler integration step."""
    deriv, new_sl = _derivatives(state, vehicle, shock_lengths_prev, dt)
    return state + dt * deriv, new_sl


def _solve_all_corners(state: np.ndarray, vehicle, cog_static: np.ndarray) -> list:
    """Solve for all corners and transform to world frame.""" 
    z_cog = state[_IDX_Z_COG]
    phi = state[_IDX_PHI]
    theta = state[_IDX_THETA]
    z_wu = state[_IDX_Z_WU]

    cog_world = np.array([cog_static[0], cog_static[1], z_cog])
    R_body = _body_rotation(phi, theta)

    corner_steps = []
    for i, attr in enumerate(_CORNERS_ATTR):
        corner = getattr(vehicle, attr)
        hp = corner.hardpoints
        wc_stat = np.array(hp.wc, float)
        bump_z = _bump_z_for_corner(z_wu[i], z_cog, phi, theta, wc_stat, cog_static)
        step = corner.solver.solve(bump_z=bump_z)
        step = _apply_body_transform(step, cog_static, cog_world, R_body)
        corner_steps.append(step)
    return corner_steps

def _build_step_dict(state: np.ndarray, corner_steps: list, vehicle, t: float, phase: str) -> Dict[str, Any]:
    """Build a step dictionary."""
    z_cog = float(state[_IDX_Z_COG])
    return {
        "t": t,
        "phase": phase,
        "cog_pos": np.array([float(vehicle.cog[0]), float(vehicle.cog[1]), z_cog]),
        "phi": float(state[_IDX_PHI]),
        "theta": float(state[_IDX_THETA]),
        "fl": corner_steps[0],
        "fr": corner_steps[1],
        "rl": corner_steps[2],
        "rr": corner_steps[3],
    }

def _initial_shock_lengths(vehicle) -> np.ndarray:
    """Get initial shock lengths."""
    return np.array([getattr(vehicle, a).hardpoints.shock_max for a in _CORNERS_ATTR], dtype=float)

def _is_settled(state: np.ndarray, z_cog_history: deque, z_cog_static: float, tol_frac: float = 0.02) -> bool:
    """Check if the vehicle CoG has settled to within `tol_frac` (2% by default)
    of the static ride height, sustained over the trailing history window."""
    z_cog_history.append(state[_IDX_Z_COG])
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
        state[_IDX_Z_COG] = z_cog_hoist
        for i, attr in enumerate(_CORNERS_ATTR):
            corner  = getattr(vehicle, attr)
            wc_stat = np.array(corner.hardpoints.wc, float)
            state[6 + i] = z_cog_hoist + (wc_stat[2] - z_cog_static)

        shock_lengths = _initial_shock_lengths(vehicle)

        steps: List[Dict] = []
        t = 0.0
        viz_accum = 0.0

        # Phase 1: Hoist
        self._progress(0.08, "Phase 1/2: hoisting — suspension drooping...")
        print(f"  [HOIST] t=0.000 s — starting ({hoist_steps:,} steps)")

        log_pct_next = log_10pct_h
        for step_i in range(hoist_steps):

            # Euler: body stays pinned, only advance wheel DOFs
            deriv, shock_lengths = _derivatives(state, vehicle, shock_lengths, dt)
            state[_IDX_Z_WU] = state[_IDX_Z_WU] + dt * deriv[_IDX_Z_WU]
            state[_IDX_DZ_WU] = state[_IDX_DZ_WU] + dt * deriv[_IDX_DZ_WU]

            # Body pinned
            state[_IDX_Z_COG] = z_cog_hoist
            state[_IDX_PHI] = 0.0
            state[_IDX_THETA] = 0.0
            state[_IDX_DZ_COG] = 0.0
            state[_IDX_DPHI] = 0.0
            state[_IDX_DTHETA] = 0.0

            t += dt
            viz_accum += dt

            if viz_accum >= viz_dt:
                viz_accum = 0.0
                cs = _solve_all_corners(state, vehicle, cog_static)
                steps.append(_build_step_dict(state, cs, vehicle, t, "hoist"))

            if step_i + 1 >= log_pct_next:
                pct = (step_i + 1) / hoist_steps
                z_wu = state[_IDX_Z_WU]
                print(f"  [HOIST {pct*100:3.0f}%] t={t:.3f} s  "
                      f"wheel z: FL={z_wu[0]:.0f} FR={z_wu[1]:.0f} "
                      f"RL={z_wu[2]:.0f} RR={z_wu[3]:.0f} mm")
                log_to_file(f"  [HOIST {pct*100:.0f}%] t={t:.3f}s  z_wu={z_wu.tolist()}")
                self._progress(0.08 + pct * 0.42, f"Phase 1/2: hoisting ({pct*100:.0f}%)…")
                log_pct_next += log_10pct_h

        hoist_n = len(steps)
        print(f"  [HOIST] done — {hoist_n} viz frames, "
              f"avg wheel hub = {state[_IDX_Z_WU].mean():.1f} mm")
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
            state, shock_lengths = _euler_step(state, dt, vehicle, shock_lengths)
            t += dt
            viz_accum += dt

            if viz_accum >= viz_dt:
                viz_accum = 0.0
                cs = _solve_all_corners(state, vehicle, cog_static)
                steps.append(_build_step_dict(state, cs, vehicle, t, "drop"))

            if not contact_logged[0]:
                wheel_r = vehicle.front_left.wheel.radius
                z_wu = state[_IDX_Z_WU]
                labels = ["FL", "FR", "RL", "RR"]
                in_contact = [labels[i] for i in range(4) if z_wu[i] - wheel_r <= 0.0]
                if in_contact:
                    print(f"  [DROP]  t={t:.3f} s — first ground contact: "
                          f"{', '.join(in_contact)}")
                    log_to_file(f"StaticDrop: first contact t={t:.3f} s: {in_contact}")
                    contact_logged[0] = True

            if step_i + 1 >= log_pct_next:
                pct = (step_i + 1) / drop_steps
                z_cog_now = state[_IDX_Z_COG]
                phi_deg = float(np.degrees(state[_IDX_PHI]))
                theta_deg = float(np.degrees(state[_IDX_THETA]))
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
                z_final = state[_IDX_Z_COG]
                print(f"  [DROP]  t={t:.3f} s — SETTLED  "
                      f"CoG z={z_final:.1f} mm  "
                      f"(Δ from static = {z_final - z_cog_static:+.1f} mm)")
                log_to_file(f"StaticDrop: settled t={t:.3f}s  "
                            f"z_cog={z_final:.2f}  delta={z_final - z_cog_static:+.2f} mm")
                cs = _solve_all_corners(state, vehicle, cog_static)
                steps.append(_build_step_dict(state, cs, vehicle, t, "settled"))
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
