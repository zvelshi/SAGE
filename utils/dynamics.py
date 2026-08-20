from __future__ import annotations

# default
from typing import Any, Dict

# third-party
import numpy as np

# ours
from utils.geometry import _bump_z_for_corner, _body_rotation, _apply_body_transform

G_M = 9.80665

IDX_Z_COG = 0
IDX_PHI = 1
IDX_THETA = 2
IDX_DZ_COG = 3
IDX_DPHI = 4
IDX_DTHETA = 5
IDX_Z_WU = slice(6, 10)
IDX_DZ_WU = slice(10, 14)

CORNERS_ATTR = ["front_left", "front_right", "rear_left", "rear_right"]

def derivatives(state: np.ndarray, vehicle, shock_lengths_prev: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Full EOM derivatives. Calls analytical kinematic solver for each corner."""
    deriv = np.zeros(14)

    z_cog  = state[IDX_Z_COG]
    phi    = state[IDX_PHI]
    theta  = state[IDX_THETA]
    dz_cog = state[IDX_DZ_COG]
    dphi   = state[IDX_DPHI]
    dtheta = state[IDX_DTHETA]
    z_wu   = state[IDX_Z_WU]
    dz_wu  = state[IDX_DZ_WU]

    cog_static = np.array(vehicle.cog)
    m_s  = vehicle.total_sprung_mass
    Ixx  = vehicle.inertia_matrix[0, 0]
    Iyy  = vehicle.inertia_matrix[1, 1]

    deriv[IDX_Z_COG]  = dz_cog
    deriv[IDX_PHI]    = dphi
    deriv[IDX_THETA]  = dtheta
    deriv[IDX_Z_WU]   = dz_wu

    Fz_body  = -m_s * G_M
    Mx_roll  = 0.0
    My_pitch = 0.0
    new_sl = shock_lengths_prev.copy()

    for i, attr in enumerate(CORNERS_ATTR):
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
        deriv[10 + i] = ((-m_u * G_M - Fz_shock + Fz_tire) / m_u) * 1000.0

    deriv[IDX_DZ_COG] = (Fz_body / m_s) * 1000.0
    deriv[IDX_DPHI] = Mx_roll / Ixx
    deriv[IDX_DTHETA] = My_pitch / Iyy

    return deriv, new_sl

def euler_step(state: np.ndarray, dt: float, vehicle, shock_lengths_prev: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Euler integration step."""
    deriv, new_sl = derivatives(state, vehicle, shock_lengths_prev, dt)
    return state + dt * deriv, new_sl


def solve_all_corners(state: np.ndarray, vehicle, cog_static: np.ndarray) -> list:
    """Solve for all corners and transform to world frame.""" 
    z_cog = state[IDX_Z_COG]
    phi = state[IDX_PHI]
    theta = state[IDX_THETA]
    z_wu = state[IDX_Z_WU]

    cog_world = np.array([cog_static[0], cog_static[1], z_cog])
    R_body = _body_rotation(phi, theta)

    corner_steps = []
    for i, attr in enumerate(CORNERS_ATTR):
        corner = getattr(vehicle, attr)
        hp = corner.hardpoints
        wc_stat = np.array(hp.wc, float)
        bump_z = _bump_z_for_corner(z_wu[i], z_cog, phi, theta, wc_stat, cog_static)
        step = corner.solver.solve(bump_z=bump_z)
        step = _apply_body_transform(step, cog_static, cog_world, R_body)
        corner_steps.append(step)
    return corner_steps

def build_step_dict(state: np.ndarray, corner_steps: list, vehicle, t: float, phase: str) -> Dict[str, Any]:
    """Build a step dictionary."""
    z_cog = float(state[IDX_Z_COG])
    return {
        "t": t,
        "phase": phase,
        "cog_pos": np.array([float(vehicle.cog[0]), float(vehicle.cog[1]), z_cog]),
        "phi": float(state[IDX_PHI]),
        "theta": float(state[IDX_THETA]),
        "fl": corner_steps[0],
        "fr": corner_steps[1],
        "rl": corner_steps[2],
        "rr": corner_steps[3],
    }

def initial_shock_lengths(vehicle) -> np.ndarray:
    """Actual static shock length at each corner (bump_z=0), used to seed the
    previous-length history for the first shock-velocity finite difference.

    Must be the true solved length, not shock_max/shock_min: seeding with the
    wrong value creates a huge bogus velocity on the first step
    ((sl - seed) / dt), which the damper amplifies into an unrecoverable
    blow-up before the sim ever gets going.
    """
    lengths = np.empty(4)
    for i, attr in enumerate(CORNERS_ATTR):
        corner = getattr(vehicle, attr)
        step = corner.solver.solve(bump_z=0.0)
        lengths[i] = float(step["shock_length"]) if step is not None else corner.hardpoints.shock_max
    return lengths