from __future__ import annotations

# third-party
import numpy as np
from scipy.optimize import brentq

# ours
from models.components.axle import Axle
from models.components.cv_joint import CVJoint, PlungingCVJoint
from utils.misc import log_to_file
from utils.geometry import _rodrigues, _rot_axis, _lin_trig, _nearest_branch

class SemiTrailingLinkAnalytical:
    """
    Closed-form + 1-D brentq kinematic solver for a semi-trailing link.

    Key insight: tl_f pivot is a fixed ball joint in world space, so the entire
    upright rotates about hp.tl_f:
        q_world = tl_f + Rw @ (q_static - tl_f)

    Position is determined by rotation:
        wc_world = tl_f - Rw @ tl_f_local   where tl_f_local = tl_f - wc_static

    Reduction chain (3 DOF rotation -> 0 DOF):
      1. wc[2] target -> (Rw @ tl_f_local)[2] = c_z
         -> parameterize: Rw @ tl_f_local = (r cosφ, r sinφ, c_z)  [φ is 1-D free]
      2. UCL length -> base rotation R0(φ) via Rodrigues, then spin ψ from
         A cosψ + B sinψ = k_ucl  [closed form]
      3. LCL length -> residual g(φ) = LCL_dot - k_lcl -> outer brentq on φ
    """

    def __init__(self, hp):
        self.hp       = hp
        self.len      = type(hp).link_lengths(hp)
        self._wc_z0   = float(hp.wc[2])
        self._shock0  = self.len["shock_static"]
        self.tl_f     = np.array(hp.tl_f, float)

        # tl_f_local: static body-frame vector from wc to tl_f
        self.tl_f_local = self.tl_f - np.array(hp.wc, float)
        self.tl_f_len   = float(np.linalg.norm(self.tl_f_local))
        self.tl_f_hat   = self.tl_f_local / self.tl_f_len

        # Vectors from tl_f to outboard points in static world frame
        self.w_ucl = np.array(hp.ucl_ob, float) - self.tl_f
        self.w_lcl = np.array(hp.lcl_ob, float) - self.tl_f
        self.w_s   = np.array(hp.s_ob,   float) - self.tl_f
        self.w_piv = np.array(hp.piv_ob, float) - self.tl_f
        self.w_wc  = np.array(hp.wc,     float) - self.tl_f   # = -tl_f_local

        # Inboard target vectors
        self.d_ucl = np.array(hp.ucl_ib, float) - self.tl_f
        self.d_lcl = np.array(hp.lcl_ib, float) - self.tl_f

        # Precomputed dot constants: (Rw @ w) · d = k  ↔  |w|²+|d|²-L² / 2
        L_ucl = self.len["upper_camber_link"]
        L_lcl = self.len["lower_camber_link"]
        self.k_ucl = (float(np.dot(self.w_ucl, self.w_ucl))
                      + float(np.dot(self.d_ucl, self.d_ucl)) - L_ucl ** 2) / 2.0
        self.k_lcl = (float(np.dot(self.w_lcl, self.w_lcl))
                      + float(np.dot(self.d_lcl, self.d_lcl)) - L_lcl ** 2) / 2.0

        # Static φ (angle of tl_f_local projected onto XY plane)
        self._phi_static = float(np.arctan2(self.tl_f_local[1], self.tl_f_local[0]))

        # Axle
        spindle = np.array(hp.wc, float) - np.array(hp.piv_ob, float)
        sn = np.linalg.norm(spindle)
        self.local_spindle = spindle / sn if sn > 1e-6 else np.array([0., 1., 0.])
        self.axle = Axle(
            joint1=PlungingCVJoint(max_angle=30, plunge_limit=25),
            joint2=CVJoint(max_angle=30),
            length=self.len["axle_ib_ob_static"],
        )

        self._phi_prev = self._phi_static
        self._psi_prev = 0.0

    def reset(self):
        self._phi_prev = self._phi_static
        self._psi_prev = 0.0

    def _rw(self, phi: float, c_z: float) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        """Full rotation matrix Rw and q_hat for given (φ, c_z)."""
        r_f = np.sqrt(max(self.tl_f_len ** 2 - c_z ** 2, 0.0))
        q_f = np.array([r_f * np.cos(phi), r_f * np.sin(phi), c_z])
        # q_f always has magnitude tl_f_len -> q_hat = q_f / tl_f_len
        q_hat = q_f / self.tl_f_len
        R0 = _rodrigues(self.tl_f_hat, q_hat)

        # Spin ψ from UCL: A cosψ + B sinψ = k_ucl - wp_par·d_ucl
        wp    = R0 @ self.w_ucl
        wp_p  = float(np.dot(wp, q_hat)) * q_hat
        wp_v  = wp - wp_p
        wp_c  = np.cross(q_hat, wp_v)
        A     = float(np.dot(wp_v, self.d_ucl))
        B     = float(np.dot(wp_c, self.d_ucl))
        C     = self.k_ucl - float(np.dot(wp_p, self.d_ucl))
        sols  = _lin_trig(A, B, C)
        psi   = _nearest_branch(sols, 0.0) if sols else 0.0

        Rw = _rot_axis(q_hat, psi) @ R0
        return Rw, psi

    def _lcl_residual(self, phi: float, c_z: float) -> float:
        """LCL constraint residual for outer brentq on φ."""
        Rw, _ = self._rw(phi, c_z)
        return float(np.dot(Rw @ self.w_lcl, self.d_lcl)) - self.k_lcl

    def _solve_bump(self, bump_z: float):
        target_wc = self._wc_z0 + bump_z
        c_z = self.tl_f[2] - target_wc

        if c_z ** 2 > self.tl_f_len ** 2:
            return None

        g   = lambda phi: self._lcl_residual(phi, c_z)
        p0, span = self._phi_prev, 0.25
        for _ in range(7):
            ga, gb = g(p0 - span), g(p0 + span)
            if np.isfinite(ga) and np.isfinite(gb) and ga * gb < 0.0:
                break
            span = min(span * 2.0, np.pi)
        try:
            phi = brentq(g, p0 - span, p0 + span, xtol=1e-12, maxiter=60)
        except ValueError:
            try:
                phi = brentq(g, p0 - np.pi, p0 + np.pi, xtol=1e-12, maxiter=80)
            except ValueError:
                return None

        Rw, psi = self._rw(phi, c_z)
        self._phi_prev = phi
        self._psi_prev = psi

        wc_world  = self.tl_f + Rw @ self.w_wc
        ucl_ob_w  = self.tl_f + Rw @ self.w_ucl
        lcl_ob_w  = self.tl_f + Rw @ self.w_lcl
        s_ob_w    = self.tl_f + Rw @ self.w_s
        piv_ob_w  = self.tl_f + Rw @ self.w_piv

        hp = self.hp
        n_ob = Rw @ self.local_spindle
        n_ib_dir = 1.0 if hp.piv_ib[1] > 0 else -1.0
        axle_state = self.axle.get_state(
            np.array(hp.piv_ib, float), piv_ob_w,
            np.array([0., n_ib_dir, 0.]), n_ob,
        )

        return {
            "wc":           wc_world,
            "ucl_ib":       np.array(hp.ucl_ib, float),
            "ucl_ob":       ucl_ob_w,
            "lcl_ib":       np.array(hp.lcl_ib, float),
            "lcl_ob":       lcl_ob_w,
            "piv_ib":       np.array(hp.piv_ib, float),
            "piv_ob":       piv_ob_w,
            "s_ib":         np.array(hp.s_ib, float),
            "s_ob":         s_ob_w,
            "tl_f":         self.tl_f.copy(),
            "tl_f_upright": self.tl_f.copy(),
            "wheel_axis":   n_ob,
            "axle_data":    axle_state,
            "shock_length": float(np.linalg.norm(np.array(hp.s_ib, float) - s_ob_w)),
        }

    def solve(self, *, travel_mm: float | None = None, bump_z: float | None = None,
              steer_mm: float = 0.0):
        if (travel_mm is None) == (bump_z is None):
            raise ValueError("Specify exactly ONE of travel_mm or bump_z")

        if bump_z is not None:
            return self._solve_bump(bump_z)

        # travel_mm: find bump_z s.t. shock_length = target_shock
        target_shock = self._shock0 - travel_mm
        hp = self.hp
        if not (hp.shock_min <= target_shock <= hp.shock_max):
            log_to_file(f"[WARN] shock {target_shock:.1f} out of range")
            return None

        def f(bz: float) -> float:
            step = self._solve_bump(bz)
            return (step["shock_length"] - target_shock) if step else np.nan

        bz_range = hp.shock_max - hp.shock_min
        try:
            bz = brentq(f, -bz_range, bz_range, xtol=1e-6, maxiter=40)
        except ValueError:
            return None
        return self._solve_bump(bz)
