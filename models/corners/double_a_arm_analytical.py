from __future__ import annotations

# third-party
import numpy as np
from scipy.optimize import brentq

# ours
from models.components.axle import Axle
from models.components.cv_joint import CVJoint, PlungingCVJoint
from utils.misc import log_to_file
from utils.geometry import _circle_from_spheres, _rodrigues, _rot_axis, _lin_trig, _nearest_branch

class DoubleAArmAnalytical:
    """
    Closed-form + 1-D brentq kinematic solver for a double A-arm.

    Reduction:
      1. Lower A-arm (2 constraints) -> lbj on circle C_lower
      2. Upper A-arm (2 constraints) -> ubj on circle C_upper
      3. Rigid upright |ubj - lbj| = L -> β(α) closed form via A cosβ + B sinβ = C
      4. Tie-rod length -> upright spin ψ(α) closed form, same trick
      5. Outer 1-D brentq on α -> wc[2] = target (bump_z) or shock = target (travel_mm)
    """

    def __init__(self, hp):
        self.hp      = hp
        self.len     = type(hp).link_lengths(hp)
        self._wc0    = float(hp.wc[2])
        self._shock0 = self.len["shock_static"]
        self.L_tr    = self.len["tie_rod"]

        lf = np.array(hp.lf, float)
        lr = np.array(hp.lr, float)
        uf = np.array(hp.uf, float)
        ur = np.array(hp.ur, float)

        res = _circle_from_spheres(lf, self.len["lower_front"], lr, self.len["lower_rear"])
        assert res is not None, "Lower A-arm circle degenerate"
        self.c_lo, self.r_lo, self.u_lo, self.v_lo = res
        self.n_lo = np.cross(self.u_lo, self.v_lo)
        self.lf = lf

        res = _circle_from_spheres(uf, self.len["upper_front"], ur, self.len["upper_rear"])
        assert res is not None, "Upper A-arm circle degenerate"
        self.c_hi, self.r_hi, self.u_hi, self.v_hi = res

        lbj = np.array(hp.lbj, float)
        ubj = np.array(hp.ubj, float)
        self.ubj_loc    = ubj - lbj
        self.wc_loc     = np.array(hp.wc, float) - lbj
        self.tr_ob_loc  = np.array(hp.tr_ob, float) - lbj
        self.piv_ob_loc = np.array(hp.piv_ob, float) - lbj
        # Shock outboard: attached to the lower A-arm body
        self.s_ob_static = np.array(hp.s_ob, float)
        self.L_upright   = float(np.linalg.norm(self.ubj_loc))

        lbj_prj = lbj - self.c_lo
        self._alpha_static = float(np.arctan2(np.dot(lbj_prj, self.v_lo), np.dot(lbj_prj, self.u_lo)))
        ubj_prj = ubj - self.c_hi
        self._beta_static  = float(np.arctan2(np.dot(ubj_prj, self.v_hi), np.dot(ubj_prj, self.u_hi)))

        spindle = np.array(hp.wc, float) - np.array(hp.piv_ob, float)
        sn = np.linalg.norm(spindle)
        self.local_spindle = spindle / sn if sn > 1e-6 else np.array([0., 1., 0.])
        self.axle = Axle(
            joint1=PlungingCVJoint(max_angle=30, plunge_limit=30.0),
            joint2=CVJoint(max_angle=30),
            length=self.len["axle_ib_ob_static"],
        )

        self._alpha_prev = self._alpha_static
        self._psi_prev   = 0.0

    def reset(self):
        self._alpha_prev = self._alpha_static
        self._psi_prev   = 0.0

    def _lbj(self, a: float) -> np.ndarray:
        return self.c_lo + self.r_lo * (np.cos(a) * self.u_lo + np.sin(a) * self.v_lo)

    def _ubj(self, lbj: np.ndarray) -> np.ndarray | None:
        p = self.c_hi - lbj
        A = float(np.dot(p, self.u_hi))
        B = float(np.dot(p, self.v_hi))
        C = (self.L_upright ** 2 - float(np.dot(p, p)) - self.r_hi ** 2) / (2.0 * self.r_hi)
        sols = _lin_trig(A, B, C)
        if not sols:
            return None
        best_ubj = None
        best_dot = -float('inf')
        for b in sols:
            ubj_cand = self.c_hi + self.r_hi * (np.cos(b) * self.u_hi + np.sin(b) * self.v_hi)
            vec = ubj_cand - lbj
            d = float(np.dot(vec, self.ubj_loc))
            if d > best_dot:
                best_dot = d
                best_ubj = ubj_cand
        return best_ubj

    def _spin_psi(self, lbj: np.ndarray, R0: np.ndarray, sw: np.ndarray,
                  steer_mm: float = 0.0) -> float:
        """Closed-form spin angle ψ from tie-rod constraint."""
        tr_ib = np.array(self.hp.tr_ib, float) + np.array([0., steer_mm, 0.])
        q    = R0 @ self.tr_ob_loc
        qp   = float(np.dot(q, sw)) * sw
        qv   = q - qp
        qc   = np.cross(sw, qv)
        d    = (tr_ib - lbj) - qp
        A    = float(np.dot(d, qv))
        B    = float(np.dot(d, qc))
        C    = (float(np.dot(d, d)) + float(np.dot(qv, qv)) - self.L_tr ** 2) / 2.0
        sols = _lin_trig(A, B, C)
        return _nearest_branch(sols, 0.0) if sols else 0.0

    def _build(self, a: float, steer_mm: float = 0.0):
        lbj = self._lbj(a)
        ubj = self._ubj(lbj)
        if ubj is None:
            return None
        sl  = self.ubj_loc / self.L_upright
        sw  = (ubj - lbj) / self.L_upright
        R0  = _rodrigues(sl, sw)
        psi = self._spin_psi(lbj, R0, sw, steer_mm)
        Rw  = _rot_axis(sw, psi) @ R0
        return lbj, ubj, Rw, sw, psi

    def _wc_z(self, a: float, steer_mm: float = 0.0) -> float:
        cfg = self._build(a, steer_mm)
        if cfg is None:
            return np.nan
        lbj, _, Rw, _, _ = cfg
        return float((lbj + Rw @ self.wc_loc)[2])

    def _shock_len(self, a: float) -> float:
        delta_a = a - self._alpha_static
        R_lo = _rot_axis(self.n_lo, delta_a)
        sha = self.lf + R_lo @ (self.s_ob_static - self.lf)
        return float(np.linalg.norm(np.array(self.hp.s_ib, float) - sha))

    def _brentq_alpha(self, residual) -> float | None:
        a0, span = self._alpha_prev, 0.25
        for _ in range(7):
            fa, fb = residual(a0 - span), residual(a0 + span)
            if np.isfinite(fa) and np.isfinite(fb) and fa * fb < 0.0:
                break
            span = min(span * 2.0, np.pi)
        try:
            return brentq(residual, a0 - span, a0 + span, xtol=1e-5, maxiter=60)
        except ValueError:
            pass
        try:
            return brentq(residual, a0 - np.pi, a0 + np.pi, xtol=1e-5, maxiter=80)
        except ValueError:
            return None

    def solve(self, *, travel_mm: float | None = None, bump_z: float | None = None,
              steer_mm: float = 0.0):
        if (travel_mm is None) == (bump_z is None):
            raise ValueError("Specify exactly ONE of travel_mm or bump_z")

        hp = self.hp

        if travel_mm is not None:
            target_shock = self._shock0 - travel_mm
            if not (hp.shock_min <= target_shock <= hp.shock_max):
                log_to_file(f"[WARN] target shock {target_shock:.1f} out of bounds")
                return None
            res = lambda a: self._shock_len(a) - target_shock
        else:
            target_wc = self._wc0 + bump_z
            res = lambda a: self._wc_z(a, steer_mm) - target_wc

        alpha = self._brentq_alpha(res)
        if alpha is None:
            return None

        cfg = self._build(alpha, steer_mm)
        if cfg is None:
            return None
        lbj, ubj, Rw, _, psi = cfg
        self._alpha_prev = alpha
        self._psi_prev   = psi

        wc     = lbj + Rw @ self.wc_loc
        tr_ob  = lbj + Rw @ self.tr_ob_loc
        piv_ob = lbj + Rw @ self.piv_ob_loc
        
        delta_a = alpha - self._alpha_static
        R_lo = _rot_axis(self.n_lo, delta_a)
        sha = self.lf + R_lo @ (self.s_ob_static - self.lf)

        n_ob = Rw @ self.local_spindle
        n_ib_dir = 1.0 if hp.piv_ib[1] > 0 else -1.0
        axle_state = self.axle.get_state(
            np.array(hp.piv_ib, float), piv_ob,
            np.array([0., n_ib_dir, 0.]), n_ob,
        )

        return {
            "lbj":        lbj,
            "ubj":        ubj,
            "uf":         np.array(hp.uf, float),
            "ur":         np.array(hp.ur, float),
            "lf":         np.array(hp.lf, float),
            "lr":         np.array(hp.lr, float),
            "wc":         wc,
            "s_ib":       np.array(hp.s_ib, float),
            "s_ob":       sha,
            "piv_ib":     np.array(hp.piv_ib, float),
            "piv_ob":     piv_ob,
            "tr_ib":      np.array(hp.tr_ib, float),
            "tr_ob":      tr_ob,
            "wheel_axis": n_ob,
            "axle_data":  axle_state,
            "shock_length": float(np.linalg.norm(np.array(hp.s_ib, float) - sha)),
        }
