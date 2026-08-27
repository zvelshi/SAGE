# default
from __future__ import annotations

# ours
from models.components.axle import Axle
from models.corners._kinematics import euler_xyz, cross3
from utils.logging_setup import get_logger

log = get_logger(__name__)

# third-party
import numpy as np
from scipy.optimize import least_squares

class DoubleAArmNumeric:
    def __init__(self, hp, axle: Axle):
        self.hp = hp
        self.len = type(hp).link_lengths(hp)

        self._x_prev = np.hstack([hp.lbj, np.zeros(3)]) # seed guess
        self._wc0 = hp.wc[2]
        self._shock0 = self.len["shock_static"]
        self._tierod0 = self.len["tie_rod"]

        self.s_rel_pt = hp.ubj
        if hp.s_loc == 'upper':
            self.s_rel_pt = hp.ubj
        else:
            self.s_rel_pt = hp.lbj
        self.sh_vec = hp.s_ob - self.s_rel_pt

        # The shock/damper outboard point is rigidly attached to whichever wishbone
        # owns s_rel_pt ("Damper to Lower/Upper Wishbone"), and that wishbone rotates
        # about its OWN fixed inboard-pivot axis -- a different (single-DOF) rotation
        # from the upright's full 6-DOF pose solved for in solve(). Precompute that
        # axis and the ball joint's static radius vector from it so solve() can derive
        # the wishbone's live rotation angle from how far the ball joint has swung.
        axis_p1, axis_p2 = (hp.uf, hp.ur) if hp.s_loc == 'upper' else (hp.lf, hp.lr)
        axis_dir = axis_p2 - axis_p1
        self._sh_axis_n = axis_dir / np.linalg.norm(axis_dir)
        foot = axis_p1 + np.dot(self.s_rel_pt - axis_p1, self._sh_axis_n) * self._sh_axis_n
        self._sh_axis_foot = foot
        self._sh_v0 = self.s_rel_pt - foot
        self._sh_v0_normsq = float(np.dot(self._sh_v0, self._sh_v0))

        # passive axle
        self.axle_static_len = np.linalg.norm(hp.piv_ob - hp.piv_ib)
        self.piv_ob_loc = hp.piv_ob - hp.lbj

        self.axle = axle

        # static camber calculation
        spindle_vec = self.hp.wc - self.hp.piv_ob
        norm = np.linalg.norm(spindle_vec)
        if norm < 1e-6: # check they aren't the same point
            self.local_spindle_axis = np.array([0.0, 1.0, 0.0])
        else:
            self.local_spindle_axis = spindle_vec / norm
        static_camber = -np.rad2deg(np.arcsin(self.local_spindle_axis[2]))

    def reset(self):
        # reset the prev x to the default guess
        self._x_prev = np.hstack([self.hp.lbj, np.zeros(3)])

    @staticmethod
    def _rot(eul: np.ndarray) -> np.ndarray:
        return euler_xyz(eul)

    def _shock_outboard(self, current_ball: np.ndarray) -> np.ndarray:
        """Current shock/damper outboard point, given the live position of the ball
        joint (ubj or lbj, matching s_loc) that its owning wishbone also carries.
        Derives that wishbone's rotation angle about its fixed pivot axis from how
        current_ball has swung relative to its static position, via Rodrigues'
        rotation formula (cheaper per-call than building a scipy Rotation)."""
        axis = self._sh_axis_n
        v1 = current_ball - self._sh_axis_foot
        denom = np.sqrt(self._sh_v0_normsq * np.dot(v1, v1))
        cos_t = np.dot(self._sh_v0, v1) / denom
        sin_t = np.dot(cross3(self._sh_v0, v1), axis) / denom
        v = self.sh_vec
        v_rot = (v * cos_t + cross3(axis, v) * sin_t
                 + axis * np.dot(axis, v) * (1.0 - cos_t))
        return current_ball + v_rot

    def solve(
            self,
            travel_mm : float | None = None,
            bump_z    : float | None = None,
            steer_mm  : float = 0.0,
        ):
        if (travel_mm is None) == (bump_z is None):
            raise ValueError("Specify exactly ONE of travel_mm or bump_z")

        hp = self.hp
        target_shock = self._shock0 - travel_mm if travel_mm is not None else None
        if target_shock and not (hp.shock_min <= target_shock <= hp.shock_max):
            log.debug("target shock length %.2fmm out of bounds (%s-%smm)",
                      target_shock, hp.shock_min, hp.shock_max)
            return None

        target_wheel = self._wc0 + bump_z if bump_z is not None else None

        target_tie   = self.len["tie_rod"]
        tr_ib_offset = hp.tr_ib + np.array([0.0, steer_mm, 0.0])

        # local coords (lbj frame)
        ubj_loc = hp.ubj - hp.lbj
        tr_ob_loc = hp.tr_ob - hp.lbj
        wc_loc = hp.wc - hp.lbj

        def res(x):
            p, e = x[:3], x[3:]
            Rw = self._rot(e)
            world = lambda v: p + Rw @ v

            lbj   = p
            ubj   = world(ubj_loc)
            tr_ob = world(tr_ob_loc)
            wc    = world(wc_loc)

            s_rel_pt_loc = ubj if hp.s_loc == 'upper' else lbj
            sha   = self._shock_outboard(s_rel_pt_loc)

            r = np.empty(6)
            
            # 4 a arms
            r[0] = np.linalg.norm(hp.uf - ubj) - self.len["upper_front"]
            r[1] = np.linalg.norm(hp.ur - ubj) - self.len["upper_rear"]
            r[2] = np.linalg.norm(hp.lf - lbj) - self.len["lower_front"]
            r[3] = np.linalg.norm(hp.lr - lbj) - self.len["lower_rear"]
            
            # tie-rod
            r[4] = np.linalg.norm(tr_ib_offset - tr_ob) - target_tie
            
            # shock / wheel
            if target_shock:
                r[5] = np.linalg.norm(hp.s_ib - sha) - target_shock
            else:
                r[5] = wc[2] - target_wheel

            return r

        sol = least_squares(res, self._x_prev, method="lm", xtol=1e-7, ftol=1e-7, gtol=1e-7)
        if not sol.success:
            return
        if (hp.wc[1] >= 0) != (sol.x[1] >= 0) or abs(sol.x[4]) > np.pi / 2:
            log.debug("DoubleAArm solve landed on a mirrored/flipped root: x=%s", sol.x)
            return
        self._x_prev = sol.x.copy()

        p, e = sol.x[:3], sol.x[3:]
        Rw   = self._rot(e) # wheel rot matrix
        world = lambda v: p + Rw @ v

        lbj = p
        ubj = world(ubj_loc)
        wc = world(wc_loc)
        tr_ob = world(tr_ob_loc)
        
        s_rel_pt_loc = ubj if hp.s_loc == 'upper' else lbj
        sha = self._shock_outboard(s_rel_pt_loc)

        # axle calcs
        piv_ob = world(self.piv_ob_loc)
        n_ib_dir = 1.0 if hp.piv_ib[1] > 0 else -1.0
        n_ib = np.array([0.0, n_ib_dir, 0.0])
        n_ob = Rw @ self.local_spindle_axis
        axle_state = self.axle.get_state(hp.piv_ib, piv_ob, n_ib, n_ob)

        step = {
            "lbj": lbj,
            "ubj": ubj,
            "uf": hp.uf,
            "ur": hp.ur,
            "lf": hp.lf,
            "lr": hp.lr,
            "wc": wc,
            "s_ib": hp.s_ib,
            "s_ob": sha,
            "piv_ib": hp.piv_ib,
            "piv_ob": piv_ob,
            "tr_ib": tr_ib_offset,
            "tr_ob": tr_ob,
            "wheel_axis": n_ob,
            "axle_data": axle_state,
            "shock_length": np.linalg.norm(hp.s_ib - sha),
        }
        return step