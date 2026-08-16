# default
from typing import Dict

# ours
from models.components.axle import Axle
from utils.misc import log_to_file

# third-party
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

class SemiTrailingLinkNumeric:
    def __init__(self, hp, axle: Axle):
        self.hp = hp
        self.len = type(hp).link_lengths(hp)

        # state vector = [wc_x, wc_y, wc_z, eul_x, eul_y, eul_z]
        self._x_prev = np.hstack([hp.wc, np.zeros(3)])

        self._wc_z0   = hp.wc[2]
        self._shock_0 = self.len["shock_static"]

        self._tl_f_local = hp.tl_f - hp.wc

        self.axle = axle

        # static vectors
        self._axle_ob_vec = hp.piv_ob - hp.wc
        self._static_forward = np.array([1.0, 0.0, 0.0])

        # static camber calculation
        spindle_vec = hp.wc - hp.piv_ob
        norm = np.linalg.norm(spindle_vec)
        if norm < 1e-6:
            self.local_spindle_axis = np.array([0.0, 1.0, 0.0])
        else:
            self.local_spindle_axis = spindle_vec / norm

    def reset(self):
        self._x_prev = np.hstack([self.hp.wc, np.zeros(3)])

    @staticmethod
    def _rot(eul: np.ndarray) -> np.ndarray:
        return R.from_euler("xyz", eul).as_matrix()

    def solve(
        self,
        *,
        travel_mm: float | None = None,
        bump_z: float | None = None,
    ) -> Dict[str, np.ndarray] | None:

        if (travel_mm is None) == (bump_z is None):
            raise ValueError("Specify exactly *one* of travel_mm or bump_z")

        hp = self.hp

        targ_shock = self._shock_0 - travel_mm if travel_mm is not None else None
        if targ_shock is not None and not (hp.shock_min <= targ_shock <= hp.shock_max):
            log_to_file(f"[WARN] Target shock length {targ_shock:.2f}mm out of bounds ({hp.shock_min}-{hp.shock_max}mm)")
            return None
        targ_wcz = self._wc_z0 + bump_z if bump_z is not None else None

        # Vectors from wheel center to outboard points in static
        ucl_vec = hp.ucl_ob - hp.wc
        lcl_vec = hp.lcl_ob - hp.wc
        s_ob_vec = hp.s_ob - hp.wc
        axle_ob_vec = self._axle_ob_vec

        def res(x):
            p, e = x[:3], x[3:]
            Rw = self._rot(e)

            # Transform outboard points
            ucl_ob_w = p + Rw @ ucl_vec
            lcl_ob_w = p + Rw @ lcl_vec
            s_ob_w   = p + Rw @ s_ob_vec
            
            # Front trailing pivot (rigidly attached to upright)
            tl_f_current = p + Rw @ self._tl_f_local

            r = np.empty(6)

            # [0, 1, 2] Trailing link Front Pivot 
            delta_pivot = tl_f_current - hp.tl_f
            r[0] = delta_pivot[0]
            r[1] = delta_pivot[1]
            r[2] = delta_pivot[2]

            # [3] Upper Camber Link
            r[3] = np.linalg.norm(ucl_ob_w - hp.ucl_ib) - self.len["upper_camber_link"]

            # [4] Lower Camber Link
            r[4] = np.linalg.norm(lcl_ob_w - hp.lcl_ib) - self.len["lower_camber_link"]

            # [5] Travel Constraint
            if targ_shock is not None:
                r[5] = np.linalg.norm(hp.s_ib - s_ob_w) - targ_shock
            else:
                r[5] = p[2] - targ_wcz

            return r

        sol = least_squares(
            res,
            self._x_prev,
            xtol=1e-9,
        )

        if not sol.success:
            print(f"Solution failed: {sol.message}")
            return None

        self._x_prev = sol.x.copy()
        p, e = sol.x[:3], sol.x[3:]
        Rw = self._rot(e)

        # Re-calculate final points for plotting
        ucl_ob_w = p + Rw @ ucl_vec
        lcl_ob_w = p + Rw @ lcl_vec
        s_ob_w   = p + Rw @ s_ob_vec
        piv_ob_w = p + Rw @ axle_ob_vec
        tl_f_current = p + Rw @ self._tl_f_local

        # Axle state calculation
        n_ib_dir = 1.0 if hp.piv_ib[1] > 0 else -1.0
        n_ib = np.array([0.0, n_ib_dir, 0.0])
        n_ob = Rw @ self.local_spindle_axis
        axle_state = self.axle.get_state(hp.piv_ib, piv_ob_w, n_ib, n_ob)

        step = {
            "wc": p,
            "ucl_ib": hp.ucl_ib,
            "ucl_ob": ucl_ob_w,
            "lcl_ib": hp.lcl_ib,
            "lcl_ob": lcl_ob_w,
            "piv_ib": hp.piv_ib,
            "piv_ob": piv_ob_w,
            "s_ib": hp.s_ib,
            "s_ob": s_ob_w,
            "tl_f": hp.tl_f,
            "tl_f_upright": tl_f_current,
            "wheel_axis": n_ob,
            "axle_data": axle_state,
            "shock_length": np.linalg.norm(hp.s_ib - s_ob_w),
        }        
        return step