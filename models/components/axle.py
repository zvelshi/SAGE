from __future__ import annotations
from dataclasses import dataclass
from typing import Dict

# ours
from .joint import Joint
from .axle_type import AxleType

# third-party
import numpy as np

@dataclass
class Axle:
    """
    The Axle container.
    Manages the two joints and the connecting shaft.
    """

    joint1: Joint
    joint2: Joint
    length: float # static center-to-center length
    axle_type: AxleType
    piv_ib_static: np.ndarray # static (hardpoints-file) inboard pivot position

    def resolve_inboard(self, piv_ob: np.ndarray) -> np.ndarray | None:
        """Where the inboard pivot sits this step, per this axle's plunge model
        (see `AxleType`). Returns None if `piv_ob` is out of this axle type's reach."""
        return self.axle_type.resolve_inboard(self.piv_ib_static, piv_ob, self.length)

    def constraints(
        self, 
        p_inboard: np.ndarray, 
        p_outboard: np.ndarray, 
        n_inboard: np.ndarray, 
        n_outboard: np.ndarray
    ) -> np.ndarray:

        # Calculate shaft vector and length 
        shaft_vec = p_outboard - p_inboard
        current_len = np.linalg.norm(shaft_vec)

        # Collect residuals from both joints
        # Joint 1 (Inboard)
        res1 = self.joint1.residual(shaft_vec, n_inboard, current_len, self.length)

        # Joint 2 (Outboard) - Note the negative shaft_vec for the opposing joint
        res2 = self.joint2.residual(-shaft_vec, n_outboard, current_len, self.length)

        # Flatten into a single array for solver
        return np.array(res1 + res2)

    def get_state(
        self, 
        p_inboard: np.ndarray, 
        p_outboard: np.ndarray, 
        n_inboard: np.ndarray, 
        n_outboard: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculates the current state (Plunge mm, Angles deg) for reporting/plotting.
        Does not apply penalties.
        """
        shaft_vec = p_outboard - p_inboard

        # Plunge Calculation (+ve = extension, -ve = compression) -- meaning
        # depends on axle_type: joint slide distance for CVPlunge, shaft
        # length change for InternalPlunge. See AxleType.plunge_mm.
        plunge = self.axle_type.plunge_mm(self.piv_ib_static, p_inboard, p_outboard, self.length)

        # Helper to calculate the joint's operating (bend) angle.
        # The joint's normal is a directionless axis (a rotation about it, or
        # about its reverse, is physically identical), so the raw angle
        # between shaft and normal can land anywhere in [0, 180] depending on
        # which way the normal happens to point. Fold it into [0, 90] so the
        # readout always reflects the actual articulation of the joint.
        def get_true_angle(vec, normal):
            v_norm = np.linalg.norm(vec)
            n_norm = np.linalg.norm(normal)
            if v_norm < 1e-9 or n_norm < 1e-9: return 0.0
            dot = np.clip(np.dot(vec/v_norm, normal/n_norm), -1.0, 1.0)
            angle = np.arccos(dot)
            return min(angle, np.pi - angle)

        angle_ib = get_true_angle(shaft_vec, n_inboard)
        angle_ob = get_true_angle(-shaft_vec, n_outboard)

        return {
            "plunge_mm": plunge,
            "angle_ib_deg": np.degrees(angle_ib),
            "angle_ob_deg": np.degrees(angle_ob),
        }