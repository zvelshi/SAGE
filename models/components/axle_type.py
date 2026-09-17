from __future__ import annotations

# default
from abc import ABC, abstractmethod

# third-party
import numpy as np


class AxleType(ABC):
    """Strategy for where an axle's inboard pivot sits at a given outboard
    position -- the two front/rear driveline layouts this project supports.
    Owned by `Axle` (see `Axle.resolve_inboard`); selected per corner via the
    hardpoints YAML's `axle_type` key.
    """

    @abstractmethod
    def resolve_inboard(self, piv_ib_static: np.ndarray, piv_ob: np.ndarray,
                         static_len: float) -> np.ndarray | None:
        """The inboard pivot's position for this step, or None if the current
        outboard position can't be reached by this axle type."""

    @abstractmethod
    def plunge_mm(self, piv_ib_static: np.ndarray, piv_ib: np.ndarray,
                   piv_ob: np.ndarray, static_len: float) -> float:
        """How far this axle type's own plunge mechanism has moved from rest.
        Not the same computation for every type -- see each subclass."""


class CVPlunge(AxleType):
    """Legacy CV joint (`axle_type: cv_plunge`). The shaft stays at its rigid
    static length, so the inboard joint's centre slides along its fixed
    differential-output axis (X, Z pinned at the static pivot_inboard; only Y
    slides) to take up whatever length change the outboard point's travel
    demands.
    """

    def resolve_inboard(self, piv_ib_static, piv_ob, static_len):
        dx = piv_ob[0] - piv_ib_static[0]
        dz = piv_ob[2] - piv_ib_static[2]
        rem = static_len**2 - dx**2 - dz**2
        if rem < 0:
            return None
        dy = np.sqrt(rem)
        y = min((piv_ob[1] + dy, piv_ob[1] - dy), key=lambda yy: abs(yy - piv_ib_static[1]))
        return np.array([piv_ib_static[0], y, piv_ib_static[2]])

    def plunge_mm(self, piv_ib_static, piv_ib, piv_ob, static_len):
        # The shaft is rigid by construction (see resolve_inboard) -- its
        # length never changes. What plunges is the joint itself, sliding
        # along its fixed axis away from its static position.
        return float(piv_ib[1] - piv_ib_static[1])


class InternalPlunge(AxleType):
    """U-joint driveline (`axle_type: internal_plunge`). Both joint centres
    are fixed at their static hardpoints; whatever length change results is
    absorbed internally -- a mid-shaft slip-spline, or (as on the rear's CV
    cups) plunge within the joint itself -- reported as plunge_mm regardless
    of which.
    """

    def resolve_inboard(self, piv_ib_static, piv_ob, static_len):
        return piv_ib_static

    def plunge_mm(self, piv_ib_static, piv_ib, piv_ob, static_len):
        # Both pivots are fixed -- what plunges is the shaft's own length.
        return float(np.linalg.norm(piv_ob - piv_ib) - static_len)


AXLE_TYPES: dict[str, type[AxleType]] = {
    "cv_plunge": CVPlunge,
    "internal_plunge": InternalPlunge,
}


def axle_type_from_name(name: str) -> AxleType:
    try:
        return AXLE_TYPES[name]()
    except KeyError:
        raise ValueError(f"unknown axle_type '{name}', expected one of {sorted(AXLE_TYPES)}") from None
