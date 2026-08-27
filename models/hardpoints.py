from __future__ import annotations

# default
from dataclasses import dataclass, field, fields
from typing import List, Dict

# third-party
import numpy as np

@dataclass
class Hardpoints:
    # shock properties
    shock_min: float = field(default=0.0, init=False)
    shock_max: float = field(default=0.0, init=False)

    # wheel properties
    wr: float = field(default=0.0, init=False)
    ww: float = field(default=0.0, init=False)
    wheel_stiffness: float = field(default=0.0, init=False)  # [N/mm]
    wheel_damping:   float = field(default=0.0, init=False)  # [N·s/mm]

    @property
    def names(self) -> List[str]:
        """Dynamically returns a list of all defined field names in the dataclass."""
        return [f.name for f in fields(self)]

    def _fill_vehicle_properties(self, config):
        wp = config.wheel_properties
        self.shock_min = config.shock_min
        self.shock_max = config.shock_max
        self.wr = wp.radius
        self.ww = wp.width
        self.wheel_stiffness = wp.stiffness
        self.wheel_damping = wp.damping

    @classmethod
    def from_config(cls, corner) -> Hardpoints:
        raise NotImplementedError

    @classmethod
    def link_lengths(cls, hp) -> Dict[str, float]:
        raise NotImplementedError

    @classmethod
    def points_to_yaml(cls, step: dict) -> dict:
        """Map a solved step dict (keyed by attribute name, e.g. 'wc', 's_ib') back
        to the point keys used in the hardpoints YAML, via `cls._YAML_MAP`."""
        return {
            yaml_key: [round(float(v), 3) for v in step[attr]]
            for attr, yaml_key in cls._YAML_MAP.items()
            if attr in step
        }
    
    @classmethod
    def mirror_points(cls, hp: Hardpoints) -> Hardpoints:
        """Return a new Hardpoints instance with left/right points mirrored about the xz plane."""
        mirrored_data = {}
        for attr, value in hp.__dict__.items():
            if isinstance(value, np.ndarray) and value.shape == (3,):
                mirrored_data[attr] = np.array([value[0], -value[1], value[2]])
            else:
                mirrored_data[attr] = value
        return cls(**mirrored_data)

@dataclass
class DoubleAArm(Hardpoints):

    # inboard a arm points
    uf: np.ndarray          # upper front
    ur: np.ndarray          # upper rear
    lf: np.ndarray          # lower front
    lr: np.ndarray          # lower rear

    # upright joints
    ubj: np.ndarray         # upper ball joint
    lbj: np.ndarray         # lower ball joint

    # steering points
    tr_ib: np.ndarray       # tie rod inboard
    tr_ob: np.ndarray       # tie rod outboard

    # shock points
    s_loc: str              # mounting location of outboard shock point
    s_ib: np.ndarray        # shock inboard
    s_ob: np.ndarray        # shock outboard

    # pivot points
    piv_ib:  np.ndarray      # in-board pivot center (cv)
    piv_ob:  np.ndarray      # out-board pivot center (cv)

    # wheel points
    wc: np.ndarray          # wheel center point

    _YAML_MAP = {
        "uf":     "upper_a_arm_front",
        "ur":     "upper_a_arm_rear",
        "lf":     "lower_a_arm_front",
        "lr":     "lower_a_arm_rear",
        "ubj":    "upper_ball_joint",
        "lbj":    "lower_ball_joint",
        "tr_ib":  "tie_rod_inboard",
        "tr_ob":  "tie_rod_outboard",
        "s_ib":   "shock_inboard",
        "s_ob":   "shock_outboard",
        "piv_ib": "pivot_inboard",
        "piv_ob": "pivot_outboard",
        "wc":     "wheel_center",
    }

    @classmethod
    def from_config(cls, corner) -> "DoubleAArm":
        return cls(
            **{attr: np.array(getattr(corner, yaml_key)) for attr, yaml_key in cls._YAML_MAP.items()},
            s_loc=corner.shock_location,
        )

    @classmethod
    def link_lengths(cls, hp: "DoubleAArm") -> Dict[str, float]:
        return {
            "upper_front": float(np.linalg.norm(hp.ubj - hp.uf)),
            "upper_rear": float(np.linalg.norm(hp.ubj - hp.ur)),
            "lower_front": float(np.linalg.norm(hp.lbj - hp.lf)),
            "lower_rear": float(np.linalg.norm(hp.lbj - hp.lr)),
            "tie_rod": float(np.linalg.norm(hp.tr_ib - hp.tr_ob)),
            "shock_static": float(np.linalg.norm(hp.s_ib - hp.s_ob)),
            "axle_ib_ob_static": float(np.linalg.norm(hp.piv_ib - hp.piv_ob)),
            "axle_ob_wc": float(np.linalg.norm(hp.piv_ob - hp.wc)),
        }

@dataclass
class SemiTrailingLink(Hardpoints):
    
    # trailing link points
    tl_f: np.ndarray         # front trailing link mount

    # camber link points
    ucl_ib: np.ndarray       # upper camber link inboard
    ucl_ob: np.ndarray       # upper camber link outboard
    lcl_ib: np.ndarray       # lower camber link inboard
    lcl_ob: np.ndarray       # lower camber link outboard

    # shock points
    s_ib: np.ndarray         # shock inboard
    s_ob: np.ndarray         # shock outboard

    # pivot points
    piv_ib:  np.ndarray      # in-board pivot center (cv)
    piv_ob:  np.ndarray      # out-board pivot center (cv)

    # wheel points
    wc: np.ndarray           # wheel center point

    _YAML_MAP = {
        "tl_f":   "trailing_link_front",
        "ucl_ib": "upper_camber_link_inboard",
        "ucl_ob": "upper_camber_link_outboard",
        "lcl_ib": "lower_camber_link_inboard",
        "lcl_ob": "lower_camber_link_outboard",
        "s_ib":   "shock_inboard",
        "s_ob":   "shock_outboard",
        "piv_ib": "pivot_inboard",
        "piv_ob": "pivot_outboard",
        "wc":     "wheel_center",
    }

    @classmethod
    def from_config(cls, corner) -> "SemiTrailingLink":
        return cls(
            **{attr: np.array(getattr(corner, yaml_key)) for attr, yaml_key in cls._YAML_MAP.items()}
        )

    @classmethod
    def link_lengths(cls, hp: "SemiTrailingLink") -> Dict[str, float]:
        return {
            "upper_trailing_link":  float(np.linalg.norm(hp.tl_f - hp.ucl_ob)),
            "lower_trailing_link":  float(np.linalg.norm(hp.tl_f - hp.lcl_ob)),
            "upper_camber_link":    float(np.linalg.norm(hp.ucl_ib - hp.ucl_ob)),
            "lower_camber_link":    float(np.linalg.norm(hp.lcl_ib - hp.lcl_ob)),
            "shock_static":         float(np.linalg.norm(hp.s_ib - hp.s_ob)),
            "axle_ib_ob_static":    float(np.linalg.norm(hp.piv_ib - hp.piv_ob)),
            "axle_ob_wc":           float(np.linalg.norm(hp.piv_ob - hp.wc)),
        }