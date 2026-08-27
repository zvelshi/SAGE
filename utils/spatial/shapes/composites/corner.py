# default
from __future__ import annotations

# ours
from utils.spatial.primitives import Point
from utils.spatial.shapes._common import AX_DEFAULT
from utils.spatial.shapes.composites.composite import Composite
from utils.spatial.shapes.composites.a_arm import AArm
from utils.spatial.shapes.composites.axle import Axle
from utils.spatial.shapes.dashed_line import DashedLine
from utils.spatial.shapes.composites.link import Link
from utils.spatial.shapes.composites.shock import Shock
from utils.spatial.shapes.composites.trailing_link import TrailingLink
from utils.spatial.shapes.composites.wheel import Wheel


def _pt(step: dict, key: str, hp, default=None) -> Point:
    """A hardpoint from the solved step, falling back to the static hardpoints
    object, then to ``default``."""
    src = step.get(key)
    if src is None and hp is not None and hasattr(hp, key):
        src = getattr(hp, key)
    if src is None:
        src = default
    return Point(src)


class DoubleAArmCorner(Composite):
    """Front double-A-arm corner: upper + lower A-arms, upright, tie rod,
    coil-over, driveshaft, wheel."""

    def __init__(self, step: dict, hp, struct_color="#1e1e1e", tie_color="#009944",
                 shock_color="#6e6e82", axle_color="#cc2828", show_guides: bool = True):
        self.step, self.hp = step, hp
        self.struct_color = struct_color
        self.tie_color = tie_color
        self.shock_color = shock_color
        self.axle_color = axle_color
        self.show_guides = show_guides

    def parts(self):
        s, hp = self.step, self.hp
        ubj, lbj = _pt(s, "ubj", hp), _pt(s, "lbj", hp)
        p = {
            "upper_arm": AArm(_pt(s, "uf", hp), _pt(s, "ur", hp), ubj, self.struct_color),
            "lower_arm": AArm(_pt(s, "lf", hp), _pt(s, "lr", hp), lbj, self.struct_color),
            "upright": Link(lbj, ubj, link_color=self.struct_color),
            "tie_rod": Link(_pt(s, "tr_ib", hp), _pt(s, "tr_ob", hp), link_color=self.tie_color),
            "shock": Shock(_pt(s, "s_ib", hp), _pt(s, "s_ob", hp), hp.shock_min, self.shock_color),
            "wheel": Wheel(_pt(s, "wc", hp), _pt(s, "wheel_axis", hp, AX_DEFAULT), hp.wr, hp.ww),
        }
        if "piv_ob" in s:
            plunge = (s.get("axle_data") or {}).get("plunge_mm", 0.0)
            p["axle"] = Axle(_pt(s, "piv_ib", hp), _pt(s, "piv_ob", hp), _pt(s, "wc", hp),
                             plunge, self.axle_color)
        return p

    def static_parts(self):
        if not self.show_guides:
            return {}
        s, hp = self.step, self.hp
        g = {"tie_guide": DashedLine(_pt(s, "tr_ib", hp), AX_DEFAULT, 250.0, color=self.tie_color)}
        if "piv_ob" in s:
            g["axle_guide"] = DashedLine(_pt(s, "piv_ib", hp), AX_DEFAULT, 250.0, color=self.axle_color)
        return g


class SemiTrailingLinkCorner(Composite):
    """Rear semi-trailing-link corner: trailing link, upper + lower camber links,
    coil-over, driveshaft, wheel."""

    def __init__(self, step: dict, hp, struct_color="#1e1e1e", tie_color="#009944",
                 shock_color="#6e6e82", axle_color="#cc2828", show_guides: bool = True):
        self.step, self.hp = step, hp
        self.struct_color = struct_color
        self.shock_color = shock_color
        self.axle_color = axle_color
        self.show_guides = show_guides

    def parts(self):
        s, hp = self.step, self.hp
        p = {
            "trailing_link": TrailingLink(_pt(s, "tl_f", hp), _pt(s, "ucl_ob", hp),
                                          _pt(s, "lcl_ob", hp), self.struct_color),
            "upper_camber_link": Link(_pt(s, "ucl_ib", hp), _pt(s, "ucl_ob", hp),
                                      link_color=self.struct_color),
            "lower_camber_link": Link(_pt(s, "lcl_ib", hp), _pt(s, "lcl_ob", hp),
                                      link_color=self.struct_color),
            "shock": Shock(_pt(s, "s_ib", hp), _pt(s, "s_ob", hp), hp.shock_min, self.shock_color),
            "wheel": Wheel(_pt(s, "wc", hp), _pt(s, "wheel_axis", hp, AX_DEFAULT), hp.wr, hp.ww),
        }
        if "piv_ob" in s:
            plunge = (s.get("axle_data") or {}).get("plunge_mm", 0.0)
            p["axle"] = Axle(_pt(s, "piv_ib", hp), _pt(s, "piv_ob", hp), _pt(s, "wc", hp),
                             plunge, self.axle_color)
        return p

    def static_parts(self):
        s, hp = self.step, self.hp
        if not self.show_guides or "piv_ob" not in s:
            return {}
        return {"axle_guide": DashedLine(_pt(s, "piv_ib", hp), AX_DEFAULT, 250.0,
                                         color=self.axle_color)}


def corner_shape(step: dict, hp, **style) -> Composite:
    """The right corner composite for a hardpoints type."""
    from models.hardpoints import DoubleAArm
    cls = DoubleAArmCorner if isinstance(hp, DoubleAArm) else SemiTrailingLinkCorner
    return cls(step, hp, **style)
