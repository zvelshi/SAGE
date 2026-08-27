"""Composite shapes -- physical suspension components, each a bundle of
primitive shapes (and/or other composites). ``Composite`` is the base."""

from utils.spatial.shapes.composites.composite import Composite
from utils.spatial.shapes.composites.link import Link
from utils.spatial.shapes.composites.a_arm import AArm
from utils.spatial.shapes.composites.trailing_link import TrailingLink
from utils.spatial.shapes.composites.shock import Shock
from utils.spatial.shapes.composites.axle import Axle
from utils.spatial.shapes.composites.wheel import Wheel
from utils.spatial.shapes.composites.corner import (
    DoubleAArmCorner, SemiTrailingLinkCorner, corner_shape,
)

__all__ = [
    "Composite", "Link", "AArm", "TrailingLink", "Shock", "Axle", "Wheel",
    "DoubleAArmCorner", "SemiTrailingLinkCorner", "corner_shape",
]
