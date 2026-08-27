# third-party
import numpy as np
from scipy.spatial.transform import Rotation as _Rot

# ours
from utils.spatial.primitives import Point

# nicegui's ``ui.scene`` works in metres; the rest of SAGE works in millimetres.
SCENE_SCALE = 1.0 / 1000.0

# Shared defaults for suspension components.
AX_DEFAULT = Point(0.0, 1.0, 0.0)   # fallback wheel spin axis
POINT_COLOR = "#222222"             # joint / pickup markers

def xyz(p, scale: float) -> tuple[float, float, float]:
    """A Point (or anything Point-constructible) as an (x, y, z) tuple in scene units."""
    p = Point(p)
    return (p.x * scale, p.y * scale, p.z * scale)

def align_y_to_direction(direction) -> tuple[float, float, float]:
    """Euler angles (xyz) that rotate the +Y axis onto ``direction``. Used to
    orient any Y-aligned scene primitive (cylinder, wheel disc, plate) along an
    arbitrary 3-D direction -- a link axis, a wheel spin axis, a plane normal."""
    d = Point(direction).array
    n = np.linalg.norm(d)
    d = d / n if n > 1e-9 else np.array([0., 1., 0.])
    Y = np.array([0., 1., 0.])
    dot = float(np.clip(np.dot(Y, d), -1.0, 1.0))
    if dot >= 0.9999:
        return 0.0, 0.0, 0.0
    if dot <= -0.9999:
        return float(np.pi), 0.0, 0.0
    ax = np.cross(Y, d)
    ax /= np.linalg.norm(ax)
    rx, ry, rz = _Rot.from_rotvec(ax * np.arccos(dot)).as_euler("xyz")
    return float(rx), float(ry), float(rz)
