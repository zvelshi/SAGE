# default
from typing import Dict

# third-party
import numpy as np

def _circle_from_spheres(c1: np.ndarray, r1: float, c2: np.ndarray, r2: float):
    """Intersection circle of two spheres. Returns (center, radius, u, v) or None."""
    d = c2 - c1
    d2 = float(np.dot(d, d))
    if d2 < 1e-18:
        return None
    t = (r1 * r1 - r2 * r2 + d2) / (2.0 * d2)
    center = c1 + t * d
    r2c = r1 * r1 - t * t * d2
    if r2c < 0.0:
        return None
    rc = np.sqrt(r2c)
    n = d / np.sqrt(d2)
    u = np.cross(n, np.array([1., 0., 0.])) if abs(n[0]) < 0.9 else np.cross(n, np.array([0., 1., 0.]))
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    return center, rc, u, v

def _rodrigues(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation matrix R s.t. R @ a_hat = b_hat (both unit vectors)."""
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = float(np.linalg.norm(v))
    if s < 1e-9:
        if c > 0.0:
            return np.eye(3)
        ax = np.array([1., 0., 0.]) if abs(a[0]) < 0.9 else np.array([0., 1., 0.])
        ax -= a * float(np.dot(a, ax))
        ax /= np.linalg.norm(ax)
        return 2.0 * np.outer(ax, ax) - np.eye(3)
    K = np.array([[0., -v[2], v[1]], [v[2], 0., -v[0]], [-v[1], v[0], 0.]])
    return np.eye(3) + K + K @ K * (1.0 - c) / (s * s)

def _rot_axis(ax: np.ndarray, ang: float) -> np.ndarray:
    """Rotation matrix: rotate `ang` radians about unit vector `ax`."""
    c, s = np.cos(ang), np.sin(ang)
    K = np.array([[0., -ax[2], ax[1]], [ax[2], 0., -ax[0]], [-ax[1], ax[0], 0.]])
    return c * np.eye(3) + s * K + (1.0 - c) * np.outer(ax, ax)

def _lin_trig(A: float, B: float, C: float) -> list[float]:
    """Solve A cos θ + B sin θ = C. Returns up to 2 solutions."""
    R = np.sqrt(A * A + B * B)
    if R < 1e-12:
        return []
    ratio = np.clip(C / R, -1.0, 1.0)
    if abs(C / R) > 1.0 + 1e-9:
        return []
    phi, d = np.arctan2(B, A), np.arccos(ratio)
    return [phi + d, phi - d]

def _nearest_branch(candidates: list[float], ref: float) -> float:
    """Pick candidate angle nearest to ref (on circle)."""
    return min(candidates, key=lambda x: abs((x - ref + np.pi) % (2 * np.pi) - np.pi))

# Keys in solver step dicts that are absolute positions (translate + rotate)
_VIZ_POS_KEYS = frozenset({
    "lbj", "ubj", "uf", "ur", "lf", "lr",
    "wc", "s_ib", "s_ob", "piv_ib", "piv_ob", "tr_ib", "tr_ob",
    "ucl_ib", "ucl_ob", "lcl_ib", "lcl_ob",
    "tl_f", "tl_f_upright",
})
# Keys that are unit direction vectors (rotate only, no translation)
_VIZ_DIR_KEYS = frozenset({"wheel_axis"})

def _bump_z_for_corner(z_wu_i: float, z_cog: float, phi: float, theta: float, wc_static: np.ndarray, cog_static: np.ndarray) -> float:
    """Relative vertical displacement of wheel hub from body attachment point [mm]."""
    dx = wc_static[0] - cog_static[0]
    dy = wc_static[1] - cog_static[1]
    dz_static = wc_static[2] - cog_static[2]
    z_body_at_corner = z_cog + phi * dy - theta * dx
    z_wc_expected    = z_body_at_corner + dz_static
    return z_wu_i - z_wc_expected

def _body_rotation(phi: float, theta: float) -> np.ndarray:
    """Exact body rotation matrix R_y(theta) @ R_x(phi)."""
    cp, sp = np.cos(phi), np.sin(phi)
    ct, st = np.cos(theta), np.sin(theta)
    return np.array([
        [ct,   st * sp,  st * cp],
        [0.0,  cp,      -sp     ],
        [-st,  ct * sp,  ct * cp],
    ])

def _apply_body_transform(step: dict | None, cog_static: np.ndarray, cog_world: np.ndarray, R_body: np.ndarray) -> dict | None:
    """Rotate and translate every 3-D position in `step` from body frame to world frame.
    Direction vectors (wheel_axis) are rotated only.  Scalars/dicts are unchanged."""
    if step is None:
        return None
    out = {}
    for k, v in step.items():
        if isinstance(v, np.ndarray) and v.shape == (3,):
            if k in _VIZ_POS_KEYS:
                out[k] = cog_world + R_body @ (v - cog_static)
            elif k in _VIZ_DIR_KEYS:
                out[k] = R_body @ v
            else:
                out[k] = v
        else:
            out[k] = v
    return out

def get_wheel_attitude(step: Dict[str, np.ndarray]) -> Dict[str, float]:
    return {
        "camber": get_camber_angle(step), 
        "toe": get_toe_angle(step), 
        "caster": get_caster_angle(step),
    }

def get_camber_angle(step: Dict) -> float:
    n = step["wheel_axis"]
    camber_rad = np.arcsin(n[2])
    return -np.rad2deg(camber_rad)

def get_toe_angle(step: Dict) -> float:
    n = step["wheel_axis"]
    toe_rad = np.arcsin(n[0])
    return -np.rad2deg(toe_rad)

def get_caster_angle(step: Dict) -> float:
    if "ubj" not in step or "lbj" not in step:
        return 0.0
    v = step["ubj"] - step["lbj"]
    return np.rad2deg(np.arctan2(v[0], v[2]))
 
def calculate_ackermann_percentage(
    inner_toe: float, 
    outer_toe: float, 
    track_width: float, 
    wheelbase: float
) -> float:

    theta_3 = abs(inner_toe)  # Actual Inner
    theta_4 = abs(outer_toe)  # Actual Outer

    # Calculate centerline angle - this is the "requested steer angle"
    avg_angle = (theta_3 + theta_4) / 2.0
    
    # Calculate ideal angles based on centerline
    # cot(angle) = 1/tan(angle)
    cot_center = 1.0 / np.tan(np.deg2rad(avg_angle))
    
    # Half-width ratio for geometry calc
    hw_ratio = (track_width / 2.0) / wheelbase
    
    # Handle ideal inner angle
    if (cot_center - hw_ratio) == 0:
        theta_1 = 90.0
    else:
        theta_1 = np.rad2deg(np.arctan(1.0 / (cot_center - hw_ratio)))

    theta_2 = np.rad2deg(np.arctan(1.0 / (cot_center + hw_ratio)))

    return ((theta_3 - theta_4) / (theta_1 - theta_2)) * 100.0