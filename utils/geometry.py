# default
from typing import Dict

# third-party
import numpy as np
from scipy.spatial.transform import Rotation as _Rot

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

def get_contact_patch(step: Dict, wr: float) -> np.ndarray:
    """Tyre contact centre: the wheel-radius vector from wc, tilted by camber,
    that points most toward the ground."""
    wc = np.asarray(step["wc"], float)
    axis = np.asarray(step["wheel_axis"], float)
    down = np.array([0., 0., -1.])
    m = down - np.dot(down, axis) * axis
    n = np.linalg.norm(m)
    if n < 1e-9:
        return wc + np.array([0., 0., -wr])
    return wc + wr * (m / n)

def _steering_axis_point_at_z(step: Dict, z: float) -> np.ndarray | None:
    """Point where the steering axis (line through lbj/ubj) crosses height z."""
    lbj = np.asarray(step["lbj"], float)
    ubj = np.asarray(step["ubj"], float)
    axis_up = ubj - lbj
    if abs(axis_up[2]) < 1e-9:
        return None
    t = (z - lbj[2]) / axis_up[2]
    return lbj + t * axis_up

def get_kingpin_angle(step: Dict) -> float:
    """Front-elevation angle of the steering axis from vertical.
    Positive when the axis leans inward at the top."""
    lbj = np.asarray(step["lbj"], float)
    ubj = np.asarray(step["ubj"], float)
    axis_up = ubj - lbj
    return -np.rad2deg(np.arctan2(axis_up[1], axis_up[2]))

def get_caster_trail(step: Dict) -> float:
    """Side-elevation X distance, at wheel-centre height, between the steering
    axis and the wheel centre. Positive when the axis is forward of the wheel."""
    wc = np.asarray(step["wc"], float)
    p = _steering_axis_point_at_z(step, wc[2])
    if p is None:
        return 0.0
    return float(p[0] - wc[0])

def get_caster_offset(step: Dict, wr: float) -> float:
    """Side-elevation X distance, at ground height, between the steering axis
    and the tyre contact centre. Positive when the axis is forward of contact."""
    contact = get_contact_patch(step, wr)
    p = _steering_axis_point_at_z(step, contact[2])
    if p is None:
        return 0.0
    return float(contact[0] - p[0])

def get_kingpin_offset_wheel(step: Dict) -> float:
    """Front-elevation Y distance, at wheel-centre height, from the wheel
    centre to the steering axis. Positive when the wheel is outboard."""
    wc = np.asarray(step["wc"], float)
    p = _steering_axis_point_at_z(step, wc[2])
    if p is None:
        return 0.0
    return float(wc[1] - p[1])

def get_kingpin_offset_ground(step: Dict, wr: float) -> float:
    """Front-elevation Y distance, at ground height, between the steering axis
    and the tyre contact centre. Positive when contact is outboard of the axis."""
    contact = get_contact_patch(step, wr)
    p = _steering_axis_point_at_z(step, contact[2])
    if p is None:
        return 0.0
    return float(contact[1] - p[1])

def get_mechanical_trail(step: Dict, wr: float) -> float:
    """Side-elevation perpendicular distance between the steering axis and the
    tyre contact centre."""
    lbj = np.asarray(step["lbj"], float)
    ubj = np.asarray(step["ubj"], float)
    contact = get_contact_patch(step, wr)
    axis_side = np.array([ubj[0] - lbj[0], ubj[2] - lbj[2]])
    n = np.linalg.norm(axis_side)
    if n < 1e-9:
        return 0.0
    axis_side /= n
    v = np.array([contact[0] - lbj[0], contact[2] - lbj[2]])
    perp = v - np.dot(v, axis_side) * axis_side
    return float(np.linalg.norm(perp))

def contact_patch_z_series(steps: list, wr: float) -> np.ndarray:
    """Tyre contact-centre height across a sequence of steps."""
    if not wr:
        return np.array([s["wc"][2] for s in steps], dtype=float)
    return np.array([get_contact_patch(s, wr)[2] for s in steps], dtype=float)

def motion_ratio_series(steps: list, wr: float = 0.0) -> np.ndarray:
    """Instantaneous Motion Ratio per the Lotus definition: the ratio of change
    in vertical height of the tyre contact centre to change in spring/shock
    length (unsigned; >1 when the wheel moves more than the spring).
    MR = |d(contact_patch_z)/d(shock_length)|. Computed as a true finite-
    difference derivative w.r.t. shock length (not index), which is robust to
    uneven step spacing (e.g. steer sweeps)."""
    contact_z = contact_patch_z_series(steps, wr)
    shock_len = np.array([s["shock_length"] for s in steps], dtype=float)
    if len(steps) < 2:
        return np.zeros_like(shock_len)
    with np.errstate(divide="ignore", invalid="ignore"):
        motion_ratio = np.gradient(contact_z, shock_len)
    motion_ratio = np.abs(motion_ratio)
    motion_ratio[~np.isfinite(motion_ratio)] = 0.0
    return motion_ratio

def static_ride_height_index(steps: list) -> int:
    """Index of the step closest to 0mm shock travel / 0mm steer (ride height, centered)."""
    def dist(s):
        t  = s.get("travel_mm", 0.0) or 0.0
        st = s.get("steer_mm", 0.0) or 0.0
        return t * t + st * st
    return min(range(len(steps)), key=lambda i: dist(steps[i]))

def get_steering_axis_geometry(step: Dict, wr: float) -> Dict[str, float]:
    return {
        "kingpin_angle":      get_kingpin_angle(step),
        "caster_trail":       get_caster_trail(step),
        "caster_offset":      get_caster_offset(step, wr),
        "kingpin_offset_wc":  get_kingpin_offset_wheel(step),
        "kingpin_offset_gnd": get_kingpin_offset_ground(step, wr),
        "mechanical_trail":   get_mechanical_trail(step, wr),
    }

def align_y_to_direction(direction: np.ndarray) -> tuple[float, float, float]:
    """Euler angles (xyz) that rotate the +Y axis onto unit vector `direction`.
    Used to orient any Y-aligned primitive (cylinder, wheel disc) along an
    arbitrary 3-D direction (a link axis, a wheel spin axis, ...)."""
    d = np.asarray(direction, float)
    n = np.linalg.norm(d)
    d = d / n if n > 1e-9 else np.array([0., 1., 0.])
    Y = np.array([0., 1., 0.])
    dot = float(np.clip(np.dot(Y, d), -1.0, 1.0))
    if dot >= 0.9999:
        return 0.0, 0.0, 0.0
    if dot <= -0.9999:
        return np.pi, 0.0, 0.0
    ax = np.cross(Y, d)
    ax /= np.linalg.norm(ax)
    rx, ry, rz = _Rot.from_rotvec(ax * np.arccos(dot)).as_euler('xyz')
    return float(rx), float(ry), float(rz)

def shock_body_end(s_ib: np.ndarray, s_ob: np.ndarray, shock_min_mm: float) -> np.ndarray:
    """Outboard end of the (fixed-length) shock body, given the current shock
    inboard/outboard mount points and the body's static length."""
    s_ib = np.asarray(s_ib, float)
    s_ob = np.asarray(s_ob, float)
    d = s_ob - s_ib
    L = float(np.linalg.norm(d))
    if L < 1e-9:
        return s_ib + np.array([0.0, float(shock_min_mm), 0.0])
    return s_ib + float(shock_min_mm) * (d / L)

def axle_plunge_point(piv_ib: np.ndarray, piv_ob: np.ndarray, plunge_mm: float) -> np.ndarray:
    """Offset the inboard axle (CV plunge joint) point along the shaft axis by
    the current plunge amount, so it visually slides relative to the fixed
    chassis mount."""
    piv_ib = np.asarray(piv_ib, float)
    piv_ob = np.asarray(piv_ob, float)
    d = piv_ob - piv_ib
    L = float(np.linalg.norm(d))
    if L < 1e-9:
        return piv_ib
    return piv_ib + (d / L) * float(plunge_mm)

def dash_segments(center: np.ndarray, axis: np.ndarray, length_mm: float,
                   n_dashes: int = 14) -> list[tuple[np.ndarray, np.ndarray]]:
    """(p1, p2) endpoint pairs for a dashed reference line of total `length_mm`,
    centered on `center` and running along `axis`."""
    center = np.asarray(center, float)
    axis   = np.asarray(axis, float)
    n = np.linalg.norm(axis)
    axis = axis / n if n > 1e-9 else np.array([0., 1., 0.])
    half = float(length_mm) / 2.0
    step_len = float(length_mm) / (2 * n_dashes - 1)
    segments = []
    for i in range(n_dashes):
        t0 = -half + i * 2 * step_len
        t1 = t0 + step_len
        segments.append((center + axis * t0, center + axis * t1))
    return segments

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