# default
from typing import Dict

# third-party
import numpy as np
from scipy.spatial.transform import Rotation as _Rot

def segment_segment_distance(p1: np.ndarray, p2: np.ndarray,
                              q1: np.ndarray, q2: np.ndarray) -> float:
    """Closest distance between segment p1-p2 and segment q1-q2 (3-D)."""
    p1, p2, q1, q2 = (np.asarray(v, float) for v in (p1, p2, q1, q2))
    d1 = p2 - p1
    d2 = q2 - q1
    r  = p1 - q1

    a = np.dot(d1, d1)
    e = np.dot(d2, d2)
    f = np.dot(d2, r)

    if a <= 1e-12 and e <= 1e-12:
        return float(np.linalg.norm(r))

    if a <= 1e-12:
        s = 0.0
        t = np.clip(f / e, 0.0, 1.0)
    else:
        c = np.dot(d1, r)
        if e <= 1e-12:
            t = 0.0
            s = np.clip(-c / a, 0.0, 1.0)
        else:
            b = np.dot(d1, d2)
            denom = a * e - b * b
            s = np.clip((b * f - c * e) / denom, 0.0, 1.0) if denom > 1e-12 else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t = 0.0
                s = np.clip(-c / a, 0.0, 1.0)
            elif t > 1.0:
                t = 1.0
                s = np.clip((b - c) / a, 0.0, 1.0)

    closest_p = p1 + s * d1
    closest_q = q1 + t * d2
    return float(np.linalg.norm(closest_p - closest_q))

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

def _bump_z_for_corner(z_wu: float, z_cog: float, phi: float, theta: float,
                        wc_stat: np.ndarray, cog_static: np.ndarray) -> float:
    """Vertical suspension travel input for the corner's kinematic solver: the
    wheel-centre world z minus where it would sit if rigidly attached to the
    body at its static offset, given the body's current heave/roll/pitch."""
    dx_m = (wc_stat[0] - cog_static[0]) * 0.001
    dy_m = (wc_stat[1] - cog_static[1]) * 0.001
    dz_static = wc_stat[2] - cog_static[2]
    z_body = z_cog + dz_static + (phi * dy_m - theta * dx_m) * 1000.0
    return float(z_wu - z_body)

def _body_rotation(phi: float, theta: float) -> _Rot:
    """Body rotation (roll about x = phi, pitch about y = theta) as a
    scipy Rotation, applied to static-frame points/directions."""
    return _Rot.from_euler('xy', [phi, theta])

def _apply_body_transform(step: Dict, cog_static: np.ndarray, cog_world: np.ndarray,
                           R_body: _Rot) -> Dict:
    """Transform a solved corner step (in the static body frame) into the
    world frame: positions are rotated about the CoG and translated,
    direction vectors (e.g. wheel_axis) are rotated only."""
    if step is None:
        return step
    cog_static = np.asarray(cog_static, float)
    cog_world = np.asarray(cog_world, float)
    # All world-space points a corner solver can return, across both corner
    # types (double A-arm and semi-trailing link) — every one of these must
    # move with the body, or it renders stuck at its static/ground position
    # while the rest of the corner follows the body into the air.
    pos_keys = (
        "wc", "ubj", "lbj", "uf", "ur", "lf", "lr", "s_ib", "s_ob",
        "piv_ib", "piv_ob", "tr_ib", "tr_ob",
        "ucl_ib", "ucl_ob", "lcl_ib", "lcl_ob", "tl_f", "tl_f_upright",
    )
    dir_keys = ("wheel_axis",)
    out = dict(step)
    for k in pos_keys:
        if out.get(k) is not None:
            p = np.asarray(out[k], float)
            out[k] = cog_world + R_body.apply(p - cog_static)
    for k in dir_keys:
        if out.get(k) is not None:
            v = np.asarray(out[k], float)
            out[k] = R_body.apply(v)
    return out

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