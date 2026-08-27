# default
from typing import Dict

# third-party
import numpy as np
from scipy.spatial.transform import Rotation as _Rot

# ours
from utils.spatial import Point, Line

def get_wheel_attitude(step: Dict[str, np.ndarray]) -> Dict[str, float]:
    return {
        "camber": get_camber_angle(step),
        "toe": get_toe_angle(step),
        "caster": get_caster_angle(step),
    }

def get_camber_angle(step: Dict) -> float:
    n = step["wheel_axis"]
    return float(-np.rad2deg(np.arcsin(n[2])))

def get_toe_angle(step: Dict) -> float:
    n = step["wheel_axis"]
    return float(-np.rad2deg(np.arcsin(n[0])))

def get_caster_angle(step: Dict) -> float:
    if "ubj" not in step or "lbj" not in step:
        return 0.0
    v = Point(step["ubj"]) - Point(step["lbj"])
    return float(np.rad2deg(np.arctan2(v.x, v.z)))

def get_contact_patch(step: Dict | None, wr: float) -> np.ndarray:
    """Tire contact center: the wheel-radius vector from wc, tilted by camber,
    that points most toward the ground. NaN vector for a missing (failed-solve) step."""
    if not step:
        return np.full(3, np.nan)
    wc = Point(step["wc"])
    axis = Point(step["wheel_axis"])
    down = Point(0.0, 0.0, -1.0)
    # component of straight-down perpendicular to the spin axis
    m = down - axis * down.dot(axis)
    if m.norm < 1e-9:
        return (wc + Point(0.0, 0.0, -wr)).array
    return (wc + m.unit() * wr).array

def _steering_axis(step: Dict) -> Line | None:
    """The steering axis as a Line through the lower and upper ball joints."""
    if "lbj" not in step or "ubj" not in step:
        return None
    try:
        return Line.from_points(step["lbj"], step["ubj"])
    except ValueError:
        return None

def _steering_axis_point_at_z(step: Dict, z: float) -> np.ndarray | None:
    """Point where the steering axis crosses height ``z``."""
    axis = _steering_axis(step)
    if axis is None:
        return None
    p = axis.at_z(z)
    return None if p is None else p.array

def get_kingpin_angle(step: Dict) -> float:
    """Front-elevation angle of the steering axis from vertical.
    Positive when the axis leans inward at the top."""
    up = Point(step["ubj"]) - Point(step["lbj"])
    return float(-np.rad2deg(np.arctan2(up.y, up.z)))

def get_caster_trail(step: Dict) -> float:
    """Side-elevation X distance, at wheel-center height, between the steering
    axis and the wheel center. Positive when the axis is forward of the wheel."""
    wc = Point(step["wc"])
    p = _steering_axis_point_at_z(step, wc.z)
    return 0.0 if p is None else float(p[0] - wc.x)

def get_caster_offset(step: Dict, wr: float) -> float:
    """Side-elevation X distance, at ground height, between the steering axis
    and the tire contact center. Positive when the axis is forward of contact."""
    contact = Point(get_contact_patch(step, wr))
    p = _steering_axis_point_at_z(step, contact.z)
    return 0.0 if p is None else float(contact.x - p[0])

def get_kingpin_offset_wheel(step: Dict) -> float:
    """Front-elevation Y distance, at wheel-center height, from the wheel
    center to the steering axis. Positive when the wheel is outboard."""
    wc = Point(step["wc"])
    p = _steering_axis_point_at_z(step, wc.z)
    return 0.0 if p is None else float(wc.y - p[1])

def get_kingpin_offset_ground(step: Dict, wr: float) -> float:
    """Front-elevation Y distance, at ground height, between the steering axis
    and the tire contact center. Positive when contact is outboard of the axis."""
    contact = Point(get_contact_patch(step, wr))
    p = _steering_axis_point_at_z(step, contact.z)
    return 0.0 if p is None else float(contact.y - p[1])

def get_mechanical_trail(step: Dict, wr: float) -> float:
    """Side-elevation (X-Z) perpendicular distance between the steering axis and
    the tire contact center."""
    lbj, ubj = Point(step["lbj"]), Point(step["ubj"])
    contact = Point(get_contact_patch(step, wr))
    try:
        axis_xz = Line.from_points(Point(lbj.x, 0.0, lbj.z), Point(ubj.x, 0.0, ubj.z))
    except ValueError:
        return 0.0
    return axis_xz.distance_to_point(Point(contact.x, 0.0, contact.z))

def _contact_patch_perp_line(step: Dict, step_plus: Dict, step_minus: Dict, wr: float) -> Line | None:
    """Line through the contact patch, lying in the X=0 plane, perpendicular to
    the contact patch's own Y-Z path tangent under a small +/- bump perturbation.
    The Lotus Shark roll-center construction. None if a step failed or the path
    has no in-plane motion."""
    if not step or not step_plus or not step_minus:
        return None
    cp = Point(get_contact_patch(step, wr))
    cp_p = Point(get_contact_patch(step_plus, wr))
    cp_m = Point(get_contact_patch(step_minus, wr))
    tangent = Point(0.0, cp_p.y - cp_m.y, cp_p.z - cp_m.z)
    if tangent.norm < 1e-9:
        return None
    perp = Point(0.0, -tangent.z, tangent.y)   # 90 deg rotation in the Y-Z plane
    try:
        return Line(Point(0.0, cp.y, cp.z), perp)
    except ValueError:
        return None

def roll_center_yz(step_l: Dict, step_l_plus: Dict, step_l_minus: Dict,
                    step_r: Dict, step_r_plus: Dict, step_r_minus: Dict, wr: float):
    """Roll center: intersection of each side's contact-patch perpendicular-to-path
    line (see _contact_patch_perp_line). Returns [y, z] or None if unavailable."""
    line_l = _contact_patch_perp_line(step_l, step_l_plus, step_l_minus, wr)
    line_r = _contact_patch_perp_line(step_r, step_r_plus, step_r_minus, wr)
    if line_l is None or line_r is None:
        return None
    p = line_l.intersection(line_r)
    return None if p is None else np.array([p.y, p.z])

def contact_patch_z_series(steps: list, wr: float) -> np.ndarray:
    """Tire contact-center height across a sequence of steps. NaN for any failed-solve step."""
    if not wr:
        return np.array([s["wc"][2] if s else np.nan for s in steps], dtype=float)
    return np.array([get_contact_patch(s, wr)[2] for s in steps], dtype=float)

def motion_ratio_series(steps: list, wr: float = 0.0) -> np.ndarray:
    """Instantaneous Motion Ratio per the Lotus definition: the ratio of change
    in vertical height of the tire contact center to change in spring/shock
    length (unsigned; >1 when the wheel moves more than the spring).
    MR = |d(contact_patch_z)/d(shock_length)|. Computed as a true finite-
    difference derivative w.r.t. shock length (not index), which is robust to
    uneven step spacing (e.g. steer sweeps)."""
    contact_z = contact_patch_z_series(steps, wr)
    shock_len = np.array([s["shock_length"] if s else np.nan for s in steps], dtype=float)
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

def _bump_z_for_corner(z_wu: float, z_cog: float, phi: float, theta: float,
                        wc_stat: np.ndarray, cog_static: np.ndarray) -> float:
    """Vertical suspension travel input for the corner's kinematic solver: the
    wheel-center world z minus where it would sit if rigidly attached to the
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

    # Negated to match Lotus Shark's sign convention (verified against its front-steer
    # report: e.g. actual/ideal toe-split ratio +1.27% there reads as Ackermann% = -1.27).
    return -((theta_3 - theta_4) / (theta_1 - theta_2)) * 100.0
