import numpy as np
from scipy.spatial.transform import Rotation as _Rot
from models.hardpoints import DoubleAArm, SemiTrailingLink

_S = 1.0 / 1000.0   # mm → scene units

def _v(arr):
    return [float(arr[0]) * _S, float(arr[1]) * _S, float(arr[2]) * _S]

def _place_cyl(cyl, p1s, p2s):
    """Move + rotate an existing cylinder to span p1→p2 (scene units).
    The cylinder must already have the correct height baked into its geometry."""
    d = p2s - p1s
    L = float(np.linalg.norm(d))
    if L < 1e-9:
        return
    mid    = (p1s + p2s) * 0.5
    d_norm = d / L
    Y      = np.array([0., 1., 0.])
    dot    = float(np.clip(np.dot(Y, d_norm), -1.0, 1.0))
    cyl.move(float(mid[0]), float(mid[1]), float(mid[2]))
    if dot < 0.9999:
        if dot > -0.9999:
            ax = np.cross(Y, d_norm);  ax /= np.linalg.norm(ax)
            rx, ry, rz = _Rot.from_rotvec(ax * np.arccos(dot)).as_euler('XYZ')
        else:
            rx, ry, rz = np.pi, 0.0, 0.0
        cyl.rotate(float(rx), float(ry), float(rz))
    else:
        cyl.rotate(0.0, 0.0, 0.0)

def _make_stick(scene, p1, p2, color, radius=0.004):
    """Create a cylinder with height = exact distance(p1, p2), correctly placed."""
    p1s = np.asarray(p1, float) * _S
    p2s = np.asarray(p2, float) * _S
    L   = max(float(np.linalg.norm(p2s - p1s)), 1e-9)
    cyl = scene.cylinder(top_radius=radius, bottom_radius=radius, height=L).material(color)
    _place_cyl(cyl, p1s, p2s)
    return cyl

def _move_stick(cyl, p1, p2):
    """Reposition a FIXED-LENGTH stick (height geometry unchanged, only move+rotate)."""
    _place_cyl(cyl, np.asarray(p1, float) * _S, np.asarray(p2, float) * _S)

def _swap_stick(scene, o, key, p1, p2, color, radius=0.004):
    """For VARIABLE-LENGTH sticks: park old cylinder off-screen, create fresh one."""
    o[key].move(1e6, 0.0, 0.0)   # hide old (no per-object removal in NiceGUI)
    with scene:
        o[key] = _make_stick(scene, p1, p2, color, radius)

def _shock_body_end(s_ib, s_ob, shock_min_mm):
    """Return the outboard end of the shock body (fixed length = shock_min_mm, in mm)."""
    s_ib = np.asarray(s_ib, float);  s_ob = np.asarray(s_ob, float)
    d = s_ob - s_ib
    L = float(np.linalg.norm(d))
    if L < 1e-9:
        return s_ib + np.array([0.0, float(shock_min_mm), 0.0])
    return s_ib + float(shock_min_mm) * (d / L)

def _make_wheel(scene, wc, wheel_axis, wr_mm, ww_mm, color="#888888"):
    """Create wheel cylinder with correct radius and width baked in, then place it."""
    radius = float(wr_mm) * _S
    width  = float(ww_mm) * _S
    cyl = scene.cylinder(top_radius=radius, bottom_radius=radius, height=width).material(color, opacity=0.35)
    _move_wheel(cyl, wc, wheel_axis)
    return cyl

def _move_wheel(cyl, wc, wheel_axis):
    """Reposition existing wheel cylinder (radius/width geometry unchanged)."""
    wc_s = np.asarray(wc, float) * _S
    ax   = np.asarray(wheel_axis, float)
    ax   = ax / (np.linalg.norm(ax) or 1.0)
    Y    = np.array([0., 1., 0.])
    dot  = float(np.clip(np.dot(Y, ax), -1.0, 1.0))
    cyl.move(float(wc_s[0]), float(wc_s[1]), float(wc_s[2]))
    if dot < 0.9999:
        if dot > -0.9999:
            rot_ax = np.cross(Y, ax);  rot_ax /= np.linalg.norm(rot_ax)
            rx, ry, rz = _Rot.from_rotvec(rot_ax * np.arccos(dot)).as_euler('XYZ')
        else:
            rx, ry, rz = np.pi, 0.0, 0.0
        cyl.rotate(float(rx), float(ry), float(rz))
    else:
        cyl.rotate(0.0, 0.0, 0.0)

def _build_corner_objects(scene, step, hp,
                           c_struct="#1e1e1e", c_tie="#009944",
                           c_shock="#6e6e82", c_axle="#cc2828"):
    """Create all 3-D objects for one corner. Returns dict of Object3D refs."""
    o = {"_scene": scene, "_c_shock": c_shock, "_c_axle": c_axle}
    ax_default = np.array([0., 1., 0.])

    if isinstance(hp, DoubleAArm):
        ubj = np.asarray(step["ubj"]);  lbj = np.asarray(step["lbj"])
        o["uf_arm"]  = _make_stick(scene, hp.uf,  ubj, c_struct)
        o["ur_arm"]  = _make_stick(scene, hp.ur,  ubj, c_struct)
        o["lf_arm"]  = _make_stick(scene, hp.lf,  lbj, c_struct)
        o["lr_arm"]  = _make_stick(scene, hp.lr,  lbj, c_struct)
        o["upright"] = _make_stick(scene, lbj,    ubj, c_struct)
        o["tierod"]  = _make_stick(scene, step["tr_ib"], step["tr_ob"], c_tie)
        _sbe = _shock_body_end(hp.s_ib, step["s_ob"], hp.shock_min)
        # plunger = thin rod spanning full shock axis (s_ib→s_ob); body overlaps from s_ib end
        o["shock_plunger"] = _make_stick(scene, hp.s_ib, step["s_ob"], c_shock, radius=0.003)
        o["shock_body"]    = _make_stick(scene, hp.s_ib, _sbe,         c_shock, radius=0.007)
        if "piv_ob" in step:
            o["axle_in"]  = _make_stick(scene, hp.piv_ib,     step["piv_ob"], c_axle)
            o["axle_out"] = _make_stick(scene, step["piv_ob"], step["wc"],     c_axle)
        for k, pt in [("sp_uf", hp.uf), ("sp_ur", hp.ur), ("sp_lf", hp.lf),
                      ("sp_lr", hp.lr), ("sp_s_ib", hp.s_ib)]:
            o[k] = scene.sphere(radius=0.010).material("#222222").move(*_v(pt))
        for k, pt in [("sp_ubj", ubj), ("sp_lbj", lbj),
                      ("sp_tr_ib", step["tr_ib"]), ("sp_tr_ob", step["tr_ob"]),
                      ("sp_s_ob", step["s_ob"])]:
            o[k] = scene.sphere(radius=0.010).material("#222222").move(*_v(pt))
        o["wheel"] = _make_wheel(scene, step["wc"], step.get("wheel_axis", ax_default), hp.wr, hp.ww)
        o["sp_wc"] = scene.sphere(radius=0.014).material("#4466bb").move(*_v(step["wc"]))

    elif isinstance(hp, SemiTrailingLink):
        o["tl_f_ucl"] = _make_stick(scene, hp.tl_f,   step["ucl_ob"], c_struct)
        o["tl_f_lcl"] = _make_stick(scene, hp.tl_f,   step["lcl_ob"], c_struct)
        o["ucl_link"] = _make_stick(scene, hp.ucl_ib,  step["ucl_ob"], c_struct)
        o["lcl_link"] = _make_stick(scene, hp.lcl_ib,  step["lcl_ob"], c_struct)
        _sbe = _shock_body_end(hp.s_ib, step["s_ob"], hp.shock_min)
        o["shock_body"]    = _make_stick(scene, hp.s_ib, _sbe,         c_shock, radius=0.007)
        o["shock_plunger"] = _make_stick(scene, _sbe,    step["s_ob"], c_shock, radius=0.003)
        if float(np.linalg.norm(np.asarray(step["s_ob"]) - np.asarray(_sbe))) < 1.0:
            o["shock_plunger"].move(1e6, 0.0, 0.0)
        if "piv_ob" in step:
            o["axle_in"]  = _make_stick(scene, hp.piv_ib,     step["piv_ob"], c_axle)
            o["axle_out"] = _make_stick(scene, step["piv_ob"], step["wc"],     c_axle)
        for k, pt in [("sp_tl_f", hp.tl_f), ("sp_ucl_ib", hp.ucl_ib),
                      ("sp_lcl_ib", hp.lcl_ib), ("sp_s_ib", hp.s_ib)]:
            o[k] = scene.sphere(radius=0.010).material("#222222").move(*_v(pt))
        for k, pt in [("sp_ucl_ob", step["ucl_ob"]), ("sp_lcl_ob", step["lcl_ob"]),
                      ("sp_s_ob", step["s_ob"])]:
            o[k] = scene.sphere(radius=0.010).material("#222222").move(*_v(pt))
        o["wheel"] = _make_wheel(scene, step["wc"], step.get("wheel_axis", ax_default), hp.wr, hp.ww)
        o["sp_wc"] = scene.sphere(radius=0.014).material("#4466bb").move(*_v(step["wc"]))

    return o

def _update_corner_objects(o, step, hp):
    """Update scene objects for a new step.
    Fixed-length links: move+rotate only.  Variable-length links: swap cylinder."""
    sc         = o["_scene"]
    c_shock    = o["_c_shock"]
    c_axle     = o["_c_axle"]
    ax_default = np.array([0., 1., 0.])

    if isinstance(hp, DoubleAArm):
        ubj = np.asarray(step["ubj"]);  lbj = np.asarray(step["lbj"])
        # fixed-length: a-arms, upright, tie rod, axle_out — only move+rotate
        _move_stick(o["uf_arm"],  hp.uf, ubj)
        _move_stick(o["ur_arm"],  hp.ur, ubj)
        _move_stick(o["lf_arm"],  hp.lf, lbj)
        _move_stick(o["lr_arm"],  hp.lr, lbj)
        _move_stick(o["upright"], lbj,   ubj)
        _move_stick(o["tierod"],  step["tr_ib"], step["tr_ob"])
        if "axle_out" in o and "piv_ob" in step:
            _move_stick(o["axle_out"], step["piv_ob"], step["wc"])
        # shock: swap both parts every frame so rotation starts from zero each time
        _sbe = _shock_body_end(hp.s_ib, step["s_ob"], hp.shock_min)
        _swap_stick(sc, o, "shock_body", hp.s_ib, _sbe, c_shock, radius=0.007)
        plunger_mm = float(np.linalg.norm(np.asarray(step["s_ob"]) - np.asarray(_sbe)))
        if plunger_mm > 1.0:
            _swap_stick(sc, o, "shock_plunger", _sbe, step["s_ob"], c_shock, radius=0.003)
        else:
            o["shock_plunger"].move(1e6, 0.0, 0.0)
        # axle_in: variable length → swap
        if "axle_in" in o and "piv_ob" in step:
            _swap_stick(sc, o, "axle_in", hp.piv_ib, step["piv_ob"], c_axle)
        # sphere updates
        o["sp_ubj"].move(*_v(ubj));  o["sp_lbj"].move(*_v(lbj))
        o["sp_tr_ib"].move(*_v(step["tr_ib"]))
        o["sp_tr_ob"].move(*_v(step["tr_ob"]))
        o["sp_s_ob"].move(*_v(step["s_ob"]))
        _move_wheel(o["wheel"], step["wc"], step.get("wheel_axis", ax_default))
        o["sp_wc"].move(*_v(step["wc"]))

    elif isinstance(hp, SemiTrailingLink):
        _move_stick(o["tl_f_ucl"], hp.tl_f,  step["ucl_ob"])
        _move_stick(o["tl_f_lcl"], hp.tl_f,  step["lcl_ob"])
        _move_stick(o["ucl_link"], hp.ucl_ib, step["ucl_ob"])
        _move_stick(o["lcl_link"], hp.lcl_ib, step["lcl_ob"])
        if "axle_out" in o and "piv_ob" in step:
            _move_stick(o["axle_out"], step["piv_ob"], step["wc"])
        _sbe = _shock_body_end(hp.s_ib, step["s_ob"], hp.shock_min)
        _swap_stick(sc, o, "shock_body", hp.s_ib, _sbe, c_shock, radius=0.007)
        plunger_mm = float(np.linalg.norm(np.asarray(step["s_ob"]) - np.asarray(_sbe)))
        if plunger_mm > 1.0:
            _swap_stick(sc, o, "shock_plunger", _sbe, step["s_ob"], c_shock, radius=0.003)
        else:
            o["shock_plunger"].move(1e6, 0.0, 0.0)
        if "axle_in" in o and "piv_ob" in step:
            _swap_stick(sc, o, "axle_in", hp.piv_ib, step["piv_ob"], c_axle)
        o["sp_ucl_ob"].move(*_v(step["ucl_ob"]))
        o["sp_lcl_ob"].move(*_v(step["lcl_ob"]))
        o["sp_s_ob"].move(*_v(step["s_ob"]))
        _move_wheel(o["wheel"], step["wc"], step.get("wheel_axis", ax_default))
        o["sp_wc"].move(*_v(step["wc"]))

def _build_scene(scene, step, sim_type, vehicle, hp):
    """Build all scene objects once. Returns scene_objs dict."""
    with scene:
        if sim_type == "ackermann":
            objs = {}
            if "left"  in step:
                objs["left"]  = _build_corner_objects(scene, step["left"],
                                                       vehicle.front_left.hardpoints,
                                                       c_struct="#003cb4")
            if "right" in step:
                objs["right"] = _build_corner_objects(scene, step["right"],
                                                       vehicle.front_right.hardpoints,
                                                       c_struct="#b42800")
        else:
            objs = _build_corner_objects(scene, step, hp)
    return objs

def _update_scene(scene_objs, step, sim_type, vehicle, hp):
    """Update all scene objects in-place for the new step. Zero allocation."""
    if sim_type == "ackermann":
        if "left"  in scene_objs and "left"  in step:
            _update_corner_objects(scene_objs["left"],  step["left"],
                                   vehicle.front_left.hardpoints)
        if "right" in scene_objs and "right" in step:
            _update_corner_objects(scene_objs["right"], step["right"],
                                   vehicle.front_right.hardpoints)
    else:
        _update_corner_objects(scene_objs, step, hp)

def _fit_camera(scene, step, sim_type, vehicle, hp):
    """Position camera to frame the geometry."""
    all_pts = []

    def collect(s, h):
        for k in ["ubj", "lbj", "wc", "tr_ob", "s_ob", "ucl_ob", "lcl_ob"]:
            if k in s:
                all_pts.append(np.asarray(s[k]))
        for attr in ["uf", "ur", "lf", "lr", "s_ib", "tl_f", "ucl_ib", "lcl_ib"]:
            if hasattr(h, attr):
                all_pts.append(np.asarray(getattr(h, attr)))

    if sim_type == "ackermann":
        if "left"  in step: collect(step["left"],  vehicle.front_left.hardpoints)
        if "right" in step: collect(step["right"], vehicle.front_right.hardpoints)
    else:
        collect(step, hp)

    if not all_pts:
        return
    arr    = np.array(all_pts) * _S
    ctr    = arr.mean(axis=0)
    span   = float(np.max(arr.max(axis=0) - arr.min(axis=0)))
    dist   = span * 1.8
    scene.move_camera(
        x=float(ctr[0]) + dist * 0.4,
        y=float(ctr[1]) - dist * 1.1,
        z=float(ctr[2]) + dist * 0.7,
        look_at_x=float(ctr[0]),
        look_at_y=float(ctr[1]),
        look_at_z=float(ctr[2]),
        up_x=0, up_y=0, up_z=1,
    )
