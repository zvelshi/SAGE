# third-party
import numpy as np

# ours
from models.hardpoints import DoubleAArm, SemiTrailingLink
from utils.geometry import (
    align_y_to_direction, shock_body_end as _shock_body_end,
    axle_plunge_point as _axle_plunge_point, dash_segments,
)
# constants
_S = 1.0 / 1000.0 # mm -> scene units

_DYN_CORNERS = [
    ("fl", "front_left",  "#003cb4", "#b42800"),
    ("fr", "front_right", "#b42800", "#003cb4"),
    ("rl", "rear_left",   "#1a6e1a", "#6e1a1a"),
    ("rr", "rear_right",  "#6e1a1a", "#1a6e1a"),
]

def _v(arr: np.ndarray | list[float]) -> list[float]:
    """Convert mm to scene units."""
    return [float(arr[0]) * _S, float(arr[1]) * _S, float(arr[2]) * _S]

def _place_cyl(cyl, p1s: np.ndarray, p2s: np.ndarray):
    """Move + rotate an existing cylinder to span p1->p2 (scene units).
    The cylinder must already have the correct height baked into its geometry."""
    d = p2s - p1s
    L = float(np.linalg.norm(d))
    if L < 1e-9:
        return
    mid = (p1s + p2s) * 0.5
    cyl.move(float(mid[0]), float(mid[1]), float(mid[2]))
    cyl.rotate(*align_y_to_direction(d))

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

def _make_variable_stick(scene, p1, p2, color, radius=0.004):
    """Create a cylinder with height=1.0 and scale its length to fit p1->p2."""
    p1s = np.asarray(p1, float) * _S
    p2s = np.asarray(p2, float) * _S
    L   = max(float(np.linalg.norm(p2s - p1s)), 1e-9)
    cyl = scene.cylinder(top_radius=radius, bottom_radius=radius, height=1.0).material(color)
    cyl.scale(1.0, L, 1.0)
    _place_cyl(cyl, p1s, p2s)
    return cyl

def _move_variable_stick(cyl, p1, p2):
    """Dynamically scale and rotate an existing variable-length stick."""
    p1s = np.asarray(p1, float) * _S
    p2s = np.asarray(p2, float) * _S
    L   = max(float(np.linalg.norm(p2s - p1s)), 1e-9)
    cyl.scale(1.0, L, 1.0)
    _place_cyl(cyl, p1s, p2s)

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
    cyl.move(float(wc_s[0]), float(wc_s[1]), float(wc_s[2]))
    cyl.rotate(*align_y_to_direction(np.asarray(wheel_axis, float)))

def _make_dashed_line(scene, center, axis, length_mm, color, n_dashes=14, radius=0.0018, opacity=0.55):
    """Static reference line made of short dashes, centered on `center` along `axis`."""
    dashes = []
    for p1, p2 in dash_segments(center, axis, length_mm, n_dashes):
        cyl = _make_stick(scene, p1, p2, color, radius=radius)
        cyl.material(color, opacity=opacity)
        dashes.append(cyl)
    return dashes

def _build_corner_objects(scene, step, hp,
                           c_struct="#1e1e1e", c_tie="#009944",
                           c_shock="#6e6e82", c_axle="#cc2828",
                           show_guides=True):
    """Create all 3-D objects for one corner. Returns dict of Object3D refs."""
    o = {"_scene": scene, "_c_shock": c_shock, "_c_axle": c_axle}
    ax_default = np.array([0., 1., 0.])

    if isinstance(hp, DoubleAArm):
        ubj    = np.asarray(step["ubj"]);  lbj = np.asarray(step["lbj"])
        uf     = np.asarray(step.get("uf",     hp.uf))
        ur     = np.asarray(step.get("ur",     hp.ur))
        lf     = np.asarray(step.get("lf",     hp.lf))
        lr     = np.asarray(step.get("lr",     hp.lr))
        s_ib   = np.asarray(step.get("s_ib",   hp.s_ib))
        piv_ib = np.asarray(step.get("piv_ib", hp.piv_ib))
        tr_ib  = np.asarray(step.get("tr_ib",  hp.tr_ib))
        o["uf_arm"]  = _make_stick(scene, uf,  ubj, c_struct)
        o["ur_arm"]  = _make_stick(scene, ur,  ubj, c_struct)
        o["lf_arm"]  = _make_stick(scene, lf,  lbj, c_struct)
        o["lr_arm"]  = _make_stick(scene, lr,  lbj, c_struct)
        o["upright"] = _make_stick(scene, lbj, ubj, c_struct)
        o["tierod"]  = _make_stick(scene, tr_ib, step["tr_ob"], c_tie)
        _sbe = _shock_body_end(s_ib, step["s_ob"], hp.shock_min)
        o["shock_plunger"] = _make_variable_stick(scene, s_ib, step["s_ob"], c_shock, radius=0.003)
        o["shock_body"]    = _make_stick(scene, s_ib, _sbe,         c_shock, radius=0.012)
        if "piv_ob" in step:
            plunge_mm = step.get("axle_data", {}).get("plunge_mm", 0.0)
            piv_ib_dyn = _axle_plunge_point(piv_ib, step["piv_ob"], plunge_mm)
            o["axle_in"]  = _make_variable_stick(scene, piv_ib_dyn,     step["piv_ob"], c_axle)
            o["axle_out"] = _make_stick(scene, step["piv_ob"], step["wc"],     c_axle)
            o["sp_piv_ib"] = scene.sphere(radius=0.010).material("#000000").move(*_v(piv_ib_dyn))
            o["sp_piv_ob"] = scene.sphere(radius=0.010).material("#000000").move(*_v(step["piv_ob"]))
            if show_guides:
                o["axle_ib_guide"] = _make_dashed_line(scene, piv_ib, [0, 1, 0], 250.0, c_axle)
        for k, pt in [("sp_uf", uf), ("sp_ur", ur), ("sp_lf", lf),
                      ("sp_lr", lr), ("sp_s_ib", s_ib)]:
            o[k] = scene.sphere(radius=0.010).material("#222222").move(*_v(pt))
        for k, pt in [("sp_ubj", ubj), ("sp_lbj", lbj),
                      ("sp_tr_ib", tr_ib), ("sp_tr_ob", step["tr_ob"]),
                      ("sp_s_ob", step["s_ob"])]:
            o[k] = scene.sphere(radius=0.010).material("#222222").move(*_v(pt))
        if show_guides:
            o["tie_ib_guide"] = _make_dashed_line(scene, tr_ib, [0, 1, 0], 250.0, c_tie)
        o["wheel"] = _make_wheel(scene, step["wc"], step.get("wheel_axis", ax_default), hp.wr, hp.ww)
        o["sp_wc"] = scene.sphere(radius=0.014).material("#4466bb").move(*_v(step["wc"]))

    elif isinstance(hp, SemiTrailingLink):
        tl_f   = np.asarray(step.get("tl_f",   hp.tl_f))
        ucl_ib = np.asarray(step.get("ucl_ib", hp.ucl_ib))
        lcl_ib = np.asarray(step.get("lcl_ib", hp.lcl_ib))
        s_ib   = np.asarray(step.get("s_ib",   hp.s_ib))
        piv_ib = np.asarray(step.get("piv_ib", hp.piv_ib))
        o["tl_f_ucl"] = _make_stick(scene, tl_f,   step["ucl_ob"], c_struct)
        o["tl_f_lcl"] = _make_stick(scene, tl_f,   step["lcl_ob"], c_struct)
        o["ucl_link"] = _make_stick(scene, ucl_ib, step["ucl_ob"], c_struct)
        o["lcl_link"] = _make_stick(scene, lcl_ib, step["lcl_ob"], c_struct)
        _sbe = _shock_body_end(s_ib, step["s_ob"], hp.shock_min)
        o["shock_body"]    = _make_stick(scene, s_ib, _sbe,         c_shock, radius=0.012)
        o["shock_plunger"] = _make_variable_stick(scene, _sbe, step["s_ob"], c_shock, radius=0.003)
        if float(np.linalg.norm(np.asarray(step["s_ob"]) - np.asarray(_sbe))) < 1.0:
            o["shock_plunger"].scale(1.0, 1e-6, 1.0)
        if "piv_ob" in step:
            plunge_mm = step.get("axle_data", {}).get("plunge_mm", 0.0)
            piv_ib_dyn = _axle_plunge_point(piv_ib, step["piv_ob"], plunge_mm)
            o["axle_in"]  = _make_variable_stick(scene, piv_ib_dyn,     step["piv_ob"], c_axle)
            o["axle_out"] = _make_stick(scene, step["piv_ob"], step["wc"],     c_axle)
            o["sp_piv_ib"] = scene.sphere(radius=0.010).material("#000000").move(*_v(piv_ib_dyn))
            o["sp_piv_ob"] = scene.sphere(radius=0.010).material("#000000").move(*_v(step["piv_ob"]))
            if show_guides:
                o["axle_ib_guide"] = _make_dashed_line(scene, piv_ib, [0, 1, 0], 250.0, c_axle)
        for k, pt in [("sp_tl_f", tl_f), ("sp_ucl_ib", ucl_ib),
                      ("sp_lcl_ib", lcl_ib), ("sp_s_ib", s_ib)]:
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
        ubj    = np.asarray(step["ubj"]);  lbj = np.asarray(step["lbj"])
        uf     = np.asarray(step.get("uf",     hp.uf))
        ur     = np.asarray(step.get("ur",     hp.ur))
        lf     = np.asarray(step.get("lf",     hp.lf))
        lr     = np.asarray(step.get("lr",     hp.lr))
        s_ib   = np.asarray(step.get("s_ib",   hp.s_ib))
        piv_ib = np.asarray(step.get("piv_ib", hp.piv_ib))
        tr_ib  = np.asarray(step.get("tr_ib",  hp.tr_ib))
        _move_stick(o["uf_arm"],  uf, ubj)
        _move_stick(o["ur_arm"],  ur, ubj)
        _move_stick(o["lf_arm"],  lf, lbj)
        _move_stick(o["lr_arm"],  lr, lbj)
        _move_stick(o["upright"], lbj, ubj)
        _move_stick(o["tierod"],  tr_ib, step["tr_ob"])
        if "axle_out" in o and "piv_ob" in step:
            _move_stick(o["axle_out"], step["piv_ob"], step["wc"])
        _sbe = _shock_body_end(s_ib, step["s_ob"], hp.shock_min)
        _move_stick(o["shock_body"], s_ib, _sbe)
        plunger_mm = float(np.linalg.norm(np.asarray(step["s_ob"]) - np.asarray(_sbe)))
        if plunger_mm > 1.0:
            _move_variable_stick(o["shock_plunger"], _sbe, step["s_ob"])
        else:
            o["shock_plunger"].scale(1.0, 1e-6, 1.0)
        if "piv_ob" in step:
            plunge_mm = step.get("axle_data", {}).get("plunge_mm", 0.0)
            piv_ib_dyn = _axle_plunge_point(piv_ib, step["piv_ob"], plunge_mm)
            if "axle_in" in o:
                _move_variable_stick(o["axle_in"], piv_ib_dyn, step["piv_ob"])
            if "sp_piv_ib" in o:
                o["sp_piv_ib"].move(*_v(piv_ib_dyn))
            if "sp_piv_ob" in o:
                o["sp_piv_ob"].move(*_v(step["piv_ob"]))
        o["sp_uf"].move(*_v(uf));  o["sp_ur"].move(*_v(ur))
        o["sp_lf"].move(*_v(lf));  o["sp_lr"].move(*_v(lr))
        o["sp_s_ib"].move(*_v(s_ib))
        o["sp_ubj"].move(*_v(ubj));  o["sp_lbj"].move(*_v(lbj))
        o["sp_tr_ib"].move(*_v(tr_ib))
        o["sp_tr_ob"].move(*_v(step["tr_ob"]))
        o["sp_s_ob"].move(*_v(step["s_ob"]))
        _move_wheel(o["wheel"], step["wc"], step.get("wheel_axis", ax_default))
        o["sp_wc"].move(*_v(step["wc"]))

    elif isinstance(hp, SemiTrailingLink):
        tl_f   = np.asarray(step.get("tl_f",   hp.tl_f))
        ucl_ib = np.asarray(step.get("ucl_ib", hp.ucl_ib))
        lcl_ib = np.asarray(step.get("lcl_ib", hp.lcl_ib))
        s_ib   = np.asarray(step.get("s_ib",   hp.s_ib))
        piv_ib = np.asarray(step.get("piv_ib", hp.piv_ib))
        _move_stick(o["tl_f_ucl"], tl_f,   step["ucl_ob"])
        _move_stick(o["tl_f_lcl"], tl_f,   step["lcl_ob"])
        _move_stick(o["ucl_link"], ucl_ib, step["ucl_ob"])
        _move_stick(o["lcl_link"], lcl_ib, step["lcl_ob"])
        if "axle_out" in o and "piv_ob" in step:
            _move_stick(o["axle_out"], step["piv_ob"], step["wc"])
        _sbe = _shock_body_end(s_ib, step["s_ob"], hp.shock_min)
        _move_stick(o["shock_body"], s_ib, _sbe)
        plunger_mm = float(np.linalg.norm(np.asarray(step["s_ob"]) - np.asarray(_sbe)))
        if plunger_mm > 1.0:
            _move_variable_stick(o["shock_plunger"], _sbe, step["s_ob"])
        else:
            o["shock_plunger"].scale(1.0, 1e-6, 1.0)
        if "piv_ob" in step:
            plunge_mm = step.get("axle_data", {}).get("plunge_mm", 0.0)
            piv_ib_dyn = _axle_plunge_point(piv_ib, step["piv_ob"], plunge_mm)
            if "axle_in" in o:
                _move_variable_stick(o["axle_in"], piv_ib_dyn, step["piv_ob"])
            if "sp_piv_ib" in o:
                o["sp_piv_ib"].move(*_v(piv_ib_dyn))
            if "sp_piv_ob" in o:
                o["sp_piv_ob"].move(*_v(step["piv_ob"]))
        o["sp_tl_f"].move(*_v(tl_f))
        o["sp_ucl_ib"].move(*_v(ucl_ib));  o["sp_lcl_ib"].move(*_v(lcl_ib))
        o["sp_s_ib"].move(*_v(s_ib))
        o["sp_ucl_ob"].move(*_v(step["ucl_ob"]))
        o["sp_lcl_ob"].move(*_v(step["lcl_ob"]))
        o["sp_s_ob"].move(*_v(step["s_ob"]))
        _move_wheel(o["wheel"], step["wc"], step.get("wheel_axis", ax_default))
        o["sp_wc"].move(*_v(step["wc"]))

def _build_scene(scene, step, hp):
    """Build all scene objects once. Returns scene_objs dict."""
    with scene:
        objs = _build_corner_objects(scene, step, hp)
    return objs

def _update_scene(scene_objs, step, hp):
    """Update all scene objects in-place for the new step. Zero allocation."""
    _update_corner_objects(scene_objs, step, hp)

def _cog_color(phase: str) -> str:
    if phase == "hoist":
        return "#ff6600"
    if phase == "settled":
        return "#00cc44"
    return "#ffcc00"

def _build_dyn_scene(scene, step, vehicle) -> dict:
    """Build all 4 corners + CoG sphere for the dynamic drop scenario."""
    objs: dict = {}
    with scene:
        for key, attr, c_struct, _ in _DYN_CORNERS:
            corner = getattr(vehicle, attr)
            if step.get(key) is not None:
                objs[key] = _build_corner_objects(scene, step[key], corner.hardpoints,
                                                   c_struct=c_struct, show_guides=False)
        if step.get("cog_pos") is not None:
            objs["cog"] = (
                scene.sphere(radius=0.05)
                .material(_cog_color(step.get("phase", "drop")))
                .move(*_v(step["cog_pos"]))
            )
    return objs

def _build_shock_dyno_scene(scene, step) -> dict:
    """Build an isolated shock for the shock dyno."""
    objs = {"_scene": scene}
    
    # Position upper point 200mm above its own length
    z_fixed = step.get("shock_max", 500.0) + 200.0
    s_ib = np.array([0, 0, z_fixed])
    s_ob = np.array([0, 0, z_fixed - step["shock_len"]])
    shock_min = step.get("shock_min", 200.0)
    _sbe = _shock_body_end(s_ib, s_ob, shock_min)
    
    c_shock = "#6e6e82"
    with scene:
        objs["shock_body"]    = _make_stick(scene, s_ib, _sbe, c_shock, radius=0.012)
        objs["shock_plunger"] = _make_variable_stick(scene, _sbe, s_ob, c_shock, radius=0.003)
        objs["sp_s_ib"] = scene.sphere(radius=0.010).material("#222222").move(*_v(s_ib))
        objs["sp_s_ob"] = scene.sphere(radius=0.010).material("#222222").move(*_v(s_ob))
        
    return objs

def _update_shock_dyno_scene(objs: dict, step) -> None:
    z_fixed = step.get("shock_max", 500.0) + 200.0
    s_ib = np.array([0, 0, z_fixed])
    s_ob = np.array([0, 0, z_fixed - step["shock_len"]])
    shock_min = step.get("shock_min", 200.0)
    _sbe = _shock_body_end(s_ib, s_ob, shock_min)
    
    _move_stick(objs["shock_body"], s_ib, _sbe)
    _move_variable_stick(objs["shock_plunger"], _sbe, s_ob)
    objs["sp_s_ib"].move(*_v(s_ib))
    objs["sp_s_ob"].move(*_v(s_ob))

def _fit_camera_shock_dyno(scene, step) -> None:
    z_fixed = step.get("shock_max", 500.0) + 200.0
    z_center = (z_fixed + (z_fixed - step["shock_len"])) / 2.0 * _S
    slen = float(step["shock_max"]) * _S
    scene.move_camera(
        x=slen * 1.5,
        y=slen * 1.5,
        z=z_center + slen * 0.5,
        look_at_x=0, look_at_y=0, look_at_z=z_center,
        up_x=0, up_y=0, up_z=1
    )

def _update_dyn_scene(objs: dict, step, vehicle) -> None:
    """Update all 4 corners and CoG sphere in-place."""
    for key, attr, _, _ in _DYN_CORNERS:
        corner = getattr(vehicle, attr)
        if key in objs and step.get(key) is not None:
            _update_corner_objects(objs[key], step[key], corner.hardpoints)
    if "cog" in objs and step.get("cog_pos") is not None:
        objs["cog"].move(*_v(step["cog_pos"]))
        objs["cog"].material(_cog_color(step.get("phase", "drop")))


def _fit_camera_dyn(scene, step, vehicle) -> None:
    """Position camera to frame all 4 corners of the full vehicle."""
    all_pts = []
    for key, attr, _, _ in _DYN_CORNERS:
        corner = getattr(vehicle, attr)
        hp = corner.hardpoints
        s = step.get(key)
        if s:
            for k in ["ubj", "lbj", "wc", "tr_ob", "s_ob", "ucl_ob", "lcl_ob"]:
                if k in s:
                    all_pts.append(np.asarray(s[k]))
        for attr_name in ["uf", "ur", "lf", "lr", "s_ib", "tl_f", "ucl_ib", "lcl_ib"]:
            pt = s.get(attr_name) if s else None
            if pt is None and hasattr(hp, attr_name):
                pt = getattr(hp, attr_name)
            if pt is not None:
                all_pts.append(np.asarray(pt))
    if not all_pts:
        return
    arr  = np.array(all_pts) * _S
    ctr  = arr.mean(axis=0)
    span = float(np.max(arr.max(axis=0) - arr.min(axis=0)))
    dist = span * 1.6
    scene.move_camera(
        x=float(ctr[0]) + dist * 0.3,
        y=float(ctr[1]) - dist * 1.2,
        z=float(ctr[2]) + dist * 0.8,
        look_at_x=float(ctr[0]),
        look_at_y=float(ctr[1]),
        look_at_z=float(ctr[2]),
        up_x=0, up_y=0, up_z=1,
    )

# --- Optimizer config preview: free-point search boxes + keepout zones -----
_ZONE_PALETTE = ["#cc2828", "#2864cc", "#cc8a28", "#8a28cc", "#28ccaa", "#cc2891"]
_FREE_POINT_COLOR = "#FFEE00"
_PREVIEW_OPACITY = 0.50

_STATIC_ATTRS = ["ubj", "lbj", "uf", "ur", "lf", "lr", "tr_ib", "tr_ob",
                  "s_ib", "s_ob", "piv_ib", "piv_ob", "wc",
                  "tl_f", "ucl_ib", "ucl_ob", "lcl_ib", "lcl_ob"]

def _hp_to_static_step(hp) -> dict:
    """Build a step-like dict from a hardpoints object's rest-position attributes."""
    return {attr: getattr(hp, attr) for attr in _STATIC_ATTRS if hasattr(hp, attr)}

def _place_shape(obj, p1s: np.ndarray, p2s: np.ndarray):
    """Move + rotate any Object3D (box or cylinder) to span p1->p2 (scene units)."""
    _place_cyl(obj, p1s, p2s)

def _make_free_point_box(scene, center_xyz, deltas: dict, color=_FREE_POINT_COLOR,
                          opacity=_PREVIEW_OPACITY):
    """Axis-aligned translucent box spanning a free point's per-axis [min,max] deltas.
    `deltas` is like {'x': [lo, hi], 'y': [...], 'z': [...]} (mm, relative to center)."""
    center = np.asarray(center_xyz, float)
    dims = []
    offset = np.zeros(3)
    for i, ax in enumerate("xyz"):
        lo, hi = deltas.get(ax, [0.0, 0.0])
        span = max(float(hi) - float(lo), 2.0)  # thin slab if unconstrained
        dims.append(span * _S)
        offset[i] = (float(lo) + float(hi)) / 2.0
    box = scene.box(dims[0], dims[1], dims[2]).material(color, opacity=opacity, side="both")
    pos = (center + offset) * _S
    box.move(float(pos[0]), float(pos[1]), float(pos[2]))
    return box

def _make_keepout_zone(scene, p1, p2, shape: str, dim1: float, dim2: float | None,
                        color: str, opacity=_PREVIEW_OPACITY):
    """Translucent shape extruded along axis p1->p2. shape='cylinder' (dim1=radius)
    or 'box' (dim1, dim2 = cross-section side lengths)."""
    p1s = np.asarray(p1, float) * _S
    p2s = np.asarray(p2, float) * _S
    L = max(float(np.linalg.norm(p2s - p1s)), 1e-9)
    if shape == "box":
        obj = scene.box(float(dim1) * _S, L, float(dim2 or dim1) * _S)
    else:
        r = float(dim1) * _S
        obj = scene.cylinder(top_radius=r, bottom_radius=r, height=L)
    obj.material(color, opacity=opacity, side="both")
    _place_shape(obj, p1s, p2s)
    return obj

def zone_color(index: int) -> str:
    return _ZONE_PALETTE[index % len(_ZONE_PALETTE)]

def build_zone_colors(keepout_cfg: list, groups_cfg: dict | None) -> dict[str, str]:
    """Assign each zone a color. Zones sharing a COLLISION_GROUPS group get the
    same color; ungrouped zones (or when groups_cfg is unset) each get their own."""
    zone_to_group: dict[str, str] = {}
    if groups_cfg:
        for group_name, members in groups_cfg.items():
            for member in members or []:
                zone_to_group[member] = group_name

    color_for_key: dict = {}
    colors: dict[str, str] = {}
    for zone in keepout_cfg or []:
        name = zone.get("name", "")
        key = zone_to_group.get(name, ("__zone__", name))
        if key not in color_for_key:
            color_for_key[key] = zone_color(len(color_for_key))
        colors[name] = color_for_key[key]
    return colors

def build_legend_entries(keepout_cfg: list, groups_cfg: dict | None) -> list[tuple[str, str]]:
    """One (label, color) entry per group, plus one per ungrouped zone."""
    zone_to_group: dict[str, str] = {}
    if groups_cfg:
        for group_name, members in groups_cfg.items():
            for member in members or []:
                zone_to_group[member] = group_name

    colors = build_zone_colors(keepout_cfg, groups_cfg)
    entries: list[tuple[str, str]] = []
    seen_groups: set = set()
    for zone in keepout_cfg or []:
        name = zone.get("name", "")
        group = zone_to_group.get(name)
        if group is not None:
            if group in seen_groups:
                continue
            seen_groups.add(group)
            entries.append((group, colors[name]))
        else:
            entries.append((name, colors[name]))
    return entries

def _resolve_point_attr(hp, name: str) -> str | None:
    """Resolve a config-supplied point name to the hardpoints object's short
    attribute name. Accepts either the short attr (e.g. 'tr_ib') or the long
    YAML key (e.g. 'tie_rod_inboard') via the hardpoints class's _YAML_MAP."""
    if hasattr(hp, name):
        return name
    yaml_map = getattr(hp, "_YAML_MAP", {}) or {}
    for attr, yaml_key in yaml_map.items():
        if yaml_key == name:
            return attr
    return None

def _resolve_zone_point(name: str, hp) -> np.ndarray:
    attr = _resolve_point_attr(hp, name)
    if attr is None:
        raise ValueError(f"KEEPOUT_ZONES point '{name}' not found on hardpoints")
    return np.asarray(getattr(hp, attr), float)

def _build_config_preview_scene(scene, hp, free_points_cfg: dict, keepout_cfg: list,
                                 groups_cfg: dict | None = None) -> dict:
    """Build a static preview: the vehicle at rest, free-point search boxes (grey),
    and keepout zone shapes (colored by COLLISION_GROUPS membership, if set)."""
    objs: dict = {}
    zone_colors = build_zone_colors(keepout_cfg, groups_cfg)
    with scene:
        step = _hp_to_static_step(hp)
        objs["corner"] = _build_corner_objects(scene, step, hp, show_guides=False)

        free_boxes = {}
        for pt_name, deltas in (free_points_cfg or {}).items():
            attr = _resolve_point_attr(hp, pt_name)
            if attr is None:
                print(f"WARNING: FREE_POINTS point '{pt_name}' not found on hardpoints. Skipping preview box.")
                continue
            center = getattr(hp, attr)
            free_boxes[pt_name] = _make_free_point_box(scene, center, deltas)
        objs["free_boxes"] = free_boxes

        zones = {}
        for i, zone in enumerate(keepout_cfg or []):
            try:
                p1 = _resolve_zone_point(zone["point_a"], hp)
                p2 = _resolve_zone_point(zone["point_b"], hp)
            except ValueError as e:
                print(f"WARNING: {e}. Skipping keepout zone '{zone.get('name', i)}'.")
                continue
            name = zone.get("name", f"zone_{i}")
            color = zone_colors.get(name, zone_color(i))
            zones[name] = (
                _make_keepout_zone(scene, p1, p2, zone.get("shape", "cylinder"),
                                    zone.get("dim1", 10.0), zone.get("dim2"), color),
                color,
            )
        objs["zones"] = zones
    return objs

def _fit_camera(scene, step, hp):
    """Position camera to frame the geometry."""
    all_pts = []

    def collect(s, h):
        for k in ["ubj", "lbj", "wc", "tr_ob", "s_ob", "ucl_ob", "lcl_ob"]:
            if k in s:
                all_pts.append(np.asarray(s[k]))
        for attr in ["uf", "ur", "lf", "lr", "s_ib", "tl_f", "ucl_ib", "lcl_ib"]:
            if hasattr(h, attr):
                all_pts.append(np.asarray(getattr(h, attr)))

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
