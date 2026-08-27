# third-party
import numpy as np

# ours
from utils.spatial import Point, Sphere, Cylinder, Cuboid, Shock, corner_shape, align_y_to_direction
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

def _corner_style(c_struct="#1e1e1e", c_tie="#009944",
                   c_shock="#6e6e82", c_axle="#cc2828", show_guides=True) -> dict:
    return dict(struct_color=c_struct, tie_color=c_tie, shock_color=c_shock,
               axle_color=c_axle, show_guides=show_guides)

def _build_corner_objects(scene, step, hp,
                           c_struct="#1e1e1e", c_tie="#009944",
                           c_shock="#6e6e82", c_axle="#cc2828",
                           show_guides=True):
    """Create all 3-D objects for one corner via its corner shape
    (utils.spatial.shapes.corner). Returns a nested dict of Object3D refs."""
    return corner_shape(step, hp, **_corner_style(c_struct, c_tie, c_shock,
                                                     c_axle, show_guides)).to_3d(scene)

def _update_corner_objects(o, step, hp):
    """Re-place one corner's scene objects for a new step. Colors/guides don't
    change between steps, so a default-styled assembly is enough to drive place()."""
    corner_shape(step, hp).place(o)

def _build_scene(scene, step, hp):
    """Build all scene objects once. Returns scene_objs dict."""
    with scene:
        objs = _build_corner_objects(scene, step, hp)
    return objs

def _update_scene(scene_objs, step, hp):
    """Re-place all scene objects for the new step (no new scene objects created)."""
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
        if step.get("gc_viz") is not None:
            objs["gc"] = _build_ground_clearance_objs(scene, step["gc_viz"])
    return objs

_GC_GROUND_COLOR  = "#3a7bd5"
_GC_CHASSIS_COLOR = "#d59a3a"
_GC_POINT_COLOR   = "#111111"

def _gc_plane_span(viz) -> tuple[float, float]:
    """(x span, y span) for the drawn plane slabs, from the contact-patch extent."""
    c = np.asarray(viz["contacts"], float)
    dx = max(float(np.ptp(c[:, 0])), 100.0) * 1.35
    dy = max(float(np.ptp(c[:, 1])), 100.0) * 1.35
    return dx, dy

def _gauge_color(clearance: float) -> str:
    return "#d11a2a" if (clearance is not None and clearance < 0) else "#22aa44"

def _gc_shape_specs(viz) -> dict:
    """Build the drawable shape specs (utils.spatial.shapes) for the ground-
    clearance overlay from a step's ``gc_viz`` dict. Shared by build + update so
    the two paths can never drift."""
    dx, dy = _gc_plane_span(viz)
    ground_c = Point(viz["ground_centroid"])
    cx = (viz["front_x"] + viz["rear_x"]) / 2.0
    return {
        "ground":  Cuboid(ground_c, Point(viz["ground_normal"]), dx, dy, 3.0, _GC_GROUND_COLOR, 0.28),
        "chassis": Cuboid(Point(cx, ground_c.y, viz["chassis_bottom_z"]), Point(0.0, 0.0, 1.0),
                          dx, dy, 3.0, _GC_CHASSIS_COLOR, 0.30),
        "contacts": [Sphere(Point(p), 12.0, _GC_GROUND_COLOR) for p in viz["contacts"]],
        "low_pt":  Sphere(Point(viz["chassis_low_point"]), 14.0, _GC_POINT_COLOR),
        "cog":     Sphere(Point(viz["cog_point"]), 16.0, "#cc22aa"),
        "gauge_f": Cylinder(Point(viz["front_x"], 0.0, viz["front_ground_z"]),
                            Point(viz["front_x"], 0.0, viz["chassis_bottom_z"]),
                            6.0, _gauge_color(viz.get("_front_clearance"))),
        "gauge_r": Cylinder(Point(viz["rear_x"], 0.0, viz["rear_ground_z"]),
                            Point(viz["rear_x"], 0.0, viz["chassis_bottom_z"]),
                            6.0, _gauge_color(viz.get("_rear_clearance"))),
    }

def _build_ground_clearance_objs(scene, viz) -> dict:
    """Ground plane (through the 4 contact patches), fixed chassis-bottom plane,
    the points they're built from, and the front/rear vertical clearance gauges.
    Starts hidden -- revealed via the parts tree beside the 3D view."""
    o: dict = {}
    for key, spec in _gc_shape_specs(viz).items():
        o[key] = [s.to_3d(scene) for s in spec] if isinstance(spec, list) else spec.to_3d(scene)
    _set_subtree_visible(o, False)
    return o

def _update_ground_clearance_objs(o, viz) -> None:
    # restyle=True so the front/rear gauge color tracks the sign of the clearance
    for key, spec in _gc_shape_specs(viz).items():
        if isinstance(spec, list):
            for s, obj in zip(spec, o[key]):
                s.place(obj, restyle=True)
        else:
            spec.place(o[key], restyle=True)

def _iter_leaf_objs(node):
    """Yield every Object3D beneath a scene-objs node."""
    if isinstance(node, dict):
        for v in node.values():
            yield from _iter_leaf_objs(v)
    elif isinstance(node, (list, tuple)):
        for v in node:
            yield from _iter_leaf_objs(v)
    elif node is not None:
        yield node

def _set_subtree_visible(node, show: bool) -> None:
    for obj in _iter_leaf_objs(node):
        obj.visible(show)

# ----------------------- #
#  Parts visibility tree  #
# ----------------------- #

_TREE_LABELS = {
    "fl": "Front Left", "fr": "Front Right", "rl": "Rear Left", "rr": "Rear Right",
    "cog": "Center of Gravity", "gc": "Ground Clearance",
    "upper_arm": "Upper A-Arm", "lower_arm": "Lower A-Arm", "upright": "Upright",
    "tie_rod": "Tie Rod", "shock": "Coilover", "axle": "Axle", "wheel": "Wheel",
    "trailing_link": "Trailing Link", "upper_camber_link": "Upper Camber Link",
    "lower_camber_link": "Lower Camber Link",
    "tie_guide": "Tie Rod Axis", "axle_guide": "Axle Axis",
    "ground": "Ground Plane", "chassis": "Chassis Plane", "contacts": "Contact Patches",
    "low_pt": "Reference Point", "gauge_f": "Front Gauge", "gauge_r": "Rear Gauge",
    "corner": "Suspension", "free_boxes": "Free-Point Ranges", "zones": "Keepout Zones",
}

# Composites that start hidden (their parts-tree node is unticked on load).
_HIDDEN_BY_DEFAULT = {"gc"}

def _label_for(key: str) -> str:
    return _TREE_LABELS.get(key, key.replace("_", " ").title())

def _is_branch(val) -> bool:
    """A dict node is a branch only if it nests further dicts (corners); a dict of
    plain leaves (one composite) is itself a single toggle."""
    return isinstance(val, dict) and any(isinstance(v, dict) for v in val.values())

def scene_parts_tree(scene_objs, _prefix: str = "") -> list[dict]:
    """meshcat-style node list [{id, label, children}] for every togglable part."""
    nodes = []
    for key, val in (scene_objs or {}).items():
        node_id = f"{_prefix}{key}"
        children = scene_parts_tree(val, f"{node_id}/") if _is_branch(val) else []
        nodes.append({"id": node_id, "label": _label_for(key), "children": children})
    return nodes

def _tree_leaf_ids(nodes: list[dict]) -> list[str]:
    out = []
    for n in nodes:
        if n["children"]:
            out.extend(_tree_leaf_ids(n["children"]))
        else:
            out.append(n["id"])
    return out

def scene_parts_tree_defaults(scene_objs) -> tuple[list[dict], list[str], list[str]]:
    """(tree, all leaf ids, leaf ids that start ticked/visible)."""
    tree = scene_parts_tree(scene_objs)
    leaves = _tree_leaf_ids(tree)
    ticked = [lid for lid in leaves if lid.split("/")[0] not in _HIDDEN_BY_DEFAULT]
    return tree, leaves, ticked

def resolve_scene_node(scene_objs, node_id: str):
    node = scene_objs
    for part in node_id.split("/"):
        node = node[part]
    return node

def set_scene_node_visible(scene_objs, node_id: str, show: bool) -> None:
    """Show/hide a parts-tree node's whole subtree; hidden objects are skipped by
    the per-frame ``place`` so the renderer does less work."""
    _set_subtree_visible(resolve_scene_node(scene_objs, node_id), show)

def _dyno_shock(step) -> Shock:
    """The isolated shock for the dyno, positioned vertically with its inboard
    mount fixed 200mm above the fully-extended length."""
    z_fixed = step.get("shock_max", 500.0) + 200.0
    s_ib = Point(0.0, 0.0, z_fixed)
    s_ob = Point(0.0, 0.0, z_fixed - step["shock_len"])
    return Shock(s_ib, s_ob, step.get("shock_min", 200.0))

def _build_shock_dyno_scene(scene, step) -> dict:
    """Build an isolated shock for the shock dyno."""
    with scene:
        return {"shock": _dyno_shock(step).to_3d(scene)}

def _update_shock_dyno_scene(objs: dict, step) -> None:
    _dyno_shock(step).place(objs["shock"])

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
    if "cog" in objs and step.get("cog_pos") is not None and getattr(objs["cog"], "visible_", True):
        objs["cog"].move(*_v(step["cog_pos"]))
        objs["cog"].material(_cog_color(step.get("phase", "drop")))
    if "gc" in objs and step.get("gc_viz") is not None:
        _update_ground_clearance_objs(objs["gc"], step["gc_viz"])

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
    """Move + rotate any Object3D (box or cylinder) to span p1->p2 (scene units).
    The object must already have the correct length baked into its geometry."""
    d = p2s - p1s
    L = float(np.linalg.norm(d))
    if L < 1e-9:
        return
    mid = (p1s + p2s) * 0.5
    obj.move(float(mid[0]), float(mid[1]), float(mid[2]))
    obj.rotate(*align_y_to_direction(d))

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
        if free_boxes:
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
            zones[name] = _make_keepout_zone(scene, p1, p2, zone.get("shape", "cylinder"),
                                              zone.get("dim1", 10.0), zone.get("dim2"), color)
        if zones:
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
