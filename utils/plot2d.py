# default
from typing import Any

# third-party
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ours
from utils.geometry import (
    get_wheel_attitude, get_steering_axis_geometry,
    motion_ratio_series, static_ride_height_index,
)

_LAYOUT_BASE = dict(
    margin=dict(l=44, r=8, t=30, b=32),
    height=200,
    template="plotly_white",
    font=dict(size=10),
    plot_bgcolor="#fafafa",
)

_COLORS = ["#059669", "#65a30d", "#14b8a6", "#047857", "#10b981", "#84cc16"]
_C_FRONT = "#059669"
_C_REAR = "#65a30d"
_C_TIE = "#14b8a6"
_C_CMP = "#9333ea"
_C_CMP2 = "#c026d3"

def _vline_shape(x: float) -> dict[str, Any]:
    """Return a vertical line shape for plotly."""
    return dict(type="line", x0=x, x1=x, y0=0, y1=1,
                xref="x", yref="paper",
                line=dict(color="rgba(16,185,129,0.8)", width=1.5, dash="dot"))

def _build_dyno_figures(steps: list, cmp_steps: list | None = None,
                         cmp_label: str = "Compare") -> tuple[list[tuple[str, Any]], list[float]]:
    """Build figures for a Shock Dyno test. cmp_steps: optional second test overlaid dashed."""
    if not steps:
        return [], []

    t = [s["t"] for s in steps]
    vel = [s["velocity"] for s in steps]
    disp = [s["displacement"] for s in steps]
    force_tot = [s["force_total"] for s in steps]
    force_spr = [s["force_spring"] for s in steps]
    force_damp = [s["force_damper"] for s in steps]

    c_t = c_vel = c_disp = c_force_tot = c_force_spr = c_force_damp = None
    if cmp_steps:
        c_t = [s["t"] for s in cmp_steps]
        c_vel = [s["velocity"] for s in cmp_steps]
        c_disp = [s["displacement"] for s in cmp_steps]
        c_force_tot = [s["force_total"] for s in cmp_steps]
        c_force_spr = [s["force_spring"] for s in cmp_steps]
        c_force_damp = [s["force_damper"] for s in cmp_steps]

    named_figs = []

    # Force vs Velocity (The classic dyno plot)
    fig_fv = make_subplots(rows=1, cols=1)
    fig_fv.add_trace(go.Scatter(x=vel, y=force_damp, mode="lines", name="Damping Force", line=dict(color=_C_FRONT, width=2)))
    fig_fv.add_trace(go.Scatter(x=vel, y=force_tot, mode="lines", name="Total Force", line=dict(color=_C_REAR, width=2, dash="dash")))
    if cmp_steps:
        fig_fv.add_trace(go.Scatter(x=c_vel, y=c_force_damp, mode="lines", name=f"{cmp_label} Damping",
                                     line=dict(color=_C_CMP, width=2)))
        fig_fv.add_trace(go.Scatter(x=c_vel, y=c_force_tot, mode="lines", name=f"{cmp_label} Total",
                                     line=dict(color=_C_CMP2, width=2, dash="dash")))
    fig_fv.update_layout(
        title="Force vs Velocity",
        xaxis_title="Velocity [mm/s] (+ Compression, - Rebound)",
        yaxis_title="Force [N]",
        height=450,
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(x=0.02, y=0.98),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(240,240,240,0.5)'
    )
    fig_fv.update_xaxes(zeroline=True, zerolinewidth=1, zerolinecolor='black')
    fig_fv.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor='black')
    named_figs.append(("Force vs Velocity", fig_fv))

    # Force vs Absolute Velocity
    fig_fav = make_subplots(rows=1, cols=1)

    comp_vel = [abs(v) for v in vel if v >= 0]
    comp_f = [f for v, f in zip(vel, force_damp) if v >= 0]
    reb_vel = [abs(v) for v in vel if v < 0]
    reb_f = [f for v, f in zip(vel, force_damp) if v < 0]

    fig_fav.add_trace(go.Scatter(x=comp_vel, y=comp_f, mode="lines", name="Compression", line=dict(color=_C_FRONT, width=2)))
    fig_fav.add_trace(go.Scatter(x=reb_vel, y=reb_f, mode="lines", name="Rebound", line=dict(color=_C_REAR, width=2)))
    if cmp_steps:
        c_comp_vel = [abs(v) for v in c_vel if v >= 0]
        c_comp_f   = [f for v, f in zip(c_vel, c_force_damp) if v >= 0]
        c_reb_vel  = [abs(v) for v in c_vel if v < 0]
        c_reb_f    = [f for v, f in zip(c_vel, c_force_damp) if v < 0]
        fig_fav.add_trace(go.Scatter(x=c_comp_vel, y=c_comp_f, mode="lines", name=f"{cmp_label} Compression",
                                      line=dict(color=_C_CMP, width=2)))
        fig_fav.add_trace(go.Scatter(x=c_reb_vel, y=c_reb_f, mode="lines", name=f"{cmp_label} Rebound",
                                      line=dict(color=_C_CMP2, width=2)))
    fig_fav.update_layout(
        title="Force vs Absolute Velocity",
        xaxis_title="Absolute Velocity [mm/s]",
        yaxis_title="Damping Force [N]",
        height=450,
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(x=0.02, y=0.98),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(240,240,240,0.5)'
    )
    fig_fav.update_xaxes(zeroline=True, zerolinewidth=1, zerolinecolor='black')
    fig_fav.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor='black')
    named_figs.append(("Force vs Abs Velocity", fig_fav))

    # Force vs Displacement
    fig_fx = make_subplots(rows=1, cols=1)
    fig_fx.add_trace(go.Scatter(x=disp, y=force_spr, mode="lines", name="Spring Force", line=dict(color=_C_FRONT, width=2)))
    fig_fx.add_trace(go.Scatter(x=disp, y=force_tot, mode="lines", name="Total Force", line=dict(color=_C_REAR, width=2, dash="dash")))
    if cmp_steps:
        fig_fx.add_trace(go.Scatter(x=c_disp, y=c_force_spr, mode="lines", name=f"{cmp_label} Spring",
                                     line=dict(color=_C_CMP, width=2)))
        fig_fx.add_trace(go.Scatter(x=c_disp, y=c_force_tot, mode="lines", name=f"{cmp_label} Total",
                                     line=dict(color=_C_CMP2, width=2, dash="dash")))
    fig_fx.update_layout(
        title="Force vs Displacement",
        xaxis_title="Displacement [mm] (+ Compression, - Rebound)",
        yaxis_title="Force [N]",
        height=450,
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(x=0.02, y=0.98),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(240,240,240,0.5)'
    )
    named_figs.append(("Force vs Displacement", fig_fx))

    # Time series
    fig_t = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08)
    fig_t.add_trace(go.Scatter(x=t, y=disp, mode="lines", name="Displacement", line=dict(color=_C_FRONT)), row=1, col=1)
    fig_t.add_trace(go.Scatter(x=t, y=vel, mode="lines", name="Velocity", line=dict(color=_C_REAR)), row=2, col=1)
    fig_t.add_trace(go.Scatter(x=t, y=force_tot, mode="lines", name="Total Force", line=dict(color=_C_TIE)), row=3, col=1)
    if cmp_steps:
        fig_t.add_trace(go.Scatter(x=c_t, y=c_disp, mode="lines", name=f"{cmp_label} Displacement",
                                    line=dict(color=_C_CMP, dash="dash")), row=1, col=1)
        fig_t.add_trace(go.Scatter(x=c_t, y=c_vel, mode="lines", name=f"{cmp_label} Velocity",
                                    line=dict(color=_C_CMP, dash="dash")), row=2, col=1)
        fig_t.add_trace(go.Scatter(x=c_t, y=c_force_tot, mode="lines", name=f"{cmp_label} Total Force",
                                    line=dict(color=_C_CMP, dash="dash")), row=3, col=1)

    fig_t.update_yaxes(title_text="Disp [mm]", row=1, col=1)
    fig_t.update_yaxes(title_text="Vel [mm/s]", row=2, col=1)
    fig_t.update_yaxes(title_text="Force [N]", row=3, col=1)

    fig_t.update_layout(
        title="Time Series",
        xaxis_title="Time [s]",
        height=600,
        margin=dict(l=40, r=20, t=40, b=40),
        showlegend=bool(cmp_steps),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(240,240,240,0.5)'
    )
    named_figs.append(("Time Series", fig_t))

    return named_figs, t

def _kin_x_series(steps):
    if "x_val" in steps[0]:
        return [s["x_val"] for s in steps], steps[0].get("x_label", "Input")
    if "travel_mm" in steps[0]:
        return [s["travel_mm"] for s in steps], "Shock Travel [mm]"
    if "steer_mm" in steps[0]:
        return [s["steer_mm"] for s in steps], "Rack Travel [mm]"
    return list(range(len(steps))), "Step"

def _build_kin_figures(steps, half_label="Front", wr=0.0, sim_type=None,
                        cmp_steps=None, cmp_wr=0.0, cmp_label="Compare"):
    """cmp_steps: optional second dataset (e.g. a previous run) overlaid as a dashed trace
    on every applicable figure, for the Web UI's run-comparison feature."""
    atts = [get_wheel_attitude(s) for s in steps]
    plunge, a_ib, a_ob = [], [], []
    for s in steps:
        d = s.get("axle_data", {})
        plunge.append(d.get("plunge_mm", 0))
        a_ib.append(d.get("angle_ib_deg", 0))
        a_ob.append(d.get("angle_ob_deg", 0))

    xs, xl = _kin_x_series(steps)
    x0 = xs[0]
    motion_ratio = motion_ratio_series(steps, wr=wr)

    c_atts = c_plunge = c_a_ib = c_a_ob = c_mr = None
    c_xs = None
    if cmp_steps:
        c_atts = [get_wheel_attitude(s) for s in cmp_steps]
        c_plunge, c_a_ib, c_a_ob = [], [], []
        for s in cmp_steps:
            d = s.get("axle_data", {})
            c_plunge.append(d.get("plunge_mm", 0))
            c_a_ib.append(d.get("angle_ib_deg", 0))
            c_a_ob.append(d.get("angle_ob_deg", 0))
        c_xs, _ = _kin_x_series(cmp_steps)
        c_mr = motion_ratio_series(cmp_steps, wr=cmp_wr)

    def mfig(title, primary_name, y, extra_traces=None):
        traces = [go.Scatter(x=xs, y=y, mode="lines",
                              name=primary_name if cmp_steps else None,
                              line=dict(color=_COLORS[0], width=2))]
        if extra_traces:
            traces.extend(extra_traces)
        f = go.Figure(data=traces)
        f.update_layout(**_LAYOUT_BASE,
                        title=dict(text=title, font=dict(size=11)),
                        xaxis_title=xl,
                        showlegend=bool(cmp_steps) or (extra_traces is not None),
                        shapes=[_vline_shape(x0)])
        return f

    def cmp_trace(y, name=None, color=_C_CMP, dash="dash"):
        return go.Scatter(x=c_xs, y=y, mode="lines", name=name or cmp_label,
                           line=dict(color=color, width=2, dash=dash))

    figs = [
        ("camber", mfig("Camber [°]", half_label, [a["camber"] for a in atts],
            [cmp_trace([a["camber"] for a in c_atts])] if cmp_steps else None)),
        ("caster", mfig("Caster [°]", half_label, [a["caster"] for a in atts],
            [cmp_trace([a["caster"] for a in c_atts])] if cmp_steps else None)),
        ("toe",    mfig("Toe [°]", half_label, [a["toe"] for a in atts],
            [cmp_trace([a["toe"] for a in c_atts])] if cmp_steps else None)),
        ("motion_ratio", mfig(f"Motion Ratio — {half_label} [-]", half_label, motion_ratio.tolist(),
            [cmp_trace(c_mr.tolist())] if cmp_steps else None)),
    ]

    if sim_type != "sweep_space":
        figs.append(("plunge", mfig("Axle Plunge [mm]", half_label, plunge,
            [cmp_trace(c_plunge)] if cmp_steps else None)))

        cv_traces = [
            go.Scatter(x=xs, y=a_ib, mode="lines", name="Inboard",
                       line=dict(color=_COLORS[4], width=2)),
            go.Scatter(x=xs, y=a_ob, mode="lines", name="Outboard",
                       line=dict(color=_COLORS[4], width=2, dash="dash")),
        ]
        if cmp_steps:
            cv_traces.append(cmp_trace(c_a_ib, name=f"{cmp_label} Inboard", color=_C_CMP))
            cv_traces.append(cmp_trace(c_a_ob, name=f"{cmp_label} Outboard", color=_C_CMP2))
        cv_fig = go.Figure(data=cv_traces)
        cv_fig.update_layout(**_LAYOUT_BASE, title=dict(text="CV Angles [°]", font=dict(size=11)),
                              xaxis_title=xl, showlegend=True, shapes=[_vline_shape(x0)])
        figs.append(("cv", cv_fig))

    return figs, xs

def _build_sweep_space_figures(steps):
    """Build 3D surface plots (shock travel x rack travel) of CV plunge and joint angle."""
    travel_u = sorted({round(s.get("travel_mm", 0.0), 6) for s in steps})
    steer_u = sorted({round(s.get("steer_mm", 0.0), 6) for s in steps})
    t_idx = {v: i for i, v in enumerate(travel_u)}
    s_idx = {v: i for i, v in enumerate(steer_u)}

    plunge_grid = np.full((len(travel_u), len(steer_u)), np.nan)
    angle_grid = np.full((len(travel_u), len(steer_u)), np.nan)

    for s in steps:
        i = t_idx[round(s.get("travel_mm", 0.0), 6)]
        j = s_idx[round(s.get("steer_mm", 0.0), 6)]
        d = s.get("axle_data") or {}
        plunge_grid[i, j] = d.get("plunge_mm", np.nan)
        ib, ob = d.get("angle_ib_deg", np.nan), d.get("angle_ob_deg", np.nan)
        angle_grid[i, j] = max(ib, ob)

    def surf(title, z, colorscale):
        f = go.Figure(data=[go.Surface(x=steer_u, y=travel_u, z=z,
                                        colorscale=colorscale, showscale=False)])
        f.update_layout(
            title=dict(text=title, font=dict(size=11)),
            scene=dict(
                xaxis_title="Rack Travel [mm]",
                yaxis_title="Shock Travel [mm]",
                zaxis_title=title,
            ),
            margin=dict(l=6, r=6, t=30, b=6),
            height=420,
            template="plotly_white",
        )
        return f

    return [
        ("plunge_3d", surf("CV Plunge [mm]", plunge_grid, "Greens")),
        ("cv_3d", surf("CV Joint Angle [°]", angle_grid, "Tealgrn")),
    ]

def _build_kin_stats(steps, wr=0.0):
    """Static (0mm shock travel / 0mm steer) value cards for a corner kinematic sweep."""
    idx = static_ride_height_index(steps)
    static_step = steps[idx]
    att = get_wheel_attitude(static_step)
    mr  = motion_ratio_series(steps, wr=wr)[idx]
    stats = [
        ("Motion Ratio (Static) [-]", f"{mr:.3f}"),
        ("Camber (Static) [°]", f"{att['camber']:.2f}"),
        ("Caster (Static) [°]", f"{att['caster']:.2f}"),
        ("Toe (Static) [°]", f"{att['toe']:.2f}"),
    ]
    if "lbj" in static_step and "ubj" in static_step:
        sa = get_steering_axis_geometry(static_step, wr)
        stats.extend([
            ("Kingpin Angle [°]", f"{sa['kingpin_angle']:.2f}"),
            ("Caster Trail (Hub) [mm]", f"{sa['caster_trail']:.2f}"),
            ("Caster Offset (Ground) [mm]", f"{sa['caster_offset']:.2f}"),
            ("Kingpin Offset (Wheel Centre) [mm]", f"{sa['kingpin_offset_wc']:.2f}"),
            ("Kingpin Offset (Ground) [mm]", f"{sa['kingpin_offset_gnd']:.2f}"),
            ("Mechanical Trail (Ground) [mm]", f"{sa['mechanical_trail']:.2f}"),
        ])
    return stats

def _build_dyn_stats(steps, corner_wr=None):
    """After-settle (last frame) value cards for a dynamic drop / terrain run, per corner."""
    corner_wr = corner_wr or {}
    last = steps[-1]
    stats = []
    for key, label in [("fl", "FL"), ("fr", "FR"), ("rl", "RL"), ("rr", "RR")]:
        corner_steps = [s[key] for s in steps if s.get(key)]
        if not corner_steps or not last.get(key):
            continue
        mr  = motion_ratio_series(corner_steps, wr=corner_wr.get(key, 0.0))[-1]
        att = get_wheel_attitude(last[key])
        stats.append((f"{label} Motion Ratio (After Settle) [-]", f"{mr:.3f}"))
        stats.append((f"{label} Camber (After Settle) [°]", f"{att['camber']:.2f}"))
        stats.append((f"{label} Caster (After Settle) [°]", f"{att['caster']:.2f}"))
        stats.append((f"{label} Toe (After Settle) [°]", f"{att['toe']:.2f}"))
    return stats

def _build_front_steer_figures(steps, cmp_steps=None, cmp_label="Compare"):
    xs = [s["input"] for s in steps]
    ys = [s["ackermann_pct"] for s in steps]
    traces = [go.Scatter(x=xs, y=ys, mode="lines", name="Current" if cmp_steps else None,
                          line=dict(color=_COLORS[4], width=2))]
    if cmp_steps:
        c_xs = [s["input"] for s in cmp_steps]
        c_ys = [s["ackermann_pct"] for s in cmp_steps]
        traces.append(go.Scatter(x=c_xs, y=c_ys, mode="lines", name=cmp_label,
                                  line=dict(color=_C_CMP, width=2, dash="dash")))
    f = go.Figure(data=traces)
    f.add_hline(y=100, line_color="gray", line_dash="dash", opacity=0.4)
    f.update_layout(**{**_LAYOUT_BASE, "height": 280},
                    title="Ackermann %",
                    xaxis_title="Rack Travel [mm]",
                    yaxis_title="%",
                    showlegend=bool(cmp_steps),
                    shapes=[_vline_shape(xs[0])])

    def mfig(title, y, extra_traces=None, y_title=None):
        traces = [go.Scatter(x=xs, y=y, mode="lines", name="Current" if cmp_steps else None,
                              line=dict(color=_COLORS[0], width=2))]
        if extra_traces:
            traces.extend(extra_traces)
        fig = go.Figure(data=traces)
        fig.update_layout(**_LAYOUT_BASE,
                          title=dict(text=title, font=dict(size=11)),
                          xaxis_title="Rack Travel [mm]",
                          yaxis_title=y_title,
                          showlegend=bool(cmp_steps) or (extra_traces is not None),
                          shapes=[_vline_shape(xs[0])])
        return fig

    def cmp_trace(y, name=None, color=_C_CMP, dash="dash"):
        return go.Scatter(x=c_xs, y=y, mode="lines", name=name or cmp_label,
                           line=dict(color=color, width=2, dash=dash))

    track_change_ys = [s["track_change_mm"] for s in steps]
    if cmp_steps:
        c_track_change_ys = [s["track_change_mm"] for s in cmp_steps]

    track_change_fig = mfig("Track Change [mm]", track_change_ys,
        [cmp_trace(c_track_change_ys)] if cmp_steps else None)

    toe_traces = [
        go.Scatter(x=xs, y=[s["toe_l_deg"] for s in steps], mode="lines", name="Left",
                   line=dict(color=_COLORS[4], width=2)),
        go.Scatter(x=xs, y=[s["toe_r_deg"] for s in steps], mode="lines", name="Right",
                   line=dict(color=_COLORS[4], width=2, dash="dash")),
    ]
    if cmp_steps:
        toe_traces.append(cmp_trace([s["toe_l_deg"] for s in cmp_steps], name=f"{cmp_label} Left", color=_C_CMP))
        toe_traces.append(cmp_trace([s["toe_r_deg"] for s in cmp_steps], name=f"{cmp_label} Right", color=_C_CMP2))
    toe_fig = go.Figure(data=toe_traces)
    toe_fig.update_layout(**_LAYOUT_BASE, title=dict(text="Toe [°]", font=dict(size=11)),
                          xaxis_title="Rack Travel [mm]", showlegend=True, shapes=[_vline_shape(xs[0])])

    return [
        ("ackermann_pct", f),
        ("track_change", track_change_fig),
        ("toe", toe_fig),
    ], xs

def _build_full_vehicle_figures(steps, mode="heave", wr_front=0.0, wr_rear=0.0,
                                 cmp_steps=None, cmp_label="Compare"):
    """Camber/toe/caster/motion-ratio FL/FR/RL/RR overlays plus track-change, wheelbase-change,
    pitch-angle, roll-angle, and (front, where the corner geometry supports it) roll-center
    Y/Z, for a full-vehicle pitch or roll sweep."""
    xs = [s["input"] for s in steps]
    x0 = xs[0] if xs else 0.0
    x_label = "Shock Travel [mm]"

    _NO_ATT = {"camber": None, "caster": None, "toe": None}

    def _atts(steps_list, key):
        return [get_wheel_attitude(s[key]) if s.get(key) else _NO_ATT for s in steps_list]

    def _mr(steps_list, key, wr):
        return motion_ratio_series([s[key] for s in steps_list], wr=wr)

    corner_wr = {"fl": wr_front, "fr": wr_front, "rl": wr_rear, "rr": wr_rear}
    atts = {k: _atts(steps, k) for k in ("fl", "fr", "rl", "rr")}
    mr = {k: _mr(steps, k, corner_wr[k]) for k in ("fl", "fr", "rl", "rr")}

    c_xs = c_atts = c_mr = None
    if cmp_steps:
        c_xs = [s["input"] for s in cmp_steps]
        c_atts = {k: _atts(cmp_steps, k) for k in ("fl", "fr", "rl", "rr")}
        c_mr = {k: _mr(cmp_steps, k, corner_wr[k]) for k in ("fl", "fr", "rl", "rr")}

    def cmp_trace(y, name=None, color=_C_CMP, dash="dash"):
        return go.Scatter(x=c_xs, y=y, mode="lines", name=name or cmp_label,
                           line=dict(color=color, width=2, dash=dash))

    _CORNER_STYLE = [
        ("fl", _C_FRONT, "solid"), ("fr", _C_FRONT, "dot"),
        ("rl", _C_REAR, "solid"), ("rr", _C_REAR, "dot"),
    ]

    def corner_fig(title, y_by_corner, c_y_by_corner=None):
        traces = [go.Scatter(x=xs, y=y_by_corner[k], mode="lines", name=k.upper(),
                              line=dict(color=color, width=2, dash=dash))
                  for k, color, dash in _CORNER_STYLE]
        if cmp_steps:
            for k, color, dash in _CORNER_STYLE:
                traces.append(cmp_trace(c_y_by_corner[k], name=f"{cmp_label} {k.upper()}",
                                         color=_C_CMP if k in ("fl", "rl") else _C_CMP2, dash=dash))
        fig = go.Figure(data=traces)
        fig.update_layout(**_LAYOUT_BASE, title=dict(text=title, font=dict(size=11)),
                          xaxis_title=x_label, showlegend=True, shapes=[_vline_shape(x0)])
        return fig

    def mfig(title, y, c_y=None):
        traces = [go.Scatter(x=xs, y=y, mode="lines", name="Current" if cmp_steps else None,
                              line=dict(color=_COLORS[0], width=2))]
        if cmp_steps:
            traces.append(cmp_trace(c_y))
        fig = go.Figure(data=traces)
        fig.update_layout(**_LAYOUT_BASE, title=dict(text=title, font=dict(size=11)),
                          xaxis_title=x_label, showlegend=bool(cmp_steps), shapes=[_vline_shape(x0)])
        return fig

    def fr_fig(title, front_y, rear_y, c_front_y=None, c_rear_y=None):
        traces = [
            go.Scatter(x=xs, y=front_y, mode="lines", name="Front", line=dict(color=_COLORS[0], width=2)),
            go.Scatter(x=xs, y=rear_y, mode="lines", name="Rear", line=dict(color=_COLORS[0], width=2, dash="dash")),
        ]
        if cmp_steps:
            traces.append(cmp_trace(c_front_y, name=f"{cmp_label} Front", color=_C_CMP))
            traces.append(cmp_trace(c_rear_y, name=f"{cmp_label} Rear", color=_C_CMP2))
        fig = go.Figure(data=traces)
        fig.update_layout(**_LAYOUT_BASE, title=dict(text=title, font=dict(size=11)),
                          xaxis_title=x_label, showlegend=True, shapes=[_vline_shape(x0)])
        return fig

    figs = [
        ("camber", corner_fig("Camber [°]", {k: [a["camber"] for a in atts[k]] for k in atts},
            {k: [a["camber"] for a in c_atts[k]] for k in c_atts} if cmp_steps else None)),
        ("toe", corner_fig("Toe [°]", {k: [a["toe"] for a in atts[k]] for k in atts},
            {k: [a["toe"] for a in c_atts[k]] for k in c_atts} if cmp_steps else None)),
        ("caster", corner_fig("Caster [°]", {k: [a["caster"] for a in atts[k]] for k in atts},
            {k: [a["caster"] for a in c_atts[k]] for k in c_atts} if cmp_steps else None)),
        ("motion_ratio", corner_fig("Motion Ratio [-]", {k: mr[k].tolist() for k in mr},
            {k: c_mr[k].tolist() for k in c_mr} if cmp_steps else None)),
        ("track_change", fr_fig("Track Change [mm]",
            [s["front_track_change_mm"] for s in steps], [s["rear_track_change_mm"] for s in steps],
            [s["front_track_change_mm"] for s in cmp_steps] if cmp_steps else None,
            [s["rear_track_change_mm"] for s in cmp_steps] if cmp_steps else None)),
        ("wheelbase_change", mfig("Wheelbase Change [mm]", [s["wheelbase_change_mm"] for s in steps],
            [s["wheelbase_change_mm"] for s in cmp_steps] if cmp_steps else None)),
        ("pitch_angle", mfig("Pitch Angle [°]", [s["pitch_angle_deg"] for s in steps],
            [s["pitch_angle_deg"] for s in cmp_steps] if cmp_steps else None)),
        ("roll_angle", fr_fig("Roll Angle [°]",
            [s["front_roll_angle_deg"] for s in steps], [s["rear_roll_angle_deg"] for s in steps],
            [s["front_roll_angle_deg"] for s in cmp_steps] if cmp_steps else None,
            [s["rear_roll_angle_deg"] for s in cmp_steps] if cmp_steps else None)),
    ]

    if any(s.get("front_roll_center_z_mm") is not None or s.get("rear_roll_center_z_mm") is not None for s in steps):
        figs.append(("roll_center_z", fr_fig("Roll Center Height [mm]",
            [s["front_roll_center_z_mm"] for s in steps], [s["rear_roll_center_z_mm"] for s in steps],
            [s["front_roll_center_z_mm"] for s in cmp_steps] if cmp_steps else None,
            [s["rear_roll_center_z_mm"] for s in cmp_steps] if cmp_steps else None)))
    if any(s.get("front_ground_clearance_mm") is not None or s.get("rear_ground_clearance_mm") is not None for s in steps):
        gc_fig = fr_fig("Ground Clearance [mm]",
            [s.get("front_ground_clearance_mm") for s in steps],
            [s.get("rear_ground_clearance_mm") for s in steps],
            [s.get("front_ground_clearance_mm") for s in cmp_steps] if cmp_steps else None,
            [s.get("rear_ground_clearance_mm") for s in cmp_steps] if cmp_steps else None)
        gc_fig.add_hline(y=0, line_color="red", line_dash="dash", opacity=0.5)
        figs.append(("ground_clearance", gc_fig))
    if any(s.get("chassis_ground_angle_deg") is not None for s in steps):
        figs.append(("chassis_ground_angle", mfig("Chassis–Ground Plane Angle [°]",
            [s.get("chassis_ground_angle_deg") for s in steps],
            [s.get("chassis_ground_angle_deg") for s in cmp_steps] if cmp_steps else None)))

    if any(s.get("front_roll_center_y_mm") is not None or s.get("rear_roll_center_y_mm") is not None for s in steps):
        figs.append(("roll_center_y", fr_fig("Roll Center Lateral Position [mm]",
            [s["front_roll_center_y_mm"] for s in steps], [s["rear_roll_center_y_mm"] for s in steps],
            [s["front_roll_center_y_mm"] for s in cmp_steps] if cmp_steps else None,
            [s["rear_roll_center_y_mm"] for s in cmp_steps] if cmp_steps else None)))

    return figs, xs

def rank_solutions(F: np.ndarray) -> np.ndarray:
    """Return indices into F sorted best-first, by normalized Euclidean distance
    to the ideal point (0,...,0) across all objectives. Works for any n_obj >= 1."""
    Fn = (F - F.min(0)) / (np.ptp(F, 0) + 1e-9)
    scores = np.linalg.norm(Fn, axis=1)
    return np.argsort(scores)

def _pareto_hull_2d(F_front: np.ndarray):
    """Smooth filled curve tracing the sorted Pareto front (2 objectives)."""
    if len(F_front) < 3:
        return None
    order = np.argsort(F_front[:, 0])
    x, y = F_front[order, 0], F_front[order, 1]
    return go.Scatter(x=x, y=y, mode="lines",
                       line=dict(color=_COLORS[0], width=2, shape="spline", smoothing=1.0),
                       fill="tozeroy", fillcolor="rgba(5,150,105,0.10)",
                       name="Pareto Hull", hoverinfo="skip")

def _pareto_hull_3d(F_front: np.ndarray):
    """Smooth continuous surface interpolated over the Pareto front points,
    clipped to the convex hull of their (x, y) projection (3 objectives)."""
    from scipy.interpolate import griddata
    from scipy.spatial import Delaunay, QhullError

    if len(F_front) < 4:
        return None
    x, y, z = F_front[:, 0], F_front[:, 1], F_front[:, 2]
    if np.ptp(x) < 1e-12 or np.ptp(y) < 1e-12:
        return None

    res = 40
    xi = np.linspace(x.min(), x.max(), res)
    yi = np.linspace(y.min(), y.max(), res)
    XI, YI = np.meshgrid(xi, yi)

    try:
        ZI = griddata((x, y), z, (XI, YI), method="cubic")
    except Exception:
        ZI = griddata((x, y), z, (XI, YI), method="linear")

    try:
        hull = Delaunay(np.column_stack([x, y]))
        outside = hull.find_simplex(np.column_stack([XI.ravel(), YI.ravel()])) < 0
        ZI = ZI.ravel()
        ZI[outside] = np.nan
        ZI = ZI.reshape(XI.shape)
    except QhullError:
        pass

    if np.all(np.isnan(ZI)):
        return None

    return go.Surface(x=XI, y=YI, z=ZI, opacity=0.35, colorscale="Greens",
                       showscale=False, name="Pareto Hull", hoverinfo="skip")

def _build_opt_figures(F_all, F_front, obj_names):
    """F_all: every design evaluated during the run. F_front: the final
    non-dominated Pareto front (subset of F_all)."""
    n = F_front.shape[1]
    if n == 1:
        f = go.Figure([go.Histogram(x=F_all.flatten(), nbinsx=25,
                                     marker_color=_COLORS[0], opacity=0.85)])
        f.add_vline(x=float(F_front.min()), line_color="red", line_dash="dash",
                    annotation_text=f"Best: {F_front.min():.4f}")
        f.update_layout(**{**_LAYOUT_BASE, "height": 340},
                         title=f"Objective: {obj_names[0]} ({len(F_all)} evaluated)",
                         xaxis_title="Cost", yaxis_title="Count")
    elif n == 2:
        Fn   = (F_front - F_front.min(0)) / (np.ptp(F_front, 0) + 1e-9)
        best = int(np.argmin(np.linalg.norm(Fn, axis=1)))
        traces = []
        hull = _pareto_hull_2d(F_front)
        if hull is not None:
            traces.append(hull)
        traces.extend([
            go.Scatter(x=F_all[:,0], y=F_all[:,1], mode="markers",
                       marker=dict(size=5, color="#9ca3af", opacity=0.45),
                       name=f"All Evaluated ({len(F_all)})"),
            go.Scatter(x=F_front[:,0], y=F_front[:,1], mode="markers",
                       marker=dict(size=8, color=_COLORS[0], opacity=0.9,
                                   line=dict(width=1, color="white")),
                       name="Pareto Front"),
            go.Scatter(x=[F_front[best,0]], y=[F_front[best,1]], mode="markers",
                       marker=dict(size=14, symbol="star", color="red"),
                       name="Best balance"),
        ])
        f = go.Figure(traces)
        f.update_layout(**{**_LAYOUT_BASE, "height": 380},
                         title="Pareto Front",
                         xaxis_title=obj_names[0], yaxis_title=obj_names[1],
                         showlegend=True)
    else:
        traces = []
        surface = _pareto_hull_3d(F_front)
        if surface is not None:
            traces.append(surface)
        traces.append(go.Scatter3d(
            x=F_all[:,0], y=F_all[:,1], z=F_all[:,2], mode="markers",
            marker=dict(size=3, color="#9ca3af", opacity=0.35),
            name=f"All Evaluated ({len(F_all)})",
        ))
        traces.append(go.Scatter3d(
            x=F_front[:,0], y=F_front[:,1], z=F_front[:,2], mode="markers",
            marker=dict(size=5, color=F_front[:,2], colorscale="Viridis",
                        showscale=True,
                        colorbar=dict(title=obj_names[2], thickness=12),
                        line=dict(width=1, color="white")),
            name="Pareto Front",
        ))
        f = go.Figure(traces)
        f.update_layout(height=480, title="Pareto Front (3D)",
                         margin=dict(l=0, r=0, t=40, b=0),
                         showlegend=True,
                         scene=dict(xaxis_title=obj_names[0],
                                    yaxis_title=obj_names[1],
                                    zaxis_title=obj_names[2]))
    return [("pareto", f)]

def _move_vline(fig: go.Figure, x_val: float) -> None:
    """Move the vertical line to the new x value."""    
    if fig.layout.shapes:
        fig.layout.shapes[0].x0 = x_val
        fig.layout.shapes[0].x1 = x_val

def _build_dyn_figures(steps: list[dict[str, Any]], cmp_steps: list[dict[str, Any]] | None = None,
                        cmp_label: str = "Compare"):
    """Build figures for a Static Drop run. cmp_steps: optional second run overlaid dashed."""
    xs = [s["t"] for s in steps]
    x0 = xs[0] if xs else 0.0

    z_cog = [s["cog_pos"][2] for s in steps]
    roll = [np.degrees(s["phi"]) for s in steps]
    pitch = [np.degrees(s["theta"]) for s in steps]

    sl_fl = [s["fl"]["shock_length"] if s["fl"] else None for s in steps]
    sl_fr = [s["fr"]["shock_length"] if s["fr"] else None for s in steps]
    sl_rl = [s["rl"]["shock_length"] if s["rl"] else None for s in steps]
    sl_rr = [s["rr"]["shock_length"] if s["rr"] else None for s in steps]

    c_xs = c_z_cog = c_roll = c_pitch = None
    c_sl_fl = c_sl_fr = c_sl_rl = c_sl_rr = None
    if cmp_steps:
        c_xs = [s["t"] for s in cmp_steps]
        c_z_cog = [s["cog_pos"][2] for s in cmp_steps]
        c_roll  = [np.degrees(s["phi"]) for s in cmp_steps]
        c_pitch = [np.degrees(s["theta"]) for s in cmp_steps]
        c_sl_fl = [s["fl"]["shock_length"] if s["fl"] else None for s in cmp_steps]
        c_sl_fr = [s["fr"]["shock_length"] if s["fr"] else None for s in cmp_steps]
        c_sl_rl = [s["rl"]["shock_length"] if s["rl"] else None for s in cmp_steps]
        c_sl_rr = [s["rr"]["shock_length"] if s["rr"] else None for s in cmp_steps]

    def mfig(title, traces, show_legend=False):
        f = go.Figure(data=traces)
        f.update_layout(**_LAYOUT_BASE,
                        title=dict(text=title, font=dict(size=11)),
                        xaxis_title="Time [s]",
                        showlegend=show_legend,
                        shapes=[_vline_shape(x0)])
        return f

    def _attitude(steps_list, key, field):
        out = []
        for s in steps_list:
            corner = s.get(key)
            out.append(get_wheel_attitude(corner)[field] if corner else None)
        return out

    def _lr_fig(title, field, l_key, r_key, color):
        traces = [
            go.Scatter(x=xs, y=_attitude(steps, l_key, field), mode="lines", name=l_key.upper(),
                       line=dict(color=color, width=1.5)),
            go.Scatter(x=xs, y=_attitude(steps, r_key, field), mode="lines", name=r_key.upper(),
                       line=dict(color=color, width=1.5, dash="dot")),
        ]
        if cmp_steps:
            traces.append(go.Scatter(x=c_xs, y=_attitude(cmp_steps, l_key, field), mode="lines",
                                      name=f"{cmp_label} {l_key.upper()}",
                                      line=dict(color=_C_CMP, width=1.5)))
            traces.append(go.Scatter(x=c_xs, y=_attitude(cmp_steps, r_key, field), mode="lines",
                                      name=f"{cmp_label} {r_key.upper()}",
                                      line=dict(color=_C_CMP2, width=1.5, dash="dot")))
        return mfig(title, traces, show_legend=True)

    cog_z_traces = [go.Scatter(x=xs, y=z_cog, mode="lines", name="Current" if cmp_steps else None,
                                line=dict(color=_COLORS[0], width=2))]
    roll_traces  = [go.Scatter(x=xs, y=roll, mode="lines", name="Current" if cmp_steps else None,
                                line=dict(color=_COLORS[1], width=2))]
    pitch_traces = [go.Scatter(x=xs, y=pitch, mode="lines", name="Current" if cmp_steps else None,
                                line=dict(color=_COLORS[2], width=2))]
    shock_traces = [
        go.Scatter(x=xs, y=sl_fl, mode="lines", name="FL", line=dict(color=_COLORS[0], width=1.5)),
        go.Scatter(x=xs, y=sl_fr, mode="lines", name="FR", line=dict(color=_COLORS[1], width=1.5)),
        go.Scatter(x=xs, y=sl_rl, mode="lines", name="RL", line=dict(color=_COLORS[2], width=1.5, dash="dash")),
        go.Scatter(x=xs, y=sl_rr, mode="lines", name="RR", line=dict(color=_COLORS[3], width=1.5, dash="dash")),
    ]
    if cmp_steps:
        cog_z_traces.append(go.Scatter(x=c_xs, y=c_z_cog, mode="lines", name=cmp_label,
                                        line=dict(color=_C_CMP, width=2, dash="dash")))
        roll_traces.append(go.Scatter(x=c_xs, y=c_roll, mode="lines", name=cmp_label,
                                       line=dict(color=_C_CMP, width=2, dash="dash")))
        pitch_traces.append(go.Scatter(x=c_xs, y=c_pitch, mode="lines", name=cmp_label,
                                        line=dict(color=_C_CMP, width=2, dash="dash")))
        shock_traces.extend([
            go.Scatter(x=c_xs, y=c_sl_fl, mode="lines", name=f"{cmp_label} FL", line=dict(color=_C_CMP, width=1.5)),
            go.Scatter(x=c_xs, y=c_sl_fr, mode="lines", name=f"{cmp_label} FR", line=dict(color=_C_CMP2, width=1.5)),
            go.Scatter(x=c_xs, y=c_sl_rl, mode="lines", name=f"{cmp_label} RL", line=dict(color=_C_CMP, width=1.5, dash="dash")),
            go.Scatter(x=c_xs, y=c_sl_rr, mode="lines", name=f"{cmp_label} RR", line=dict(color=_C_CMP2, width=1.5, dash="dash")),
        ])

    return [
        ("cog_z", mfig("CoG Z [mm]", cog_z_traces, show_legend=bool(cmp_steps))),
        ("cog_roll", mfig("Roll [°]", roll_traces, show_legend=bool(cmp_steps))),
        ("cog_pitch", mfig("Pitch [°]", pitch_traces, show_legend=bool(cmp_steps))),
        ("shock_lengths", mfig("Shock Lengths [mm]", shock_traces, show_legend=True)),
        ("camber_front", _lr_fig("Front Camber [°]", "camber", "fl", "fr", _C_FRONT)),
        ("caster_front", _lr_fig("Front Caster [°]", "caster", "fl", "fr", _C_FRONT)),
        ("toe_front",    _lr_fig("Front Toe [°]",    "toe",    "fl", "fr", _C_FRONT)),
        ("camber_rear",  _lr_fig("Rear Camber [°]",  "camber", "rl", "rr", _C_REAR)),
        ("caster_rear",  _lr_fig("Rear Caster [°]",  "caster", "rl", "rr", _C_REAR)),
        ("toe_rear",     _lr_fig("Rear Toe [°]",     "toe",    "rl", "rr", _C_REAR)),
    ], xs
