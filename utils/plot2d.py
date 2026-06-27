# default
from typing import Any

# third-party
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ours
from utils.geometry import get_wheel_attitude

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

def _vline_shape(x: float) -> dict[str, Any]:
    """Return a vertical line shape for plotly."""
    return dict(type="line", x0=x, x1=x, y0=0, y1=1,
                xref="x", yref="paper",
                line=dict(color="rgba(16,185,129,0.8)", width=1.5, dash="dot"))

def _build_dyno_figures(steps: list) -> tuple[list[tuple[str, Any]], list[float]]:
    """Build figures for a Shock Dyno test."""
    if not steps:
        return [], []
        
    t = [s["t"] for s in steps]
    vel = [s["velocity"] for s in steps]
    disp = [s["displacement"] for s in steps]
    force_tot = [s["force_total"] for s in steps]
    force_spr = [s["force_spring"] for s in steps]
    force_damp = [s["force_damper"] for s in steps]

    named_figs = []

    # Force vs Velocity (The classic dyno plot)
    fig_fv = make_subplots(rows=1, cols=1)
    fig_fv.add_trace(go.Scatter(x=vel, y=force_damp, mode="lines", name="Damping Force", line=dict(color=_C_FRONT, width=2)))
    fig_fv.add_trace(go.Scatter(x=vel, y=force_tot, mode="lines", name="Total Force", line=dict(color=_C_REAR, width=2, dash="dash")))
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
    
    fig_t.update_yaxes(title_text="Disp [mm]", row=1, col=1)
    fig_t.update_yaxes(title_text="Vel [mm/s]", row=2, col=1)
    fig_t.update_yaxes(title_text="Force [N]", row=3, col=1)
    
    fig_t.update_layout(
        title="Time Series",
        xaxis_title="Time [s]",
        height=600,
        margin=dict(l=40, r=20, t=40, b=40),
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(240,240,240,0.5)'
    )
    named_figs.append(("Time Series", fig_t))

    return named_figs, t

def _build_kin_figures(steps):
    atts = [get_wheel_attitude(s) for s in steps]
    plunge, a_ib, a_ob = [], [], []
    for s in steps:
        d = s.get("axle_data", {})
        plunge.append(d.get("plunge_mm", 0))
        a_ib.append(d.get("angle_ib_deg", 0))
        a_ob.append(d.get("angle_ob_deg", 0))

    if "x_val" in steps[0]:
        xs, xl = [s["x_val"] for s in steps], steps[0].get("x_label", "Input")
    elif "travel_mm" in steps[0]:
        xs, xl = [s["travel_mm"] for s in steps], "Shock Travel [mm]"
    elif "steer_mm" in steps[0]:
        xs, xl = [s["steer_mm"] for s in steps], "Rack Travel [mm]"
    else:
        xs, xl = list(range(len(steps))), "Step"

    x0 = xs[0]

    def mfig(title, traces, show_legend=False):
        f = go.Figure(data=traces)
        f.update_layout(**_LAYOUT_BASE,
                        title=dict(text=title, font=dict(size=11)),
                        xaxis_title=xl,
                        showlegend=show_legend,
                        shapes=[_vline_shape(x0)])
        return f

    return [
        ("camber", mfig("Camber [°]",
            [go.Scatter(x=xs, y=[a["camber"] for a in atts],
                        mode="lines", line=dict(color=_COLORS[0], width=2))])),
        ("caster", mfig("Caster [°]",
            [go.Scatter(x=xs, y=[a["caster"] for a in atts],
                        mode="lines", line=dict(color=_COLORS[1], width=2))])),
        ("toe",    mfig("Toe [°]",
            [go.Scatter(x=xs, y=[a["toe"] for a in atts],
                        mode="lines", line=dict(color=_COLORS[2], width=2))])),
        ("plunge", mfig("Axle Plunge [mm]",
            [go.Scatter(x=xs, y=plunge,
                        mode="lines", line=dict(color=_COLORS[3], width=2))])),
        ("cv",     mfig("CV Angles [°]", [
            go.Scatter(x=xs, y=a_ib, mode="lines", name="Inboard",
                       line=dict(color=_COLORS[4], width=2)),
            go.Scatter(x=xs, y=a_ob, mode="lines", name="Outboard",
                       line=dict(color=_COLORS[4], width=2, dash="dash")),
        ], show_legend=True)),
    ], xs

def _build_ackermann_figures(steps):
    xs = [s["input"] for s in steps]
    ys = [s["ackermann_pct"] for s in steps]
    f  = go.Figure(data=[go.Scatter(x=xs, y=ys, mode="lines",
                                     line=dict(color=_COLORS[4], width=2))])
    f.add_hline(y=100, line_color="gray", line_dash="dash", opacity=0.4)
    f.update_layout(**{**_LAYOUT_BASE, "height": 280},
                    title="Ackermann %",
                    xaxis_title="Rack Travel [mm]",
                    yaxis_title="%",
                    shapes=[_vline_shape(xs[0])])
    return [("ackermann_pct", f)], xs

def _build_opt_figures(F, obj_names):
    n = F.shape[1]
    if n == 1:
        f = go.Figure([go.Histogram(x=F.flatten(), nbinsx=25,
                                     marker_color=_COLORS[0], opacity=0.85)])
        f.add_vline(x=float(F.min()), line_color="red", line_dash="dash",
                    annotation_text=f"Best: {F.min():.4f}")
        f.update_layout(**{**_LAYOUT_BASE, "height": 340},
                         title=f"Objective: {obj_names[0]}",
                         xaxis_title="Cost", yaxis_title="Count")
    elif n == 2:
        Fn   = (F - F.min(0)) / (np.ptp(F, 0) + 1e-9)
        best = int(np.argmin(np.linalg.norm(Fn, axis=1)))
        f = go.Figure([
            go.Scatter(x=F[:,0], y=F[:,1], mode="markers",
                       marker=dict(size=8, color=_COLORS[0], opacity=0.75,
                                   line=dict(width=1, color="white")),
                       name="Pareto"),
            go.Scatter(x=[F[best,0]], y=[F[best,1]], mode="markers",
                       marker=dict(size=14, symbol="star", color="red"),
                       name="Best balance"),
        ])
        f.update_layout(**{**_LAYOUT_BASE, "height": 380},
                         title="Pareto Front",
                         xaxis_title=obj_names[0], yaxis_title=obj_names[1],
                         showlegend=True)
    else:
        f = go.Figure([go.Scatter3d(
            x=F[:,0], y=F[:,1], z=F[:,2], mode="markers",
            marker=dict(size=5, color=F[:,2], colorscale="Viridis",
                        showscale=True,
                        colorbar=dict(title=obj_names[2], thickness=12))
        )])
        f.update_layout(height=480, title="Pareto Front (3D)",
                         margin=dict(l=0, r=0, t=40, b=0),
                         scene=dict(xaxis_title=obj_names[0],
                                    yaxis_title=obj_names[1],
                                    zaxis_title=obj_names[2]))
    return [("pareto", f)]

def _move_vline(fig: go.Figure, x_val: float) -> None:
    """Move the vertical line to the new x value."""    
    if fig.layout.shapes:
        fig.layout.shapes[0].x0 = x_val
        fig.layout.shapes[0].x1 = x_val

def _build_dyn_figures(steps: list[dict[str, Any]]):
    """Build figures for a Shock Dyno test."""  
    xs = [s["t"] for s in steps]
    x0 = xs[0] if xs else 0.0

    z_cog = [s["cog_pos"][2] for s in steps]
    roll = [np.degrees(s["phi"]) for s in steps]
    pitch = [np.degrees(s["theta"]) for s in steps]

    sl_fl = [s["fl"]["shock_length"] if s["fl"] else None for s in steps]
    sl_fr = [s["fr"]["shock_length"] if s["fr"] else None for s in steps]
    sl_rl = [s["rl"]["shock_length"] if s["rl"] else None for s in steps]
    sl_rr = [s["rr"]["shock_length"] if s["rr"] else None for s in steps]

    def mfig(title, traces, show_legend=False):
        f = go.Figure(data=traces)
        f.update_layout(**_LAYOUT_BASE,
                        title=dict(text=title, font=dict(size=11)),
                        xaxis_title="Time [s]",
                        showlegend=show_legend,
                        shapes=[_vline_shape(x0)])
        return f

    return [
        ("cog_z", mfig("CoG Z [mm]", [go.Scatter(x=xs, y=z_cog, mode="lines", line=dict(color=_COLORS[0], width=2))])),
        ("cog_roll", mfig("Roll [°]", [go.Scatter(x=xs, y=roll, mode="lines", line=dict(color=_COLORS[1], width=2))])),
        ("cog_pitch", mfig("Pitch [°]", [go.Scatter(x=xs, y=pitch, mode="lines", line=dict(color=_COLORS[2], width=2))])),
        ("shock_lengths", mfig("Shock Lengths [mm]", [
            go.Scatter(x=xs, y=sl_fl, mode="lines", name="FL", line=dict(color=_COLORS[0], width=1.5)),
            go.Scatter(x=xs, y=sl_fr, mode="lines", name="FR", line=dict(color=_COLORS[1], width=1.5)),
            go.Scatter(x=xs, y=sl_rl, mode="lines", name="RL", line=dict(color=_COLORS[2], width=1.5, dash="dash")),
            go.Scatter(x=xs, y=sl_rr, mode="lines", name="RR", line=dict(color=_COLORS[3], width=1.5, dash="dash")),
        ], show_legend=True)),
    ], xs
