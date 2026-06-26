import numpy as np
import plotly.graph_objects as go
from utils.geometry import get_wheel_attitude

_LAYOUT_BASE = dict(
    margin=dict(l=44, r=8, t=30, b=32),
    height=200,
    template="plotly_white",
    font=dict(size=10),
    plot_bgcolor="#fafafa",
)

_COLORS = ["#059669", "#65a30d", "#14b8a6", "#047857", "#10b981", "#84cc16"]


def _vline_shape(x):
    return dict(type="line", x0=x, x1=x, y0=0, y1=1,
                xref="x", yref="paper",
                line=dict(color="rgba(16,185,129,0.8)", width=1.5, dash="dot"))


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


def _move_vline(fig, x_val):
    if fig.layout.shapes:
        fig.layout.shapes[0].x0 = x_val
        fig.layout.shapes[0].x1 = x_val
