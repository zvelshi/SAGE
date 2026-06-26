import asyncio
import traceback
import yaml
import numpy as np
from nicegui import ui, run

from utils.sim_runners import _run_kin, _run_opt
from utils.scene3d import _build_scene, _update_scene, _fit_camera
from utils.plot2d import _build_kin_figures, _build_ackermann_figures, _build_opt_figures, _move_vline

# ── config paths ───────────────────────────────────────────────────────────────
SCRUB_FPS = 48   # steps per second during auto-play

KIN_PATH = "config/kin_config.yml"
DYN_PATH = "config/dyn_config.yml"
OPT_PATH = "config/opt_config.yml"

TAB_PATH  = {"kin": KIN_PATH, "dyn": DYN_PATH, "opt": OPT_PATH}
SIM_TYPES = {
    "kin": ["travel", "steer", "droop_steer", "jounce_steer", "extreme", "ackermann"],
    "dyn": ["terrain"],
    "opt": ["run"],
}


# ══════════════════════════════════════════════════════════════════════════════
# PAGE
# ══════════════════════════════════════════════════════════════════════════════

@ui.page("/")
def main_page():
    editors    = {}
    mode_ref   = {"v": "kin"}
    type_ref   = {"v": "travel"}
    active_tab = {"v": "kin"}
    rtask      = {"task": None}

    # sim result cache
    cache = {
        "steps":      [],
        "xs":         [],
        "sim_type":   None,
        "vehicle":    None,
        "hp":         None,
        "named_figs": [],
        "plot_elems": [],
        "scene_objs": None,
    }
    scrub = {"dirty": False, "idx": 0, "playing": False, "last_t": 0.0}

    # ── styles ─────────────────────────────────────────────────────────────────
    ui.add_head_html("""<style>
        body,html{margin:0;padding:0;height:100vh;overflow:hidden}
        .nicegui-content{padding:0!important;height:100vh!important;overflow:hidden!important}
        .cm-editor{height:100%!important}.cm-scroller{overflow:auto!important}
    </style>""")

    # ── outer split ────────────────────────────────────────────────────────────
    with ui.row().classes("w-full gap-0").style("height:100vh;overflow:hidden"):

        # ══ LEFT ═══════════════════════════════════════════════════════════════
        with ui.column().classes("border-r border-gray-300").style(
            "width:400px;min-width:400px;height:100vh;display:flex;"
            "flex-direction:column;overflow:hidden"
        ):
            with ui.row().classes("items-center gap-2 px-3 py-2 bg-gray-900").style("flex-shrink:0"):
                ui.label("SAGE").classes("text-white font-bold tracking-widest")
                ui.label("Suspension Analysis & Geometry Engine").classes("text-gray-400 text-xs")

            with ui.tabs().classes("w-full bg-gray-100 border-b border-gray-200").props("dense no-caps") as tabs:
                ui.tab("kin", label="Kinematic")
                ui.tab("dyn", label="Dynamic")
                ui.tab("opt", label="Optimizer")

            with ui.tab_panels(tabs, value="kin").style(
                "flex:1;min-height:0;display:flex;flex-direction:column;overflow:hidden"
            ) as panels:
                for tab_name, path in TAB_PATH.items():
                    with ui.tab_panel(tab_name).style(
                        "height:100%;padding:0;display:flex;flex-direction:column;overflow:hidden"
                    ):
                        ed = ui.codemirror(value=open(path).read(), language="yaml",
                                           theme="githubLight"
                                           ).style("flex:1;min-height:0;font-size:12px")
                        editors[tab_name] = ed

            panels.on_value_change(lambda e: active_tab.update({"v": e.value}))

            with ui.row().classes("w-full items-center gap-2 px-2 py-2 border-t border-gray-200 bg-gray-50").style("flex-shrink:0"):
                mode_sel    = ui.select(list(SIM_TYPES), value="kin",  label="Mode").classes("w-20").props("dense outlined")
                subtype_sel = ui.select(SIM_TYPES["kin"], value=SIM_TYPES["kin"][0], label="Type").classes("w-36").props("dense outlined")
                save_btn = ui.button("Save").props("unelevated dense").classes("bg-gray-700 text-white text-sm px-3")
                run_btn  = ui.button("Run" ).props("unelevated dense").classes("bg-blue-600 text-white text-sm px-3")

        # ══ RIGHT ══════════════════════════════════════════════════════════════
        with ui.column().style(
            "flex:1;height:100vh;display:flex;flex-direction:column;overflow:hidden;background:#f8fafc"
        ):
            # status bar
            with ui.row().classes("items-center gap-3 px-4 pt-2 pb-1").style("flex-shrink:0"):
                spinner  = ui.spinner("dots", size="sm", color="blue")
                spinner.visible = False
                status_lbl = ui.label("Ready — configure and press Run.").classes("text-gray-500 text-sm italic")
                progress = ui.linear_progress(value=0).classes("flex-1 ml-2").props("instant-feedback rounded")
                progress.visible = False

            # scrollable viz body
            with ui.column().classes("w-full px-4 pb-2 gap-3").style("flex:1;overflow-y:auto;min-height:0") as viz_area:
                pass

            # playback footer
            with ui.row().classes("w-full items-center gap-3 px-4 py-2 border-t border-gray-200 bg-white").style("flex-shrink:0"):
                play_btn = ui.button(icon="play_arrow").props("round dense unelevated color=blue")
                step_lbl = ui.label("—").classes("text-xs text-gray-500 font-mono")
                play_btn.visible = False

    # ── callbacks ──────────────────────────────────────────────────────────────
    def on_mode_change(e):
        mode_ref["v"] = e.value
        opts = SIM_TYPES[e.value]
        subtype_sel.options = opts
        subtype_sel.value   = opts[0]
        type_ref["v"]       = opts[0]
        tabs.set_value(e.value)
        active_tab["v"] = e.value

    def on_type_change(e):
        type_ref["v"] = e.value

    def do_save():
        tab, path = active_tab["v"], TAB_PATH[active_tab["v"]]
        text = editors[tab].value
        try:
            yaml.safe_load(text)
        except yaml.YAMLError as exc:
            ui.notify(f"YAML error: {exc}", type="negative"); return
        open(path, "w").write(text)
        ui.notify(f"Saved → {path}", type="positive", position="bottom-right")

    async def do_run():
        mode, sim_type = mode_ref["v"], type_ref["v"]
        rtask["task"] = asyncio.current_task()

        run_btn.disable(); save_btn.disable()
        spinner.visible  = True
        play_btn.visible = False
        progress.visible = True
        progress.set_value(0)
        status_lbl.text  = f"Running {mode} / {sim_type}…"
        viz_area.clear()
        cache["named_figs"].clear(); cache["plot_elems"].clear()
        cache["steps"] = []; cache["scene_objs"] = None

        try:
            if mode == "kin":
                result = await run.io_bound(_run_kin, editors["kin"].value, sim_type)
                progress.set_value(0.85)
                await asyncio.sleep(0)
                _render_kin(result)

            elif mode == "opt":
                result = await run.io_bound(_run_opt, editors["kin"].value, editors["opt"].value)
                progress.set_value(0.85)
                await asyncio.sleep(0)
                _render_opt(result)

            elif mode == "dyn":
                with viz_area:
                    ui.label("Dynamic simulation not yet supported in web UI.").classes("text-orange-500 text-sm")

            progress.set_value(1.0)
            status_lbl.text = f"Done — {mode} / {sim_type}  ({len(cache['steps'])} steps)"

        except asyncio.CancelledError:
            status_lbl.text = "Stopped."
            ui.notify("Simulation stopped.", type="warning", position="bottom-right")
        except Exception as exc:
            status_lbl.text = f"Error: {exc}"
            ui.notify(str(exc), type="negative", timeout=10_000)
            with viz_area:
                ui.label(traceback.format_exc()).classes("text-red-500 text-xs font-mono whitespace-pre-wrap")
        finally:
            rtask["task"]    = None
            spinner.visible  = False
            progress.visible = False
            run_btn.enable(); save_btn.enable()

    def do_play_pause():
        scrub["playing"] = not scrub["playing"]
        play_btn.props(f'icon={"pause" if scrub["playing"] else "play_arrow"}')


    async def _apply_scrub():
        import time
        steps = cache["steps"]
        if not steps:
            return

        # auto-advance when playing
        if scrub["playing"]:
            now = time.monotonic()
            if now - scrub["last_t"] >= 1.0 / SCRUB_FPS:
                scrub["last_t"] = now
                nxt = scrub["idx"] + 1
                if nxt >= len(steps):
                    nxt = 0
                scrub["idx"] = nxt
                scrub["dirty"] = True

        if not scrub["dirty"]:
            return
        scrub["dirty"] = False

        idx = scrub["idx"]
        xs  = cache["xs"]
        n   = len(steps)

        x_str = f"  x={xs[idx]:.2f}" if xs else ""
        step_lbl.text = f"Step {idx+1} / {n}{x_str}"

        so = cache["scene_objs"]
        if so is not None:
            _update_scene(so, steps[idx], cache["sim_type"],
                          cache["vehicle"], cache["hp"])

        await asyncio.sleep(0)   # yield so UI events can be processed

        for (_, fig), pel in zip(cache["named_figs"], cache["plot_elems"]):
            if xs:
                _move_vline(fig, xs[idx])
            pel.update()

    ui.timer(1.0 / (SCRUB_FPS * 2), _apply_scrub)   # 2× SCRUB_FPS polling

    # ── result renderers ────────────────────────────────────────────────────────

    def _setup_scrubber(n):
        play_btn.props("icon=play_arrow")
        play_btn.visible = True
        step_lbl.text    = f"Step 1 / {n}"
        scrub["idx"]     = 0
        scrub["dirty"]   = False
        scrub["playing"] = False

    def _render_kin(result):
        sim_type, steps, vehicle, cfg, corner_id = result

        with viz_area:
            if not steps:
                ui.label("No valid solution steps returned.").classes("text-red-500 text-sm"); return

            if sim_type == "extreme":
                _render_extreme(steps); return

            corner = vehicle.get_corner_from_id(corner_id)
            hp     = corner.hardpoints

            cache.update(steps=steps, sim_type=sim_type,
                         vehicle=vehicle, hp=hp)

            if sim_type == "ackermann":
                named_figs, xs = _build_ackermann_figures(steps)
            else:
                named_figs, xs = _build_kin_figures(steps)

            cache["xs"]         = xs
            cache["named_figs"] = named_figs

            # ── 3-D view ──────────────────────────────────────────────────────
            with ui.card().classes("w-full p-0 overflow-hidden").style("flex-shrink:0"):
                with ui.row().classes("items-center px-3 py-1 bg-gray-100 border-b border-gray-200"):
                    ui.label("3D View").classes("text-sm font-semibold text-gray-700")
                    ui.label("drag to rotate · scroll to zoom · right-drag to pan"
                             ).classes("text-xs text-gray-400 ml-2")
                scene3d = ui.scene(width=900, height=420,
                                   background_color="#f0f4f8").classes("w-full")

            # build objects on last step (widest extent) for camera fit, then update to step 0
            scene_objs = _build_scene(scene3d, steps[-1], sim_type, vehicle, hp)
            _fit_camera(scene3d, steps[-1], sim_type, vehicle, hp)
            _update_scene(scene_objs, steps[0], sim_type, vehicle, hp)
            cache["scene_objs"] = scene_objs

            # ── 2-D plots ─────────────────────────────────────────────────────
            ncols = 3 if len(named_figs) >= 3 else len(named_figs)
            with ui.grid(columns=ncols).classes("w-full gap-2"):
                plot_elems = []
                for _, fig in named_figs:
                    pel = ui.plotly(fig).classes("w-full")
                    plot_elems.append(pel)
            cache["plot_elems"] = plot_elems

            _setup_scrubber(len(steps))

    def _render_extreme(data):
        with viz_area:
            ui.label("Extreme Points Results").classes("font-bold text-base mt-1")
            for half, sides in data.items():
                ui.label(half.upper()).classes("font-semibold text-sm mt-2 text-gray-700")
                for side, conditions in sides.items():
                    for cond, steer_data in conditions.items():
                        with ui.expansion(f"{side} / {cond}",
                                          icon="expand_more").classes("w-full border rounded bg-white"):
                            for steer, pts in steer_data.items():
                                ui.label(steer).classes("font-semibold text-xs px-2 pt-1 text-gray-600")
                                rows = []
                                for pt_name, val in pts.items():
                                    try:
                                        v = list(val)
                                        rows.append({"point": pt_name,
                                                     "x": f"{v[0]:.3f}",
                                                     "y": f"{v[1]:.3f}",
                                                     "z": f"{v[2]:.3f}"})
                                    except Exception:
                                        pass
                                if rows:
                                    ui.table(
                                        columns=[
                                            {"name":"point","label":"Point","field":"point","align":"left"},
                                            {"name":"x","label":"X [mm]","field":"x","align":"right"},
                                            {"name":"y","label":"Y [mm]","field":"y","align":"right"},
                                            {"name":"z","label":"Z [mm]","field":"z","align":"right"},
                                        ],
                                        rows=rows,
                                    ).classes("text-xs w-full").props("dense flat")

    def _render_opt(result):
        res, optimizer, cfg = result
        F = res.F
        obj_names = [o.name for o in optimizer.objectives]

        with viz_area:
            if F is None:
                ui.label("Optimization failed.").classes("text-red-500 text-sm"); return
            if F.ndim == 1:
                F = F.reshape(-1, 1)
            F = F[np.all(F <= 1e2, axis=1)]
            if not len(F):
                ui.label("No feasible solutions found.").classes("text-orange-500 text-sm"); return

            ui.label(f"Pareto front — {len(F)} solutions").classes("font-bold text-base")
            with ui.column().classes("gap-0.5"):
                for i, row in enumerate(F[:10]):
                    s = ",  ".join(f"{n}: {v:.4f}" for n, v in zip(obj_names, row))
                    ui.label(f"Sol {i:2d}:  {s}").classes("text-xs font-mono")

            for _, fig in _build_opt_figures(F, obj_names):
                ui.plotly(fig).classes("w-full")

    # wire
    mode_sel.on_value_change(on_mode_change)
    subtype_sel.on_value_change(on_type_change)
    save_btn.on_click(do_save)
    run_btn.on_click(do_run)
    play_btn.on_click(do_play_pause)


# ── entry point ────────────────────────────────────────────────────────────────
ui.run(title="SAGE - Suspension Analysis", port=8080, reload=False, show=True, favicon="⚙️")
