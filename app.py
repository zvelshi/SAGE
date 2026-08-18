# default
import asyncio
import traceback
import time
import os

# third-party
import yaml
import numpy as np
from nicegui import ui, run

# ours
from utils.sim_runners import _run_kin, _run_opt, _run_dyn
from utils.scene3d import (_build_scene, _update_scene, _fit_camera,
                            _build_dyn_scene, _update_dyn_scene, _fit_camera_dyn,
                            _build_shock_dyno_scene, _update_shock_dyno_scene, _fit_camera_shock_dyno)
from utils.plot2d import (_build_kin_figures, _build_ackermann_figures, _build_opt_figures,
                           _move_vline, _build_dyn_figures, _build_dyno_figures,
                           _build_kin_stats, _build_dyn_stats)

# global constants
FPS = 60

KIN_PATH = "config/kin_config.yml"
DYN_PATH = "config/dyn_config.yml"
OPT_PATH = "config/opt_config.yml"

TAB_PATH  = {"kin": KIN_PATH, "dyn": DYN_PATH, "opt": OPT_PATH}
SIM_TYPES = {
    "kin": ["travel", "steer", "droop_steer", "jounce_steer", "left_travel", "right_travel", "extreme", "ackermann"],
    "dyn": ["static", "shock_dyno"],
    "opt": ["run"],
}

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
    scrub    = {"dirty": False, "idx": 0, "playing": False, "last_t": 0.0}
    dyn_prog = {"fraction": 0.0, "message": ""}
    display_items: list = []  # [{"key","label","element","default"}, ...] rebuilt each render

    ui.add_head_html("""<style>
        body,html{margin:0;padding:0;height:100vh;overflow:hidden}
        .nicegui-content{padding:0!important;height:100vh!important;overflow:hidden!important}
        .cm-editor{height:100%!important}.cm-scroller{overflow:auto!important}
    </style>""")

    with ui.row().classes("w-full gap-0").style("height:100vh;overflow:hidden"):

        with ui.column().classes("border-r border-stone-300").style(
            "width:400px;min-width:400px;height:100vh;display:flex;"
            "flex-direction:column;overflow:hidden"
        ):
            with ui.row().classes("items-center gap-2 px-3 py-2 bg-stone-900").style("flex-shrink:0"):
                ui.label("SAGE").classes("text-emerald-500 font-bold tracking-widest")

            def get_hp_name():
                if "kin" in editors:
                    try:
                        cfg = yaml.safe_load(editors["kin"].value)
                        if cfg and isinstance(cfg, dict):
                            return cfg.get("HARDPOINTS", "unknown")
                    except Exception:
                        pass
                return "unknown"

            def open_hp_dialog():
                name = get_hp_name()
                path = f"config/hardpoints/{name}.yml"
                if os.path.exists(path):
                    hp_editor.value = open(path).read()
                    hp_dialog_title.text = f"{path}"
                else:
                    hp_editor.value = ""
                    hp_dialog_title.text = f"{path} (Not Found)"
                hp_dialog.open()

            with ui.row().classes("w-full px-3 py-2 bg-stone-100 justify-center border-b border-stone-200"):
                hp_button = ui.button("Hardpoints", icon="edit_document", on_click=open_hp_dialog).props("outline size=sm").classes("w-full bg-white text-stone-700 border-stone-300 shadow-sm hover:bg-stone-50 text-base")

            hp_dialog = ui.dialog()
            with hp_dialog, ui.card().style("width: 700px; height: 85vh; max-width: 100vw; display: flex; flex-direction: column; padding: 0;"):
                with ui.row().classes("w-full px-3 py-2 bg-stone-100 border-b border-stone-200 items-center justify-between").style("flex-shrink:0"):
                    hp_dialog_title = ui.label("Hardpoints YAML").classes("font-bold text-sm text-stone-700")
                    ui.button(icon="close", on_click=hp_dialog.close).props("flat dense round size=sm").classes("text-stone-500")
                
                hp_editor = ui.codemirror(language="yaml", theme="githubLight").style("flex:1; min-height: 0; font-size: 12px;")
                
                with ui.row().classes("w-full px-3 py-2 bg-stone-50 border-t border-stone-200 justify-end gap-2").style("flex-shrink:0"):
                    ui.button("Cancel", on_click=hp_dialog.close).props("flat dense").classes("text-stone-600 px-3")
                    def save_hp():
                        name = get_hp_name()
                        path = f"config/hardpoints/{name}.yml"
                        try:
                            yaml.safe_load(hp_editor.value)
                            open(path, "w").write(hp_editor.value)
                            ui.notify(f"Saved -> {path}", type="positive", position="bottom-right")
                            hp_dialog.close()
                        except yaml.YAMLError as exc:
                            ui.notify(f"YAML error: {exc}", type="negative")
                    ui.button("Save", on_click=save_hp).props("unelevated dense").classes("bg-emerald-600 text-white px-3")

            with ui.tabs().classes("w-full bg-stone-100 border-b border-stone-200").props("dense no-caps") as tabs:
                ui.tab("kin", label="Kinematic")
                ui.tab("dyn", label="Dynamic")
                ui.tab("opt", label="Optimizer")

            with ui.tab_panels(tabs, value="kin").style("flex:1;min-height:0;display:flex;flex-direction:column;overflow:hidden") as panels:
                for tab_name, path in TAB_PATH.items():
                    with ui.tab_panel(tab_name).style("height:100%;padding:0;display:flex;flex-direction:column;overflow:hidden"):
                        ed = ui.codemirror(value=open(path).read(), language="yaml", theme="githubLight").style("flex:1;min-height:0;font-size:12px")
                        editors[tab_name] = ed

            panels.on_value_change(lambda e: active_tab.update({"v": e.value}))

            with ui.row().classes("w-full items-center gap-2 px-2 py-2 border-t border-stone-200 bg-stone-50").style("flex-shrink:0"):
                mode_sel = ui.select(list(SIM_TYPES), value="kin",  label="Mode").classes("w-20").props("dense outlined")
                subtype_sel = ui.select(SIM_TYPES["kin"], value=SIM_TYPES["kin"][0], label="Type").classes("w-36").props("dense outlined")
                save_btn = ui.button("Save").props("unelevated dense").classes("bg-stone-700 text-white text-sm px-3")
                run_btn  = ui.button("Run" ).props("unelevated dense").classes("bg-emerald-600 text-white text-sm px-3")

        with ui.column().style("flex:1;height:100vh;display:flex;flex-direction:column;overflow:hidden;background:#f8fafc"):
            # status bar
            with ui.row().classes("items-center gap-3 px-4 pt-2 pb-1").style("flex-shrink:0"):
                spinner  = ui.spinner("dots", size="sm", color="teal")
                spinner.visible = False
                status_lbl = ui.label("Ready — configure and press Run.").classes("text-stone-500 text-sm italic")
                progress = ui.linear_progress(value=0).classes("flex-1 ml-2").props("instant-feedback rounded color=teal")
                progress.visible = False
                edit_display_btn = ui.button("Add Graphs", icon="tune", on_click=lambda: edit_display_dialog.open()) \
                    .props("outline dense size=sm").classes("text-stone-700 border-stone-300 text-base")
                edit_display_btn.visible = False

            edit_display_dialog = ui.dialog()
            with edit_display_dialog, ui.card().style("width: 380px; max-width: 90vw;"):
                ui.label("Add Graphs").classes("font-bold text-sm text-stone-700")
                ui.label("Choose which panels are shown below.").classes("text-xs text-stone-400 mb-1")
                display_dialog_body = ui.column().classes("w-full gap-1")
                with ui.row().classes("w-full justify-end pt-2"):
                    ui.button("Close", on_click=edit_display_dialog.close).props("flat dense").classes("text-stone-600 px-3")

            # scrollable viz body
            with ui.column().classes("w-full px-4 pb-2 gap-3").style("flex:1;overflow-y:auto;min-height:0") as viz_area:
                pass

            # playback footer
            with ui.row().classes("w-full items-center gap-3 px-4 py-2 border-t border-stone-200 bg-white").style("flex-shrink:0"):
                play_btn = ui.button(icon="play_arrow").props("round dense unelevated color=teal")
                step_lbl = ui.label("—").classes("text-xs text-stone-500 font-mono")
                play_btn.visible = False

    # callbacks
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
        ui.notify(f"Saved -> {path}", type="positive", position="bottom-right")
        if tab == "kin":
            hp_button.text = f"Hardpoints: '{get_hp_name()}'"

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
        _reset_display_items()
        cache["named_figs"].clear(); cache["plot_elems"].clear()
        cache["steps"] = []; cache["scene_objs"] = None
        cache["xs"] = []

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
                dyn_prog["fraction"] = 0.0
                dyn_prog["message"]  = "Initializing..."
                dyn_poll_timer.active = True
                result = await run.io_bound(_run_dyn, editors["kin"].value, editors["dyn"].value, sim_type, dyn_prog)
                dyn_poll_timer.active = False
                progress.set_value(0.95)
                await asyncio.sleep(0)
                _render_dyn(result)

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
        steps = cache["steps"]
        if not steps:
            return

        # auto-advance when playing
        if scrub["playing"]:
            now = time.monotonic()
            if now - scrub["last_t"] >= 1.0 / FPS:
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
            if cache["sim_type"] == "static":
                _update_dyn_scene(so, steps[idx], cache["vehicle"])
            elif cache["sim_type"] == "shock_dyno":
                _update_shock_dyno_scene(so, steps[idx])
            else:
                _update_scene(so, steps[idx], cache["sim_type"], cache["vehicle"], cache["hp"])

        await asyncio.sleep(0) # yield so UI events can be processed

        for (_, fig), pel in zip(cache["named_figs"], cache["plot_elems"]):
            if xs:
                _move_vline(fig, xs[idx])
            pel.update()

    ui.timer(1.0 / (FPS * 2), _apply_scrub)

    def _poll_dyn_progress():
        progress.set_value(dyn_prog["fraction"])
        if dyn_prog["message"]:
            status_lbl.text = dyn_prog["message"]

    dyn_poll_timer = ui.timer(0.25, _poll_dyn_progress, active=False)

    # result renderers 
    def _setup_scrubber(n):
        play_btn.props("icon=play_arrow")
        play_btn.visible = True
        step_lbl.text = f"Step 1 / {n}"
        scrub["idx"] = 0
        scrub["dirty"] = False
        scrub["playing"] = False

    _DISPLAY_CATEGORIES = [("3d", "3D"), ("2d", "2D"), ("value", "Values")]

    def _reset_display_items():
        display_items.clear()
        edit_display_btn.visible = False

    def _add_display_item(label, element, default_visible, category="value"):
        element.visible = default_visible
        display_items.append({"label": label, "element": element,
                               "default": default_visible, "category": category})

    def _rebuild_edit_display_dialog():
        display_dialog_body.clear()
        with display_dialog_body:
            for cat_key, cat_label in _DISPLAY_CATEGORIES:
                items = [it for it in display_items if it["category"] == cat_key]
                if not items:
                    continue
                ui.label(cat_label).classes("text-xs font-semibold text-stone-500 uppercase mt-2 first:mt-0")
                for item in items:
                    def _on_change(e, it=item):
                        it["element"].visible = e.value
                    ui.checkbox(item["label"], value=item["default"]).on_value_change(_on_change).classes("ml-1")
        edit_display_btn.visible = bool(display_items)

    def _render_kin(result):
        sim_type, steps, vehicle, cfg, corner_id, run_dir = result

        with viz_area:
            if not steps:
                ui.label("No valid solution steps returned.").classes("text-red-500 text-sm"); return

            _reset_display_items()

            if sim_type == "extreme":
                _render_extreme(steps, run_dir, cfg); return

            corner = vehicle.get_corner_from_id(corner_id)
            hp = corner.hardpoints

            cache.update(steps=steps, sim_type=sim_type, vehicle=vehicle, hp=hp)

            stat_pairs = []  # (label, value_str)
            if sim_type == "ackermann":
                named_figs, xs = _build_ackermann_figures(steps)
            else:
                half_label = "Rear" if corner_id[1] == 1 else "Front"
                named_figs, xs = _build_kin_figures(steps, half_label=half_label, wr=hp.wr)

                axle_steps = [s["axle_data"] for s in steps if s.get("axle_data")]
                if axle_steps:
                    abs_plunge = max(abs(a["plunge_mm"]) for a in axle_steps)
                    abs_angle  = max(max(a["angle_ib_deg"], a["angle_ob_deg"]) for a in axle_steps)
                    stat_pairs.append((f"{half_label} — Absolute Plunge [mm]", f"{abs_plunge:.2f}"))
                    stat_pairs.append((f"{half_label} — Absolute Max Joint Angle [deg]", f"{abs_angle:.2f}"))
                stat_pairs.extend(_build_kin_stats(steps, wr=hp.wr))

            cache["xs"] = xs
            cache["named_figs"] = named_figs

            # static value cards
            if stat_pairs:
                with ui.row().classes("w-full gap-2 flex-wrap"):
                    for label, value in stat_pairs:
                        with ui.card().classes("px-4 py-2") as stat_card:
                            ui.label(label).classes("text-xs text-stone-500")
                            ui.label(value).classes("text-lg font-semibold text-emerald-700")
                        _add_display_item(label, stat_card, default_visible=False, category="value")

            # 3D view
            with ui.card().classes("w-full p-0 overflow-hidden").style("flex-shrink:0") as card_3d:
                with ui.row().classes("items-center px-3 py-1 bg-stone-100 border-b border-stone-200"):
                    ui.label("3D View").classes("text-sm font-semibold text-stone-700")
                    ui.label("drag to rotate · scroll to zoom · right-drag to pan").classes("text-xs text-stone-400 ml-2")
                scene3d = ui.scene(width=900, height=420,background_color="#f0f4f8", grid=(10, 100)).classes("w-full")
            _add_display_item("3D View", card_3d, default_visible=True, category="3d")

            # build objects on last step (widest extent) for camera fit, then update to step 0
            scene_objs = _build_scene(scene3d, steps[-1], sim_type, vehicle, hp)
            _fit_camera(scene3d, steps[-1], sim_type, vehicle, hp)
            _update_scene(scene_objs, steps[0], sim_type, vehicle, hp)
            cache["scene_objs"] = scene_objs

            # 2D plots
            ncols = 3 if len(named_figs) >= 3 else len(named_figs)
            with ui.grid(columns=ncols).classes("w-full gap-2"):
                plot_elems = []
                for key, fig in named_figs:
                    pel = ui.plotly(fig).classes("w-full")
                    plot_elems.append(pel)
                    _add_display_item(key.replace("_", " ").title(), pel, default_visible=False, category="2d")
            cache["plot_elems"] = plot_elems

            _rebuild_edit_display_dialog()
            _setup_scrubber(len(steps))

    def _render_extreme(data, run_dir, cfg):
        with viz_area:
            ui.label("Extreme Points Results").classes("font-bold text-base mt-1 text-emerald-800")
            
            hp_name = cfg.get("HARDPOINTS", "UNKNOWN")
            out_file = os.path.abspath(os.path.join(run_dir, f"HARDPOINTS_{hp_name}.xlsx"))
            with ui.row().classes("items-center gap-2 mt-1 mb-2 bg-emerald-50 p-2 rounded w-full"):
                ui.icon("folder", color="teal")
                ui.label(f"Exported to: {out_file}").classes("text-sm text-stone-700 font-mono")
                
            ui.link("How to import this data into SolidWorks", "https://docs.google.com/document/d/1YMDovPIkaAoIByOL9fQeDe5OqUxFQ4b42RDFjEYWVxo/edit?usp=sharing", new_tab=True).classes("text-emerald-600 text-sm underline mb-4 block")

            for half, sides in data.items():
                ui.label(half.upper()).classes("font-semibold text-sm mt-2 text-stone-700")
                for side, conditions in sides.items():
                    for cond, steer_data in conditions.items():
                        with ui.expansion(f"{side} / {cond}", icon="expand_more").classes("w-full border rounded bg-white"):
                            for steer, pts in steer_data.items():
                                ui.label(steer).classes("font-semibold text-xs px-2 pt-1 text-stone-600")
                                rows = []
                                for pt_name, val in pts.items():
                                    try:
                                        v = list(val)
                                        rows.append({"point": pt_name, "x": f"{v[0]:.3f}", "y": f"{v[1]:.3f}", "z": f"{v[2]:.3f}" })
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

    def _render_dyn(result):
        if len(result) == 4:
            steps, vehicle, run_dir, dyn_cfg = result
        else:
            steps, vehicle, run_dir = result
            dyn_cfg = {"SIMULATION": "static"}

        sim_type = dyn_cfg.get("SIMULATION", "static")

        with viz_area:
            if not steps:
                ui.label("No simulation frames returned.").classes("text-red-500 text-sm"); return

            cache.update(steps=steps, sim_type=sim_type, vehicle=vehicle, hp=None)
            _reset_display_items()

            if sim_type == "shock_dyno":
                out_file = os.path.abspath(os.path.join(run_dir, "shock_dyno_results.csv"))
                with ui.row().classes("items-center gap-2 mt-1 mb-2 bg-emerald-50 p-2 rounded w-full"):
                    ui.icon("folder", color="teal")
                    ui.label(f"Dyno Results Exported to: {out_file}").classes("text-sm text-stone-700 font-mono")

            if sim_type != "shock_dyno":
                corner_wr = {
                    "fl": vehicle.front_left.hardpoints.wr,
                    "fr": vehicle.front_right.hardpoints.wr,
                    "rl": vehicle.rear_left.hardpoints.wr,
                    "rr": vehicle.rear_right.hardpoints.wr,
                }
                stat_pairs = _build_dyn_stats(steps, corner_wr=corner_wr)
                if stat_pairs:
                    with ui.row().classes("w-full gap-2 flex-wrap"):
                        for label, value in stat_pairs:
                            with ui.card().classes("px-4 py-2") as stat_card:
                                ui.label(label).classes("text-xs text-stone-500")
                                ui.label(value).classes("text-lg font-semibold text-emerald-700")
                            _add_display_item(label, stat_card, default_visible=False, category="value")

            with ui.card().classes("w-full p-0 overflow-hidden").style("flex-shrink:0") as card_3d:
                with ui.row().classes("items-center px-3 py-1 bg-stone-100 border-b border-stone-200"):
                    if sim_type == "shock_dyno":
                        ui.label("3D View — Isolated Shock").classes("text-sm font-semibold text-stone-700")
                    else:
                        ui.label("3D View — Full Vehicle Drop").classes("text-sm font-semibold text-stone-700")

                    ui.label("drag to rotate · scroll to zoom · right-drag to pan").classes("text-xs text-stone-400 ml-2")

                    if sim_type != "shock_dyno":
                        ui.label("● hoist  ● drop  ● settled").classes("text-xs text-stone-400 ml-auto")
                scene3d = ui.scene(width=900, height=500, background_color="#f0f4f8", grid=(10, 100)).classes("w-full")
            _add_display_item("3D View", card_3d, default_visible=True, category="3d")

            if sim_type == "shock_dyno":
                scene_objs = _build_shock_dyno_scene(scene3d, steps[-1])
                _fit_camera_shock_dyno(scene3d, steps[-1])
                _update_shock_dyno_scene(scene_objs, steps[0])
            else:
                scene_objs = _build_dyn_scene(scene3d, steps[-1], vehicle)
                _fit_camera_dyn(scene3d, steps[-1], vehicle)
                _update_dyn_scene(scene_objs, steps[0], vehicle)
                
            cache["scene_objs"] = scene_objs

            if sim_type == "shock_dyno":
                named_figs, xs = _build_dyno_figures(steps)
            else:
                named_figs, xs = _build_dyn_figures(steps)
                
            cache["xs"] = xs
            cache["named_figs"] = named_figs

            ncols = 1 if sim_type == "shock_dyno" else (3 if len(named_figs) >= 3 else len(named_figs))
            with ui.grid(columns=ncols).classes("w-full gap-2 mt-2"):
                plot_elems = []
                for key, fig in named_figs:
                    pel = ui.plotly(fig).classes("w-full")
                    plot_elems.append(pel)
                    _add_display_item(key.replace("_", " ").title(), pel, default_visible=False, category="2d")
            cache["plot_elems"] = plot_elems

            _rebuild_edit_display_dialog()
            _setup_scrubber(len(steps))

    def _render_opt(result):
        res, optimizer, cfg, run_dir = result
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

            ui.label(f"Pareto front — {len(F)} solutions").classes("font-bold text-base text-emerald-800")
            with ui.column().classes("gap-0.5"):
                for i, row in enumerate(F[:10]):
                    s = ",  ".join(f"{n}: {v:.4f}" for n, v in zip(obj_names, row))
                    ui.label(f"Sol {i:2d}:  {s}").classes("text-xs font-mono text-stone-600")

            for _, fig in _build_opt_figures(F, obj_names):
                ui.plotly(fig).classes("w-full")

    # wire
    mode_sel.on_value_change(on_mode_change)
    subtype_sel.on_value_change(on_type_change)
    save_btn.on_click(do_save)
    run_btn.on_click(do_run)
    play_btn.on_click(do_play_pause)

    # initialize button text
    try:
        cfg = yaml.safe_load(open(KIN_PATH).read())
        if cfg and isinstance(cfg, dict):
            hp_button.text = f"EDIT HARDPOINTS"
    except Exception:
        pass

ui.run(title="SAGE", port=8080, reload=False, show=True, favicon="🌿")
