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
from utils.sim_runners import _run_kin, _run_opt, _run_dyn, _load_kin_run, _load_dyn_run, _load_opt_run
from utils.export import (list_available_runs, build_kin_static_values, build_dyn_static_values,
                           load_kin_run_data, load_dyn_run_data, NO_COMPARE_SIM_TYPES)
from utils import scene3d
from utils.scene3d import (_build_scene, _update_scene, _fit_camera,
                            _build_dyn_scene, _update_dyn_scene, _fit_camera_dyn,
                            _build_shock_dyno_scene, _update_shock_dyno_scene, _fit_camera_shock_dyno,
                            _build_config_preview_scene, build_legend_entries, _FREE_POINT_COLOR,
                            _resolve_point_attr)
from utils.plot2d import (_build_kin_figures, _build_front_steer_figures,
                           _build_full_vehicle_figures, _build_opt_figures, _move_vline, _build_dyn_figures,
                           _build_dyno_figures, _build_sweep_space_figures, rank_solutions)
from simulations.scenarios.kin.full_vehicle import FULL_VEHICLE_TYPES
from models.vehicle import Vehicle
from utils.misc import add_console_subscriber, remove_console_subscriber

# global constants
FPS = 30

KIN_PATH = "config/kin_config.yml"
DYN_PATH = "config/dyn_config.yml"
OPT_PATH = "config/opt_config.yml"

TAB_PATH  = {"kin": KIN_PATH, "dyn": DYN_PATH, "opt": OPT_PATH}

KIN_SIM_GROUPS = [
    ("Corner Vehicle", ["travel", "steer", "droop_steer", "jounce_steer", "left_travel", "right_travel", "sweep_space"]),
    ("Half Vehicle", ["front_steer"]),
    ("Full Vehicle", ["extreme", "heave", "roll"]),
]

def _grouped_options(groups):
    opts = {}
    for title, items in groups:
        opts[f"__sep__{title}"] = f"── {title} ──"
        for it in items:
            opts[it] = it
    return opts

SIM_TYPES = {
    "kin": _grouped_options(KIN_SIM_GROUPS),
    "dyn": ["static", "shock_dyno"],
    "opt": ["run"],
}

def _step_geometry_complete(s):
    if "left" in s or "right" in s:
        return bool(s.get("left")) and bool(s.get("right"))
    if "fl" in s:
        return all(s.get(k) for k in ("fl", "fr", "rl", "rr"))
    return True

def _last_scene_step(steps):
    for s in reversed(steps):
        if _step_geometry_complete(s):
            return s
    return steps[-1]

@ui.page("/")
def main_page():
    editors    = {}
    mode_ref   = {"v": "kin"}
    type_ref   = {"v": "travel"}
    active_tab = {"v": "kin"}
    rtask      = {"task": None}

    # sim result cache
    cache = {
        "steps": [],
        "xs": [],
        "sim_type": None,
        "vehicle": None,
        "hp": None,
        "named_figs": [],
        "plot_elems": [],
        "scene_objs": None,
        "mode": None,
        "last_result": None,
    }
    scrub = {"dirty": False, "idx": 0, "playing": False, "last_t": 0.0}
    dyn_prog = {"fraction": 0.0, "message": ""}
    run_lookup = {}
    compare_state = {"active": False, "run_dir": None}
    display_items: list = []

    ui.add_head_html("""<style>
        body,html{margin:0;padding:0;height:100vh;overflow:hidden}
        .nicegui-content{padding:0!important;height:100vh!important;overflow:hidden!important}
        .cm-editor{height:100%!important}.cm-scroller{overflow:auto!important}
    </style>""")

    with ui.row().classes("w-full gap-0").style("height:100vh;overflow:hidden"):

        with ui.column().classes("border-r border-stone-300").style(
            "width:37.5vw;min-width:37.5vw;height:100vh;display:flex;"
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
                subtype_sel = ui.select(SIM_TYPES["kin"], value="travel", label="Type").classes("w-36").props("dense outlined")
                save_btn = ui.button("Save").props("unelevated dense").classes("bg-stone-700 text-white text-sm px-3")
                preview_btn = ui.button("Preview", icon="visibility").props("outline dense").classes("text-stone-700 border-stone-300 text-sm px-3")
                preview_btn.visible = False
                run_btn  = ui.button("Run" ).props("unelevated dense").classes("bg-emerald-600 text-white text-sm px-3")

            with ui.row().classes("w-full items-center gap-2 px-2 py-2 border-t border-stone-200 bg-stone-50").style("flex-shrink:0"):
                run_browse_sel = ui.select({}, label="Load Past Run").classes("flex-1").props("dense outlined")
                refresh_runs_btn = ui.button(icon="refresh").props("outline dense size=sm").classes("text-stone-600")
                load_run_btn = ui.button("Load", icon="folder_open").props("outline dense size=sm").classes("text-stone-700 border-stone-300")

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
                compare_sel = ui.select({}, label="Compare vs").classes("w-56").props("dense outlined")
                compare_btn = ui.button("Compare", icon="compare_arrows").props("outline dense size=sm").classes("text-stone-700 border-stone-300")
                compare_clear_btn = ui.button(icon="close").props("flat dense round size=sm").classes("text-stone-500")
                compare_sel.visible = compare_btn.visible = compare_clear_btn.visible = False

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
        first = next(iter(opts)) if isinstance(opts, dict) else opts[0]
        if isinstance(opts, dict) and first.startswith("__sep__"):
            first = next(v for v in opts if not v.startswith("__sep__"))
        subtype_sel.value = first
        type_ref["v"]     = first
        tabs.set_value(e.value)
        active_tab["v"] = e.value
        preview_btn.visible = (e.value == "opt")

    def on_type_change(e):
        if isinstance(e.value, str) and e.value.startswith("__sep__"):
            subtype_sel.value = type_ref["v"]
            return
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

    def _refresh_run_options():
        runs = list_available_runs()
        run_lookup.clear()
        options = {}
        for r in runs:
            run_lookup[r["run_dir"]] = r
            options[r["run_dir"]] = r["label"]
        run_browse_sel.options = options
        if options and run_browse_sel.value not in options:
            run_browse_sel.value = next(iter(options))
        run_browse_sel.update()

    def _current_run_dir():
        lr = cache.get("last_result")
        if not lr:
            return None
        return lr[5] if cache["mode"] == "kin" else lr[2]

    def _refresh_compare_options():
        mode = cache.get("mode")
        sim_type = cache.get("sim_type")
        if mode not in ("kin", "dyn") or sim_type in NO_COMPARE_SIM_TYPES or not cache.get("steps"):
            compare_sel.visible = compare_btn.visible = compare_clear_btn.visible = False
            return

        this_run = _current_run_dir()
        matches = [r for r in list_available_runs()
                   if r["mode"] == mode and r["sim_type"] == sim_type and r["run_dir"] != this_run]
        run_lookup.update({r["run_dir"]: r for r in matches})  # keep labels usable even if run_lookup
                                                                 # hasn't been refreshed since this run was created
        options = {r["run_dir"]: r["label"] for r in matches}
        compare_sel.options = options
        if options and compare_sel.value not in options:
            compare_sel.value = next(iter(options))
        compare_sel.update()

        compare_sel.visible = bool(options)
        compare_btn.visible = bool(options)
        compare_clear_btn.visible = compare_state["active"]

    def _rerender_current():
        result = cache.get("last_result")
        if not result:
            return
        _reset_display_items()
        cache["named_figs"].clear(); cache["plot_elems"].clear()
        cache["steps"] = []; cache["scene_objs"] = None
        cache["xs"] = []
        viz_area.clear()
        if cache["mode"] == "kin":
            _render_kin(result)
        else:
            _render_dyn(result)
        _refresh_compare_options()

    def do_compare():
        run_dir = compare_sel.value
        if not run_dir:
            ui.notify("Select a run to compare against first.", type="warning"); return
        compare_state["active"], compare_state["run_dir"] = True, run_dir
        try:
            _rerender_current()
        except Exception as exc:
            compare_state["active"], compare_state["run_dir"] = False, None
            ui.notify(f"Could not load comparison run: {exc}", type="negative", timeout=8000)

    def do_clear_compare():
        compare_state["active"], compare_state["run_dir"] = False, None
        _rerender_current()

    def do_load_run():
        run_dir = run_browse_sel.value
        meta = run_lookup.get(run_dir)
        if not meta:
            ui.notify("Select a run to load first.", type="warning"); return

        run_btn.disable(); save_btn.disable()
        try:
            _reset_display_items()
            cache["named_figs"].clear(); cache["plot_elems"].clear()
            cache["steps"] = []; cache["scene_objs"] = None
            cache["xs"] = []
            compare_state["active"], compare_state["run_dir"] = False, None
            viz_area.clear()

            if meta["mode"] == "kin":
                result = _load_kin_run(run_dir)
                cache["mode"], cache["last_result"] = "kin", result
                _render_kin(result)
                _refresh_compare_options()
            elif meta["mode"] == "dyn":
                result = _load_dyn_run(run_dir)
                cache["mode"], cache["last_result"] = "dyn", result
                _render_dyn(result)
                _refresh_compare_options()
            else:  # opt -- persistence/reload only, no comparison support
                result = _load_opt_run(run_dir)
                cache["mode"], cache["last_result"] = "opt", result
                compare_sel.visible = compare_btn.visible = compare_clear_btn.visible = False
                _render_opt(result)

            status_lbl.text = f"Loaded (not re-run) — {meta['label']}"
            ui.notify(f"Loaded {meta['label']}", type="positive", position="bottom-right")
        except Exception as exc:
            status_lbl.text = f"Error loading run: {exc}"
            ui.notify(str(exc), type="negative", timeout=10_000)
            with viz_area:
                ui.label(traceback.format_exc()).classes("text-red-500 text-xs font-mono whitespace-pre-wrap")
        finally:
            run_btn.enable(); save_btn.enable()

    async def do_run():
        mode, sim_type = mode_ref["v"], type_ref["v"]
        rtask["task"] = asyncio.current_task()

        run_btn.disable(); save_btn.disable()
        spinner.visible  = True
        play_btn.visible = False
        progress.visible = True
        progress.set_value(0)
        status_lbl.text  = f"Running {mode} / {sim_type}…"
        _reset_display_items()
        cache["named_figs"].clear(); cache["plot_elems"].clear()
        cache["steps"] = []; cache["scene_objs"] = None
        cache["xs"] = []
        compare_state["active"], compare_state["run_dir"] = False, None

        _start_console()

        try:
            if mode == "kin":
                result = await run.io_bound(_run_kin, editors["kin"].value, sim_type)
                progress.set_value(0.85)
                await asyncio.sleep(0)
                _stop_console()
                viz_area.clear()
                cache["mode"], cache["last_result"] = "kin", result
                _render_kin(result)
                _refresh_compare_options()

            elif mode == "opt":
                result = await run.io_bound(_run_opt, editors["kin"].value, editors["opt"].value)
                progress.set_value(0.85)
                await asyncio.sleep(0)
                _stop_console()
                viz_area.clear()
                cache["mode"], cache["last_result"] = "opt", result
                compare_sel.visible = compare_btn.visible = compare_clear_btn.visible = False
                _render_opt(result)

            elif mode == "dyn":
                dyn_prog["fraction"] = 0.0
                dyn_prog["message"]  = "Initializing..."
                dyn_poll_timer.active = True
                result = await run.io_bound(_run_dyn, editors["kin"].value, editors["dyn"].value, sim_type, dyn_prog)
                dyn_poll_timer.active = False
                progress.set_value(0.95)
                await asyncio.sleep(0)
                _stop_console()
                viz_area.clear()
                cache["mode"], cache["last_result"] = "dyn", result
                _render_dyn(result)
                _refresh_compare_options()

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
            _stop_console()
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
            if cache["sim_type"] == "static" or cache["sim_type"] == "front_steer" or cache["sim_type"] in FULL_VEHICLE_TYPES:
                _update_dyn_scene(so, steps[idx], cache["vehicle"])
            elif cache["sim_type"] == "shock_dyno":
                _update_shock_dyno_scene(so, steps[idx])
            else:
                _update_scene(so, steps[idx], cache["hp"])

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

    # live console output shown in viz_area while a run is in progress
    console_state = {"lines": [], "pushed": 0}
    console_ref   = {"log": None}

    def _on_console_line(line: str) -> None:
        console_state["lines"].append(line)

    def _poll_console():
        log_widget = console_ref["log"]
        if log_widget is None:
            return
        lines = console_state["lines"]
        n = len(lines)
        if n > console_state["pushed"]:
            for line in lines[console_state["pushed"]:n]:
                log_widget.push(line)
            console_state["pushed"] = n

    console_poll_timer = ui.timer(0.2, _poll_console, active=False)

    def _start_console():
        console_state["lines"].clear()
        console_state["pushed"] = 0
        viz_area.clear()
        with viz_area:
            console_ref["log"] = ui.log(max_lines=4000).classes("w-full").style(
                "flex:1;min-height:300px;background:#111827;color:#d1d5db;"
                "font-family:Consolas,monospace;font-size:12px;border-radius:6px;padding:8px;"
            )
        add_console_subscriber(_on_console_line)
        console_poll_timer.active = True

    def _stop_console():
        console_poll_timer.active = False
        remove_console_subscriber(_on_console_line)
        console_ref["log"] = None

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

    def _fit_preview_camera(scene, hp):
        pts = [np.asarray(getattr(hp, a)) for a in
               ("ubj", "lbj", "wc", "tr_ob", "s_ob", "ucl_ob", "lcl_ob") if hasattr(hp, a)]
        if not pts:
            return
        arr  = np.array(pts) / 1000.0
        ctr  = arr.mean(axis=0)
        span = float(np.max(arr.max(axis=0) - arr.min(axis=0)))
        dist = span * 1.9
        scene.move_camera(
            x=float(ctr[0]) + dist * 0.4, y=float(ctr[1]) - dist * 1.1, z=float(ctr[2]) + dist * 0.7,
            look_at_x=float(ctr[0]), look_at_y=float(ctr[1]), look_at_z=float(ctr[2]),
            up_x=0, up_y=0, up_z=1,
        )

    def _render_legend(keepout_cfg, groups_cfg=None):
        with ui.column().classes("gap-1 p-2 bg-white/90 rounded shadow").style(
            "position:absolute; top:8px; right:8px; z-index:10;"
        ):
            ui.label("Legend").classes("text-xs font-bold text-stone-700")
            with ui.row().classes("items-center gap-1"):
                ui.element("div").style(f"width:10px;height:10px;background:{_FREE_POINT_COLOR};opacity:0.6;border-radius:2px")
                ui.label("Free Variable Range").classes("text-xs text-stone-600")
            for label, color in build_legend_entries(keepout_cfg, groups_cfg):
                with ui.row().classes("items-center gap-1"):
                    ui.element("div").style(f"width:10px;height:10px;background:{color};opacity:0.6;border-radius:2px")
                    ui.label(label).classes("text-xs text-stone-600")

    def _mount_parts_tree(scene_objs, anchor: str = "right"):
        """meshcat-style show/hide tree, overlaid in a top corner of the 3D view
        (`anchor` = "left" or "right"). Call inside the scene's position:relative
        wrapper, after scene_objs is built."""
        tree_nodes, leaf_ids, ticked = scene3d.scene_parts_tree_defaults(scene_objs)
        for lid in leaf_ids:
            scene3d.set_scene_node_visible(scene_objs, lid, lid in ticked)
        if not tree_nodes:
            return

        def _apply(ticked_ids):
            keep = set(ticked_ids)
            for lid in leaf_ids:
                scene3d.set_scene_node_visible(cache["scene_objs"], lid, lid in keep)

        with ui.column().classes("bg-white/95 rounded shadow").style(
            f"position:absolute; top:8px; {anchor}:8px; z-index:10; width:216px; "
            "max-height:calc(100% - 16px); overflow:auto"
        ):
            with ui.expansion("Parts", icon="account_tree", value=True).props(
                "dense expand-separator"
            ).classes("w-full text-xs"):
                tree = ui.tree(tree_nodes, label_key="label", node_key="id",
                               tick_strategy="leaf",
                               on_tick=lambda e: _apply(e.value)) \
                    .props("dense no-connectors").classes("text-xs")
                tree.tick(ticked)
                tree.expand([n["id"] for n in tree_nodes if n["children"]])

    def _corner_for(vehicle, cfg):
        corner_id = [1 if cfg.get("SIDE") == "right" else 0, 1 if cfg.get("HALF") == "rear" else 0]
        return vehicle.get_corner_from_id(corner_id)

    def do_preview():
        try:
            kin_cfg = yaml.safe_load(editors["kin"].value)
            opt_cfg = yaml.safe_load(editors["opt"].value)
        except yaml.YAMLError as exc:
            ui.notify(f"YAML error: {exc}", type="negative"); return
        if not kin_cfg or not isinstance(kin_cfg, dict):
            ui.notify("Invalid kinematic config.", type="negative"); return

        hp_name = kin_cfg.get("HARDPOINTS")
        hp_path = f"config/hardpoints/{hp_name}.yml"
        if not os.path.exists(hp_path):
            ui.notify(f"Hardpoints file not found: {hp_path}", type="negative"); return

        try:
            with open(hp_path) as f:
                hp_data = yaml.safe_load(f)
            vehicle = Vehicle(hp_data)
        except Exception as exc:
            ui.notify(f"Failed to build vehicle: {exc}", type="negative"); return

        hp = _corner_for(vehicle, kin_cfg).hardpoints
        free_points_cfg = (opt_cfg or {}).get("FREE_POINTS", {})
        keepout_cfg = (opt_cfg or {}).get("KEEPOUT_ZONES", [])
        groups_cfg = (opt_cfg or {}).get("COLLISION_GROUPS")

        viz_area.clear()
        _reset_display_items()
        cache["scene_objs"] = None
        cache["steps"] = []

        with viz_area:
            ui.label("Optimizer Preview — free-variable ranges & keepout zones").classes(
                "font-bold text-base text-emerald-800")
            with ui.card().classes("w-full p-0 overflow-hidden").style("flex-shrink:0"):
                with ui.row().classes("items-center px-3 py-1 bg-stone-100 border-b border-stone-200"):
                    ui.label("3D Preview").classes("text-sm font-semibold text-stone-700")
                    ui.label("drag to rotate · scroll to zoom · right-drag to pan").classes("text-xs text-stone-400 ml-2")
                with ui.element("div").style("position:relative;width:100%") as scene_wrap:
                    scene3d = ui.scene(width=900, height=460, background_color="#f0f4f8", grid=False).classes("w-full")
                    _render_legend(keepout_cfg, groups_cfg)

        scene_objs = _build_config_preview_scene(scene3d, hp, free_points_cfg, keepout_cfg, groups_cfg)
        _fit_preview_camera(scene3d, hp)
        cache["scene_objs"] = scene_objs
        with scene_wrap:
            _mount_parts_tree(scene_objs, anchor="left")
        status_lbl.text = "Preview ready — configure and press Run to optimize."

    def _parse_num(value_str):
        try:
            return float(value_str)
        except (TypeError, ValueError):
            return None

    def _render_stat_compare_table(current_pairs, cmp_pairs, cmp_label):
        cmp_map = {p[0]: p[1] for p in cmp_pairs}
        rows = []
        for label, cur_val, *_ in current_pairs:
            cmp_val = cmp_map.get(label, "—")
            cur_num, cmp_num = _parse_num(cur_val), _parse_num(cmp_val)
            delta = f"{cur_num - cmp_num:+.3f}" if cur_num is not None and cmp_num is not None else "—"
            rows.append({"metric": label, "current": cur_val, "compare": cmp_val, "delta": delta})
        ui.label("Static Values — Current vs Compare").classes("font-bold text-sm text-stone-700 mt-1")
        ui.table(
            columns=[
                {"name": "metric", "label": "Metric", "field": "metric", "align": "left"},
                {"name": "current", "label": "Current", "field": "current", "align": "right"},
                {"name": "compare", "label": cmp_label, "field": "compare", "align": "right"},
                {"name": "delta", "label": "Δ", "field": "delta", "align": "right"},
            ],
            rows=rows,
        ).classes("text-xs w-full mb-2").props("dense flat")

    def _render_kin(result):
        sim_type, steps, vehicle, cfg, corner_id, run_dir = result

        with viz_area:
            if not steps:
                ui.label("No valid solution steps returned.").classes("text-red-500 text-sm"); return

            _reset_display_items()
            cache.update(steps=steps, sim_type=sim_type, vehicle=vehicle, hp=None)

            if sim_type == "extreme":
                _render_extreme(steps, run_dir, cfg); return

            corner = vehicle.get_corner_from_id(corner_id)
            hp = corner.hardpoints

            cache.update(steps=steps, sim_type=sim_type, vehicle=vehicle, hp=hp)

            cmp_steps = cmp_hp = cmp_label = None
            if compare_state["active"] and compare_state["run_dir"] and sim_type not in NO_COMPARE_SIM_TYPES:
                try:
                    cmp_payload = load_kin_run_data(compare_state["run_dir"])
                    if cmp_payload["sim_type"] == sim_type:
                        cmp_steps = cmp_payload["steps"]
                        cmp_corner_id = cmp_payload.get("corner_id") or corner_id
                        cmp_hp_name = cmp_payload.get("hardpoints_name")
                        cmp_hp_path = os.path.join(compare_state["run_dir"], f"{cmp_hp_name}.yml")
                        if not os.path.exists(cmp_hp_path):
                            cmp_hp_path = f"config/hardpoints/{cmp_hp_name}.yml"
                        cmp_vehicle = Vehicle(yaml.safe_load(open(cmp_hp_path)))
                        cmp_hp = cmp_vehicle.get_corner_from_id(cmp_corner_id).hardpoints
                        cmp_label = run_lookup.get(compare_state["run_dir"], {}).get(
                            "timestamp", compare_state["run_dir"])
                except Exception as exc:
                    ui.notify(f"Comparison run failed to load: {exc}", type="negative")

            if cmp_steps:
                ui.label(f"Comparing against: {cmp_label}").classes("text-xs text-purple-700 bg-purple-50 px-2 py-1 rounded w-fit")

            half_label = "Rear" if corner_id[1] == 1 else "Front"
            if sim_type == "front_steer":
                named_figs, xs = _build_front_steer_figures(steps, cmp_steps=cmp_steps, cmp_label=cmp_label or "Compare")
            elif sim_type in FULL_VEHICLE_TYPES:
                named_figs, xs = _build_full_vehicle_figures(steps, mode=sim_type,
                                                               wr_front=vehicle.front_left.hardpoints.wr,
                                                               wr_rear=vehicle.rear_left.hardpoints.wr,
                                                               cmp_steps=cmp_steps, cmp_label=cmp_label or "Compare")
            else:
                named_figs, xs = _build_kin_figures(steps, half_label=half_label, wr=hp.wr, sim_type=sim_type,
                                                     cmp_steps=cmp_steps, cmp_wr=(cmp_hp.wr if cmp_hp else 0.0),
                                                     cmp_label=cmp_label or "Compare")
            stat_pairs = build_kin_static_values(steps, sim_type, hp, half_label)

            cache["xs"] = xs
            cache["named_figs"] = named_figs

            if cmp_steps:
                cmp_stat_pairs = build_kin_static_values(cmp_steps, sim_type, cmp_hp, half_label)
                _render_stat_compare_table(stat_pairs, cmp_stat_pairs, cmp_label or "Compare")
            elif stat_pairs:
                with ui.row().classes("w-full gap-2 flex-wrap"):
                    for label, value, *rest in stat_pairs:
                        bad = bool(rest and isinstance(rest[0], dict) and rest[0].get("bad"))
                        color = "text-red-600" if bad else "text-emerald-700"
                        with ui.card().classes("px-4 py-2") as stat_card:
                            ui.label(label).classes("text-xs text-stone-500")
                            ui.label(value).classes(f"text-lg font-semibold {color}")
                        _add_display_item(label, stat_card, default_visible=False, category="value")

            # 3D view
            with ui.card().classes("w-full p-0 overflow-hidden").style("flex-shrink:0") as card_3d:
                with ui.row().classes("items-center px-3 py-1 bg-stone-100 border-b border-stone-200"):
                    ui.label("3D View").classes("text-sm font-semibold text-stone-700")
                    ui.label("drag to rotate · scroll to zoom · right-drag to pan").classes("text-xs text-stone-400 ml-2")
                with ui.element("div").style("position:relative;width:100%") as scene_wrap:
                    scene3d = ui.scene(width=900, height=420, background_color="#f0f4f8", grid=False).classes("w-full")
            _add_display_item("3D View", card_3d, default_visible=True, category="3d")

            # build objects on last valid step (widest extent) for camera fit, then update to step 0
            scene_step = _last_scene_step(steps)
            if sim_type == "front_steer" or sim_type in FULL_VEHICLE_TYPES:
                scene_objs = _build_dyn_scene(scene3d, scene_step, vehicle)
                _fit_camera_dyn(scene3d, scene_step, vehicle)
                _update_dyn_scene(scene_objs, steps[0], vehicle)
            else:
                scene_objs = _build_scene(scene3d, scene_step, hp)
                _fit_camera(scene3d, scene_step, hp)
                _update_scene(scene_objs, steps[0], hp)
            cache["scene_objs"] = scene_objs
            with scene_wrap:
                _mount_parts_tree(scene_objs)

            # 3D data plots
            if sim_type == "sweep_space":
                sweep_figs = _build_sweep_space_figures(steps)
                with ui.grid(columns=2 if len(sweep_figs) >= 2 else len(sweep_figs)).classes("w-full gap-2"):
                    for key, fig in sweep_figs:
                        pel = ui.plotly(fig).classes("w-full")
                        _add_display_item(key.replace("_", " ").title(), pel, default_visible=True, category="3d")

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

            cmp_steps = cmp_vehicle = cmp_label = None
            if compare_state["active"] and compare_state["run_dir"] and sim_type not in NO_COMPARE_SIM_TYPES:
                try:
                    cmp_payload = load_dyn_run_data(compare_state["run_dir"])
                    if cmp_payload["sim_type"] == sim_type:
                        cmp_steps = cmp_payload["steps"]
                        cmp_hp_name = cmp_payload.get("hardpoints_name")
                        cmp_hp_path = os.path.join(compare_state["run_dir"], f"{cmp_hp_name}.yml")
                        if not os.path.exists(cmp_hp_path):
                            cmp_hp_path = f"config/hardpoints/{cmp_hp_name}.yml"
                        cmp_vehicle = Vehicle(yaml.safe_load(open(cmp_hp_path)))
                        cmp_label = run_lookup.get(compare_state["run_dir"], {}).get(
                            "timestamp", compare_state["run_dir"])
                except Exception as exc:
                    ui.notify(f"Comparison run failed to load: {exc}", type="negative")

            if cmp_steps:
                ui.label(f"Comparing against: {cmp_label}").classes("text-xs text-purple-700 bg-purple-50 px-2 py-1 rounded w-fit")

            if sim_type != "shock_dyno":
                corner_wr = {
                    "fl": vehicle.front_left.hardpoints.wr,
                    "fr": vehicle.front_right.hardpoints.wr,
                    "rl": vehicle.rear_left.hardpoints.wr,
                    "rr": vehicle.rear_right.hardpoints.wr,
                }
                # Built via build_dyn_static_values() (a thin wrapper around _build_dyn_stats)
                # so the current/compare sides always use the exact same label text.
                stat_pairs = build_dyn_static_values(steps, sim_type, vehicle)
                if cmp_steps:
                    cmp_stat_pairs = build_dyn_static_values(cmp_steps, sim_type, cmp_vehicle)
                    _render_stat_compare_table(stat_pairs, cmp_stat_pairs, cmp_label or "Compare")
                elif stat_pairs:
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
                with ui.element("div").style("position:relative;width:100%") as scene_wrap:
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
            with scene_wrap:
                _mount_parts_tree(scene_objs)

            if sim_type == "shock_dyno":
                named_figs, xs = _build_dyno_figures(steps, cmp_steps=cmp_steps, cmp_label=cmp_label or "Compare")
            else:
                named_figs, xs = _build_dyn_figures(steps, cmp_steps=cmp_steps, cmp_label=cmp_label or "Compare")
                
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
        X = res.X
        obj_names = [o.name for o in optimizer.objectives]

        with viz_area:
            if F is None:
                ui.label("Optimization failed.").classes("text-red-500 text-sm"); return
            if F.ndim == 1:
                F = F.reshape(-1, 1)
            if X is not None and X.ndim == 1:
                X = X.reshape(1, -1)

            mask = np.all(F <= 1e2, axis=1)
            F = F[mask]
            X = X[mask] if X is not None else None
            if not len(F):
                ui.label("No feasible solutions found.").classes("text-orange-500 text-sm"); return

            F_all = np.array(optimizer.all_F) if optimizer.all_F else F.copy()
            if F_all.ndim == 1:
                F_all = F_all.reshape(-1, 1)
            mask_all = np.all(F_all <= 1e2, axis=1)
            F_all = F_all[mask_all] if mask_all.any() else F

            ui.label(f"Pareto front — {len(F)} solutions (of {len(F_all)} evaluated)").classes(
                "font-bold text-base text-emerald-800")

            for _, fig in _build_opt_figures(F_all, F, obj_names):
                ui.plotly(fig).classes("w-full")

            if X is None or not len(X):
                return

            order = rank_solutions(F)

            def sol_label(i):
                s = ", ".join(f"{n}: {v:.4f}" for n, v in zip(obj_names, F[i]))
                return f"Sol {i} — {s}"

            options = {int(i): sol_label(int(i)) for i in order}
            default_idx = int(order[0])

            sol_viz = ui.column().classes("w-full gap-2")

            def render_solution(idx: int):
                sol_viz.clear()
                with sol_viz:
                    with ui.card().classes("w-full p-0 overflow-hidden").style("flex-shrink:0"):
                        with ui.row().classes("items-center px-3 py-1 bg-stone-100 border-b border-stone-200"):
                            ui.label("3D View — Selected Solution").classes("text-sm font-semibold text-stone-700")
                            ui.label("drag to rotate · scroll to zoom · right-drag to pan").classes("text-xs text-stone-400 ml-2")
                        with ui.element("div").style("position:relative;width:100%") as scene_wrap:
                            scene3d = ui.scene(width=900, height=460, background_color="#f0f4f8", grid=False).classes("w-full")
                            _render_legend(cfg.get("KEEPOUT_ZONES", []), cfg.get("COLLISION_GROUPS"))

                    vehicle = optimizer.create_vehicle_from_ref(X[idx])
                    hp = _corner_for(vehicle, cfg).hardpoints
                    scene_objs = _build_config_preview_scene(scene3d, hp, cfg.get("FREE_POINTS", {}),
                                                             cfg.get("KEEPOUT_ZONES", []), cfg.get("COLLISION_GROUPS"))
                    _fit_preview_camera(scene3d, hp)
                    cache["scene_objs"] = scene_objs
                    with scene_wrap:
                        _mount_parts_tree(scene_objs, anchor="left")

                    free_points_cfg = cfg.get("FREE_POINTS", {})
                    if free_points_cfg:
                        ui.label("Free Point Positions").classes("font-bold text-sm text-stone-700 mt-1")
                        rows = []
                        for pt_name in free_points_cfg:
                            attr = _resolve_point_attr(hp, pt_name)
                            if attr is None:
                                continue
                            x, y, z = (float(v) for v in getattr(hp, attr))
                            rows.append({"point": pt_name, "x": f"{x:.3f}", "y": f"{y:.3f}", "z": f"{z:.3f}"})
                        if rows:
                            ui.table(
                                columns=[
                                    {"name": "point", "label": "Point", "field": "point", "align": "left"},
                                    {"name": "x", "label": "X [mm]", "field": "x", "align": "right"},
                                    {"name": "y", "label": "Y [mm]", "field": "y", "align": "right"},
                                    {"name": "z", "label": "Z [mm]", "field": "z", "align": "right"},
                                ],
                                rows=rows,
                            ).classes("text-xs w-full").props("dense flat")

            with ui.row().classes("w-full items-center gap-2"):
                ui.label("Viewing solution:").classes("text-sm text-stone-600")
                sol_select = ui.select(options=options, value=default_idx).classes("w-96").props("dense outlined")
            sol_select.on_value_change(lambda e: render_solution(int(e.value)))

            render_solution(default_idx)

    # wire
    mode_sel.on_value_change(on_mode_change)
    subtype_sel.on_value_change(on_type_change)
    save_btn.on_click(do_save)
    preview_btn.on_click(do_preview)
    run_btn.on_click(do_run)
    play_btn.on_click(do_play_pause)
    refresh_runs_btn.on_click(lambda: _refresh_run_options())
    load_run_btn.on_click(do_load_run)
    compare_btn.on_click(do_compare)
    compare_clear_btn.on_click(do_clear_compare)

    # initialize button text
    try:
        cfg = yaml.safe_load(open(KIN_PATH).read())
        if cfg and isinstance(cfg, dict):
            hp_button.text = f"EDIT HARDPOINTS"
    except Exception:
        pass

    _refresh_run_options()

ui.run(title="SAGE", port=8080, reload=False, show=True, favicon="🌿")
