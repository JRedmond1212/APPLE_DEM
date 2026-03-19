from __future__ import annotations

import time
from typing import Any, Dict, Tuple

import pandas as pd
import streamlit as st

try:
    import Pre_processing as prep
    from MCS.MonteCarlo_module import run_monte_carlo

    # ✅ Macro growth now lives inside DES/
    from DES.Macro_growth import run_macro_growth

    from Post_processing.Post_processing import (
        render_overview_tab,
        render_growth_tab,
        render_harvest_tab,
        render_storage_tab,
    )

    from Post_processing.Post_processing import render_macro_overview_tab

except Exception as e:
    prep = None
    run_monte_carlo = None
    run_macro_growth = None
    render_overview_tab = None
    render_growth_tab = None
    render_harvest_tab = None
    render_storage_tab = None
    render_macro_overview_tab = None
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None


st.set_page_config(page_title="APPLE DEM", layout="wide", initial_sidebar_state="expanded")
st.title("APPLE Fruit — Discrete Event Model")


def _init_state() -> None:
    defaults = {
        "draft_config": None,
        "active_config": None,
        "weather_cache": None,
        "sim_results": None,
        "last_run_time_s": None,
        "last_run_timestamp": None,
        "run_counter": 0,
        "weather_cache_store": {},

        # optional macro-specific cache bucket
        "macro_weather_cache_by_county": {},

        "progress": {
            "stage": "Idle",
            "tasks_done": 0,
            "tasks_total": 0,
            "elapsed_s": 0.0,
            "eta_s": None,
            "mc_index": 0,
            "mc_total": 0,
            "macro_orchard_index": 0,
            "macro_orchard_total": 0,
            "macro_county": "",
        },
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


def _sidebar_config() -> Dict[str, Any]:
    if prep is None:
        st.sidebar.error("Modules missing. Add project files then refresh.")
        st.sidebar.exception(_IMPORT_ERROR)
        return {}

    st.sidebar.header("Simulation Controls")

    sim_mode = st.sidebar.radio(
        "Select Simulation Mode",
        ["Micro (Single Orchard)", "Macro (Multiple Orchards)", "Digital Shadow (Custom Micro)"],
        index=0,
    )

    mc_runs = st.sidebar.slider("Monte Carlo runs", 10, 5000, 2000, 10)

    # MICRO + DT use your existing builder
    if sim_mode != "Macro (Multiple Orchards)":
        cfg = prep.build_sidebar_config(sim_mode=sim_mode, mc_runs=mc_runs)
        if not isinstance(cfg, dict):
            raise TypeError("Pre_processing.build_sidebar_config must return dict.")
        return cfg

    # --------------------------------------------------------
    # MACRO sidebar: simplified
    # --------------------------------------------------------
    cfg = prep.normalize_config({
        "sim_mode": sim_mode,
        "mc_runs": int(mc_runs),
        "macro_start_year": 1985,
        "macro_end_year": 2024,
        "base_seed": 12345,
    })
    return cfg


st.session_state["draft_config"] = _sidebar_config()


run_col_l, _ = st.columns([1, 2], vertical_alignment="center")
with run_col_l:
    run_pressed = st.button("▶ Run Sim", type="primary", use_container_width=True)

progress_box = st.container()
with progress_box:
    st.markdown("### Run progress")
    prog_stage = st.empty()
    prog_meta = st.empty()
    prog_bar = st.progress(0.0)


def _render_progress(p: Dict[str, Any]) -> None:
    stage = p.get("stage", "Idle")
    tasks_done = int(p.get("tasks_done", 0) or 0)
    tasks_total = int(p.get("tasks_total", 0) or 0)
    elapsed_s = float(p.get("elapsed_s", 0.0) or 0.0)
    eta_s = p.get("eta_s", None)

    mc_i = int(p.get("mc_index", 0) or 0)
    mc_n = int(p.get("mc_total", 0) or 0)

    macro_county = str(p.get("macro_county", "") or "")

    eta_txt = "—" if eta_s is None else f"{float(eta_s):.1f}s"
    prog_stage.markdown(f"**Current task:** {stage}")

    county_txt = f" | **Current orchard:** {macro_county}" if macro_county else ""

    prog_meta.markdown(
        f"**Tasks:** {tasks_done}/{tasks_total} | **MC:** {mc_i}/{mc_n} | "
        f"**Elapsed:** {elapsed_s:.1f}s | **ETA:** {eta_txt}{county_txt}"
    )

    frac = 0.0
    if tasks_total > 0:
        frac = min(1.0, tasks_done / tasks_total)
    prog_bar.progress(frac)


_render_progress(st.session_state["progress"])


def _run_pipeline_micro_dt(active_cfg: Dict[str, Any]) -> Tuple[Dict[str, Any], Any, Any, float]:
    if prep is None or run_monte_carlo is None:
        raise RuntimeError("Required modules missing.")

    t0 = time.time()
    mc_total = int(active_cfg.get("mc_runs", 0))
    tasks_total = 1 + mc_total
    last_ui_update = 0.0

    def progress_cb(stage: str, i: int, n: int) -> None:
        nonlocal last_ui_update
        elapsed = time.time() - t0

        eta = None
        if stage.startswith("Monte Carlo") and i > 0 and n > 0:
            rate = elapsed / max(1, i)
            eta = max(0.0, (n - i) * rate)

        if stage.startswith("Pre-processing"):
            tasks_done = 0
        elif stage.startswith("Monte Carlo"):
            tasks_done = 1 + i
        else:
            tasks_done = min(tasks_total, 1 + n)

        st.session_state["progress"] = {
            "stage": stage,
            "tasks_done": tasks_done,
            "tasks_total": tasks_total,
            "elapsed_s": elapsed,
            "eta_s": eta,
            "mc_index": i,
            "mc_total": n,
            "macro_orchard_index": 0,
            "macro_orchard_total": 0,
            "macro_county": "",
        }

        now = time.time()
        if now - last_ui_update >= 0.2:
            _render_progress(st.session_state["progress"])
            last_ui_update = now

    progress_cb("Pre-processing (weather + arrays + weekly demand)", 0, 0)
    weather_cache = prep.get_or_build_weather_cache(active_cfg)

    progress_cb("Monte Carlo + DES", 0, mc_total)
    sim_results = run_monte_carlo(
        config=active_cfg,
        weather_cache=weather_cache,
        progress_callback=progress_cb,
    )

    runtime_s = time.time() - t0
    progress_cb("Done", mc_total, mc_total)
    _render_progress(st.session_state["progress"])
    return active_cfg, weather_cache, sim_results, runtime_s


def _run_pipeline_macro(active_cfg: Dict[str, Any]) -> Tuple[Dict[str, Any], Any, Any, float]:
    if prep is None or run_macro_growth is None:
        raise RuntimeError("Macro modules missing.")

    t0 = time.time()
    last_ui_update = 0.0

    start_year = int(active_cfg.get("macro_start_year", 1985))
    end_year = int(active_cfg.get("macro_end_year", 2024))
    if end_year < start_year:
        end_year = start_year

    base_defaults = prep.macro_defaults_from_micro() if hasattr(prep, "macro_defaults_from_micro") else prep.normalize_config({})
    base_defaults["sim_mode"] = "Macro (Multiple Orchards)"

    def set_progress(
        *,
        stage: str,
        tasks_done: int,
        tasks_total: int,
        eta_s: float | None = None,
        mc_index: int = 0,
        mc_total: int = 0,
        macro_orchard_index: int = 0,
        macro_orchard_total: int = 0,
        macro_county: str = "",
    ) -> None:
        nonlocal last_ui_update
        st.session_state["progress"] = {
            "stage": stage,
            "tasks_done": int(tasks_done),
            "tasks_total": int(tasks_total),
            "elapsed_s": time.time() - t0,
            "eta_s": eta_s,
            "mc_index": int(mc_index),
            "mc_total": int(mc_total),
            "macro_orchard_index": int(macro_orchard_index),
            "macro_orchard_total": int(macro_orchard_total),
            "macro_county": str(macro_county),
        }
        now = time.time()
        if now - last_ui_update >= 0.15 or tasks_done >= tasks_total:
            _render_progress(st.session_state["progress"])
            last_ui_update = now

    # --------------------------------------------------------
    # Stage 1: load points from templates
    # --------------------------------------------------------
    set_progress(stage="Macro: loading template points", tasks_done=0, tasks_total=1)

    points_df = prep.build_macro_points_from_templates(
        base_cfg=base_defaults,
        start_year=start_year,
        end_year=end_year,
    )

    # --------------------------------------------------------
    # Stage 2: build county area + age density
    # --------------------------------------------------------
    set_progress(stage="Macro: building county area + age density", tasks_done=0, tasks_total=1)

    from DES.Macro_growth import build_county_area_by_year_from_points, build_county_age_density_by_year

    county_area_by_year = build_county_area_by_year_from_points(points_df)
    county_age_by_year = build_county_age_density_by_year(
        county_area_by_year,
        start_year=start_year,
        end_year=end_year,
    )

    # --------------------------------------------------------
    # Stage 3: load weather caches
    # --------------------------------------------------------
    set_progress(stage="Macro: loading weather caches", tasks_done=0, tasks_total=1)

    wc_by_county = prep.build_macro_weather_cache_by_county(
        base_cfg=base_defaults,
        county_area_by_year=county_area_by_year,
        start_year=start_year,
        end_year=end_year,
    )

    # --------------------------------------------------------
    # Stage 4: macro growth with detailed progress
    # --------------------------------------------------------
    counties = []
    if hasattr(points_df, "columns") and "county" in points_df.columns:
        counties = sorted(pd.Series(points_df["county"]).dropna().astype(str).unique().tolist())

    macro_total_runs = max(1, len(counties) * int(active_cfg.get("mc_runs", 2000)))

    def macro_progress_cb(stage: str, done: int, total: int, mc_i: int, mc_n: int, county_name: str) -> None:
        elapsed = time.time() - t0
        eta = None
        if done > 0 and total > 0:
            rate = elapsed / float(done)
            eta = max(0.0, (total - done) * rate)

        set_progress(
            stage=stage,
            tasks_done=done,
            tasks_total=total,
            eta_s=eta,
            mc_index=mc_i,
            mc_total=mc_n,
            macro_orchard_index=done,
            macro_orchard_total=total,
            macro_county=county_name,
        )

    macro_progress_cb(
        "Macro: running orchard simulations",
        0,
        macro_total_runs,
        0,
        int(active_cfg.get("mc_runs", 2000)),
        "",
    )

    macro_results = run_macro_growth(
        base_micro_defaults=base_defaults,
        county_area_by_year=county_area_by_year,
        county_age_by_year=county_age_by_year,
        weather_cache_by_county=wc_by_county,
        mc_runs=int(active_cfg.get("mc_runs", 2000)),
        base_seed=int(active_cfg.get("base_seed", 12345)),
        progress_callback=macro_progress_cb,
    )

    set_progress(
        stage="Done",
        tasks_done=macro_total_runs,
        tasks_total=macro_total_runs,
        eta_s=0.0,
        mc_index=int(active_cfg.get("mc_runs", 2000)),
        mc_total=int(active_cfg.get("mc_runs", 2000)),
        macro_orchard_index=macro_total_runs,
        macro_orchard_total=macro_total_runs,
        macro_county="",
    )

    runtime_s = time.time() - t0
    return active_cfg, None, macro_results, runtime_s


if run_pressed:
    draft = st.session_state.get("draft_config") or {}
    if not draft:
        st.error("Draft config invalid.")
    else:
        st.session_state["run_counter"] += 1
        st.session_state["active_config"] = dict(draft)
        st.session_state["last_run_timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

        try:
            if str(draft.get("sim_mode")) == "Macro (Multiple Orchards)":
                cfg, weather_cache, sim_results, runtime_s = _run_pipeline_macro(st.session_state["active_config"])
            else:
                cfg, weather_cache, sim_results, runtime_s = _run_pipeline_micro_dt(st.session_state["active_config"])
        except Exception as e:
            st.session_state["weather_cache"] = None
            st.session_state["sim_results"] = None
            st.session_state["last_run_time_s"] = None
            st.error("Simulation failed:")
            st.exception(e)
        else:
            st.session_state["weather_cache"] = weather_cache
            st.session_state["sim_results"] = sim_results
            st.session_state["last_run_time_s"] = float(runtime_s)


active_cfg = st.session_state.get("active_config")
sim_results = st.session_state.get("sim_results")
weather_cache = st.session_state.get("weather_cache")

if not active_cfg or sim_results is None:
    st.stop()


# ============================================================
# UI ROUTING
# ============================================================

mode = str(active_cfg.get("sim_mode", ""))

if mode == "Macro (Multiple Orchards)":
    tabs = st.tabs(["Macro Overview"])
    with tabs[0]:
        render_macro_overview_tab(sim_results=sim_results)
    st.stop()


tabs = st.tabs(["Overview", "Growth", "Harvest&Grading", "Storage&Distrubtion"])

with tabs[0]:
    render_overview_tab(config=active_cfg, sim_results=sim_results)

    #if "Digital Twin" in str(active_cfg.get("sim_mode", "")):
        #with st.expander("Digital Twin: Save Micro weather template", expanded=False):
         #   st.caption("Download weather array (Doesn't work on web app).")

          #  default_name = (
           #     f"DT_lat{active_cfg.get('lat',0):.4f}_lon{active_cfg.get('lon',0):.4f}_"
            #    f"{active_cfg.get('planting_year',0)}_{active_cfg.get('years_to_sim',0)}y"
        #    ).replace(":", "_").replace("/", "_")

         #   template_name = st.text_input("Template name", value=default_name)

          #  colA, colB = st.columns([1, 2])
           # with colA:
            #    save_pressed = st.button("💾 Save template to WeatherTemplates/templates", type="primary", use_container_width=True)
          #  with colB:
          #      st.write("Target folder: `WeatherTemplates/templates/`")

           # if save_pressed:
          #      try:
           #         csv_path, npz_path, json_path = prep.save_weather_template_to_project_folder(
            #            weather_cache,
             #           template_name.strip(),
          #    #      )
           #     except Exception as e:
            #        st.error("Template save failed:")
             #       st.exception(e)
              #  else:
              #      st.success("Template saved.")
               #     st.code(f"{npz_path}\n{csv_path}\n{json_path}")

with tabs[1]:
    render_growth_tab(config=active_cfg, sim_results=sim_results, weather_cache=weather_cache)

with tabs[2]:
    render_harvest_tab(config=active_cfg, sim_results=sim_results)

with tabs[3]:
    render_storage_tab(config=active_cfg, sim_results=sim_results)