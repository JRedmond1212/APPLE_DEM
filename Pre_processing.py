from __future__ import annotations

import hashlib
import json
import os
from datetime import date, datetime, timedelta
from typing import Any, Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import requests
import streamlit as st

from WeatherTemplates.weather_template_io import (
    list_templates,
    load_template_from_folder,
    save_template_to_folder,
    weather_cache_to_template_bytes,
)

# ✅ Macro logic lives in DES/
from DES.Macro_growth import (
    build_county_area_by_year_from_points,
    build_county_age_density_by_year,
)

TEMPLATE_FOLDER = os.path.join("WeatherTemplates", "templates")
MACRO_TEMPLATE_FOLDER = os.path.join("WeatherTemplates", "macro")

MICRO_FIXED_CONFIG: Dict[str, Any] = {
    "lat": 54.4460,
    "lon": -6.5292,
    "variety": "Bramley Seedling",
    "rootstock": "M25",
    "tree_density": 450,
    "orchard_area": 12.0,
    "planting_year": 1985,
    "years_to_sim": 50,
    "pruning_target": 20.0,
    "thinning_target": 30,
    "kg_per_bin": 350.0,
    "field_capacity": 100,
    "pregrading_capacity": 50,
    "long_term_capacity": 2000,
    "latent_grade_mix": {"Extra": 20, "Class1": 40, "Class2": 30, "Processor": 10},
    "workers": {
        "EmptyBinShuttle": {"n": 1.0, "mu": 60.0, "sigma": 10.0},
        "Harvesters": {"n": 8.0, "mu": 6.0, "sigma": 2.0},
        "FilledBinShuttle": {"n": 1.0, "mu": 60.0, "sigma": 10.0},
        "Graders": {"n": 1.0, "mu": 120.0, "sigma": 20.0},
    },
    "decay_constants": {
        "BinsOnTrees": {"Extra": 0.031878, "Class1": 0.024384, "Class2": 0.01758, "Processor": 0.14657},
        "FieldBins": {"Extra": 0.09381, "Class1": 0.069179, "Class2": 0.09010, "Processor": 0.28210},
        "PreGrading": {"Extra": 0.11878, "Class1": 0.094384, "Class2": 0.13758, "Processor": 0.334657},
        "LongTermStorage": {"Extra": 0.002479, "Class1": 0.003196, "Class2": 0.003379, "Processor": 0.017329},
    },
    "demand_grade_mix": {"Extra": 0.15, "Class1": 0.40, "Class2": 0.10, "Processor": 0.35},
    "policy_harvest": "FIFO",
    "policy_grading": "FIFO",
    "policy_storage": "FEFO",

    # ✅ changed default from None to Case_Study
    "micro_template_name": "Case_Study",
}

MONTH_THA = {
    9: 2.25, 10: 6.18, 11: 7.52, 12: 6.55,
    1: 7.87, 2: 7.27, 3: 8.21, 4: 6.98,
    5: 5.31, 6: 3.51, 7: 1.27, 8: 0.82,
}

GRADES = ["Extra", "Class1", "Class2", "Processor"]


def build_weekly_demand_profile_once(config: Dict[str, Any]) -> Dict[str, Any]:
    orchard_area = float(config.get("orchard_area", 1.0))
    kg_per_bin = float(config.get("kg_per_bin", 350.0))
    demand_factor = float(config.get("what_if_demand_factor", 1.0))

    base_seed = int(config.get("demand_seed", 12345))
    seed_blob = json.dumps(
        {
            "orchard_area": orchard_area,
            "kg_per_bin": kg_per_bin,
            "demand_factor": demand_factor,
            "base_seed": base_seed,
            "month_tha": MONTH_THA,
        },
        sort_keys=True,
    ).encode("utf-8")
    seed = int(hashlib.md5(seed_blob).hexdigest()[:8], 16) & 0xFFFFFFFF
    rng = np.random.default_rng(seed)

    ref_year = 2001
    month_lengths = {m: pd.Period(f"{ref_year}-{m:02d}").days_in_month for m in range(1, 13)}

    daily_bins = np.zeros(365, dtype=float)
    day_idx = 0
    for m in range(1, 13):
        tha = float(MONTH_THA.get(m, 0.0))
        L = int(month_lengths[m])
        if tha <= 0:
            day_idx += L
            continue

        monthly_tonnes = tha * max(0.0, orchard_area)
        monthly_kg = monthly_tonnes * 1000.0
        monthly_bins = (monthly_kg / max(1e-9, kg_per_bin)) * max(0.0, demand_factor)

        w = rng.gamma(shape=2.0, scale=1.0, size=L)
        ws = float(np.sum(w))
        w = (np.ones(L, dtype=float) / float(L)) if ws <= 1e-12 else (w / ws)
        daily_bins[day_idx:day_idx + L] = monthly_bins * w
        day_idx += L

    weekly = np.zeros(52, dtype=float)
    for w in range(51):
        weekly[w] = float(np.sum(daily_bins[w * 7:(w + 1) * 7]))
    weekly[51] = float(np.sum(daily_bins[51 * 7:]))

    grade_mix = config.get(
        "demand_grade_mix",
        {"Extra": 0.25, "Class1": 0.40, "Class2": 0.25, "Processor": 0.10},
    )
    mix_sum = float(sum(float(grade_mix.get(g, 0.0)) for g in GRADES)) or 1.0
    mix = {g: float(grade_mix.get(g, 0.0)) / mix_sum for g in GRADES}

    profile = {
        "weeks": 52,
        "weekly_total_bins": weekly.astype(np.float32),
        "grade_mix": mix,
    }

    by_grade = {g: (profile["weekly_total_bins"] * float(mix.get(g, 0.0))).astype(np.float32) for g in GRADES}
    profile["by_grade"] = by_grade
    return profile


def normalize_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(cfg)

    out.setdefault("mc_runs", 100)
    out.setdefault("base_seed", 12345)

    out.setdefault("fast_mode", True)
    out.setdefault("quality_samples_cap", 6000)
    out.setdefault("store_storage_quality_hist", True)
    out.setdefault("demand_seed", 12345)

    out.setdefault("orchard_area", 1.0)
    out.setdefault("tree_density", 1500)
    out.setdefault("pruning_target", 20.0)
    out.setdefault("thinning_target", 30)
    out.setdefault("planting_year", 1985)
    out.setdefault("years_to_sim", 50)

    out.setdefault("variety", "Bramley Seedling")
    out.setdefault("rootstock", "M9")

    out.setdefault("chill_temp_min", 0.0)
    out.setdefault("chill_temp_max", 7.0)
    out.setdefault("forcing_temp_min", 3.5)

    out.setdefault("frost_kill_10", -2.0)
    out.setdefault("frost_kill_90", -4.0)

    out.setdefault("heat_loss_10", 25.0)
    out.setdefault("heat_loss_40", 30.0)

    out.setdefault("no_bee_flight_rain_mm", 1.0)
    out.setdefault("rain_off_mm", 5.0)
    out.setdefault("sun_hour_threshold_wm2", 120.0)

    out.setdefault("kg_per_bin", 350.0)
    out.setdefault("field_capacity", 500)
    out.setdefault("pregrading_capacity", 300)
    out.setdefault("long_term_capacity", 2000)

    out.setdefault("latent_grade_mix", {"Extra": 30, "Class1": 40, "Class2": 20, "Processor": 10})
    out.setdefault(
        "workers",
        {
            "EmptyBinShuttle": {"n": 5.0, "mu": 50.0, "sigma": 10.0},
            "Harvesters": {"n": 5.0, "mu": 50.0, "sigma": 10.0},
            "FilledBinShuttle": {"n": 5.0, "mu": 50.0, "sigma": 10.0},
            "Graders": {"n": 5.0, "mu": 50.0, "sigma": 10.0},
        },
    )

    out.setdefault(
        "decay_constants",
        {
            "BinsOnTrees": {"Extra": 0.005, "Class1": 0.005, "Class2": 0.005, "Processor": 0.005},
            "FieldBins": {"Extra": 0.005, "Class1": 0.005, "Class2": 0.005, "Processor": 0.005},
            "PreGrading": {"Extra": 0.005, "Class1": 0.005, "Class2": 0.005, "Processor": 0.005},
            "LongTermStorage": {"Extra": 0.0008, "Class1": 0.0008, "Class2": 0.0008, "Processor": 0.0008},
        },
    )

    out.setdefault("demand_grade_mix", {"Extra": 0.25, "Class1": 0.40, "Class2": 0.25, "Processor": 0.10})

    out.setdefault("what_if_workers_factor", 1.0)
    out.setdefault("what_if_bins_factor", 1.0)
    out.setdefault("what_if_storage_factor", 1.0)
    out.setdefault("what_if_demand_factor", 1.0)

    out.setdefault("policy_harvest", "FIFO")
    out.setdefault("policy_grading", "FIFO")
    out.setdefault("policy_storage", "FEFO")

    out.setdefault("cc_ramp_years", 30)
    out.setdefault("cc_winter_temp_warm_c", 2.0)
    out.setdefault("cc_winter_precip_wet_frac", 0.15)
    out.setdefault("cc_summer_temp_warm_c", 3.0)
    out.setdefault("cc_summer_precip_dry_frac", 0.30)

    out.setdefault("lat", 54.446)
    out.setdefault("lon", -6.5292)

    # ✅ changed default from None to Case_Study
    out.setdefault("micro_template_name", "Case_Study")

    out.setdefault("long_term_capacity_base", int(out.get("long_term_capacity", 2000)))

    return out


def build_sidebar_config(sim_mode: str, mc_runs: int) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {"sim_mode": sim_mode, "mc_runs": int(mc_runs)}

    mode = str(sim_mode or "").strip()
    is_micro = (mode == "Micro (Single Orchard)")
    is_macro = (mode == "Macro (Multiple Orchards)")
    is_dt = ("Digital Twin" in mode)

    if is_micro or is_dt:
        cfg.update(dict(MICRO_FIXED_CONFIG))

    if is_micro:
        st.sidebar.subheader("Micro Orchard Location")
        names = list_templates(TEMPLATE_FOLDER)

        if not names:
            st.sidebar.error(
                f"No templates found in: {TEMPLATE_FOLDER}\n\n"
                "Run Digital Twin and save a template, then come back here."
            )
            template_name = None
        else:
            # ✅ default to Case_Study if present, otherwise first template
            default_template = str(cfg.get("micro_template_name", "Case_Study") or "Case_Study")
            if default_template in names:
                default_index = names.index(default_template)
            else:
                default_index = 0

            template_name = st.sidebar.selectbox(
                "Location",
                names,
                index=default_index,
                key="micro_template_select",
            )

        cfg["micro_template_name"] = template_name

    if is_macro:
        cfg.update(dict(MICRO_FIXED_CONFIG))
        cfg["micro_template_name"] = None

    if is_dt:
        st.sidebar.subheader("Digital Twin — Module Inputs")

        eff = dict(cfg)

        with st.sidebar.expander("Growth", expanded=False):
            lat = st.number_input("Latitude", value=float(eff.get("lat", 54.4460)), format="%.6f", key="dt_lat")
            lon = st.number_input("Longitude", value=float(eff.get("lon", -6.5292)), format="%.6f", key="dt_lon")

            variety_list = ["Bramley Seedling", "Discovery", "Cox's Orange Pippin", "Egremont Russet", "Elstar"]
            v_cur = str(eff.get("variety", "Bramley Seedling"))
            variety = st.selectbox(
                "Variety",
                variety_list,
                index=variety_list.index(v_cur) if v_cur in variety_list else 0,
                key="dt_variety",
            )

            rootstock_list = ["M27", "M9", "M26", "MM106", "MM111", "M25"]
            r_cur = str(eff.get("rootstock", "M25"))
            rootstock = st.selectbox(
                "Rootstock",
                rootstock_list,
                index=rootstock_list.index(r_cur) if r_cur in rootstock_list else len(rootstock_list) - 1,
                key="dt_rootstock",
            )

            tree_density = st.slider(
                "Tree density (trees/ha)",
                200,
                3000,
                int(eff.get("tree_density", 450)),
                step=50,
                key="dt_tree_density",
            )

            orchard_area = st.number_input(
                "Orchard area (ha)",
                value=float(eff.get("orchard_area", 12.0)),
                min_value=0.1,
                step=0.1,
                key="dt_orch_area",
            )

            planting_year = st.number_input(
                "Planting year",
                value=int(eff.get("planting_year", 1985)),
                step=1,
                key="dt_planting_year",
            )

            years_to_sim = st.slider(
                "Years to simulate",
                1,
                50,
                int(eff.get("years_to_sim", 50)),
                key="dt_years_to_sim",
            )

            pruning_target = st.number_input(
                "Leaf area target after pruning (m²/tree)",
                value=float(eff.get("pruning_target", 20.0)),
                min_value=0.1,
                step=0.1,
                key="dt_pruning_target",
            )

            thinning_target = st.slider(
                "Thinning target (%)",
                0,
                90,
                int(eff.get("thinning_target", 30)),
                key="dt_thinning_target",
            )

        with st.sidebar.expander("Harvest & Grading", expanded=False):
            kg_per_bin = st.number_input(
                "Kg of apples per bin",
                value=float(eff.get("kg_per_bin", 350.0)),
                min_value=1.0,
                step=5.0,
                key="dt_kg_per_bin",
            )

            st.markdown("**Latent grade mix (%)**")
            lg = eff.get("latent_grade_mix", {"Extra": 20, "Class1": 40, "Class2": 30, "Processor": 10}) or {}
            lg_extra = st.slider("Extra (%)", 0, 100, int(lg.get("Extra", 20)), key="dt_lg_extra")
            lg_c1 = st.slider("Class1 (%)", 0, 100, int(lg.get("Class1", 40)), key="dt_lg_c1")
            lg_c2 = st.slider("Class2 (%)", 0, 100, int(lg.get("Class2", 30)), key="dt_lg_c2")
            lg_proc = st.slider("Processor (%)", 0, 100, int(lg.get("Processor", 10)), key="dt_lg_proc")
            lg_sum = max(1, lg_extra + lg_c1 + lg_c2 + lg_proc)
            latent_grade_mix = {
                "Extra": int(round(100 * lg_extra / lg_sum)),
                "Class1": int(round(100 * lg_c1 / lg_sum)),
                "Class2": int(round(100 * lg_c2 / lg_sum)),
                "Processor": int(round(100 * lg_proc / lg_sum)),
            }

            st.markdown("**Workers**")
            workers_base = eff.get("workers_base", None)
            if not isinstance(workers_base, dict):
                workers_base = eff.get("workers", {}) or {}

            def _role_inputs(role: str, default: Dict[str, Any]) -> Dict[str, float]:
                st.markdown(f"*{role}*")
                n = st.number_input(
                    f"{role} — workers",
                    0.0,
                    200.0,
                    float(default.get("n", 1.0)),
                    step=0.25,
                    key=f"dt_w_n_{role}",
                )
                mu = st.number_input(
                    f"{role} — mean throughput/day (bins)",
                    0.0,
                    2000.0,
                    float(default.get("mu", 60.0)),
                    step=1.0,
                    key=f"dt_w_mu_{role}",
                )
                sigma = st.number_input(
                    f"{role} — std dev throughput/day (bins)",
                    0.0,
                    2000.0,
                    float(default.get("sigma", 10.0)),
                    step=1.0,
                    key=f"dt_w_sigma_{role}",
                )
                return {"n": float(n), "mu": float(mu), "sigma": float(sigma)}

            workers_base_new = {
                "EmptyBinShuttle": _role_inputs("EmptyBinShuttle", workers_base.get("EmptyBinShuttle", {"n": 1.0, "mu": 60.0, "sigma": 10.0})),
                "Harvesters": _role_inputs("Harvesters", workers_base.get("Harvesters", {"n": 8.0, "mu": 6.0, "sigma": 2.0})),
                "FilledBinShuttle": _role_inputs("FilledBinShuttle", workers_base.get("FilledBinShuttle", {"n": 1.0, "mu": 60.0, "sigma": 10.0})),
                "Graders": _role_inputs("Graders", workers_base.get("Graders", {"n": 1.0, "mu": 120.0, "sigma": 20.0})),
            }

            field_capacity = int(st.number_input("Field capacity (bins)", value=int(eff.get("field_capacity", 100)), min_value=0, step=10, key="dt_field_cap"))
            pregrading_capacity = int(st.number_input("Pre-grading capacity (bins)", value=int(eff.get("pregrading_capacity", 50)), min_value=0, step=10, key="dt_pregrade_cap"))

        with st.sidebar.expander("Storage & Distribution", expanded=False):
            long_term_capacity_base = int(
                st.number_input(
                    "Long-term storage capacity (bins)",
                    value=int(eff.get("long_term_capacity_base", eff.get("long_term_capacity", 2000))),
                    min_value=0,
                    step=50,
                    key="dt_ltc_base",
                )
            )

            st.markdown("**Decay constants (k)**")
            decay_constants = eff.get("decay_constants", {}) or {}
            stages = ["BinsOnTrees", "FieldBins", "PreGrading", "LongTermStorage"]
            grades = ["Extra", "Class1", "Class2", "Processor"]
            decay_out: Dict[str, Dict[str, float]] = {}
            for stage in stages:
                st.markdown(f"*{stage}*")
                decay_out[stage] = {}
                for grade in grades:
                    default_k = 0.005 if stage != "LongTermStorage" else 0.0008
                    v = float(decay_constants.get(stage, {}).get(grade, default_k))
                    decay_out[stage][grade] = float(
                        st.number_input(
                            f"k {stage}/{grade}",
                            value=v,
                            format="%.6f",
                            key=f"dt_decay_{stage}_{grade}",
                        )
                    )

            st.markdown("**Demand grade mix (%)**")
            dm = eff.get("demand_grade_mix", {"Extra": 0.15, "Class1": 0.40, "Class2": 0.10, "Processor": 0.35}) or {}
            dm_extra = st.slider("Demand Extra", 0, 100, int(round(100 * float(dm.get("Extra", 0.15)))), key="dt_dm_extra")
            dm_c1 = st.slider("Demand Class1", 0, 100, int(round(100 * float(dm.get("Class1", 0.40)))), key="dt_dm_c1")
            dm_c2 = st.slider("Demand Class2", 0, 100, int(round(100 * float(dm.get("Class2", 0.10)))), key="dt_dm_c2")
            dm_proc = st.slider("Demand Processor", 0, 100, int(round(100 * float(dm.get("Processor", 0.35)))), key="dt_dm_proc")
            dm_sum = max(1, dm_extra + dm_c1 + dm_c2 + dm_proc)
            demand_grade_mix = {
                "Extra": float(dm_extra) / dm_sum,
                "Class1": float(dm_c1) / dm_sum,
                "Class2": float(dm_c2) / dm_sum,
                "Processor": float(dm_proc) / dm_sum,
            }

        cfg.update(
            {
                "lat": float(lat),
                "lon": float(lon),
                "variety": str(variety),
                "rootstock": str(rootstock),
                "tree_density": int(tree_density),
                "orchard_area": float(orchard_area),
                "planting_year": int(planting_year),
                "years_to_sim": int(years_to_sim),
                "pruning_target": float(pruning_target),
                "thinning_target": int(thinning_target),
                "kg_per_bin": float(kg_per_bin),
                "latent_grade_mix": latent_grade_mix,
                "workers_base": workers_base_new,
                "field_capacity": int(field_capacity),
                "pregrading_capacity": int(pregrading_capacity),
                "long_term_capacity_base": int(long_term_capacity_base),
                "decay_constants": decay_out,
                "demand_grade_mix": demand_grade_mix,
            }
        )

    st.sidebar.divider()
    with st.sidebar.expander("what if", expanded=False):
        workers_pct = float(st.slider("Available workers (%)", 0, 200, 100, 5, key="wi_workers"))
        bins_pct = float(st.slider("Available bins (%)", 0, 200, 100, 5, key="wi_bins"))
        storage_pct = float(st.slider("Cold storage capacity (%)", 0, 200, 100, 5, key="wi_storage"))
        demand_pct = float(st.slider("Demand (%)", 0, 200, 100, 5, key="wi_demand"))

        policy_options = ["FIFO", "FEFO", "Highest Quality First"]
        pol_harvest = st.selectbox("Harvester priority", policy_options, index=0, key="wi_pol_h")
        pol_grading = st.selectbox("Graders priority", policy_options, index=0, key="wi_pol_g")
        pol_storage = st.selectbox("Storage priority", policy_options, index=1, key="wi_pol_s")

    workers_base = cfg.get("workers_base", None)
    if not isinstance(workers_base, dict):
        workers_base = cfg.get("workers", {}) or {}

    w_factor = float(workers_pct) / 100.0
    workers: Dict[str, Dict[str, float]] = {}
    for role, d in workers_base.items():
        n0 = float(d.get("n", 0.0))
        n_eff = max(0.0, n0 * w_factor)
        workers[role] = {
            "n": float(n_eff),
            "mu": float(d.get("mu", 0.0)),
            "sigma": float(d.get("sigma", 0.0)),
        }

    long_term_capacity_base = cfg.get("long_term_capacity_base", None)
    if long_term_capacity_base is None:
        long_term_capacity_base = int(cfg.get("long_term_capacity", 2000))

    storage_factor = float(storage_pct) / 100.0
    long_term_capacity = int(round(float(long_term_capacity_base) * storage_factor))
    long_term_capacity = max(0, long_term_capacity)

    cfg.update(
        {
            "workers": workers,
            "long_term_capacity_base": int(long_term_capacity_base),
            "long_term_capacity": int(long_term_capacity),
            "what_if_workers_factor": float(workers_pct) / 100.0,
            "what_if_bins_factor": float(bins_pct) / 100.0,
            "what_if_storage_factor": float(storage_pct) / 100.0,
            "what_if_demand_factor": float(demand_pct) / 100.0,
            "policy_harvest": str(pol_harvest),
            "policy_grading": str(pol_grading),
            "policy_storage": str(pol_storage),
        }
    )

    return normalize_config(cfg)


def get_or_build_weather_cache(config: Dict[str, Any]) -> Dict[str, Any]:
    config = normalize_config(config)
    mode = str(config.get("sim_mode", "")).strip()

    def attach_demand(wc: Dict[str, Any]) -> Dict[str, Any]:
        wc = dict(wc)
        prof = build_weekly_demand_profile_once(config)
        wc["demand_weekly"] = prof
        wc["weekly_demand_profile_by_grade"] = prof.get("by_grade", {})
        return wc

    if mode == "Micro (Single Orchard)":
        name = config.get("micro_template_name", None)
        if not name:
            raise RuntimeError("Micro mode requires a template saved in WeatherTemplates/templates.")
        wc = load_template_from_folder(name, folder=TEMPLATE_FOLDER)
        return attach_demand(wc)

    key = _weather_cache_key(config)
    store = st.session_state.get("weather_cache_store", {})
    if key in store:
        return attach_demand(store[key])

    cache = _build_weather_cache(config)
    store[key] = cache
    st.session_state["weather_cache_store"] = store
    return attach_demand(cache)


def _weather_cache_key(config: Dict[str, Any]) -> str:
    relevant = {
        "lat": config.get("lat"),
        "lon": config.get("lon"),
        "planting_year": config.get("planting_year"),
        "years_to_sim": config.get("years_to_sim"),
        "sim_mode": config.get("sim_mode"),
        "sun_thr": config.get("sun_hour_threshold_wm2", 120.0),
        "cc_ramp": int(config.get("cc_ramp_years", 30)),
        "cc_wt": float(config.get("cc_winter_temp_warm_c", 2.0)),
        "cc_wp": float(config.get("cc_winter_precip_wet_frac", 0.15)),
        "cc_st": float(config.get("cc_summer_temp_warm_c", 3.0)),
        "cc_sp": float(config.get("cc_summer_precip_dry_frac", 0.30)),
        "demand_seed": int(config.get("demand_seed", 12345)),
        "demand_factor": float(config.get("what_if_demand_factor", 1.0)),
        "demand_mix": config.get("demand_grade_mix", {}),
        "today": date.today().isoformat(),
    }
    blob = json.dumps(relevant, sort_keys=True).encode("utf-8")
    return hashlib.md5(blob).hexdigest()


def export_weather_template(weather_cache: Dict[str, Any]) -> Tuple[bytes, bytes, bytes]:
    return weather_cache_to_template_bytes(weather_cache)


def save_weather_template_to_project_folder(weather_cache: Dict[str, Any], template_name: str) -> Tuple[str, str, str]:
    daily_csv_bytes, arrays_npz_bytes, meta_json_bytes = export_weather_template(weather_cache)
    return save_template_to_folder(
        folder=TEMPLATE_FOLDER,
        template_name=template_name,
        daily_csv_bytes=daily_csv_bytes,
        arrays_npz_bytes=arrays_npz_bytes,
        meta_json_bytes=meta_json_bytes,
    )


def _build_weather_cache(config: Dict[str, Any]) -> Dict[str, Any]:
    lat = float(config["lat"])
    lon = float(config["lon"])
    planting_year = int(config["planting_year"])
    years_to_sim = int(config.get("years_to_sim", 10))

    first_year = planting_year
    last_year = planting_year + years_to_sim - 1

    start_dt = datetime(first_year - 1, 11, 1, 0)
    end_dt = datetime(last_year, 12, 11, 23)

    start_ts = pd.Timestamp(start_dt, tz="UTC")
    end_ts = pd.Timestamp(end_dt, tz="UTC")

    today_utc = datetime.utcnow().date()
    safe_archive_end = today_utc - timedelta(days=2)

    hist_start_date = start_ts.date().isoformat()
    hist_end_date = min(safe_archive_end, end_ts.date()).isoformat()

    hourly_hist = _fetch_open_meteo_archive_hourly(lat, lon, hist_start_date, hist_end_date)

    try:
        hourly_fcst = _fetch_open_meteo_ukmo_forecast_hourly(lat, lon)
    except Exception:
        hourly_fcst = pd.DataFrame(columns=hourly_hist.columns)

    hourly = pd.concat([hourly_hist, hourly_fcst], ignore_index=True)
    hourly = hourly.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
    hourly = hourly[hourly["time"] >= start_ts].copy()

    if hourly.empty:
        raise RuntimeError("No weather data after combining historic + forecast.")

    if hourly["time"].max() < end_ts:
        seed = int(hashlib.md5(f"{lat:.4f}|{lon:.4f}|{start_ts.date()}|{end_ts.date()}".encode()).hexdigest()[:8], 16)
        hourly = _extend_with_synthetic_future_by_day(
            hourly=hourly,
            start_fill=hourly["time"].max() + pd.Timedelta(hours=1),
            end_fill=end_ts,
            lat=lat,
            lon=lon,
            years_back=10,
            safe_archive_end=safe_archive_end,
            rng=np.random.default_rng(seed),
            config=config,
        )

    hourly = hourly[(hourly["time"] >= start_ts) & (hourly["time"] <= end_ts)].copy()
    hourly = hourly.sort_values("time").reset_index(drop=True)

    daily = _hourly_to_daily(config=config, hourly=hourly)
    year_arrays = build_year_arrays(daily)

    return {
        "hourly_df": hourly,
        "daily_df": daily,
        "weather_array": daily,
        "year_arrays": year_arrays,
        "meta": {
            "lat": lat,
            "lon": lon,
            "sim_first_year": first_year,
            "sim_last_year": last_year,
            "start_dt": str(start_dt),
            "end_dt": str(end_dt),
            "safe_archive_end": safe_archive_end.isoformat(),
            "generated_at_utc": datetime.utcnow().isoformat(),
        },
    }


def _fetch_open_meteo_archive_hourly(lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    start = max(date.fromisoformat(start_date), date(1940, 1, 1))
    end = date.fromisoformat(end_date)
    if end < start:
        end = start

    url = "https://archive-api.open-meteo.com/v1/archive"
    hourly_vars = "temperature_2m,precipitation,shortwave_radiation,relative_humidity_2m"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "hourly": hourly_vars,
        "timezone": "UTC",
    }
    r = requests.get(url, params=params, timeout=60)
    if r.status_code != 200:
        raise requests.HTTPError(f"Archive error {r.status_code}: {r.text}", response=r)

    hourly = r.json().get("hourly", {})
    if not hourly or "time" not in hourly:
        return pd.DataFrame(columns=["time", "temperature_2m", "precipitation", "shortwave_radiation", "relative_humidity_2m"])

    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return _normalize_hourly_columns(df)


def _fetch_open_meteo_ukmo_forecast_hourly(lat: float, lon: float) -> pd.DataFrame:
    url = "https://api.open-meteo.com/v1/forecast"
    hourly_vars = "temperature_2m,precipitation,shortwave_radiation,relative_humidity_2m"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": hourly_vars,
        "models": "ukmo_seamless",
        "timezone": "UTC",
    }
    r = requests.get(url, params=params, timeout=60)
    if r.status_code != 200:
        raise requests.HTTPError(f"Forecast error {r.status_code}: {r.text}", response=r)

    hourly = r.json().get("hourly", {})
    if not hourly or "time" not in hourly:
        return pd.DataFrame(columns=["time", "temperature_2m", "precipitation", "shortwave_radiation", "relative_humidity_2m"])

    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return _normalize_hourly_columns(df)


def _normalize_hourly_columns(df: pd.DataFrame) -> pd.DataFrame:
    for c in ["temperature_2m", "precipitation", "shortwave_radiation", "relative_humidity_2m"]:
        if c not in df.columns:
            df[c] = 0.0
    df = df[["time", "temperature_2m", "precipitation", "shortwave_radiation", "relative_humidity_2m"]].copy()
    for c in ["temperature_2m", "precipitation", "shortwave_radiation", "relative_humidity_2m"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df


def _season_of_month(m: int) -> str:
    if m in (12, 1, 2):
        return "winter"
    if m in (6, 7, 8):
        return "summer"
    return "shoulder"


def _apply_future_climate_change_to_synth(synth: pd.DataFrame, start_fill: pd.Timestamp, config: Dict[str, Any]) -> pd.DataFrame:
    if synth.empty:
        return synth

    ramp_years = max(1, int(config.get("cc_ramp_years", 30)))

    wt = float(config.get("cc_winter_temp_warm_c", 2.0))
    wp = float(config.get("cc_winter_precip_wet_frac", 0.15))
    st_ = float(config.get("cc_summer_temp_warm_c", 3.0))
    sp = float(config.get("cc_summer_precip_dry_frac", 0.30))

    df = synth.copy()
    t = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.loc[t.notna()].copy()
    t = pd.to_datetime(df["time"], utc=True, errors="coerce")

    years_from = (t - start_fill) / np.timedelta64(365, "D")
    ramp = np.clip(years_from / float(ramp_years), 0.0, 1.0)

    months = t.dt.month.to_numpy()
    seasons = np.array([_season_of_month(int(m)) for m in months], dtype=object)

    temp = pd.to_numeric(df["temperature_2m"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    add = np.zeros_like(temp)

    winter_mask = seasons == "winter"
    summer_mask = seasons == "summer"
    add[winter_mask] = ramp[winter_mask] * wt
    add[summer_mask] = ramp[summer_mask] * st_
    df["temperature_2m"] = temp + add

    precip = pd.to_numeric(df["precipitation"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    mult = np.ones_like(precip)
    mult[winter_mask] = 1.0 + ramp[winter_mask] * wp
    mult[summer_mask] = 1.0 - ramp[summer_mask] * sp
    mult = np.clip(mult, 0.0, None)
    df["precipitation"] = precip * mult

    return df


def _extend_with_synthetic_future_by_day(
    hourly: pd.DataFrame,
    start_fill: pd.Timestamp,
    end_fill: pd.Timestamp,
    lat: float,
    lon: float,
    years_back: int,
    safe_archive_end: date,
    rng: np.random.Generator,
    config: Dict[str, Any],
) -> pd.DataFrame:
    pool_end = safe_archive_end
    pool_start = max(date(1940, 1, 1), pool_end - timedelta(days=365 * years_back))
    hist = _fetch_open_meteo_archive_hourly(lat, lon, pool_start.isoformat(), pool_end.isoformat())
    if hist.empty:
        raise RuntimeError("Synthetic fill cannot build pools: historical archive returned empty data.")

    hist = hist.copy()
    hist["date"] = hist["time"].dt.date
    hist["month"] = hist["time"].dt.month
    hist["day"] = hist["time"].dt.day
    hist["hour"] = hist["time"].dt.hour

    pools: Dict[Tuple[int, int], List[pd.DataFrame]] = {}
    for _, g in hist.groupby("date"):
        if len(g) < 24:
            continue
        g = g.sort_values("time")
        if g["hour"].nunique() < 24:
            continue
        key = (int(g["month"].iloc[0]), int(g["day"].iloc[0]))
        pools.setdefault(key, []).append(g)

    if not pools:
        raise RuntimeError("Synthetic fill failed: no complete 24-hour days in historical pool.")

    def pick_day_block(month: int, day: int) -> pd.DataFrame:
        key = (month, day)
        if key in pools and pools[key]:
            return pools[key][int(rng.integers(0, len(pools[key])))]
        if month == 2 and day == 29:
            key2 = (2, 28)
            if key2 in pools and pools[key2]:
                return pools[key2][int(rng.integers(0, len(pools[key2])))]
        any_key = next(iter(pools.keys()))
        return pools[any_key][int(rng.integers(0, len(pools[any_key])))]

    start_day = start_fill.floor("D")
    end_day = end_fill.floor("D")
    days = pd.date_range(start_day, end_day, freq="D", tz="UTC")

    rows = []
    for day_ts in days:
        donor = pick_day_block(int(day_ts.month), int(day_ts.day))
        for h in range(24):
            t = day_ts + pd.Timedelta(hours=h)
            if t < start_fill or t > end_fill:
                continue
            donor_row = donor[donor["hour"] == h].iloc[0]
            rows.append(
                {
                    "time": t,
                    "temperature_2m": float(donor_row["temperature_2m"]),
                    "precipitation": float(donor_row["precipitation"]),
                    "shortwave_radiation": float(donor_row["shortwave_radiation"]),
                    "relative_humidity_2m": float(donor_row["relative_humidity_2m"]),
                }
            )

    synth = pd.DataFrame(rows)
    synth = _normalize_hourly_columns(synth)
    synth = _apply_future_climate_change_to_synth(synth, start_fill=start_fill, config=config)

    out = pd.concat([hourly, synth], ignore_index=True)
    out = out.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
    return out


def _hourly_to_daily(config: Dict[str, Any], hourly: pd.DataFrame) -> pd.DataFrame:
    h = hourly.copy()
    h["date"] = h["time"].dt.date
    h["season_year"] = np.where(h["time"].dt.month >= 11, h["time"].dt.year + 1, h["time"].dt.year)

    chill_min = float(config.get("chill_temp_min", 0.0))
    chill_max = float(config.get("chill_temp_max", 7.0))
    forcing_min = float(config.get("forcing_temp_min", 3.5))

    h["chill_hour"] = ((h["temperature_2m"] >= chill_min) & (h["temperature_2m"] <= chill_max)).astype(int)
    h["forcing_hour"] = (h["temperature_2m"] >= forcing_min).astype(int)

    sun_thr = float(config.get("sun_hour_threshold_wm2", 120.0))
    h["sun_hour"] = (h["shortwave_radiation"] > sun_thr).astype(int)

    daily = h.groupby(["season_year", "date"], as_index=False).agg(
        tmin=("temperature_2m", "min"),
        tmax=("temperature_2m", "max"),
        tavg=("temperature_2m", "mean"),
        rain=("precipitation", "sum"),
        chill_day=("chill_hour", "sum"),
        forcing_day=("forcing_hour", "sum"),
        sun_hours_day=("sun_hour", "sum"),
    )

    daily["chill_accum"] = daily.groupby("season_year")["chill_day"].cumsum()
    daily["forcing_accum"] = daily.groupby("season_year")["forcing_day"].cumsum()

    frost10 = float(config.get("frost_kill_10", -2.0))
    frost90 = float(config.get("frost_kill_90", -4.0))
    heat10 = float(config.get("heat_loss_10", 25.0))
    heat40 = float(config.get("heat_loss_40", 35.0))
    frost_df, heat_df = _two_consecutive_hour_flags(h, frost10, frost90, heat10, heat40)
    daily = daily.merge(frost_df, on=["season_year", "date"], how="left")
    daily = daily.merge(heat_df, on=["season_year", "date"], how="left")

    no_bee_rain = float(config.get("no_bee_flight_rain_mm", 0.1))
    rain_off = float(config.get("rain_off_mm", 5.0))
    daily["rain_flag"] = np.select(
        [daily["rain"] >= rain_off, daily["rain"] >= no_bee_rain],
        [2, 1],
        default=0,
    )
    daily["no_bee_flight"] = (daily["rain_flag"] > 0).astype(int)

    daily["precip_accum"] = daily.groupby("season_year")["rain"].cumsum()
    daily["temp_accum"] = daily.groupby("season_year")["tavg"].cumsum()
    daily["sun_hours_accum"] = daily.groupby("season_year")["sun_hours_day"].cumsum()

    daily["frost_flag"] = daily["frost_flag"].fillna(0).astype(int)
    daily["heat_flag"] = daily["heat_flag"].fillna(0).astype(int)

    return daily.sort_values(["season_year", "date"]).reset_index(drop=True)


def _two_consecutive_hour_flags(
    hourly: pd.DataFrame,
    frost10: float,
    frost90: float,
    heat10: float,
    heat40: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    h = hourly[["time", "temperature_2m", "date", "season_year"]].copy()
    h = h.sort_values(["season_year", "date", "time"]).reset_index(drop=True)

    h["frost90_hit"] = (h["temperature_2m"] <= frost90).astype(int)
    h["frost10_hit"] = (h["temperature_2m"] <= frost10).astype(int)
    h["heat40_hit"] = (h["temperature_2m"] >= heat40).astype(int)
    h["heat10_hit"] = (h["temperature_2m"] >= heat10).astype(int)

    def roll2_any(x: pd.Series) -> int:
        if len(x) < 2:
            return 0
        return int((x.rolling(2).sum() >= 2).any())

    frost90_any = h.groupby(["season_year", "date"])["frost90_hit"].apply(roll2_any).reset_index(name="f90")
    frost10_any = h.groupby(["season_year", "date"])["frost10_hit"].apply(roll2_any).reset_index(name="f10")
    heat40_any = h.groupby(["season_year", "date"])["heat40_hit"].apply(roll2_any).reset_index(name="h40")
    heat10_any = h.groupby(["season_year", "date"])["heat10_hit"].apply(roll2_any).reset_index(name="h10")

    frost = frost90_any.merge(frost10_any, on=["season_year", "date"], how="left")
    heat = heat40_any.merge(heat10_any, on=["season_year", "date"], how="left")

    frost["frost_flag"] = np.select([frost["f90"] == 1, frost["f10"] == 1], [2, 1], default=0)
    heat["heat_flag"] = np.select([heat["h40"] == 1, heat["h10"] == 1], [2, 1], default=0)

    return frost[["season_year", "date", "frost_flag"]], heat[["season_year", "date", "heat_flag"]]


def build_year_arrays(daily_df: pd.DataFrame) -> dict[int, dict[str, np.ndarray]]:
    out: dict[int, dict[str, np.ndarray]] = {}
    for year, g in daily_df.groupby("season_year"):
        g = g.sort_values("date").reset_index(drop=True)
        date_list = list(g["date"])
        date_ord = np.array([d.toordinal() for d in date_list], dtype=np.int32)

        tavg = g["tavg"].to_numpy(dtype=np.float32)

        chill = g["chill_accum"].to_numpy(dtype=np.float32)
        forcing = g["forcing_accum"].to_numpy(dtype=np.float32)

        precip = g["precip_accum"].to_numpy(dtype=np.float32)
        temp = g["temp_accum"].to_numpy(dtype=np.float32)
        sun = g["sun_hours_accum"].to_numpy(dtype=np.float32)

        tbase = 3.5
        gdd_daily = np.maximum(0.0, tavg - tbase).astype(np.float32)

        dec12_ord = date(int(year) - 1, 12, 12).toordinal()
        gdd_daily = np.where(date_ord >= dec12_ord, gdd_daily, 0.0).astype(np.float32)
        gdd_ps = np.cumsum(gdd_daily).astype(np.float32)

        frost1_ps = np.cumsum((g["frost_flag"].to_numpy() == 1).astype(np.int32))
        frost2_ps = np.cumsum((g["frost_flag"].to_numpy() == 2).astype(np.int32))
        heat1_ps = np.cumsum((g["heat_flag"].to_numpy() == 1).astype(np.int32))
        heat2_ps = np.cumsum((g["heat_flag"].to_numpy() == 2).astype(np.int32))
        bee_ok_ps = np.cumsum((g["no_bee_flight"].to_numpy() == 0).astype(np.int32))

        out[int(year)] = {
            "date_ord": date_ord,
            "date_list": np.array(date_list, dtype="datetime64[D]"),
            "chill": chill,
            "forcing": forcing,
            "precip": precip,
            "temp": temp,
            "sun": sun,
            "gdd_ps": gdd_ps,
            "frost1_ps": frost1_ps,
            "frost2_ps": frost2_ps,
            "heat1_ps": heat1_ps,
            "heat2_ps": heat2_ps,
            "bee_ok_ps": bee_ok_ps,
        }
    return out


def macro_defaults_from_micro() -> Dict[str, Any]:
    base = normalize_config({})

    base["sim_mode"] = "Macro (Multiple Orchards)"
    base["run_harvest"] = False
    base["run_harvest_and_grading"] = False
    base["run_storage"] = False
    base["detail_run"] = False
    base["store_storage_quality_hist"] = False
    base["fast_mode"] = True
    base["quality_samples_cap"] = 0
    base["micro_template_name"] = None

    return base


def build_macro_inputs(*, start_year: int, end_year: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    county_area_by_year = build_county_area_by_year(
        start_year=int(start_year),
        end_year=int(end_year),
    )

    county_age_by_year = build_county_age_density_by_year(
        county_area_by_year=county_area_by_year,
        start_year=int(start_year),
        end_year=int(end_year),
    )

    return county_area_by_year, county_age_by_year


def _macro_template_folder_ok() -> None:
    os.makedirs(MACRO_TEMPLATE_FOLDER, exist_ok=True)


def _attach_demand_profile_to_cache(wc: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    try:
        prof = build_weekly_demand_profile_once(config)
    except Exception:
        return wc

    out = dict(wc)
    out["demand_weekly"] = prof
    out["weekly_demand_profile_by_grade"] = prof.get("by_grade", {})
    return out


def _slice_weather_cache_to_year_range(
    wc: Dict[str, Any],
    *,
    start_year: int,
    end_year: int,
) -> Dict[str, Any]:
    if not isinstance(wc, dict):
        return {}

    out = dict(wc)

    daily = out.get("daily_df", None)
    if isinstance(daily, pd.DataFrame) and (not daily.empty) and ("season_year" in daily.columns):
        d = daily.copy()
        d["season_year"] = pd.to_numeric(d["season_year"], errors="coerce").astype("Int64")
        d = d.dropna(subset=["season_year"]).copy()
        d["season_year"] = d["season_year"].astype(int)
        d = d[(d["season_year"] >= int(start_year)) & (d["season_year"] <= int(end_year))].copy()
        out["daily_df"] = d
        out["weather_array"] = d

    ya = out.get("year_arrays", None)
    if isinstance(ya, dict) and ya:
        keep = {}
        for y, arrs in ya.items():
            try:
                yy = int(y)
            except Exception:
                continue
            if int(start_year) <= yy <= int(end_year):
                keep[yy] = arrs
        out["year_arrays"] = keep

    meta = out.get("meta", None)
    if isinstance(meta, dict):
        meta2 = dict(meta)
        meta2["macro_slice_start_year"] = int(start_year)
        meta2["macro_slice_end_year"] = int(end_year)
        out["meta"] = meta2

    return out


def load_macro_weather_templates(
    *,
    base_cfg: Dict[str, Any],
    start_year: int,
    end_year: int,
) -> Tuple[Dict[str, Dict[str, Any]], pd.DataFrame]:
    _macro_template_folder_ok()

    names = list_templates(MACRO_TEMPLATE_FOLDER)
    if not names:
        raise RuntimeError(
            f"No macro templates found in: {MACRO_TEMPLATE_FOLDER}\n\n"
            "Run Digital Twin for each location and Save template, then copy/move the template files into this folder."
        )

    wc_by_point: Dict[str, Dict[str, Any]] = {}
    points_rows: List[Dict[str, Any]] = []

    for name in names:
        wc = load_template_from_folder(name, folder=MACRO_TEMPLATE_FOLDER)
        if not isinstance(wc, dict):
            continue

        wc = _attach_demand_profile_to_cache(wc, base_cfg)
        wc = _slice_weather_cache_to_year_range(wc, start_year=int(start_year), end_year=int(end_year))

        meta = wc.get("meta", {}) if isinstance(wc.get("meta", None), dict) else {}
        lat = meta.get("lat", wc.get("lat", None))
        lon = meta.get("lon", wc.get("lon", None))

        county = meta.get("county", None)
        if county is None or str(county).strip() == "":
            county = os.path.splitext(os.path.basename(str(name)))[0]

        ha_2025 = meta.get("ha_2025", np.nan)

        try:
            latf = float(lat)
            lonf = float(lon)
        except Exception:
            raise RuntimeError(f"Template '{name}' is missing lat/lon in meta. Re-save from DT or fix meta_json.")

        wc_by_point[str(county)] = wc
        points_rows.append({"county": str(county), "lat": latf, "lon": lonf, "ha_2025": ha_2025})

    points_df = pd.DataFrame(points_rows)
    if points_df.empty:
        raise RuntimeError("Macro templates loaded but produced no usable points (check meta lat/lon).")

    return wc_by_point, points_df


def build_macro_weather_cache_by_county(
    *,
    base_cfg: Dict[str, Any],
    county_area_by_year: pd.DataFrame,
    start_year: int,
    end_year: int,
) -> Dict[str, Dict[str, Any]]:
    wc_by_county, _points_df = load_macro_weather_templates(
        base_cfg=base_cfg,
        start_year=int(start_year),
        end_year=int(end_year),
    )
    return wc_by_county


def build_macro_points_from_templates(
    *,
    base_cfg: Dict[str, Any],
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    _wc_by_county, points_df = load_macro_weather_templates(
        base_cfg=base_cfg,
        start_year=int(start_year),
        end_year=int(end_year),
    )
    return points_df