# Post_processing/Post_processing.py
from __future__ import annotations

from typing import Any, Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


import os



# ============================================================
# Constants
# ============================================================
GRADES = ["Extra", "Class1", "Class2", "Processor"]
ALL_GRADES = ["Extra", "Class1", "Class2", "Processor", "Waste"]

GRADE_COLORS = {
    "Extra": "#00A000",
    "Class1": "#7CFC00",
    "Class2": "#FFD700",
    "Processor": "#FF8C00",
    "Waste": "#FF0000",
}
QUARTILE_LINE_COLOR = "white"

# Quality thresholds for vertical separators in storage quality plot
Q_THRESHOLDS = {
    "Extra_min": 0.90,
    "Class1_min": 0.70,
    "Class2_min": 0.50,
    "Processor_min": 0.20,
}


# ============================================================
# Small utilities
# ============================================================
def _get_df(sim_results: Dict[str, Any], key: str) -> pd.DataFrame:
    df = sim_results.get(key)
    return df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()


def _get_nested(d: Dict[str, Any], path: List[str]) -> Any:
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _hist(vals: np.ndarray, bins: int = 60, range_: Optional[Tuple[float, float]] = None) -> Tuple[np.ndarray, np.ndarray]:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.array([0.0]), np.array([0])
    if range_ is None:
        counts, edges = np.histogram(vals, bins=bins)
    else:
        counts, edges = np.histogram(vals, bins=bins, range=range_)
    centers = (edges[:-1] + edges[1:]) / 2.0
    return centers, counts


def _cap_runs(arr: np.ndarray, cap: int = 300) -> np.ndarray:
    u = np.unique(np.asarray(arr, dtype=int))
    u.sort()
    return u[:cap] if u.size > cap else u


def _hash_df(df: pd.DataFrame) -> Tuple[int, int, int]:
    return (int(df.shape[0]), int(df.shape[1]), int(pd.util.hash_pandas_object(df.head(300), index=True).sum()))


def _to_1d_float_array(x: Any) -> Optional[np.ndarray]:
    if isinstance(x, np.ndarray):
        a = x.astype(float, copy=False).ravel()
        a = a.copy()
        a[~np.isfinite(a)] = np.nan
        return a
    if isinstance(x, list):
        try:
            a = np.asarray(x, dtype=float).ravel()
            a = a.copy()
            a[~np.isfinite(a)] = np.nan
            return a
        except Exception:
            return None
    return None


def _time_label(n: int) -> str:
    return "Week" if n <= 60 else "Day"


def _time_range_label(n: int) -> str:
    return "Week in storage" if n <= 60 else "Day in storage"


def _x_axis(n: int) -> np.ndarray:
    return np.arange(int(n), dtype=int)


# ============================================================
# Growth helpers
# ============================================================
def _ensure_schema_growth(mc: pd.DataFrame) -> pd.DataFrame:
    if mc.empty:
        return mc

    for c in ["mc_run", "season_year"]:
        if c in mc.columns:
            mc[c] = pd.to_numeric(mc[c], errors="coerce")
    mc = mc.dropna(subset=[c for c in ["mc_run", "season_year"] if c in mc.columns]).copy()

    if "mc_run" in mc.columns:
        mc["mc_run"] = mc["mc_run"].astype(int)
    if "season_year" in mc.columns:
        mc["season_year"] = mc["season_year"].astype(int)

    num_like = [
        "yield_t_ha", "fruit_mass_kg", "fruit_number_tree",
        "maturity_factor",
        "frost_multiplier", "pollination_multiplier", "heat_multiplier",
        "mass_weather_multiplier",
        "precip_ratio", "temp_ratio", "sun_ratio",
        "yield_total_kg",
        "fruit_mass_modifier", "fruit_number_modifier",
        "fruit_number_modifier_total",
        "fruit_mass_modifier_total",
        "risk_water_modifier",
        "mass_weather_precip_modifier",
        "mass_weather_temp_modifier",
        "mass_weather_sun_modifier",
        "fruit_mass_weather_modifier_total",
        "mass_number_modifier",
        "thin_factor",
    ]
    for c in num_like:
        if c in mc.columns:
            mc[c] = pd.to_numeric(mc[c], errors="coerce")

    for c in ["chill_complete_date", "budbreak_date", "blossom_date", "fruitset_date", "harvest_date"]:
        if c in mc.columns:
            mc[c] = pd.to_datetime(mc[c], errors="coerce")

    return mc


def _compute_cum(mc: pd.DataFrame) -> pd.DataFrame:
    mc = mc.sort_values(["mc_run", "season_year"]).reset_index(drop=True)
    if "yield_t_ha" in mc.columns:
        mc["cum_yield_t_ha"] = mc.groupby("mc_run")["yield_t_ha"].cumsum()
    return mc


def _build_spaghetti_base(mc: pd.DataFrame) -> Dict[str, Any]:
    runs = _cap_runs(mc["mc_run"].unique(), cap=300)
    m = mc[mc["mc_run"].isin(runs)].copy()

    fig = go.Figure()
    for _, g in m.groupby("mc_run", sort=False):
        fig.add_trace(go.Scatter(
            x=g["season_year"], y=g["yield_t_ha"],
            mode="lines", opacity=0.22, line={"width": 1},
            showlegend=False, hoverinfo="skip",
        ))
    fig.update_layout(
        title="Annual yield paths (t/ha)",
        xaxis_title="Season year",
        yaxis_title="Yield (t/ha)",
        hovermode=False,
        height=280,
        margin=dict(l=40, r=10, t=50, b=1),
    )
    return fig.to_dict()


def _figure_with_year_marker(fig_dict: Dict[str, Any], year_sel: int) -> go.Figure:
    fig = go.Figure(fig_dict)
    fig.add_vline(x=year_sel, line_width=2, line_dash="dash")
    return fig


def _summary_stats(series: pd.Series) -> Dict[str, float]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {}
    return {
        "min": float(s.min()),
        "q25": float(np.quantile(s, 0.25)),
        "median": float(np.quantile(s, 0.50)),
        "q75": float(np.quantile(s, 0.75)),
        "max": float(s.max()),
    }


def _summary_stats_dates(series: pd.Series) -> Dict[str, pd.Timestamp]:
    d = pd.to_datetime(series, errors="coerce").dropna()
    if d.empty:
        return {}
    ords = d.map(lambda x: x.toordinal()).to_numpy(dtype=float)
    return {
        "min": pd.Timestamp.fromordinal(int(np.min(ords))),
        "q25": pd.Timestamp.fromordinal(int(np.quantile(ords, 0.25))),
        "median": pd.Timestamp.fromordinal(int(np.quantile(ords, 0.50))),
        "q75": pd.Timestamp.fromordinal(int(np.quantile(ords, 0.75))),
        "max": pd.Timestamp.fromordinal(int(np.max(ords))),
    }


def _fmt_num(x: float, decimals: int = 3) -> str:
    if x is None:
        return "—"
    try:
        xf = float(x)
        if not np.isfinite(xf):
            return "—"
        return f"{xf:.{decimals}f}"
    except Exception:
        return "—"


def _fmt_mmdd(ts: pd.Timestamp) -> str:
    if ts is None or pd.isna(ts):
        return "—"
    try:
        return pd.to_datetime(ts).strftime("%b-%d")
    except Exception:
        return "—"


def _growth_requested_table(dfy: pd.DataFrame) -> pd.DataFrame:
    rows = []

    stage_items = [
        ("Dormancy", "date", "chill_complete_date"),
        ("Budbreak", "date", "budbreak_date"),
        ("Blossom", "date", "blossom_date"),
        ("Fruit set", "date", "fruitset_date"),
        ("Maturity", "date", "harvest_date"),
    ]

    core_items = [
        ("Fruit number modifier (TOTAL)", "num", "fruit_number_modifier_total"),
        ("Fruit mass modifier (TOTAL)", "num", "fruit_mass_modifier_total"),
    ]

    number_comp = [
        ("Risk — frost modifier", "num", "frost_multiplier"),
        ("Risk — pollination modifier", "num", "pollination_multiplier"),
        ("Risk — heat modifier", "num", "heat_multiplier"),
        ("Risk — water modifier", "num", "risk_water_modifier"),
        ("Tree — maturity factor", "num", "maturity_factor"),
        ("Management — thinning factor", "num", "thin_factor"),
    ]

    mass_weather = [
        ("Mass weather — precip modifier", "num", "mass_weather_precip_modifier"),
        ("Mass weather — temp modifier", "num", "mass_weather_temp_modifier"),
        ("Mass weather — sun modifier", "num", "mass_weather_sun_modifier"),
        ("Mass weather modifier (TOTAL)", "num", "fruit_mass_weather_modifier_total"),
        ("Legacy mass weather multiplier (avg ratios)", "num", "mass_weather_multiplier"),
    ]

    load_only = [
        ("Mass/number modifier (load-only)", "num", "mass_number_modifier"),
    ]

    items = stage_items + core_items + number_comp + mass_weather + load_only

    for label, typ, col in items:
        if col not in dfy.columns:
            rows.append({"Metric": label, "Min": "—", "Q1": "—", "Median": "—", "Q3": "—", "Max": "—"})
            continue

        if typ == "date":
            stt = _summary_stats_dates(dfy[col])
            if not stt:
                rows.append({"Metric": label, "Min": "—", "Q1": "—", "Median": "—", "Q3": "—", "Max": "—"})
                continue
            rows.append(
                {
                    "Metric": label,
                    "Min": _fmt_mmdd(stt["min"]),
                    "Q1": _fmt_mmdd(stt["q25"]),
                    "Median": _fmt_mmdd(stt["median"]),
                    "Q3": _fmt_mmdd(stt["q75"]),
                    "Max": _fmt_mmdd(stt["max"]),
                }
            )
        else:
            stt = _summary_stats(dfy[col])
            if not stt:
                rows.append({"Metric": label, "Min": "—", "Q1": "—", "Median": "—", "Q3": "—", "Max": "—"})
                continue
            rows.append(
                {
                    "Metric": label,
                    "Min": _fmt_num(stt["min"]),
                    "Q1": _fmt_num(stt["q25"]),
                    "Median": _fmt_num(stt["median"]),
                    "Q3": _fmt_num(stt["q75"]),
                    "Max": _fmt_num(stt["max"]),
                }
            )

    return pd.DataFrame(rows)














# ============================================================
# Overview tab (UPDATED) — WASTE + MASS-BASIS FIX:
#   ✅ "Median bins on trees" uses util["initial_bins_on_trees"] (preferred)
#      instead of trees_series[0] (which can already be reduced by day-0 decay removal).
#   ✅ Waste metric = median( harvest_waste_removed_total + storage_total_waste_bins )
#      - computed per-run if shared run-id exists
#      - else fallback sum-of-medians
#   ✅ Guardrail: waste_total_all is clamped to median initial bins (prevents impossible displays)
#   ✅ Layout compacted:
#      - starting inventory legend moved below plot
#      - utilisation heatmap moved vertical, to the right of metrics
#      - overall dashboard slightly smaller to better fit one screen
# ============================================================

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# assumes these exist in your project
# - _get_df, _get_nested, _ensure_schema_growth, _hist, _to_1d_float_array, _x_axis
# - GRADES, ALL_GRADES, Q_THRESHOLDS, GRADE_COLORS


def _pick_years_from_any(sim_results: Dict[str, Any]) -> List[int]:
    for key in ["mc_yearly", "harvest_util_by_year", "storage_by_year"]:
        df = _get_df(sim_results, key)
        if not df.empty and "season_year" in df.columns:
            yrs = sorted(pd.to_numeric(df["season_year"], errors="coerce").dropna().astype(int).unique().tolist())
            if yrs:
                return yrs
    med_storage = _get_nested(sim_results, ["median_detail", "des_out", "storage_by_year"])
    if isinstance(med_storage, pd.DataFrame) and not med_storage.empty and "season_year" in med_storage.columns:
        yrs = sorted(pd.to_numeric(med_storage["season_year"], errors="coerce").dropna().astype(int).unique().tolist())
        return yrs
    return []


def _year_slider(years: List[int]) -> int:
    if not years:
        return 0
    if len(years) == 1:
        st.slider("Year", years[0], years[0], years[0], 1, key="overview_year_slider")
        return years[0]
    return int(st.slider("Year", int(years[0]), int(years[-1]), int(years[-1]), 1, key="overview_year_slider"))


def _robust_util_df(sim_results: Dict[str, Any]) -> pd.DataFrame:
    util = _get_df(sim_results, "harvest_util_by_year")
    if not util.empty:
        return util
    maybe = _get_nested(sim_results, ["median_detail", "des_out", "harvest_yearly"])
    return maybe.copy() if isinstance(maybe, pd.DataFrame) else pd.DataFrame()


def _robust_storage_df_any(sim_results: Dict[str, Any]) -> pd.DataFrame:
    storage = _get_df(sim_results, "storage_by_year")
    if not storage.empty:
        return storage
    maybe = _get_nested(sim_results, ["median_detail", "des_out", "storage_by_year"])
    return maybe.copy() if isinstance(maybe, pd.DataFrame) else pd.DataFrame()


def _median_detail_storage_row(sim_results: Dict[str, Any], year_sel: int) -> Optional[Dict[str, Any]]:
    med = sim_results.get("median_detail", {})
    des_out = med.get("des_out") if isinstance(med, dict) else None
    storage_med = des_out.get("storage_by_year", None) if isinstance(des_out, dict) else None
    if not isinstance(storage_med, pd.DataFrame) or storage_med.empty or "season_year" not in storage_med.columns:
        return None
    df = storage_med.copy()
    df["season_year"] = pd.to_numeric(df["season_year"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["season_year"]).copy()
    df["season_year"] = df["season_year"].astype(int)
    sel = df[df["season_year"] == int(year_sel)]
    if sel.empty:
        return None
    return sel.iloc[0].to_dict()


def _median_initial_bins(util: pd.DataFrame, year_sel: int) -> Optional[float]:
    if util.empty or "season_year" not in util.columns:
        return None
    u = util.copy()
    u["season_year"] = pd.to_numeric(u["season_year"], errors="coerce")
    u = u[u["season_year"] == float(year_sel)]
    if u.empty:
        return None

    if "initial_bins_on_trees" in u.columns:
        v = pd.to_numeric(u["initial_bins_on_trees"], errors="coerce").dropna()
        if not v.empty:
            return float(v.median())

    if "trees_series" in u.columns:
        starts = []
        for vv in u["trees_series"]:
            if isinstance(vv, list) and len(vv) > 0:
                try:
                    starts.append(float(vv[0]))
                except Exception:
                    pass
        if starts:
            return float(np.nanmedian(np.asarray(starts, dtype=float)))

    return None


def _waste_total_all_sources(util: pd.DataFrame, storage: pd.DataFrame, year_sel: int) -> Optional[float]:
    """
    ALL waste (for overview) = harvest_waste_removed_total + storage_total_waste_bins (EOY)

    Harvest waste preferred->fallback:
      - waste_bins_removed_total
      - waste_bins_removed
      - waste_bins_short_term

    Storage waste:
      - total_waste_bins
    """
    u = util.copy() if isinstance(util, pd.DataFrame) else pd.DataFrame()
    s = storage.copy() if isinstance(storage, pd.DataFrame) else pd.DataFrame()

    if not u.empty and "season_year" in u.columns:
        u["season_year"] = pd.to_numeric(u["season_year"], errors="coerce")
        u = u[u["season_year"] == float(year_sel)].copy()
    else:
        u = pd.DataFrame()

    if not s.empty and "season_year" in s.columns:
        s["season_year"] = pd.to_numeric(s["season_year"], errors="coerce")
        s = s[s["season_year"] == float(year_sel)].copy()
    else:
        s = pd.DataFrame()

    harvest_col = None
    for c in ["waste_bins_removed_total", "waste_bins_removed", "waste_bins_short_term"]:
        if (not u.empty) and (c in u.columns):
            harvest_col = c
            break
    storage_col = "total_waste_bins" if ((not s.empty) and ("total_waste_bins" in s.columns)) else None

    if harvest_col is None and storage_col is None:
        return None

    run_id_candidates = ["run_id", "mc_run", "rep", "trial", "sim", "seed", "iteration"]
    shared_id = None
    for c in run_id_candidates:
        if (not u.empty) and (not s.empty) and (c in u.columns) and (c in s.columns):
            shared_id = c
            break

    if shared_id is not None:
        uu = u[[shared_id, harvest_col]].copy() if harvest_col else u[[shared_id]].copy()
        ss = s[[shared_id, storage_col]].copy() if storage_col else s[[shared_id]].copy()
        if harvest_col:
            uu[harvest_col] = pd.to_numeric(uu[harvest_col], errors="coerce").fillna(0.0)
        if storage_col:
            ss[storage_col] = pd.to_numeric(ss[storage_col], errors="coerce").fillna(0.0)
        m = uu.merge(ss, on=shared_id, how="outer").fillna(0.0)
        hw = m[harvest_col].to_numpy(dtype=float) if harvest_col else 0.0
        sw = m[storage_col].to_numpy(dtype=float) if storage_col else 0.0
        tot = np.asarray(hw, dtype=float) + np.asarray(sw, dtype=float)
        tot = tot[np.isfinite(tot)]
        return float(np.median(tot)) if tot.size else None

    total = 0.0
    got = False

    if harvest_col is not None and not u.empty:
        hw = pd.to_numeric(u[harvest_col], errors="coerce").dropna()
        if not hw.empty:
            total += float(hw.median())
            got = True

    if storage_col is not None and not s.empty:
        sw = pd.to_numeric(s[storage_col], errors="coerce").dropna()
        if not sw.empty:
            total += float(sw.median())
            got = True

    return float(total) if got else None


def _overview_colored_gradeband_hist(inv_hist: np.ndarray, title: str) -> go.Figure:
    h = np.asarray(inv_hist, dtype=float)
    h[~np.isfinite(h)] = 0.0
    h = np.maximum(0.0, h)

    q_bins = int(h.size)
    edges = np.linspace(0.0, 1.0, q_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    width = float(edges[1] - edges[0]) if q_bins > 0 else 0.02

    thr_p = float(Q_THRESHOLDS["Processor_min"])
    thr_c2 = float(Q_THRESHOLDS["Class2_min"])
    thr_c1 = float(Q_THRESHOLDS["Class1_min"])
    thr_e = float(Q_THRESHOLDS["Extra_min"])

    masks = {
        "Waste": centers < thr_p,
        "Processor": (centers >= thr_p) & (centers < thr_c2),
        "Class2": (centers >= thr_c2) & (centers < thr_c1),
        "Class1": (centers >= thr_c1) & (centers < thr_e),
        "Extra": centers >= thr_e,
    }

    fig = go.Figure()
    for g in ["Waste", "Processor", "Class2", "Class1", "Extra"]:
        m = masks[g]
        if not np.any(m):
            continue
        fig.add_trace(go.Bar(
            x=centers[m],
            y=h[m],
            name=g,
            marker=dict(color=GRADE_COLORS.get(g, "#888888")),
            width=width,
            hovertemplate=f"{g}<br>Quality=%{{x:.3f}}<br>Bins=%{{y:.1f}}<extra></extra>",
        ))

    thr_lines = [thr_p, thr_c2, thr_c1, thr_e]
    ymax = float(np.nanmax(h)) if h.size else 1.0
    ymax = max(1.0, ymax * 1.08)

    for xthr in thr_lines:
        fig.add_shape(
            type="line",
            x0=xthr, x1=xthr,
            y0=0.0, y1=ymax,
            line=dict(color="white", width=2, dash="dot"),
        )

    fig.update_layout(
        title=title,
        barmode="overlay",
        xaxis_title="Quality (0–1)",
        yaxis_title="Bins",
        height=248,
        margin=dict(l=52, r=12, t=46, b=1),
        hovermode="x unified",
        showlegend=True,
        legend=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=-0.22,
            yanchor="top",
            title_text="",
        ),
    )
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, ymax])
    return fig


def _yield_dist_fig(mc: pd.DataFrame, year_sel: int, height: int) -> Optional[go.Figure]:
    if mc.empty or "season_year" not in mc.columns or "yield_t_ha" not in mc.columns:
        return None
    dfy = mc[mc["season_year"] == int(year_sel)].copy()
    yld = pd.to_numeric(dfy["yield_t_ha"], errors="coerce").dropna().to_numpy(dtype=float)
    if yld.size == 0:
        return None
    xh, yh = _hist(yld, bins=45)
    fig = go.Figure([go.Bar(x=xh, y=yh)])
    fig.update_layout(
        title="Yield distribution",
        height=int(height),
        margin=dict(l=52, r=12, t=46, b=1),
    )
    fig.update_xaxes(title="Yield (t/ha)", range=[0, max(1e-9, float(np.nanmax(xh)) if xh.size else 1.0)])
    fig.update_yaxes(title="Frequency", range=[0, max(1.0, float(np.nanmax(yh)) * 1.12 if yh.size else 1.0)])
    return fig


def _util_heatmap_selected_year_vertical(util: pd.DataFrame, year_sel: int) -> Optional[go.Figure]:
    if util.empty or "season_year" not in util.columns:
        return None

    u = util.copy()
    u["season_year"] = pd.to_numeric(u["season_year"], errors="coerce")
    u = u[u["season_year"] == float(year_sel)].copy()
    if u.empty:
        return None

    stages = [
        ("util_grade", "Graders"),
        ("util_shuttle", "Filled shuttle"),
        ("util_pick", "Harvesters"),
        ("util_empty", "Empty shuttle"),
    ]

    ys, vals = [], []
    for col, label in stages:
        if col not in u.columns:
            continue
        v = pd.to_numeric(u[col], errors="coerce").dropna()
        med = float(np.clip(v.median(), 0.0, 1.0)) if not v.empty else 0.0
        ys.append(label)
        vals.append(med)

    if not ys:
        return None

    colorscale = [
        [0.00, "#B0B0B0"],
        [0.05, "#B0B0B0"],
        [0.30, "#FFD966"],
        [0.60, "#B7F0A0"],
        [0.90, "#00B050"],
        [1.00, "#D00000"],
    ]

    Z = np.asarray(vals, dtype=float).reshape(-1, 1)

    fig = go.Figure(data=go.Heatmap(
        z=Z,
        x=[""],
        y=ys,
        zmin=0.0,
        zmax=1.0,
        colorscale=colorscale,
        hoverongaps=False,
        colorbar=dict(
            title=dict(text="Util", side="top"),
            thickness=16,
            len=0.94,
            y=0.50,
            yanchor="middle",
            x=1.10,
        ),
    ))

    fig.update_layout(
        title="Harvest utilisation (median)",
        height=250,
        margin=dict(l=6, r=42, t=42, b=1),
    )
    fig.update_xaxes(title="", showticklabels=False)
    fig.update_yaxes(title="", automargin=True)
    return fig


def _starting_quality_fig(storage_row_med: Optional[Dict[str, Any]]) -> Optional[go.Figure]:
    if not isinstance(storage_row_med, dict):
        return None
    inv_hist_week = storage_row_med.get("inventory_quality_hist_by_week", None)
    if inv_hist_week is None:
        return None
    inv = np.asarray(inv_hist_week, dtype=float)
    if inv.ndim != 2 or inv.shape[0] <= 0:
        return None
    return _overview_colored_gradeband_hist(inv[0, :], "Harvest inventory quality")


def _estimated_revenue(storage_row_med: Optional[Dict[str, Any]], waste_total_all: Optional[float]) -> Optional[float]:
    if not isinstance(storage_row_med, dict):
        return None
    ful = storage_row_med.get("fulfilled_by_week", None)
    if not isinstance(ful, dict):
        return None

    price = {"Extra": 100.0, "Class1": 75.0, "Class2": 60.0, "Processor": 40.0, "Waste": -5.0}

    delivered = {}
    for g in GRADES:
        a = _to_1d_float_array(ful.get(g, []))
        delivered[g] = float(np.nansum(np.maximum(0.0, a))) if a is not None else 0.0

    waste_bins_all = float(waste_total_all) if (waste_total_all is not None and np.isfinite(waste_total_all)) else float(
        storage_row_med.get("total_waste_bins", 0.0) or 0.0
    )

    rev = 0.0
    for g in GRADES:
        rev += delivered.get(g, 0.0) * price[g]
    rev += waste_bins_all * price["Waste"]
    return float(rev)


def _storage_grade_mix_pie(storage_row_med: Optional[Dict[str, Any]], config: Dict[str, Any]) -> Optional[go.Figure]:
    if not isinstance(storage_row_med, dict):
        return None
    inv_by_week = storage_row_med.get("inventory_by_week", None)
    if not isinstance(inv_by_week, dict):
        return None

    vals = {}
    for g in ALL_GRADES:
        a = _to_1d_float_array(inv_by_week.get(g, []))
        vals[g] = float(np.nanmean(np.maximum(0.0, a))) if (a is not None and a.size) else 0.0

    storage_factor = float(config.get("what_if_storage_factor", 1.0))
    long_term_capacity = float(config.get("long_term_capacity", 2000))
    effective_capacity = max(0.0, long_term_capacity * storage_factor)

    used = float(sum(vals.get(g, 0.0) for g in ALL_GRADES))
    vals["Empty / unused"] = float(max(0.0, effective_capacity - used))

    labels = ["Extra", "Class1", "Class2", "Processor", "Waste", "Empty / unused"]
    pie_vals = [vals.get(k, 0.0) for k in labels]
    pie_colors = [
        GRADE_COLORS["Extra"],
        GRADE_COLORS["Class1"],
        GRADE_COLORS["Class2"],
        GRADE_COLORS["Processor"],
        GRADE_COLORS["Waste"],
        "#9E9E9E",
    ]

    fig = go.Figure(data=[go.Pie(labels=labels, values=pie_vals, hole=0.55, marker=dict(colors=pie_colors))])
    fig.update_layout(
        title="Storage usage",
        height=248,
        margin=dict(l=8, r=8, t=42, b=8),
        legend=dict(orientation="v", x=1.02, xanchor="left", y=1.0, yanchor="top"),
    )
    return fig


def _fillrate_grade_bars(storage_row_med: Optional[Dict[str, Any]], waste_total_all: Optional[float]) -> Optional[go.Figure]:
    if not isinstance(storage_row_med, dict):
        return None
    dem = storage_row_med.get("demand_by_week", None)
    ful = storage_row_med.get("fulfilled_by_week", None)
    if not isinstance(dem, dict) or not isinstance(ful, dict):
        return None

    fr = {}
    for g in GRADES:
        d = _to_1d_float_array(dem.get(g, []))
        f = _to_1d_float_array(ful.get(g, []))
        if d is None or f is None or d.size == 0 or f.size == 0:
            fr[g] = 1.0
            continue
        dt = float(np.nansum(np.maximum(0.0, d)))
        ft = float(np.nansum(np.maximum(0.0, f)))
        fr[g] = float(ft / dt) if dt > 1e-9 else 1.0

    waste_all = float(waste_total_all) if (waste_total_all is not None and np.isfinite(waste_total_all)) else float(
        storage_row_med.get("total_waste_bins", 0.0) or 0.0
    )

    fulfilled_total = float(storage_row_med.get("total_fulfilled_bins", 0.0) or 0.0)
    ending_inv = float(storage_row_med.get("ending_inventory_bins", 0.0) or 0.0)
    denom = max(1e-9, fulfilled_total + ending_inv + waste_all)
    waste_pct = float(waste_all / denom)

    labels = ["Extra", "Class1", "Class2", "Processor", "Waste"]
    values = [
        fr.get("Extra", 1.0) * 100.0,
        fr.get("Class1", 1.0) * 100.0,
        fr.get("Class2", 1.0) * 100.0,
        fr.get("Processor", 1.0) * 100.0,
        waste_pct * 100.0,
    ]
    colors = [
        GRADE_COLORS["Extra"],
        GRADE_COLORS["Class1"],
        GRADE_COLORS["Class2"],
        GRADE_COLORS["Processor"],
        GRADE_COLORS["Waste"],
    ]

    fig = go.Figure([go.Bar(x=labels, y=values, marker=dict(color=colors))])
    fig.update_layout(
        title="Fill rate by grade + waste",
        yaxis_title="Fill rate(%)",
        height=248,
        margin=dict(l=52, r=12, t=42, b=38),
    )
    fig.update_yaxes(range=[0, 100])
    return fig


def _fillrate_weekly_plot(storage_row_med: Optional[Dict[str, Any]]) -> Optional[go.Figure]:
    if not isinstance(storage_row_med, dict):
        return None
    fr = storage_row_med.get("fill_rate_by_week", None)
    if not isinstance(fr, dict):
        return None

    T = None
    for g in GRADES:
        a = _to_1d_float_array(fr.get(g, []))
        if a is not None and a.size > 0:
            T = int(a.size)
            break
    if T is None:
        return None

    x = _x_axis(T)
    fig = go.Figure()
    for g in GRADES:
        a = _to_1d_float_array(fr.get(g, []))
        if a is None or a.size == 0:
            continue
        fig.add_trace(go.Scatter(
            x=x, y=np.asarray(a, dtype=float) * 100.0,
            mode="lines", name=g,
            line=dict(color=GRADE_COLORS.get(g, None), width=3),
            line_shape="hv",
        ))
    fig.update_layout(
        title="Weekly fill rate",
        xaxis_title="Week",
        yaxis_title="Fill rate (%)",
        height=248,
        hovermode="x unified",
        margin=dict(l=52, r=90, t=42, b=38),
        legend=dict(x=1.01, xanchor="left", y=1.0, yanchor="top"),
    )
    fig.update_yaxes(range=[0, 100])
    return fig


def render_overview_tab(config: Dict[str, Any], sim_results: Dict[str, Any]) -> None:
    st.subheader("Dashboard Overview")

    years = _pick_years_from_any(sim_results)
    if not years:
        st.info("Overview unavailable (no years found).")
        return
    year_sel = _year_slider(years)

    mc = _ensure_schema_growth(_get_df(sim_results, "mc_yearly"))
    util = _robust_util_df(sim_results)
    storage = _robust_storage_df_any(sim_results)
    storage_row_med = _median_detail_storage_row(sim_results, year_sel)

    initial_bins_med = _median_initial_bins(util, int(year_sel))
    waste_total_all = _waste_total_all_sources(util=util, storage=storage, year_sel=int(year_sel))

    if (waste_total_all is not None and np.isfinite(waste_total_all)
            and initial_bins_med is not None and np.isfinite(initial_bins_med)):
        waste_total_all = float(min(waste_total_all, initial_bins_med))

    row1_h = 305
    row2_h = 248

    c1, c2, c3 = st.columns(3, vertical_alignment="top")

    with c1:
        fig_y = _yield_dist_fig(mc, year_sel, height=row1_h)
        if fig_y is None:
            st.info("Yield distribution unavailable.")
        else:
            st.plotly_chart(
                fig_y,
                use_container_width=True,
                config={"displayModeBar": False},
                key=f"ov_yield_dist_{year_sel}",
            )

    with c2:
        fig_q0 = _starting_quality_fig(storage_row_med)
        if fig_q0 is None:
            st.info("Starting quality distribution unavailable.")
        else:
            fig_q0.update_layout(height=int(row1_h))
            st.plotly_chart(
                fig_q0,
                use_container_width=True,
                config={"displayModeBar": False},
                key=f"ov_quality_week0_{year_sel}",
            )

    with c3:
        y_med = None
        if not mc.empty and "season_year" in mc.columns and "yield_t_ha" in mc.columns:
            dfy = mc[mc["season_year"] == int(year_sel)].copy()
            yv = pd.to_numeric(dfy["yield_t_ha"], errors="coerce").dropna()
            if not yv.empty:
                y_med = float(yv.median())

        bins_on_tree_med = initial_bins_med

        top_util_pct = None
        if not util.empty and "season_year" in util.columns:
            u = util.copy()
            u["season_year"] = pd.to_numeric(u["season_year"], errors="coerce")
            u = u[u["season_year"] == float(year_sel)]
            if not u.empty:
                candidates = ["util_grade", "util_shuttle", "util_pick", "util_empty"]
                best = None
                for col in candidates:
                    if col not in u.columns:
                        continue
                    v = pd.to_numeric(u[col], errors="coerce").dropna()
                    if v.empty:
                        continue
                    medp = float(np.clip(v.median(), 0.0, 1.0) * 100.0)
                    best = medp if (best is None or medp > best) else best
                top_util_pct = best

        fill_rate_med = None
        if not storage.empty and "season_year" in storage.columns and "fill_rate_overall" in storage.columns:
            s = storage.copy()
            s["season_year"] = pd.to_numeric(s["season_year"], errors="coerce")
            s = s[s["season_year"] == float(year_sel)]
            fr = pd.to_numeric(s["fill_rate_overall"], errors="coerce").dropna()
            if not fr.empty:
                fill_rate_med = float(np.clip(fr.median(), 0.0, 1.0) * 100.0)

        waste_bins_all = float(waste_total_all) if (waste_total_all is not None and np.isfinite(waste_total_all)) else None
        revenue_est = _estimated_revenue(storage_row_med, waste_total_all=waste_total_all)

        metrics_col, hm_col = st.columns([4.2, 1.4], vertical_alignment="top")

        with metrics_col:
            r1a, r1b = st.columns(2)
            r2a, r2b = st.columns(2)
            r3a, r3b = st.columns(2)

            with r1a:
                st.metric("Median yield (t/ha)", "—" if y_med is None else f"{y_med:.2f}")
            with r1b:
                st.metric("Median bins on trees", "—" if bins_on_tree_med is None else f"{bins_on_tree_med:.0f}")

            with r2a:
                st.metric("Highest utilisation (%)", "—" if top_util_pct is None else f"{top_util_pct:.0f}%")
            with r2b:
                st.metric("Fill rate (%)", "—" if fill_rate_med is None else f"{fill_rate_med:.0f}%")

            with r3a:
                st.metric("Waste bins", "—" if waste_bins_all is None else f"{waste_bins_all:.0f}")
            with r3b:
                st.metric("Estimated revenue", "—" if revenue_est is None else f"£{revenue_est:,.0f}")

        with hm_col:
            fig_hm = _util_heatmap_selected_year_vertical(util, year_sel)
            if fig_hm is None:
                st.info("Utilisation heatmap unavailable.")
            else:
                st.plotly_chart(
                    fig_hm,
                    use_container_width=True,
                    config={"displayModeBar": False},
                    key=f"ov_util_hm_{year_sel}",
                )

    st.divider()
    b1, b2, b3 = st.columns(3, vertical_alignment="top")

    with b1:
        fig_pie = _storage_grade_mix_pie(storage_row_med, config)
        if fig_pie is None:
            st.info("Storage utilisation mix unavailable.")
        else:
            fig_pie.update_layout(height=int(row2_h))
            st.plotly_chart(
                fig_pie,
                use_container_width=True,
                config={"displayModeBar": False},
                key=f"ov_storage_pie_{year_sel}",
            )

    with b2:
        fig_bar = _fillrate_grade_bars(storage_row_med, waste_total_all=waste_total_all)
        if fig_bar is None:
            st.info("Grade fill-rate bars unavailable.")
        else:
            fig_bar.update_layout(height=int(row2_h))
            st.plotly_chart(
                fig_bar,
                use_container_width=True,
                config={"displayModeBar": False},
                key=f"ov_fillrate_bars_{year_sel}",
            )

    with b3:
        fig_fr = _fillrate_weekly_plot(storage_row_med)
        if fig_fr is None:
            st.info("Weekly fill-rate plot unavailable.")
        else:
            fig_fr.update_layout(height=int(row2_h))
            st.plotly_chart(
                fig_fr,
                use_container_width=True,
                config={"displayModeBar": False},
                key=f"ov_fillrate_weekly_{year_sel}",
            )












# ============================================================
# Overview tab (UPDATED) — WASTE + MASS-BASIS FIX:
#   ✅ "Median bins on trees" uses util["initial_bins_on_trees"] (preferred)
#      instead of trees_series[0] (which can already be reduced by day-0 decay removal).
#   ✅ Waste metric = median( harvest_waste_removed_total + storage_total_waste_bins )
#      - computed per-run if shared run-id exists
#      - else fallback sum-of-medians
#   ✅ Guardrail: waste_total_all is clamped to median initial bins (prevents impossible displays)
#   ✅ Layout update:
#      - removed selected extra metrics
#      - heatmap placed underneath metrics
#      - divider moved closer to top row
#      - bottom row vertically aligned
#      - heatmap shifted left
#      - heatmap colourbar stretched further
#      - weekly fill-rate made taller
#      - starting inventory legend removed
# ============================================================

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# assumes these exist in your project
# - _get_df, _get_nested, _ensure_schema_growth, _hist, _to_1d_float_array, _x_axis
# - GRADES, ALL_GRADES, Q_THRESHOLDS, GRADE_COLORS


def _pick_years_from_any(sim_results: Dict[str, Any]) -> List[int]:
    for key in ["mc_yearly", "harvest_util_by_year", "storage_by_year"]:
        df = _get_df(sim_results, key)
        if not df.empty and "season_year" in df.columns:
            yrs = sorted(pd.to_numeric(df["season_year"], errors="coerce").dropna().astype(int).unique().tolist())
            if yrs:
                return yrs
    med_storage = _get_nested(sim_results, ["median_detail", "des_out", "storage_by_year"])
    if isinstance(med_storage, pd.DataFrame) and not med_storage.empty and "season_year" in med_storage.columns:
        yrs = sorted(pd.to_numeric(med_storage["season_year"], errors="coerce").dropna().astype(int).unique().tolist())
        return yrs
    return []


def _year_slider(years: List[int]) -> int:
    if not years:
        return 0
    if len(years) == 1:
        st.slider("Year", years[0], years[0], years[0], 1, key="overview_year_slider")
        return years[0]
    return int(st.slider("Year", int(years[0]), int(years[-1]), int(years[-1]), 1, key="overview_year_slider"))


def _robust_util_df(sim_results: Dict[str, Any]) -> pd.DataFrame:
    util = _get_df(sim_results, "harvest_util_by_year")
    if not util.empty:
        return util
    maybe = _get_nested(sim_results, ["median_detail", "des_out", "harvest_yearly"])
    return maybe.copy() if isinstance(maybe, pd.DataFrame) else pd.DataFrame()


def _robust_storage_df_any(sim_results: Dict[str, Any]) -> pd.DataFrame:
    storage = _get_df(sim_results, "storage_by_year")
    if not storage.empty:
        return storage
    maybe = _get_nested(sim_results, ["median_detail", "des_out", "storage_by_year"])
    return maybe.copy() if isinstance(maybe, pd.DataFrame) else pd.DataFrame()


def _median_detail_storage_row(sim_results: Dict[str, Any], year_sel: int) -> Optional[Dict[str, Any]]:
    med = sim_results.get("median_detail", {})
    des_out = med.get("des_out") if isinstance(med, dict) else None
    storage_med = des_out.get("storage_by_year", None) if isinstance(des_out, dict) else None
    if not isinstance(storage_med, pd.DataFrame) or storage_med.empty or "season_year" not in storage_med.columns:
        return None
    df = storage_med.copy()
    df["season_year"] = pd.to_numeric(df["season_year"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["season_year"]).copy()
    df["season_year"] = df["season_year"].astype(int)
    sel = df[df["season_year"] == int(year_sel)]
    if sel.empty:
        return None
    return sel.iloc[0].to_dict()


def _median_initial_bins(util: pd.DataFrame, year_sel: int) -> Optional[float]:
    if util.empty or "season_year" not in util.columns:
        return None
    u = util.copy()
    u["season_year"] = pd.to_numeric(u["season_year"], errors="coerce")
    u = u[u["season_year"] == float(year_sel)]
    if u.empty:
        return None

    if "initial_bins_on_trees" in u.columns:
        v = pd.to_numeric(u["initial_bins_on_trees"], errors="coerce").dropna()
        if not v.empty:
            return float(v.median())

    if "trees_series" in u.columns:
        starts = []
        for vv in u["trees_series"]:
            if isinstance(vv, list) and len(vv) > 0:
                try:
                    starts.append(float(vv[0]))
                except Exception:
                    pass
        if starts:
            return float(np.nanmedian(np.asarray(starts, dtype=float)))

    return None


def _waste_total_all_sources(util: pd.DataFrame, storage: pd.DataFrame, year_sel: int) -> Optional[float]:
    u = util.copy() if isinstance(util, pd.DataFrame) else pd.DataFrame()
    s = storage.copy() if isinstance(storage, pd.DataFrame) else pd.DataFrame()

    if not u.empty and "season_year" in u.columns:
        u["season_year"] = pd.to_numeric(u["season_year"], errors="coerce")
        u = u[u["season_year"] == float(year_sel)].copy()
    else:
        u = pd.DataFrame()

    if not s.empty and "season_year" in s.columns:
        s["season_year"] = pd.to_numeric(s["season_year"], errors="coerce")
        s = s[s["season_year"] == float(year_sel)].copy()
    else:
        s = pd.DataFrame()

    harvest_col = None
    for c in ["waste_bins_removed_total", "waste_bins_removed", "waste_bins_short_term"]:
        if (not u.empty) and (c in u.columns):
            harvest_col = c
            break
    storage_col = "total_waste_bins" if ((not s.empty) and ("total_waste_bins" in s.columns)) else None

    if harvest_col is None and storage_col is None:
        return None

    run_id_candidates = ["run_id", "mc_run", "rep", "trial", "sim", "seed", "iteration"]
    shared_id = None
    for c in run_id_candidates:
        if (not u.empty) and (not s.empty) and (c in u.columns) and (c in s.columns):
            shared_id = c
            break

    if shared_id is not None:
        uu = u[[shared_id, harvest_col]].copy() if harvest_col else u[[shared_id]].copy()
        ss = s[[shared_id, storage_col]].copy() if storage_col else s[[shared_id]].copy()
        if harvest_col:
            uu[harvest_col] = pd.to_numeric(uu[harvest_col], errors="coerce").fillna(0.0)
        if storage_col:
            ss[storage_col] = pd.to_numeric(ss[storage_col], errors="coerce").fillna(0.0)
        m = uu.merge(ss, on=shared_id, how="outer").fillna(0.0)
        hw = m[harvest_col].to_numpy(dtype=float) if harvest_col else 0.0
        sw = m[storage_col].to_numpy(dtype=float) if storage_col else 0.0
        tot = np.asarray(hw, dtype=float) + np.asarray(sw, dtype=float)
        tot = tot[np.isfinite(tot)]
        return float(np.median(tot)) if tot.size else None

    total = 0.0
    got = False
    if harvest_col is not None and not u.empty:
        hw = pd.to_numeric(u[harvest_col], errors="coerce").dropna()
        if not hw.empty:
            total += float(hw.median())
            got = True
    if storage_col is not None and not s.empty:
        sw = pd.to_numeric(s[storage_col], errors="coerce").dropna()
        if not sw.empty:
            total += float(sw.median())
            got = True

    return float(total) if got else None


def _overview_colored_gradeband_hist(inv_hist: np.ndarray, title: str) -> go.Figure:
    h = np.asarray(inv_hist, dtype=float)
    h[~np.isfinite(h)] = 0.0
    h = np.maximum(0.0, h)

    q_bins = int(h.size)
    edges = np.linspace(0.0, 1.0, q_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    width = float(edges[1] - edges[0]) if q_bins > 0 else 0.02

    thr_p = float(Q_THRESHOLDS["Processor_min"])
    thr_c2 = float(Q_THRESHOLDS["Class2_min"])
    thr_c1 = float(Q_THRESHOLDS["Class1_min"])
    thr_e = float(Q_THRESHOLDS["Extra_min"])

    masks = {
        "Waste": centers < thr_p,
        "Processor": (centers >= thr_p) & (centers < thr_c2),
        "Class2": (centers >= thr_c2) & (centers < thr_c1),
        "Class1": (centers >= thr_c1) & (centers < thr_e),
        "Extra": centers >= thr_e,
    }

    fig = go.Figure()
    for g in ["Waste", "Processor", "Class2", "Class1", "Extra"]:
        m = masks[g]
        if not np.any(m):
            continue
        fig.add_trace(go.Bar(
            x=centers[m],
            y=h[m],
            name=g,
            marker=dict(color=GRADE_COLORS.get(g, "#888888")),
            width=width,
            hovertemplate=f"{g}<br>Quality=%{{x:.3f}}<br>Bins=%{{y:.1f}}<extra></extra>",
            showlegend=False,
        ))

    thr_lines = [thr_p, thr_c2, thr_c1, thr_e]
    ymax = float(np.nanmax(h)) if h.size else 1.0
    ymax = max(1.0, ymax * 1.08)

    for xthr in thr_lines:
        fig.add_shape(
            type="line",
            x0=xthr, x1=xthr,
            y0=0.0, y1=ymax,
            line=dict(color="white", width=2, dash="dot"),
        )

    fig.update_layout(
        title=dict(text=title, x=0.0, xanchor="left"),
        barmode="overlay",
        xaxis_title="Quality (0–1)",
        yaxis_title="Bins",
        height=350,
        margin=dict(l=52, r=12, t=46, b=1),
        hovermode="x unified",
        showlegend=False,
    )
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, ymax])
    return fig


def _yield_dist_fig(mc: pd.DataFrame, year_sel: int, height: int) -> Optional[go.Figure]:
    if mc.empty or "season_year" not in mc.columns or "yield_t_ha" not in mc.columns:
        return None
    dfy = mc[mc["season_year"] == int(year_sel)].copy()
    yld = pd.to_numeric(dfy["yield_t_ha"], errors="coerce").dropna().to_numpy(dtype=float)
    if yld.size == 0:
        return None
    xh, yh = _hist(yld, bins=45)
    fig = go.Figure([go.Bar(x=xh, y=yh)])
    fig.update_layout(
        title=dict(text="Yield distribution", x=0.0, xanchor="left"),
        height=int(height),
        margin=dict(l=52, r=12, t=46, b=1),
    )
    fig.update_xaxes(title="Yield (t/ha)", range=[0, max(1e-9, float(np.nanmax(xh)) if xh.size else 1.0)])
    fig.update_yaxes(title="Frequency", range=[0, max(1.0, float(np.nanmax(yh)) * 1.12 if yh.size else 1.0)])
    return fig


def _util_heatmap_selected_year_horizontal(util: pd.DataFrame, year_sel: int) -> Optional[go.Figure]:
    if util.empty or "season_year" not in util.columns:
        return None

    u = util.copy()
    u["season_year"] = pd.to_numeric(u["season_year"], errors="coerce")
    u = u[u["season_year"] == float(year_sel)].copy()
    if u.empty:
        return None

    stages = [
        ("util_grade", "Graders"),
        ("util_shuttle", "Filled shuttle"),
        ("util_pick", "Harvesters"),
        ("util_empty", "Empty shuttle"),
    ]

    xs, vals = [], []
    for col, label in stages:
        if col not in u.columns:
            continue
        v = pd.to_numeric(u[col], errors="coerce").dropna()
        med = float(np.clip(v.median(), 0.0, 1.0)) if not v.empty else 0.0
        xs.append(label)
        vals.append(med)

    if not xs:
        return None

    colorscale = [
        [0.00, "#B0B0B0"],
        [0.05, "#B0B0B0"],
        [0.30, "#FFD966"],
        [0.60, "#B7F0A0"],
        [0.90, "#00B050"],
        [1.00, "#D00000"],
    ]

    Z = np.asarray([vals], dtype=float)
    fig = go.Figure(data=go.Heatmap(
        z=Z,
        x=xs,
        y=[""],
        zmin=0.0,
        zmax=1.0,
        colorscale=colorscale,
        hoverongaps=False,
        colorbar=dict(
            title=dict(text="Util", side="top", font=dict(size=12)),
            thickness=20,
            len=1.38,
            lenmode="fraction",
            y=0.50,
            yanchor="middle",
            x=1.03,
            tickfont=dict(size=11),
        ),
    ))

    fig.update_layout(
        title=dict(text="Harvest utilisation (median)", x=0.0, xanchor="left"),
        height=160,
        margin=dict(l=8, r=52, t=34, b=1),
    )
    fig.update_xaxes(
        title="",
        tickangle=0,
        automargin=True,
        tickfont=dict(size=10),
    )
    fig.update_yaxes(
        title="",
        showticklabels=False,
        ticks="",
    )
    return fig


def _starting_quality_fig(storage_row_med: Optional[Dict[str, Any]]) -> Optional[go.Figure]:
    if not isinstance(storage_row_med, dict):
        return None
    inv_hist_week = storage_row_med.get("inventory_quality_hist_by_week", None)
    if inv_hist_week is None:
        return None
    inv = np.asarray(inv_hist_week, dtype=float)
    if inv.ndim != 2 or inv.shape[0] <= 0:
        return None
    return _overview_colored_gradeband_hist(inv[0, :], "Harvest inventory quality")


def _estimated_revenue(storage_row_med: Optional[Dict[str, Any]], waste_total_all: Optional[float]) -> Optional[float]:
    if not isinstance(storage_row_med, dict):
        return None
    ful = storage_row_med.get("fulfilled_by_week", None)
    if not isinstance(ful, dict):
        return None

    price = {"Extra": 100.0, "Class1": 75.0, "Class2": 60.0, "Processor": 40.0, "Waste": -5.0}

    delivered = {}
    for g in GRADES:
        a = _to_1d_float_array(ful.get(g, []))
        delivered[g] = float(np.nansum(np.maximum(0.0, a))) if a is not None else 0.0

    waste_bins_all = float(waste_total_all) if (waste_total_all is not None and np.isfinite(waste_total_all)) else float(
        storage_row_med.get("total_waste_bins", 0.0) or 0.0
    )

    rev = 0.0
    for g in GRADES:
        rev += delivered.get(g, 0.0) * price[g]
    rev += waste_bins_all * price["Waste"]
    return float(rev)


def _storage_grade_mix_pie(storage_row_med: Optional[Dict[str, Any]], config: Dict[str, Any]) -> Optional[go.Figure]:
    if not isinstance(storage_row_med, dict):
        return None
    inv_by_week = storage_row_med.get("inventory_by_week", None)
    if not isinstance(inv_by_week, dict):
        return None

    vals = {}
    for g in ALL_GRADES:
        a = _to_1d_float_array(inv_by_week.get(g, []))
        vals[g] = float(np.nanmean(np.maximum(0.0, a))) if (a is not None and a.size) else 0.0

    storage_factor = float(config.get("what_if_storage_factor", 1.0))
    long_term_capacity = float(config.get("long_term_capacity", 2000))
    effective_capacity = max(0.0, long_term_capacity * storage_factor)

    used = float(sum(vals.get(g, 0.0) for g in ALL_GRADES))
    vals["Empty / unused"] = float(max(0.0, effective_capacity - used))

    labels = ["Extra", "Class1", "Class2", "Processor", "Waste", "Empty / unused"]
    pie_vals = [vals.get(k, 0.0) for k in labels]
    pie_colors = [
        GRADE_COLORS["Extra"],
        GRADE_COLORS["Class1"],
        GRADE_COLORS["Class2"],
        GRADE_COLORS["Processor"],
        GRADE_COLORS["Waste"],
        "#9E9E9E",
    ]

    fig = go.Figure(data=[go.Pie(labels=labels, values=pie_vals, hole=0.55, marker=dict(colors=pie_colors))])
    fig.update_layout(
        title=dict(text="Storage usage", x=0.0, xanchor="left"),
        height=245,
        margin=dict(l=20, r=175, t=42, b=32),
        legend=dict(orientation="v", x=0.9, xanchor="left", y=1.0, yanchor="top"),
    )
    return fig


def _fillrate_grade_bars(storage_row_med: Optional[Dict[str, Any]], waste_total_all: Optional[float]) -> Optional[go.Figure]:
    if not isinstance(storage_row_med, dict):
        return None
    dem = storage_row_med.get("demand_by_week", None)
    ful = storage_row_med.get("fulfilled_by_week", None)
    if not isinstance(dem, dict) or not isinstance(ful, dict):
        return None

    fr = {}
    for g in GRADES:
        d = _to_1d_float_array(dem.get(g, []))
        f = _to_1d_float_array(ful.get(g, []))
        if d is None or f is None or d.size == 0 or f.size == 0:
            fr[g] = 1.0
            continue
        dt = float(np.nansum(np.maximum(0.0, d)))
        ft = float(np.nansum(np.maximum(0.0, f)))
        fr[g] = float(ft / dt) if dt > 1e-9 else 1.0

    waste_all = float(waste_total_all) if (waste_total_all is not None and np.isfinite(waste_total_all)) else float(
        storage_row_med.get("total_waste_bins", 0.0) or 0.0
    )

    fulfilled_total = float(storage_row_med.get("total_fulfilled_bins", 0.0) or 0.0)
    ending_inv = float(storage_row_med.get("ending_inventory_bins", 0.0) or 0.0)
    denom = max(1e-9, fulfilled_total + ending_inv + waste_all)
    waste_pct = float(waste_all / denom)

    labels = ["Extra", "Class1", "Class2", "Processor", "Waste"]
    values = [
        fr.get("Extra", 1.0) * 100.0,
        fr.get("Class1", 1.0) * 100.0,
        fr.get("Class2", 1.0) * 100.0,
        fr.get("Processor", 1.0) * 100.0,
        waste_pct * 100.0,
    ]
    colors = [
        GRADE_COLORS["Extra"],
        GRADE_COLORS["Class1"],
        GRADE_COLORS["Class2"],
        GRADE_COLORS["Processor"],
        GRADE_COLORS["Waste"],
    ]

    fig = go.Figure([go.Bar(x=labels, y=values, marker=dict(color=colors))])
    fig.update_layout(
        title=dict(text="Fill rate by grade + waste", x=0.0, xanchor="left"),
        yaxis_title="Fill rate(%)",
        height=245,
        margin=dict(l=52, r=12, t=42, b=32),
    )
    fig.update_yaxes(range=[0, 100])
    return fig


def _fillrate_weekly_plot(storage_row_med: Optional[Dict[str, Any]]) -> Optional[go.Figure]:
    if not isinstance(storage_row_med, dict):
        return None
    fr = storage_row_med.get("fill_rate_by_week", None)
    if not isinstance(fr, dict):
        return None

    T = None
    for g in GRADES:
        a = _to_1d_float_array(fr.get(g, []))
        if a is not None and a.size > 0:
            T = int(a.size)
            break
    if T is None:
        return None

    x = _x_axis(T)
    fig = go.Figure()
    for g in GRADES:
        a = _to_1d_float_array(fr.get(g, []))
        if a is None or a.size == 0:
            continue
        fig.add_trace(go.Scatter(
            x=x, y=np.asarray(a, dtype=float) * 100.0,
            mode="lines", name=g,
            line=dict(color=GRADE_COLORS.get(g, None), width=3),
            line_shape="hv",
        ))
    fig.update_layout(
        title=dict(text="Weekly fill rate", x=0.0, xanchor="left"),
        xaxis_title="Week",
        yaxis_title="Fill rate (%)",
        height=500,
        hovermode="x unified",
        margin=dict(l=52, r=90, t=42, b=32),
        legend=dict(x=1.01, xanchor="left", y=1.0, yanchor="top"),
    )
    fig.update_yaxes(range=[0, 100])
    return fig


def render_overview_tab(config: Dict[str, Any], sim_results: Dict[str, Any]) -> None:
    st.subheader("Dashboard Overview")

    years = _pick_years_from_any(sim_results)
    if not years:
        st.info("Overview unavailable (no years found).")
        return
    year_sel = _year_slider(years)

    mc = _ensure_schema_growth(_get_df(sim_results, "mc_yearly"))
    util = _robust_util_df(sim_results)
    storage = _robust_storage_df_any(sim_results)
    storage_row_med = _median_detail_storage_row(sim_results, year_sel)

    initial_bins_med = _median_initial_bins(util, int(year_sel))
    waste_total_all = _waste_total_all_sources(util=util, storage=storage, year_sel=int(year_sel))

    if (waste_total_all is not None and np.isfinite(waste_total_all)
            and initial_bins_med is not None and np.isfinite(initial_bins_med)):
        waste_total_all = float(min(waste_total_all, initial_bins_med))

    row1_h = 350
    pie_h = 245
    bar_h = 245
    weekly_h = 275

    c1, c2, c3 = st.columns([1.05, 1.05, 1.10], vertical_alignment="top")

    with c1:
        fig_y = _yield_dist_fig(mc, year_sel, height=row1_h)
        if fig_y is None:
            st.info("Yield distribution unavailable.")
        else:
            st.plotly_chart(
                fig_y,
                use_container_width=True,
                config={"displayModeBar": False},
                key=f"ov_yield_dist_{year_sel}",
            )

    with c2:
        fig_q0 = _starting_quality_fig(storage_row_med)
        if fig_q0 is None:
            st.info("Starting quality distribution unavailable.")
        else:
            fig_q0.update_layout(height=int(row1_h))
            st.plotly_chart(
                fig_q0,
                use_container_width=True,
                config={"displayModeBar": False},
                key=f"ov_quality_week0_{year_sel}",
            )

    with c3:
        y_med = None
        if not mc.empty and "season_year" in mc.columns and "yield_t_ha" in mc.columns:
            dfy = mc[mc["season_year"] == int(year_sel)].copy()
            yv = pd.to_numeric(dfy["yield_t_ha"], errors="coerce").dropna()
            if not yv.empty:
                y_med = float(yv.median())

        fill_rate_med = None
        if not storage.empty and "season_year" in storage.columns and "fill_rate_overall" in storage.columns:
            s = storage.copy()
            s["season_year"] = pd.to_numeric(s["season_year"], errors="coerce")
            s = s[s["season_year"] == float(year_sel)]
            fr = pd.to_numeric(s["fill_rate_overall"], errors="coerce").dropna()
            if not fr.empty:
                fill_rate_med = float(np.clip(fr.median(), 0.0, 1.0) * 100.0)

        waste_bins_all = float(waste_total_all) if (waste_total_all is not None and np.isfinite(waste_total_all)) else None
        revenue_est = _estimated_revenue(storage_row_med, waste_total_all=waste_total_all)

        r1a, r1b = st.columns(2)
        r2a, r2b = st.columns(2)

        with r1a:
            st.metric("Median yield (t/ha)", "—" if y_med is None else f"{y_med:.2f}")
        with r1b:
            st.metric("Fill rate (%)", "—" if fill_rate_med is None else f"{fill_rate_med:.0f}%")

        with r2a:
            st.metric("Waste bins", "—" if waste_bins_all is None else f"{waste_bins_all:.0f}")
        with r2b:
            st.metric("Estimated revenue", "—" if revenue_est is None else f"£{revenue_est:,.0f}")

        fig_hm = _util_heatmap_selected_year_horizontal(util, year_sel)
        if fig_hm is None:
            st.info("Utilisation heatmap unavailable.")
        else:
            st.plotly_chart(
                fig_hm,
                use_container_width=True,
                config={"displayModeBar": False},
                key=f"ov_util_hm_{year_sel}",
            )

    st.markdown("<div style='margin-top:-700px;'></div>", unsafe_allow_html=True)
    st.divider()

    b1, b2, b3 = st.columns(3, vertical_alignment="top")

    with b1:
        fig_pie = _storage_grade_mix_pie(storage_row_med, config)
        if fig_pie is None:
            st.info("Storage utilisation mix unavailable.")
        else:
            fig_pie.update_layout(height=int(pie_h))
            st.plotly_chart(
                fig_pie,
                use_container_width=True,
                config={"displayModeBar": False},
                key=f"ov_storage_pie_{year_sel}",
            )

    with b2:
        fig_bar = _fillrate_grade_bars(storage_row_med, waste_total_all=waste_total_all)
        if fig_bar is None:
            st.info("Grade fill-rate bars unavailable.")
        else:
            fig_bar.update_layout(height=int(bar_h))
            st.plotly_chart(
                fig_bar,
                use_container_width=True,
                config={"displayModeBar": False},
                key=f"ov_fillrate_bars_{year_sel}",
            )

    with b3:
        fig_fr = _fillrate_weekly_plot(storage_row_med)
        if fig_fr is None:
            st.info("Weekly fill-rate plot unavailable.")
        else:
            fig_fr.update_layout(height=int(weekly_h))
            st.plotly_chart(
                fig_fr,
                use_container_width=True,
                config={"displayModeBar": False},
                key=f"ov_fillrate_weekly_{year_sel}",
            )


















# ============================================================
# Growth tab
# ============================================================
def render_growth_tab(config: Dict[str, Any], sim_results: Dict[str, Any], weather_cache=None) -> None:
    st.subheader("Growth")

    mc = _ensure_schema_growth(_get_df(sim_results, "mc_yearly"))
    if mc.empty or "yield_t_ha" not in mc.columns:
        st.info("mc_yearly missing or yield_t_ha missing.")
        return

    mc = mc.dropna(subset=["yield_t_ha"]).copy()
    mc = _compute_cum(mc)

    years = sorted(mc["season_year"].unique().tolist())
    if not years:
        st.info("No season years found.")
        return

    year_sel = st.slider("Select year", int(years[0]), int(years[-1]), int(years[-1]), 1, key="growth_year_sel")

    # -----------------------------
    # helpers (local)
    # -----------------------------
    def _set_x0(fig: go.Figure) -> None:
        fig.update_xaxes(rangemode="tozero")

    def _find_first_col_like(df: pd.DataFrame, candidates: List[str], keywords: List[str]) -> Optional[str]:
        # 1) direct candidates
        for c in candidates:
            if c in df.columns:
                return c
        # 2) keyword scan (robust for "leaf area" naming differences)
        cols = [str(c) for c in df.columns]
        low = [c.lower() for c in cols]
        for i, c in enumerate(low):
            if any(k in c for k in keywords):
                return cols[i]
        return None

    def _year_stats(df: pd.DataFrame, col: str) -> Optional[pd.DataFrame]:
        if "season_year" not in df.columns or col not in df.columns:
            return None
        tmp = df[["season_year", col]].copy()
        tmp["season_year"] = pd.to_numeric(tmp["season_year"], errors="coerce")
        tmp[col] = pd.to_numeric(tmp[col], errors="coerce")
        tmp = tmp.dropna(subset=["season_year", col]).copy()
        if tmp.empty:
            return None
        tmp["season_year"] = tmp["season_year"].astype(int)
        g = tmp.groupby("season_year")[col]
        out = g.agg(vmin="min", vmed="median", vmax="max").reset_index().sort_values("season_year").reset_index(drop=True)
        return out if not out.empty else None

    def _banded_min_med_max_figure(stats: pd.DataFrame, title: str, ylab: str) -> go.Figure:
        # Creates:
        # - red filled area between min and median (under median)
        # - green filled area between median and max (above median)
        # - median line
        x = stats["season_year"].to_numpy(dtype=int)
        vmin = stats["vmin"].to_numpy(dtype=float)
        vmed = stats["vmed"].to_numpy(dtype=float)
        vmax = stats["vmax"].to_numpy(dtype=float)

        fig = go.Figure()

        # base for red fill (min -> median)
        fig.add_trace(go.Scatter(
            x=x, y=vmin,
            mode="lines",
            line=dict(width=0),
            name="Min",
            showlegend=False,
            hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=x, y=vmed,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(255,0,0,0.18)",
            name="Min–Median",
            hovertemplate="Year=%{x}<br>Median=%{y:.3g}<extra></extra>",
            showlegend=False,
        ))

        # green fill (median -> max)
        fig.add_trace(go.Scatter(
            x=x, y=vmax,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(0,255,0,0.16)",
            name="Median–Max",
            hovertemplate="Year=%{x}<br>Max=%{y:.3g}<extra></extra>",
            showlegend=False,
        ))

        # median line on top
        fig.add_trace(go.Scatter(
            x=x, y=vmed,
            mode="lines+markers",
            name="Median",
            line=dict(width=2),
            hovertemplate="Year=%{x}<br>Median=%{y:.3g}<extra></extra>",
        ))

        fig.add_vline(x=int(year_sel), line_width=2, line_dash="dot")

        fig.update_layout(
            title=title,
            xaxis_title="Season year",
            yaxis_title=ylab,
            height=360,
            margin=dict(l=55, r=15, t=55, b=45),
            hovermode="x unified",
        )
        return fig

    # -----------------------------
    # Yield spaghetti (cached base)
    # -----------------------------
    base_key = _hash_df(mc[["mc_run", "season_year", "yield_t_ha"]])
    if st.session_state.get("_spaghetti_base_key") != base_key:
        st.session_state["_spaghetti_base_key"] = base_key
        st.session_state["_spaghetti_base_fig"] = _build_spaghetti_base(mc)

    left, right = st.columns([2.0, 1.0], vertical_alignment="top")

    with left:
        fig_spag = _figure_with_year_marker(st.session_state["_spaghetti_base_fig"], int(year_sel))
        fig_spag.update_layout(height=340, margin=dict(l=45, r=10, t=60, b=45))
        st.plotly_chart(
            fig_spag,
            use_container_width=True,
            config={"displayModeBar": False, "staticPlot": True},
            key=f"growth_spaghetti_{year_sel}",
        )

    with right:
        dfy = mc[mc["season_year"] == int(year_sel)].copy()
        cum = pd.to_numeric(dfy.get("cum_yield_t_ha", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)
        cum = cum[np.isfinite(cum)]
        dev = cum - float(np.nanmean(cum)) if cum.size else cum
        centers, counts = _hist(dev, bins=70)
        fig_dev = go.Figure([go.Bar(x=counts, y=centers, orientation="h")])
        fig_dev.update_layout(
            title=f"Cumulative deviation (t/ha) — {year_sel}",
            xaxis_title="Frequency",
            yaxis_title="Cum deviation (t/ha)",
            height=340,
            margin=dict(l=40, r=10, t=60, b=45),
        )
        st.plotly_chart(fig_dev, use_container_width=True, config={"displayModeBar": False},
                        key=f"growth_cumdev_{year_sel}")

    # -----------------------------
    # Distributions (selected year) — x-axis from 0 (taller)
    # -----------------------------
    st.markdown("### Fruit growth distributions")
    dfy = mc[mc["season_year"] == int(year_sel)].copy()
    c1, c2, c3 = st.columns(3)

    with c1:
        if "fruit_mass_kg" in dfy.columns and dfy["fruit_mass_kg"].notna().any():
            mass_g = pd.to_numeric(dfy["fruit_mass_kg"], errors="coerce").to_numpy(dtype=float) * 1000.0
            mass_g = mass_g[np.isfinite(mass_g)]
            xmax = float(np.nanmax(mass_g)) if mass_g.size else 1.0
            xh, yh = _hist(mass_g, bins=60, range_=(0.0, xmax))
            fig = go.Figure([go.Bar(x=xh, y=yh)])
            fig.update_layout(title="Fruit mass (g)", xaxis_title="g/fruit", yaxis_title="Frequency", height=340,
                              margin=dict(l=45, r=10, t=55, b=45))
            _set_x0(fig)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False},
                            key=f"growth_massdist_{year_sel}")
        else:
            st.info("Fruit mass not available.")

    with c2:
        if "fruit_number_tree" in dfy.columns and dfy["fruit_number_tree"].notna().any():
            num = pd.to_numeric(dfy["fruit_number_tree"], errors="coerce").to_numpy(dtype=float)
            num = num[np.isfinite(num)]
            xmax = float(np.nanmax(num)) if num.size else 1.0
            xh, yh = _hist(num, bins=60, range_=(0.0, xmax))
            fig = go.Figure([go.Bar(x=xh, y=yh)])
            fig.update_layout(title="Fruit number", xaxis_title="fruit/tree", yaxis_title="Frequency", height=340,
                              margin=dict(l=45, r=10, t=55, b=45))
            _set_x0(fig)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False},
                            key=f"growth_numdist_{year_sel}")
        else:
            st.info("Fruit number not available.")

    with c3:
        yld = pd.to_numeric(dfy["yield_t_ha"], errors="coerce").to_numpy(dtype=float)
        yld = yld[np.isfinite(yld)]
        xmax = float(np.nanmax(yld)) if yld.size else 1.0
        xh, yh = _hist(yld, bins=60, range_=(0.0, xmax))
        fig = go.Figure([go.Bar(x=xh, y=yh)])
        fig.update_layout(title="Yield (t/ha)", xaxis_title="t/ha", yaxis_title="Frequency", height=340,
                          margin=dict(l=45, r=10, t=55, b=45))
        _set_x0(fig)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False},
                        key=f"growth_yielddist_{year_sel}")

    # -----------------------------
    # Violin plot: biological stage completion dates (all runs, selected year)
    # y-axis ticks are MONTH ONLY (12 lines)
    # -----------------------------
    st.markdown("### Biological stage completion timings")

    stages = [
        ("Dormancy (chill complete)", "chill_complete_date"),
        ("Budbreak", "budbreak_date"),
        ("Blossom", "blossom_date"),
        ("Fruit set", "fruitset_date"),
        ("Maturity", "harvest_date"),
    ]

    viol = []
    for label, col in stages:
        if col not in dfy.columns:
            continue
        d = pd.to_datetime(dfy[col], errors="coerce").dropna()
        if d.empty:
            continue

        # encode within-month position but label axis monthly
        month = d.dt.month.to_numpy(dtype=float)
        day = d.dt.day.to_numpy(dtype=float)
        y = month + (day - 1.0) / 31.0  # keeps spread; axis labels will be month-only
        y = y[np.isfinite(y)]
        if y.size:
            viol.append((label, y))

    if viol:
        fig_v = go.Figure()
        for label, y in viol:
            fig_v.add_trace(go.Violin(
                x=[label] * len(y),
                y=y,
                name=label,
                box_visible=True,
                meanline_visible=True,
                points=False,
                scalemode="width",
                spanmode="hard",
            ))

        tickvals = list(range(1, 13))
        ticktext = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

        fig_v.update_layout(
            height=420,
            margin=dict(l=60, r=15, t=55, b=55),
            yaxis_title="Month",
            xaxis_title="Stage",
            hovermode="x unified",
            showlegend=False,
        )
        fig_v.update_yaxes(
            tickmode="array",
            tickvals=tickvals,
            ticktext=ticktext,
            range=[0.75, 12.25],
        )
        st.plotly_chart(fig_v, use_container_width=True, config={"displayModeBar": False},
                        key=f"growth_stage_violin_months_{year_sel}")
    else:
        st.info("Stage timing columns missing or empty for the selected year.")

    # -----------------------------
    # 2x2: growth curves across all years (median across runs)
    # fruit mass + fruit number include min/max band + red/green fill
    # -----------------------------
    st.markdown("### Growth curves across years (all MCS runs)")

    # Leaf area (robust column find)
    leaf_col = _find_first_col_like(
        mc,
        candidates=["leaf_area_index", "leaf_area", "lai", "LAI"],
        keywords=["leaf", "lai", "leaf_area"],
    )

    # Tree maturity (robust column find)
    maturity_col = _find_first_col_like(
        mc,
        candidates=["tree_maturity", "maturity_factor", "tree_age_maturity"],
        keywords=["maturity", "tree_maturity", "age_maturity"],
    )

    # Fruit number / mass
    fruit_num_col = "fruit_number_tree" if "fruit_number_tree" in mc.columns else None
    fruit_mass_col = "fruit_mass_kg" if "fruit_mass_kg" in mc.columns else None

    g1, g2 = st.columns(2)
    g3, g4 = st.columns(2)

    with g1:
        if leaf_col is None:
            st.info("Leaf area: not available (no leaf/lai-like column found in mc_yearly).")
        else:
            stats = _year_stats(mc, leaf_col)
            if stats is None:
                st.info("Leaf area: not available (no numeric data).")
            else:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=stats["season_year"],
                    y=stats["vmed"],
                    mode="lines+markers",
                    showlegend=False,
                ))
                fig.add_vline(x=int(year_sel), line_width=2, line_dash="dot")
                fig.update_layout(
                    title=f"Leaf area ({leaf_col})",
                    xaxis_title="Season year",
                    yaxis_title="Leaf area",
                    height=360,
                    margin=dict(l=55, r=15, t=55, b=45),
                    hovermode="x unified",
                )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False},
                                key=f"growth_curve_leaf_{year_sel}")

    with g2:
        if maturity_col is None:
            st.info("Tree maturity: not available (no maturity-like column found in mc_yearly).")
        else:
            stats = _year_stats(mc, maturity_col)
            if stats is None:
                st.info("Tree maturity: not available (no numeric data).")
            else:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=stats["season_year"],
                    y=stats["vmed"],
                    mode="lines+markers",
                    showlegend=False,
                ))
                fig.add_vline(x=int(year_sel), line_width=2, line_dash="dot")
                fig.update_layout(
                    title=f"Tree maturity ({maturity_col})",
                    xaxis_title="Season year",
                    yaxis_title="Maturity",
                    height=360,
                    margin=dict(l=55, r=15, t=55, b=45),
                    hovermode="x unified",
                )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False},
                                key=f"growth_curve_maturity_{year_sel}")

    with g3:
        if fruit_num_col is None:
            st.info("Fruit number: not available (fruit_number_tree missing).")
        else:
            stats = _year_stats(mc, fruit_num_col)
            if stats is None:
                st.info("Fruit number: not available (no numeric data).")
            else:
                fig = _banded_min_med_max_figure(stats, "Fruit number", "fruit/tree")
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False},
                                key=f"growth_curve_fruitnum_band_{year_sel}")

    with g4:
        if fruit_mass_col is None:
            st.info("Fruit mass: not available (fruit_mass_kg missing).")
        else:
            stats = _year_stats(mc, fruit_mass_col)
            if stats is None:
                st.info("Fruit mass: not available (no numeric data).")
            else:
                fig = _banded_min_med_max_figure(stats, "Fruit mass", "kg/fruit")
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False},
                                key=f"growth_curve_fruitmass_band_{year_sel}")

    # -----------------------------
    # Table: in expander (closed by default)
    # -----------------------------
    with st.expander("Biological timings + modifiers", expanded=False):
        tbl = _growth_requested_table(dfy)
        st.dataframe(tbl, use_container_width=True, hide_index=True)


# ============================================================
# Harvest tab (visuals) — FIXED:
#   (1) Window starts at earliest maturity date (no early drift)
#   (2) Trees backfilled flat; other buffers backfilled with zeros
#   (3) Window ends at latest completion (no long tail)
# ============================================================

@st.cache_data(show_spinner=False)
def _compute_harvest_window_and_completion_cached(
    dfy_use: pd.DataFrame,
    cap: int,
    eps: float,
    tail_days: int,
) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp], Optional[pd.Series]]:
    """
    Returns:
      window_start: earliest harvest maturity date in dfy (normalized)
      window_end: latest completion date across runs (normalized) + tail_days
      harvest_dates: pd.Series of per-run harvest_anchor_date (normalized)
    """
    if dfy_use.empty:
        return None, None, None

    dfy = dfy_use.copy()

    # Cap runs for speed/consistency
    if "mc_run" in dfy.columns:
        runs = _cap_runs(
            pd.to_numeric(dfy["mc_run"], errors="coerce").dropna().astype(int).unique(),
            cap=cap,
        ).tolist()
        dfy = dfy[dfy["mc_run"].isin(runs)].copy()

    # Harvest maturity dates (per run)
    if "harvest_anchor_date" not in dfy.columns:
        return None, None, None

    hd = pd.to_datetime(dfy["harvest_anchor_date"], errors="coerce").dropna()
    if hd.empty:
        return None, None, None

    harvest_dates = hd.dt.normalize()
    window_start = harvest_dates.min().normalize()

    # Compute completion date across runs:
    # last date where any buffer is > eps
    # We use date_series + the four flow series if present.
    completion_dates: List[pd.Timestamp] = []

    needed_cols = ["date_series", "trees_series", "field_empty_series", "field_filled_series", "sts_filled_series"]
    has_all = all(c in dfy.columns for c in needed_cols)

    if has_all:
        for _, r in dfy.iterrows():
            ds = r.get("date_series", None)
            if not isinstance(ds, list) or len(ds) == 0:
                continue

            d = pd.to_datetime(pd.Series(ds), errors="coerce").dropna().dt.normalize()
            if d.empty:
                continue
            d_arr = d.to_numpy(dtype="datetime64[ns]")

            def _arr(name: str) -> Optional[np.ndarray]:
                v = r.get(name, None)
                if isinstance(v, list) and len(v) > 0:
                    a = np.asarray(v, dtype=float)
                    return a
                return None

            a_t = _arr("trees_series")
            a_e = _arr("field_empty_series")
            a_f = _arr("field_filled_series")
            a_s = _arr("sts_filled_series")
            if a_t is None or a_e is None or a_f is None or a_s is None:
                continue

            L = min(len(d_arr), a_t.size, a_e.size, a_f.size, a_s.size)
            if L <= 0:
                continue

            # any buffer still active?
            active = (a_t[:L] + a_e[:L] + a_f[:L] + a_s[:L]) > float(eps)
            if not np.any(active):
                # if never active, set completion at harvest date (safe)
                # (but keep it within the plotted window)
                completion_dates.append(pd.to_datetime(harvest_dates.min()).normalize())
                continue

            last_idx = int(np.max(np.where(active)[0]))
            completion_dates.append(pd.to_datetime(d_arr[last_idx]).normalize())

    # Fallback if completion couldn't be computed:
    if completion_dates:
        window_end = max(completion_dates).normalize()
    else:
        # last resort: end at latest harvest date + 60 days
        window_end = (harvest_dates.max().normalize() + pd.Timedelta(days=60)).normalize()

    tail_days = max(0, int(tail_days))
    window_end = (window_end + pd.Timedelta(days=tail_days)).normalize()

    if window_end < window_start:
        window_end = window_start

    return window_start, window_end, harvest_dates


@st.cache_data(show_spinner=False)
def _extract_series_matrix_for_year_cached(
    dfy_use: pd.DataFrame,
    col: str,
    cap: int,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Build a run x time matrix aligned on a shared CALENDAR axis [window_start..window_end].
    Also fixes pre-start backfill:
      - trees_series: backfill with first observed value (flat line)
      - others: backfill with 0.0
    Post-end backfill:
      - forward-fill after last observed point (keeps coherent plateau rather than gaps)
    """
    if dfy_use.empty or col not in dfy_use.columns or "date_series" not in dfy_use.columns:
        return None, None

    dfy = dfy_use.copy()

    # cap runs
    if "mc_run" in dfy.columns:
        runs = _cap_runs(
            pd.to_numeric(dfy["mc_run"], errors="coerce").dropna().astype(int).unique(),
            cap=cap,
        ).tolist()
        dfy = dfy[dfy["mc_run"].isin(runs)].copy()

    window_start = pd.to_datetime(window_start, errors="coerce").normalize()
    window_end = pd.to_datetime(window_end, errors="coerce").normalize()
    if pd.isna(window_start) or pd.isna(window_end):
        return None, None
    if window_end < window_start:
        window_end = window_start

    x_dates = pd.date_range(window_start, window_end, freq="D")
    T = int(len(x_dates))
    if T <= 0:
        return None, None

    # map date -> index for fast alignment
    x_np = x_dates.to_numpy(dtype="datetime64[ns]")
    idx_map = {np.datetime64(d): i for i, d in enumerate(x_np)}

    rows: List[np.ndarray] = []

    for _, r in dfy.iterrows():
        ds = r.get("date_series", None)
        vs = r.get(col, None)

        if not (isinstance(ds, list) and isinstance(vs, list) and len(ds) > 0 and len(vs) > 0):
            continue

        d = pd.to_datetime(pd.Series(ds), errors="coerce").dropna().dt.normalize()
        if d.empty:
            continue

        d_arr = d.to_numpy(dtype="datetime64[ns]")
        v_arr = np.asarray(vs, dtype=float)

        L = min(int(d_arr.size), int(v_arr.size))
        if L <= 0:
            continue

        d_arr = d_arr[:L]
        v_arr = v_arr[:L]

        row = np.full(T, np.nan, dtype=float)

        # write points that fall inside window
        for dd, vv in zip(d_arr, v_arr):
            j = idx_map.get(np.datetime64(dd), None)
            if j is not None:
                row[j] = float(vv)

        if np.all(np.isnan(row)):
            # nothing landed in window
            continue

        # ---------- FIX 2: pre-start backfill ----------
        first = int(np.argmax(~np.isnan(row)))
        if col == "trees_series":
            # trees should exist as a flat line before flow starts
            row[:first] = row[first]
        else:
            # other buffers should be 0 before they start appearing
            row[:first] = 0.0

        # ---------- post-end forward fill ----------
        for k in range(first + 1, T):
            if np.isnan(row[k]):
                row[k] = row[k - 1]

        rows.append(row)

    if not rows:
        return None, None

    M = np.vstack(rows)
    return M, x_np


def _spaghetti_fig_dates_gl(
    M: np.ndarray,
    x_dates: np.ndarray,
    title: str,
    ylab: str,
) -> go.Figure:
    fig = go.Figure()

    for i in range(M.shape[0]):
        fig.add_trace(go.Scattergl(
            x=x_dates,
            y=M[i, :],
            mode="lines",
            opacity=0.18,
            line=dict(width=1),
            showlegend=False,
            hoverinfo="skip",
        ))

    # 7 ticks
    T = int(len(x_dates))
    tickvals = []
    ticktext = []
    if T >= 2:
        raw = np.linspace(0, T - 1, 7)
        tv = np.unique(np.clip(np.rint(raw).astype(int), 0, T - 1))
        tickvals = [pd.to_datetime(x_dates[i]) for i in tv]
        ticktext = [pd.to_datetime(x_dates[i]).strftime("%b-%d") for i in tv]
    elif T == 1:
        tickvals = [pd.to_datetime(x_dates[0])]
        ticktext = [pd.to_datetime(x_dates[0]).strftime("%b-%d")]

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=ylab,
        height=340,
        margin=dict(l=55, r=15, t=55, b=45),
        hovermode=False,
    )
    fig.update_yaxes(rangemode="tozero")

    if tickvals:
        fig.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext)

    return fig


def _robust_harvest_df(sim_results: Dict[str, Any]) -> pd.DataFrame:
    util = _get_df(sim_results, "harvest_util_by_year")
    if not util.empty:
        return util

    maybe = _get_nested(sim_results, ["median_detail", "des_out", "harvest_yearly"])
    if isinstance(maybe, pd.DataFrame) and not maybe.empty:
        return maybe.copy()

    return pd.DataFrame()


def render_harvest_tab(config: Dict[str, Any], sim_results: Dict[str, Any]) -> None:
    st.subheader("Harvest&Grading")

    util = _robust_harvest_df(sim_results)
    if util.empty:
        st.warning(
            "No harvest outputs found. I looked for:\n"
            "- sim_results['harvest_util_by_year']\n"
            "- sim_results['median_detail']['des_out']['harvest_yearly']\n\n"
            "So the harvest module likely isn't returning the utilisation/series fields."
        )
        return

    util = util.copy()
    for c in ["season_year", "mc_run"]:
        if c in util.columns:
            util[c] = pd.to_numeric(util[c], errors="coerce")

    util = util.dropna(subset=[c for c in ["season_year"] if c in util.columns]).copy()
    if "season_year" in util.columns:
        util["season_year"] = util["season_year"].astype(int)
    if "mc_run" in util.columns:
        util["mc_run"] = util["mc_run"].astype(int)

    years = sorted(util["season_year"].unique().tolist()) if "season_year" in util.columns else []
    if not years:
        st.info("No season years in harvest outputs.")
        return

    # --------------------------------------------------------
    # Heatmap (median across runs by year)
    # --------------------------------------------------------
    st.markdown("### Utilisation heatmap (median across runs)")

    stages = [
        ("util_grade", "Graders"),
        ("util_shuttle", "Filled-bin shuttle"),
        ("util_pick", "Harvesters"),
        ("util_empty", "Empty-bin shuttle"),
    ]
    cols = [c for c, _ in stages if c in util.columns]

    custom_scale = [
        [0.00, "#B0B0B0"],
        [0.05, "#B0B0B0"],
        [0.30, "#FFD966"],
        [0.50, "#B7F0A0"],
        [0.90, "#00B050"],
        [1.00, "#D00000"],
    ]

    if cols:
        hm = util.groupby("season_year")[cols].median(numeric_only=True).reset_index().sort_values("season_year")
        z_rows, y_labels = [], []
        for c, label in stages:
            if c in hm.columns:
                z_rows.append(pd.to_numeric(hm[c], errors="coerce").fillna(0.0).to_numpy())
                y_labels.append(label)

        if z_rows:
            Z = np.vstack(z_rows)
            fig_hm = go.Figure(data=go.Heatmap(
                z=Z,
                x=hm["season_year"].to_numpy(),
                y=y_labels,
                zmin=0.0, zmax=1.0,
                colorscale=custom_scale,
                hoverongaps=False,
                colorbar=dict(title="Utilisation"),
            ))
            fig_hm.update_layout(height=360, margin=dict(l=90, r=20, t=45, b=45))
            st.plotly_chart(fig_hm, use_container_width=True, config={"displayModeBar": False},
                            key="harvest_util_heatmap")

    year_sel = st.slider(
        "Select harvest year",
        int(years[0]),
        int(years[-1]),
        int(years[-1]),
        1,
        key="harvest_year_sel",
    )

    dfy = util[util["season_year"] == int(year_sel)].copy()
    if dfy.empty:
        st.info("No harvest rows for selected year.")
        return

    # --------------------------------------------------------
    # Window computation FIXES:
    #   - start = earliest maturity date (NO backfill earlier than that)
    #   - end   = latest completion date (NO long tail)
    # --------------------------------------------------------
    cap = int(config.get("plot_cap_runs", 300))
    eps = float(config.get("plot_completion_eps", 1e-6))
    tail_days = int(config.get("plot_tail_days", 3))  # small tail ok; set 0 if you want exact

    window_start, window_end, harvest_dates = _compute_harvest_window_and_completion_cached(
        dfy, cap=cap, eps=eps, tail_days=tail_days
    )

    if window_start is None or window_end is None:
        st.info("Could not compute harvest plot window (missing harvest_anchor_date or date_series).")
        return

    # --------------------------------------------------------
    # Utilisation distributions (2x2)
    # --------------------------------------------------------
    st.markdown("### Utilisation distributions")

    u_cols = [
        ("util_empty", "Empty-bin shuttle"),
        ("util_pick", "Harvesters"),
        ("util_shuttle", "Filled-bin shuttle"),
        ("util_grade", "Graders"),
    ]
    u1, u2 = st.columns(2)
    u3, u4 = st.columns(2)
    holders = [u1, u2, u3, u4]

    for (c, title), h in zip(u_cols, holders):
        with h:
            if c not in dfy.columns:
                st.info(f"{title}: missing")
                continue
            vals = pd.to_numeric(dfy[c], errors="coerce").dropna().to_numpy(dtype=float)
            xh, yh = _hist(vals, bins=25, range_=(0.0, 1.0))
            fig = go.Figure([go.Bar(x=xh, y=yh)])
            fig.update_layout(
                title=title,
                xaxis_title="Utilisation (0–1)",
                yaxis_title="Frequency",
                height=320,
                margin=dict(l=50, r=15, t=55, b=45),
            )
            fig.update_xaxes(range=[0.0, 1.0])
            fig.update_yaxes(rangemode="tozero")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False},
                            key=f"harvest_utilhist_{c}_{year_sel}")

    # --------------------------------------------------------
    # Bin flow spaghetti — TRUE date axis + correct backfill + correct end
    # --------------------------------------------------------
    st.divider()
    st.markdown("### Bin flow spaghetti")

    M_trees, x_dates = _extract_series_matrix_for_year_cached(
        dfy, "trees_series", cap=cap, window_start=window_start, window_end=window_end
    )
    M_empty, _ = _extract_series_matrix_for_year_cached(
        dfy, "field_empty_series", cap=cap, window_start=window_start, window_end=window_end
    )
    M_filled_field, _ = _extract_series_matrix_for_year_cached(
        dfy, "field_filled_series", cap=cap, window_start=window_start, window_end=window_end
    )
    M_sts, _ = _extract_series_matrix_for_year_cached(
        dfy, "sts_filled_series", cap=cap, window_start=window_start, window_end=window_end
    )

    if x_dates is None:
        st.info("No time axis for spaghetti plots (missing date_series).")
        return

    r1c1, r1c2 = st.columns(2)
    r2c1, r2c2 = st.columns(2)

    with r1c1:
        if M_trees is None:
            st.info("trees_series missing/empty.")
        else:
            fig = _spaghetti_fig_dates_gl(M_trees, x_dates, "Bins on trees (remaining)", "Bins")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False},
                            key=f"harvest_spag_trees_{year_sel}")

    with r1c2:
        if M_empty is None:
            st.info("field_empty_series missing/empty.")
        else:
            fig = _spaghetti_fig_dates_gl(M_empty, x_dates, "Empty bins in field", "Bins")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False},
                            key=f"harvest_spag_empty_{year_sel}")

    with r2c1:
        if M_filled_field is None:
            st.info("field_filled_series missing/empty.")
        else:
            fig = _spaghetti_fig_dates_gl(M_filled_field, x_dates, "Filled bins in field", "Bins")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False},
                            key=f"harvest_spag_field_{year_sel}")

    with r2c2:
        if M_sts is None:
            st.info("sts_filled_series missing/empty.")
        else:
            fig = _spaghetti_fig_dates_gl(M_sts, x_dates, "Filled bins in STS", "Bins")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False},
                            key=f"harvest_spag_sts_{year_sel}")


from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# assumes these exist in your project (same as your current app codebase)
# - _get_df, _get_nested, _to_1d_float_array, _x_axis
# - GRADES, ALL_GRADES, Q_THRESHOLDS, GRADE_COLORS


# ============================================================
# Storage & distribution tab (POST-PROCESSING) — UPDATED DATA FEED ONLY
#
# ✅ NO visuals added/removed:
#   - Year slider: kept
#   - Week slider: kept
#   - Demand cycle plot: kept
#   - Weekly fill-rate plot: kept
#   - Quality histogram + min/max whiskers: kept
#   - Any other existing visuals in your original render_storage_tab should remain
#
# ✅ Data fixes (feeds the SAME visuals):
#   1) "Bins flowing into inventory":
#      If weekly inflow is available from harvest outputs, we compute a corrected
#      start-of-week inventory series so bins don’t all appear at week 0.
#
#      Example behaviour:
#        inflow[0]=100, inflow[1]=100
#        inv_week0 = 0
#        inv_week1 = 100 - orders_week0
#        inv_week2 = 200 - orders_week0 - orders_week1
#
#   2) Min/Median/Max alignment for inventory whiskers across weeks:
#      Min/max arrays are taken from the SAME week index as the median bars.
# ============================================================


# ============================================================
# Storage tab helpers (UNCHANGED)
# ============================================================

def _robust_storage_df(sim_results: Dict[str, Any]) -> pd.DataFrame:
    storage = _get_df(sim_results, "storage_by_year")
    if not storage.empty:
        return storage

    maybe = _get_nested(sim_results, ["median_detail", "des_out", "storage_by_year"])
    if isinstance(maybe, pd.DataFrame) and not maybe.empty:
        return maybe.copy()

    return pd.DataFrame()


def _robust_storage_detail_runs(sim_results: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """
    Returns dict of detail-run storage_by_year DataFrames for envelope plotting.
    Expected (any of these will work):
      - sim_results['storage_detail_runs'] = {'min': df, 'median': df, 'max': df}
      - sim_results['min_detail']['des_out']['storage_by_year'], etc.
      - fallback: only median_detail['des_out']['storage_by_year']
    """
    out: Dict[str, pd.DataFrame] = {}

    bundle = sim_results.get("storage_detail_runs", None)
    if isinstance(bundle, dict):
        for k in ["min", "median", "max"]:
            v = bundle.get(k, None)
            if isinstance(v, pd.DataFrame) and not v.empty:
                out[k] = v.copy()

    for k, path in [
        ("min", ["min_detail", "des_out", "storage_by_year"]),
        ("median", ["median_detail", "des_out", "storage_by_year"]),
        ("max", ["max_detail", "des_out", "storage_by_year"]),
    ]:
        if k not in out:
            v = _get_nested(sim_results, path)
            if isinstance(v, pd.DataFrame) and not v.empty:
                out[k] = v.copy()

    if "median" not in out:
        v = _get_nested(sim_results, ["median_detail", "des_out", "storage_by_year"])
        if isinstance(v, pd.DataFrame) and not v.empty:
            out["median"] = v.copy()

    if not out:
        v = _get_df(sim_results, "storage_by_year")
        if isinstance(v, pd.DataFrame) and not v.empty:
            out["median"] = v.copy()

    for k in list(out.keys()):
        df = out[k]
        if "season_year" in df.columns:
            df = df.copy()
            df["season_year"] = pd.to_numeric(df["season_year"], errors="coerce").astype("Int64")
            out[k] = df.dropna(subset=["season_year"]).copy()

    return out


def _get_year_start_date_for_storage(sim_results: Dict[str, Any], year_sel: int) -> Optional[pd.Timestamp]:
    # 1) growth_df harvest_date
    growth_df = _get_nested(sim_results, ["median_detail", "des_out", "growth_df"])
    if isinstance(growth_df, pd.DataFrame) and not growth_df.empty:
        g = growth_df.copy()
        if "season_year" in g.columns:
            g["season_year"] = pd.to_numeric(g["season_year"], errors="coerce").astype("Int64")
            g = g[g["season_year"] == int(year_sel)]
            if not g.empty and "harvest_date" in g.columns:
                d = pd.to_datetime(g["harvest_date"], errors="coerce").dropna()
                if not d.empty:
                    return pd.to_datetime(d.iloc[0])

    # 2) harvest_yearly harvest_anchor_date
    util = _get_nested(sim_results, ["median_detail", "des_out", "harvest_yearly"])
    if isinstance(util, pd.DataFrame) and not util.empty and "season_year" in util.columns:
        u = util.copy()
        u["season_year"] = pd.to_numeric(u["season_year"], errors="coerce").astype("Int64")
        u = u[u["season_year"] == int(year_sel)]
        if not u.empty and "harvest_anchor_date" in u.columns:
            d = pd.to_datetime(u["harvest_anchor_date"], errors="coerce").dropna()
            if not d.empty:
                ords = d.map(lambda x: x.toordinal()).to_numpy(dtype=float)
                return pd.Timestamp.fromordinal(int(np.nanmedian(ords)))

    return None


def _week_tickvals_ticktext(T: int, start_date: Optional[pd.Timestamp]) -> Tuple[List[int], List[str]]:
    """
    Exactly 7 x-axis segments (same as demand cycle plot):
      - tickvals are week indices
      - ticktext shows:
          week_index
          MMM-DD underneath
    """
    if T <= 0:
        return [], []
    raw = np.linspace(0, T - 1, 7)
    tv = np.unique(np.clip(np.rint(raw).astype(int), 0, T - 1))
    tickvals = tv.tolist()

    if start_date is None or pd.isna(start_date):
        ticktext = [str(v) for v in tickvals]
        return tickvals, ticktext

    sd = pd.to_datetime(start_date)
    ticktext = []
    for v in tickvals:
        dt = (sd + pd.Timedelta(days=int(v) * 7)).strftime("%b-%d")
        ticktext.append(f"{v}<br>{dt}")
    return tickvals, ticktext


# Use the SAME bands your storage module uses (Storage_Distribution_module.py)
_STORAGE_GRADE_Q_BANDS = {
    "Extra": (0.90, 1.00),
    "Class1": (0.70, 0.90),
    "Class2": (0.50, 0.70),
    "Processor": (0.10, 0.50),
    "Waste": (0.00, 0.10),
}


def _storage_band_mask(centers: np.ndarray, grade: str) -> np.ndarray:
    lo, hi = _STORAGE_GRADE_Q_BANDS.get(grade, (0.0, 0.0))
    lo = float(lo)
    hi = float(hi)
    if grade == "Extra":
        return (centers >= lo) & (centers <= hi)
    return (centers >= lo) & (centers < hi)


# ============================================================
# EXISTING VISUAL FUNCTIONS (UNCHANGED)
#   - whisker histogram
#   - weekly fill-rate plot helper
#   - any other plot helpers you already had
# ============================================================

def _colored_gradeband_hist_with_whiskers(
    inv_med: np.ndarray,
    inv_min: Optional[np.ndarray],
    inv_max: Optional[np.ndarray],
    title: str,
    *,
    legend_side: str = "right",
) -> go.Figure:
    """
    Coloured bars (median).
    For each bar (quality bin), draw a white 'whisker' indicator:
      - median is the bar height
      - horizontal white tick at min and max
      - vertical white line connecting min..max
    """
    h_med = np.asarray(inv_med, dtype=float)
    h_med[~np.isfinite(h_med)] = 0.0
    h_med = np.maximum(0.0, h_med)

    q_bins = int(h_med.size)
    edges = np.linspace(0.0, 1.0, q_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    width = float(edges[1] - edges[0]) if q_bins > 0 else 0.02

    thr_p = float(Q_THRESHOLDS["Processor_min"])
    thr_c2 = float(Q_THRESHOLDS["Class2_min"])
    thr_c1 = float(Q_THRESHOLDS["Class1_min"])
    thr_e = float(Q_THRESHOLDS["Extra_min"])

    masks = {
        "Waste": centers < thr_p,
        "Processor": (centers >= thr_p) & (centers < thr_c2),
        "Class2": (centers >= thr_c2) & (centers < thr_c1),
        "Class1": (centers >= thr_c1) & (centers < thr_e),
        "Extra": centers >= thr_e,
    }

    fig = go.Figure()

    for g in ["Waste", "Processor", "Class2", "Class1", "Extra"]:
        m = masks[g]
        if not np.any(m):
            continue
        fig.add_trace(go.Bar(
            x=centers[m],
            y=h_med[m],
            name=g,
            marker=dict(color=GRADE_COLORS.get(g, "#888888")),
            width=width,
            hovertemplate=f"{g}<br>Quality=%{{x:.3f}}<br>Median bins=%{{y:.2f}}<extra></extra>",
        ))

    thr_lines = [thr_p, thr_c2, thr_c1, thr_e]
    ymax = float(np.nanmax(h_med)) if h_med.size else 1.0
    ymax = max(1.0, ymax * 1.15)

    for xthr in thr_lines:
        fig.add_shape(
            type="line",
            x0=xthr, x1=xthr,
            y0=0.0, y1=ymax,
            line=dict(color="white", width=2, dash="dot"),
        )

    if inv_min is not None and inv_max is not None:
        h_min = np.asarray(inv_min, dtype=float)
        h_max = np.asarray(inv_max, dtype=float)
        if h_min.shape == h_med.shape and h_max.shape == h_med.shape:
            h_min = np.maximum(0.0, np.nan_to_num(h_min, nan=0.0, posinf=0.0, neginf=0.0))
            h_max = np.maximum(0.0, np.nan_to_num(h_max, nan=0.0, posinf=0.0, neginf=0.0))
            lo = np.minimum(h_min, h_max)
            hi = np.maximum(h_min, h_max)

            dx = width * 0.35
            for x0, y_lo, y_hi in zip(centers, lo, hi):
                fig.add_shape(
                    type="line",
                    x0=float(x0), x1=float(x0),
                    y0=float(y_lo), y1=float(y_hi),
                    line=dict(color="white", width=2),
                )
                fig.add_shape(
                    type="line",
                    x0=float(x0 - dx), x1=float(x0 + dx),
                    y0=float(y_hi), y1=float(y_hi),
                    line=dict(color="white", width=2),
                )
                fig.add_shape(
                    type="line",
                    x0=float(x0 - dx), x1=float(x0 + dx),
                    y0=float(y_lo), y1=float(y_lo),
                    line=dict(color="white", width=2),
                )

            ymax = max(ymax, float(np.nanmax(hi)) * 1.10 if hi.size else ymax)

    if legend_side == "right":
        leg = dict(x=1.02, xanchor="left", y=1.0, yanchor="top")
    elif legend_side == "left":
        leg = dict(x=0.0, xanchor="left", y=0.98, yanchor="top")
    else:
        leg = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0)

    fig.update_layout(
        title=title,
        barmode="overlay",
        xaxis_title="Quality (0–1)",
        yaxis_title="Bins (median)",
        height=520,
        margin=dict(l=55, r=120 if legend_side == "right" else 20, t=65, b=55),
        hovermode="x unified",
        showlegend=True,
        legend=leg,
    )
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, ymax])
    return fig


def _fillrate_weekly_median_min_max_plot(
    T: int,
    tickvals: List[int],
    ticktext: List[str],
    fr_med: Dict[str, np.ndarray],
    fr_min: Optional[Dict[str, np.ndarray]],
    fr_max: Optional[Dict[str, np.ndarray]],
    *,
    title: str,
    height: int = 420,
) -> go.Figure:
    x = _x_axis(T)
    fig = go.Figure()

    for g in GRADES:
        y_med = np.asarray(fr_med.get(g, np.zeros(T, dtype=float)), dtype=float) * 100.0
        fig.add_trace(go.Scatter(
            x=x, y=y_med,
            mode="lines",
            name=f"{g} (median)",
            line=dict(color=GRADE_COLORS.get(g, None), width=3),
            line_shape="hv",
        ))

        if fr_min is not None and g in fr_min:
            y_min = np.asarray(fr_min[g], dtype=float)
            y_min = (np.clip(y_min, 0.0, 1.0) * 100.0) if y_min.size == T else (np.zeros(T) * 100.0)
            fig.add_trace(go.Scatter(
                x=x, y=y_min,
                mode="lines",
                name=f"{g} (min)",
                line=dict(color=GRADE_COLORS.get(g, None), width=2, dash="dot"),
                line_shape="hv",
            ))

        if fr_max is not None and g in fr_max:
            y_max = np.asarray(fr_max[g], dtype=float)
            y_max = (np.clip(y_max, 0.0, 1.0) * 100.0) if y_max.size == T else (np.zeros(T) * 100.0)
            fig.add_trace(go.Scatter(
                x=x, y=y_max,
                mode="lines",
                name=f"{g} (max)",
                line=dict(color=GRADE_COLORS.get(g, None), width=2, dash="dot"),
                line_shape="hv",
            ))

    fig.update_layout(
        title=title,
        xaxis_title="Week",
        yaxis_title="Fill rate (%)",
        height=int(height),
        hovermode="x unified",
        margin=dict(l=55, r=220, t=55, b=70),
        legend=dict(x=1.02, xanchor="left", y=1.0, yanchor="top"),
    )
    fig.update_yaxes(range=[0, 100])

    if tickvals:
        fig.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext)

    return fig


# ============================================================
# UPDATED DATA FEED HELPERS (new logic, NO new visuals)
# ============================================================

def _row_for_year(df: pd.DataFrame, year_sel: int) -> Optional[Dict[str, Any]]:
    if not isinstance(df, pd.DataFrame) or df.empty or "season_year" not in df.columns:
        return None
    d = df.copy()
    d["season_year"] = pd.to_numeric(d["season_year"], errors="coerce").astype("Int64")
    d = d.dropna(subset=["season_year"]).copy()
    d["season_year"] = d["season_year"].astype(int)
    r = d[d["season_year"] == int(year_sel)]
    if r.empty:
        return None
    return r.iloc[0].to_dict()


def _safe_np_2d(x: Any) -> Optional[np.ndarray]:
    if x is None:
        return None
    try:
        a = np.asarray(x, dtype=float)
    except Exception:
        return None
    if a.ndim != 2 or a.size == 0:
        return None
    a[~np.isfinite(a)] = 0.0
    a = np.maximum(0.0, a)
    return a


def _safe_weekly_dict_to_np(d: Any, weeks: int) -> Optional[Dict[str, np.ndarray]]:
    if not isinstance(d, dict):
        return None
    out: Dict[str, np.ndarray] = {}
    ok = False
    for g in ALL_GRADES:
        a = _to_1d_float_array(d.get(g, []))
        if a is None:
            continue
        arr = np.asarray(a, dtype=float)
        tmp = np.zeros(int(weeks), dtype=float)
        n = min(int(weeks), int(arr.size))
        if n > 0:
            tmp[:n] = np.maximum(0.0, arr[:n])
        out[g] = tmp
        ok = True
    return out if ok else None


def _extract_weekly_inflow_by_grade(sim_results: Dict[str, Any], year_sel: int, weeks: int) -> Optional[Dict[str, np.ndarray]]:
    """
    Pull weekly inflow into cold-store from harvest/grading outputs (best-effort).
    Returns dict: {grade: np.array(weeks)} for saleable GRADES.
    """
    # 1) preferred: harvest_year_out dict stored in median_detail
    hyo = _get_nested(sim_results, ["median_detail", "des_out", "harvest_year_out"])
    if isinstance(hyo, dict):
        cbyw = hyo.get("cold_inflow_by_week", None)
        if isinstance(cbyw, dict):
            out: Dict[str, np.ndarray] = {}
            ok = False
            for g in GRADES:
                a = cbyw.get(g, None)
                if isinstance(a, (list, np.ndarray)):
                    arr = np.asarray(a, dtype=float)
                    tmp = np.zeros(int(weeks), dtype=float)
                    n = min(int(weeks), int(arr.size))
                    if n > 0:
                        tmp[:n] = np.maximum(0.0, arr[:n])
                    out[g] = tmp
                    ok = True
            return out if ok else None

    # 2) harvest_yearly row dict
    hy = _get_nested(sim_results, ["median_detail", "des_out", "harvest_yearly"])
    if isinstance(hy, pd.DataFrame) and (not hy.empty) and ("season_year" in hy.columns):
        u = hy.copy()
        u["season_year"] = pd.to_numeric(u["season_year"], errors="coerce").astype("Int64")
        u = u.dropna(subset=["season_year"]).copy()
        u["season_year"] = u["season_year"].astype(int)
        r = u[u["season_year"] == int(year_sel)]
        if not r.empty:
            row = r.iloc[0].to_dict()

            cbyw = row.get("cold_inflow_by_week", None)
            if isinstance(cbyw, dict):
                out: Dict[str, np.ndarray] = {}
                ok = False
                for g in GRADES:
                    a = cbyw.get(g, None)
                    if isinstance(a, (list, np.ndarray)):
                        arr = np.asarray(a, dtype=float)
                        tmp = np.zeros(int(weeks), dtype=float)
                        n = min(int(weeks), int(arr.size))
                        if n > 0:
                            tmp[:n] = np.maximum(0.0, arr[:n])
                        out[g] = tmp
                        ok = True
                return out if ok else None

            # per-grade list columns
            out: Dict[str, np.ndarray] = {}
            ok = False
            for g in GRADES:
                col = f"cold_inflow_{g}"
                if col in r.columns:
                    a = r.iloc[0][col]
                    if isinstance(a, (list, np.ndarray)):
                        arr = np.asarray(a, dtype=float)
                        tmp = np.zeros(int(weeks), dtype=float)
                        n = min(int(weeks), int(arr.size))
                        if n > 0:
                            tmp[:n] = np.maximum(0.0, arr[:n])
                        out[g] = tmp
                        ok = True
            return out if ok else None

    return None


def _compute_flow_inventory_by_week(
    inflow_by_week: Dict[str, np.ndarray],
    fulfilled_by_week: Optional[Dict[str, np.ndarray]],
    *,
    weeks: int,
) -> Dict[str, np.ndarray]:
    """
    Start-of-week inventory with flow timing (matches your example):

      inv[w] = sum_{k=0..w-1} inflow[k]  -  sum_{k=0..w-1} fulfilled[k]

    so:
      inv[0] = 0
      inv[1] = inflow[0] - fulfilled[0]
      inv[2] = inflow[0]+inflow[1] - fulfilled[0]-fulfilled[1]
    """
    out: Dict[str, np.ndarray] = {g: np.zeros(int(weeks), dtype=float) for g in GRADES}

    ful = fulfilled_by_week or {g: np.zeros(int(weeks), dtype=float) for g in GRADES}

    for g in GRADES:
        infl = np.asarray(inflow_by_week.get(g, np.zeros(int(weeks))), dtype=float)
        if infl.size != int(weeks):
            tmp = np.zeros(int(weeks), dtype=float)
            n = min(int(weeks), int(infl.size))
            if n > 0:
                tmp[:n] = np.maximum(0.0, infl[:n])
            infl = tmp

        f = np.asarray(ful.get(g, np.zeros(int(weeks))), dtype=float)
        if f.size != int(weeks):
            tmp = np.zeros(int(weeks), dtype=float)
            n = min(int(weeks), int(f.size))
            if n > 0:
                tmp[:n] = np.maximum(0.0, f[:n])
            f = tmp

        infl_cum_prev = np.concatenate([[0.0], np.cumsum(np.maximum(0.0, infl))[:-1]])
        ful_cum_prev = np.concatenate([[0.0], np.cumsum(np.maximum(0.0, f))[:-1]])
        inv = np.maximum(0.0, infl_cum_prev - ful_cum_prev)

        out[g] = inv

    return out


def _dict_grades_weeks_from_row(row: Dict[str, Any], key: str, weeks: int, grades: List[str]) -> Optional[Dict[str, np.ndarray]]:
    d = row.get(key, None)
    if not isinstance(d, dict):
        return None
    out: Dict[str, np.ndarray] = {}
    ok = False
    for g in grades:
        a = _to_1d_float_array(d.get(g, []))
        if a is None:
            continue
        arr = np.asarray(a, dtype=float)
        tmp = np.zeros(int(weeks), dtype=float)
        n = min(int(weeks), int(arr.size))
        if n > 0:
            tmp[:n] = np.maximum(0.0, arr[:n])
        out[g] = tmp
        ok = True
    return out if ok else None


def _demand_cycle_plot(
    demand_by_week: Dict[str, np.ndarray],
    *,
    weeks: int,
    tickvals: List[int],
    ticktext: List[str],
    title: str = "Demand cycle",
    height: int = 320,
) -> go.Figure:
    """
    This is the same "demand cycle plot" concept: weekly demand by grade.
    If your original plot looked different, keep your original plot function and
    just feed it the same demand_by_week arrays produced below.
    """
    x = _x_axis(int(weeks))
    fig = go.Figure()
    for g in GRADES:
        y = np.asarray(demand_by_week.get(g, np.zeros(int(weeks))), dtype=float)
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode="lines",
            name=g,
            line=dict(color=GRADE_COLORS.get(g, None), width=3),
            line_shape="hv",
        ))
    fig.update_layout(
        title=title,
        xaxis_title="Week",
        yaxis_title="Demand (bins)",
        height=int(height),
        hovermode="x unified",
        margin=dict(l=55, r=140, t=55, b=70),
        legend=dict(x=1.02, xanchor="left", y=1.0, yanchor="top"),
    )
    if tickvals:
        fig.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext)
    return fig


# ============================================================
# render_storage_tab (KEEP SAME VISUALS; update data feeding)
# ============================================================

def render_storage_tab(config: Dict[str, Any], sim_results: Dict[str, Any]) -> None:
    st.subheader("Storage & distribution")

    storage_df = _robust_storage_df(sim_results)
    if storage_df.empty or "season_year" not in storage_df.columns:
        st.info("Storage results unavailable.")
        return

    years = sorted(pd.to_numeric(storage_df["season_year"], errors="coerce").dropna().astype(int).unique().tolist())
    if not years:
        st.info("Storage results unavailable (no years found).")
        return

    # ✅ (kept) Year slider
    year_sel = int(st.slider("Year", int(years[0]), int(years[-1]), int(years[-1]), 1, key="storage_year_slider"))

    detail = _robust_storage_detail_runs(sim_results)
    row_med = _row_for_year(detail.get("median", pd.DataFrame()), year_sel)
    row_min = _row_for_year(detail.get("min", pd.DataFrame()), year_sel) if "min" in detail else None
    row_max = _row_for_year(detail.get("max", pd.DataFrame()), year_sel) if "max" in detail else None

    if not isinstance(row_med, dict):
        st.info("Storage detail data unavailable for selected year.")
        return

    weeks = int(row_med.get("weeks", 52) or 52)
    weeks = max(1, min(weeks, 200))

    start_date = _get_year_start_date_for_storage(sim_results, year_sel)
    tickvals, ticktext = _week_tickvals_ticktext(weeks, start_date)

    # ✅ (kept) Week slider
    # (Your original UI used this to select which weekly histogram to show.)
    week_sel = int(st.slider("Week", 0, max(0, weeks - 1), 0, 1, key="storage_week_slider"))

    # ----------------------------
    # Pull median demand/fulfilled/fill-rate (unchanged keys)
    # ----------------------------
    demand_med = _dict_grades_weeks_from_row(row_med, "demand_by_week", weeks, GRADES) or {g: np.zeros(weeks) for g in GRADES}
    fulfilled_med = _dict_grades_weeks_from_row(row_med, "fulfilled_by_week", weeks, GRADES) or {g: np.zeros(weeks) for g in GRADES}
    fr_med_raw = _dict_grades_weeks_from_row(row_med, "fill_rate_by_week", weeks, GRADES)

    # If fill_rate_by_week missing, recompute from demand/fulfilled (keeps same visual)
    if fr_med_raw is None:
        fr_med: Dict[str, np.ndarray] = {}
        for g in GRADES:
            d = np.asarray(demand_med.get(g, np.zeros(weeks)), dtype=float)
            f = np.asarray(fulfilled_med.get(g, np.zeros(weeks)), dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                r = np.where(d > 1e-9, f / d, 1.0)
            fr_med[g] = np.clip(r, 0.0, 1.0)
    else:
        fr_med = {g: np.clip(np.asarray(fr_med_raw.get(g, np.zeros(weeks)), dtype=float), 0.0, 1.0) for g in GRADES}

    # ----------------------------
    # (DATA FEED FIX #1) Inventory "flow" series
    # Use inflow_by_week if available; otherwise fall back to whatever storage reported.
    # ----------------------------
    inflow_by_week = _extract_weekly_inflow_by_grade(sim_results, year_sel, weeks)

    inv_med_for_plots: Optional[Dict[str, np.ndarray]] = _dict_grades_weeks_from_row(row_med, "inventory_by_week", weeks, ALL_GRADES)

    if inflow_by_week is not None:
        inv_flow = _compute_flow_inventory_by_week(inflow_by_week, fulfilled_med, weeks=weeks)

        # Replace ONLY the saleable grades with flowed inventory (keeps same plot structure)
        # Waste can still come from storage if present; if not, set 0.
        inv_fixed: Dict[str, np.ndarray] = {}
        for g in ALL_GRADES:
            if g in GRADES:
                inv_fixed[g] = inv_flow[g]
            else:
                if inv_med_for_plots is not None and g in inv_med_for_plots:
                    inv_fixed[g] = np.asarray(inv_med_for_plots[g], dtype=float)
                else:
                    inv_fixed[g] = np.zeros(weeks, dtype=float)
        inv_med_for_plots = inv_fixed


    
    # ----------------------------
    # (kept) Quality hist + whiskers, selected by week slider
    # (DATA FEED FIX #2) ensure min/max are from SAME week_sel as median
    # ----------------------------
    invq_med_2d = _safe_np_2d(row_med.get("inventory_quality_hist_by_week", None))
    invq_min_2d = _safe_np_2d(row_min.get("inventory_quality_hist_by_week", None)) if isinstance(row_min, dict) else None
    invq_max_2d = _safe_np_2d(row_max.get("inventory_quality_hist_by_week", None)) if isinstance(row_max, dict) else None

    if invq_med_2d is None:
        st.info("Inventory quality histogram unavailable.")
        return

    Tmed = int(invq_med_2d.shape[0])
    week_i = int(np.clip(week_sel, 0, max(0, Tmed - 1)))

    med_vec = invq_med_2d[week_i, :]

    # ✅ align min/max to the SAME week index
    min_vec = None
    if invq_min_2d is not None and invq_min_2d.shape[1] == invq_med_2d.shape[1]:
        w = int(np.clip(week_i, 0, int(invq_min_2d.shape[0]) - 1))
        min_vec = invq_min_2d[w, :]

    max_vec = None
    if invq_max_2d is not None and invq_max_2d.shape[1] == invq_med_2d.shape[1]:
        w = int(np.clip(week_i, 0, int(invq_max_2d.shape[0]) - 1))
        max_vec = invq_max_2d[w, :]

    fig_q = _colored_gradeband_hist_with_whiskers(
        inv_med=med_vec,
        inv_min=min_vec,
        inv_max=max_vec,
        title="Inventory quality distribution",
        legend_side="right",
    )
    st.plotly_chart(fig_q, use_container_width=True, config={"displayModeBar": False}, key=f"stor_q_{year_sel}_{week_i}")


# ----------------------------
    # (kept) Weekly fill-rate plot (median/min/max)
    #   Min/max are read from detail runs if present. If present but wrong shape, align + clip.
    # ----------------------------
    fr_min: Optional[Dict[str, np.ndarray]] = None
    fr_max: Optional[Dict[str, np.ndarray]] = None

    if isinstance(row_min, dict):
        fr_min_raw = _dict_grades_weeks_from_row(row_min, "fill_rate_by_week", weeks, GRADES)
        if fr_min_raw is not None:
            fr_min = {g: np.clip(np.asarray(fr_min_raw.get(g, np.zeros(weeks)), dtype=float), 0.0, 1.0) for g in GRADES}

    if isinstance(row_max, dict):
        fr_max_raw = _dict_grades_weeks_from_row(row_max, "fill_rate_by_week", weeks, GRADES)
        if fr_max_raw is not None:
            fr_max = {g: np.clip(np.asarray(fr_max_raw.get(g, np.zeros(weeks)), dtype=float), 0.0, 1.0) for g in GRADES}

    fig_fr = _fillrate_weekly_median_min_max_plot(
        T=weeks,
        tickvals=tickvals,
        ticktext=ticktext,
        fr_med=fr_med,
        fr_min=fr_min,
        fr_max=fr_max,
        title="Weekly fill rate",
        height=420,
    )
    st.plotly_chart(fig_fr, use_container_width=True, config={"displayModeBar": False}, key=f"stor_fr_{year_sel}")



    # ----------------------------
    # (kept) Demand cycle plot (same visual concept)
    # If you had a different demand plot function originally, keep it and just feed demand_med.
    # ----------------------------
    fig_dem = _demand_cycle_plot(demand_med, weeks=weeks, tickvals=tickvals, ticktext=ticktext, title="Demand cycle")
    st.plotly_chart(fig_dem, use_container_width=True, config={"displayModeBar": False}, key=f"stor_dem_{year_sel}")

    # ----------------------------
    # NOTE:
    # Any other existing visuals you had BELOW this point should remain as-is.
    # The only intended change is:
    #   - when plotting inventory_by_week (if you do), use inv_med_for_plots (flow-corrected)
    #     instead of row_med["inventory_by_week"].
    #
    # If your original code already plots inventory_by_week, replace its data source with:
    #   inv_med_for_plots (median)
    #   and similarly min/max if you want envelopes (same pattern).
    # ----------------------------

    


































# ============================================================
# MACRO OVERVIEW TAB (INSIDE Post_processing.py) — UPDATED
#   ✅ Map shows ALL lat/lon points from:
#        - sim_results['county_points']
#        - WeatherTemplates/macro/uk_orchards_600.csv   (if present)
#   ✅ Keep:
#        0) Map
#        1) Yield
#        3) Tonnes
#        4) Correlation plots:
#             - Yield correlation
#             - Tonnes correlation
#        5) Validation statistics table
#   ✅ Remove:
#        2) Area plot
#        4) Age-density plot
#        Debug dropdown tables
#        Validation interpretation table
#   ✅ Style:
#        - DEFRA = orange
#        - Model median = blue
#        - Red band = median down to min
#        - Green band = median up to max
#        - Correlation 1:1 line = white
#   ✅ Validation table:
#        - Only R, R², MAPE (%), Bias (Model - DEFRA)
#        - R/R² colour scale:
#            weak       = red
#            moderate   = yellow
#            strong     = light green
#            very strong= dark green
# ============================================================

import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def _num_series(df: pd.DataFrame, col: str) -> pd.Series:
    if not isinstance(df, pd.DataFrame) or df.empty or col not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _int_series(df: pd.DataFrame, col: str) -> pd.Series:
    if not isinstance(df, pd.DataFrame) or df.empty or col not in df.columns:
        return pd.Series(dtype="Int64")
    s = pd.to_numeric(df[col], errors="coerce")
    return s.astype("Int64")


def _load_macro_csv_points() -> pd.DataFrame:
    """
    Best-effort loader for:
      WeatherTemplates/macro/uk_orchards_600.csv

    Expected columns can be flexible, but we need lat/lon.
    """
    try:
        csv_path = os.path.join("WeatherTemplates", "macro", "uk_orchards_600.csv")
        if not os.path.isfile(csv_path):
            return pd.DataFrame(columns=["lat", "lon"])

        df = pd.read_csv(csv_path)

        lat_col = None
        lon_col = None

        for c in df.columns:
            cl = str(c).strip().lower()
            if cl in {"lat", "latitude"}:
                lat_col = c
            if cl in {"lon", "longitude", "lng"}:
                lon_col = c

        if lat_col is None or lon_col is None:
            return pd.DataFrame(columns=["lat", "lon"])

        out = df[[lat_col, lon_col]].copy()
        out.columns = ["lat", "lon"]
        out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
        out["lon"] = pd.to_numeric(out["lon"], errors="coerce")
        out = out.dropna(subset=["lat", "lon"]).reset_index(drop=True)
        return out

    except Exception:
        return pd.DataFrame(columns=["lat", "lon"])


def _merge_map_points(county_points: pd.DataFrame) -> pd.DataFrame:
    """
    Combine:
      - macro county/template points from sim_results
      - uk_orchards_600.csv points if present
    """
    frames: List[pd.DataFrame] = []

    if isinstance(county_points, pd.DataFrame) and not county_points.empty:
        if ("lat" in county_points.columns) and ("lon" in county_points.columns):
            cp = county_points[["lat", "lon"]].copy()
            cp["lat"] = pd.to_numeric(cp["lat"], errors="coerce")
            cp["lon"] = pd.to_numeric(cp["lon"], errors="coerce")
            cp = cp.dropna(subset=["lat", "lon"])
            if not cp.empty:
                frames.append(cp)

    csv_pts = _load_macro_csv_points()
    if not csv_pts.empty:
        frames.append(csv_pts)

    if not frames:
        return pd.DataFrame(columns=["lat", "lon"])

    out = pd.concat(frames, ignore_index=True)
    out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
    out["lon"] = pd.to_numeric(out["lon"], errors="coerce")
    out = out.dropna(subset=["lat", "lon"]).copy()
    out = out.drop_duplicates(subset=["lat", "lon"]).reset_index(drop=True)
    return out


def _macro_band_figure(
    *,
    years: np.ndarray,
    med: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    title: str,
    y_label: str,
    defra_years: Optional[np.ndarray] = None,
    defra_vals: Optional[np.ndarray] = None,
    defra_name: str = "DEFRA",
    model_name: str = "Model median",
) -> go.Figure:
    """
    Styling requested:
      - model median = blue
      - DEFRA = orange
      - red fill from median down to min
      - green fill from median up to max
    """
    years = np.asarray(years, dtype=float)
    med = np.asarray(med, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)

    fig = go.Figure()

    # Red band: min -> median
    fig.add_trace(
        go.Scatter(
            x=years,
            y=lo,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
            name="Min",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=years,
            y=med,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(255,0,0,0.18)",
            name="Min",
            hovertemplate="Year=%{x}<br>Median=%{y:.3g}<extra></extra>",
        )
    )

    # Green band: median -> max
    fig.add_trace(
        go.Scatter(
            x=years,
            y=hi,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(0,180,0,0.16)",
            name="Max",
            hovertemplate="Year=%{x}<br>Max=%{y:.3g}<extra></extra>",
        )
    )

    # Model median = blue
    fig.add_trace(
        go.Scatter(
            x=years,
            y=med,
            mode="lines+markers",
            name=model_name,
            line=dict(width=3, color="blue"),
            marker=dict(color="blue"),
            hovertemplate="Year=%{x}<br>Model median=%{y:.3g}<extra></extra>",
        )
    )

    # DEFRA = orange
    if defra_years is not None and defra_vals is not None and len(defra_years) > 0:
        fig.add_trace(
            go.Scatter(
                x=np.asarray(defra_years, dtype=float),
                y=np.asarray(defra_vals, dtype=float),
                mode="lines+markers",
                name=defra_name,
                line=dict(width=2.5, dash="dot", color="orange"),
                marker=dict(color="orange"),
                hovertemplate="Year=%{x}<br>DEFRA=%{y:.3g}<extra></extra>",
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Season year",
        yaxis_title=y_label,
        height=360,
        margin=dict(l=55, r=15, t=55, b=45),
        hovermode="x unified",
    )
    fig.update_yaxes(rangemode="tozero")
    return fig


def _correlation_figure(
    *,
    model_vals: np.ndarray,
    observed_vals: np.ndarray,
    years: np.ndarray,
    title: str,
    x_label: str,
    y_label: str,
) -> go.Figure:
    """
    Scatter correlation plot:
      x = DEFRA / observed
      y = model

    Requested changes:
      - 1:1 line white
      - no year text labels on blue dots
      - no legend
      - no R / R² in title
    """
    x = np.asarray(observed_vals, dtype=float)
    y = np.asarray(model_vals, dtype=float)
    yrs = np.asarray(years, dtype=int)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    yrs = yrs[mask]

    fig = go.Figure()

    if x.size == 0 or y.size == 0:
        fig.update_layout(
            title=title,
            xaxis_title=x_label,
            yaxis_title=y_label,
            height=360,
            margin=dict(l=55, r=15, t=55, b=45),
            showlegend=False,
        )
        return fig

    xy_min = float(min(np.min(x), np.min(y)))
    xy_max = float(max(np.max(x), np.max(y)))
    pad = 0.05 * max(1e-9, (xy_max - xy_min))
    line_min = xy_min - pad
    line_max = xy_max + pad

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker=dict(
                size=9,
                color="blue",
                line=dict(width=1, color="black"),
            ),
            showlegend=False,
            hovertemplate=(
                "Year=%{customdata}<br>"
                "DEFRA=%{x:.3g}<br>"
                "Model=%{y:.3g}<extra></extra>"
            ),
            customdata=yrs,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[line_min, line_max],
            y=[line_min, line_max],
            mode="lines",
            line=dict(color="white", dash="dash", width=2),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=360,
        margin=dict(l=55, r=15, t=55, b=45),
        hovermode="closest",
        showlegend=False,
    )

    fig.update_xaxes(range=[line_min, line_max])
    fig.update_yaxes(range=[line_min, line_max])

    return fig


def _safe_metrics(model_vals: np.ndarray, observed_vals: np.ndarray) -> Dict[str, float]:
    """
    Returns:
      r, r2, rmse, mae, mape, bias
    """
    x = np.asarray(observed_vals, dtype=float)
    y = np.asarray(model_vals, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if x.size == 0 or y.size == 0:
        return {
            "r": np.nan,
            "r2": np.nan,
            "rmse": np.nan,
            "mae": np.nan,
            "mape": np.nan,
            "bias": np.nan,
        }

    err = y - x

    if x.size >= 2:
        r = float(np.corrcoef(x, y)[0, 1])
        r2 = float(r * r)
    else:
        r = np.nan
        r2 = np.nan

    rmse = float(np.sqrt(np.mean(np.square(err))))
    mae = float(np.mean(np.abs(err)))

    mask_mape = np.abs(x) > 1e-12
    if np.any(mask_mape):
        mape = float(np.mean(np.abs(err[mask_mape] / x[mask_mape])) * 100.0)
    else:
        mape = np.nan

    bias = float(np.mean(err))

    return {
        "r": r,
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "bias": bias,
    }


def _fmt_metric(v: float, decimals: int = 3, suffix: str = "") -> str:
    try:
        vf = float(v)
    except Exception:
        return "—"
    if not np.isfinite(vf):
        return "—"
    return f"{vf:.{decimals}f}{suffix}"


def _fmt_metric_by_type(metric_name: str, value: float, column_name: str) -> str:
    """
    Different rounding for Yield vs Tonnes.
    """
    if not np.isfinite(value):
        return "—"

    column_name = str(column_name).strip().lower()
    metric_name = str(metric_name).strip().lower()

    if metric_name in {"r", "r²", "r2"}:
        return _fmt_metric(value, 3)

    if metric_name == "mape (%)":
        return _fmt_metric(value, 2, "%")

    if "yield" in column_name:
        if metric_name == "bias (model - defra)":
            return _fmt_metric(value, 2)
        return _fmt_metric(value, 3)

    if "tonnes" in column_name:
        if metric_name == "bias (model - defra)":
            return _fmt_metric(value, 0)
        return _fmt_metric(value, 3)

    return _fmt_metric(value, 3)


def _colour_for_metric(metric_name: str, value: float) -> str:
    """
    Requested colour logic:
      weak       = red
      moderate   = yellow
      strong     = light green
      very strong= dark green

    Applied to R and R² only.
    """
    metric_name = str(metric_name).strip().lower()

    if not np.isfinite(value):
        return "background-color: #3a3a3a; color: white;"

    if metric_name in {"r", "r²", "r2"}:
        v = abs(float(value))

        # very strong: >0.8
        if v > 0.8:
            return "background-color: #006400; color: white;"

        # strong: 0.6–0.8
        if v >= 0.6:
            return "background-color: #90EE90; color: black;"

        # moderate: 0.4–0.6
        if v >= 0.4:
            return "background-color: #FFD966; color: black;"

        # weak: <0.4
        return "background-color: #C00000; color: white;"

    return ""


def _build_validation_display_table(
    yield_stats: Dict[str, float],
    tonnes_stats: Dict[str, float],
) -> pd.DataFrame:
    rows = [
        ("R", yield_stats["r"], tonnes_stats["r"]),
        ("R²", yield_stats["r2"], tonnes_stats["r2"]),
        ("MAPE (%)", yield_stats["mape"], tonnes_stats["mape"]),
        ("Bias (Model to DEFRA)", yield_stats["bias"], tonnes_stats["bias"]),
    ]

    out_rows = []
    for metric, yv, tv in rows:
        out_rows.append(
            {
                "Metric": metric,
                "Yield": _fmt_metric_by_type(metric, yv, "yield"),
                "Tonnes": _fmt_metric_by_type(metric, tv, "tonnes"),
                "_yield_raw": yv,
                "_tonnes_raw": tv,
            }
        )
    return pd.DataFrame(out_rows)


def _style_validation_table(df: pd.DataFrame):
    raw_lookup = {}
    if "_yield_raw" in df.columns and "_tonnes_raw" in df.columns:
        for _, r in df.iterrows():
            raw_lookup[(r["Metric"], "Yield")] = float(r["_yield_raw"]) if pd.notna(r["_yield_raw"]) else np.nan
            raw_lookup[(r["Metric"], "Tonnes")] = float(r["_tonnes_raw"]) if pd.notna(r["_tonnes_raw"]) else np.nan

    display_df = df[[c for c in df.columns if not c.startswith("_")]].copy()

    def style_fn(data: pd.DataFrame) -> pd.DataFrame:
        styles = pd.DataFrame("", index=data.index, columns=data.columns)

        for i, row in data.iterrows():
            metric = row["Metric"]

            if ("Yield" in data.columns) and ((metric, "Yield") in raw_lookup):
                raw_val = raw_lookup[(metric, "Yield")]
                styles.loc[i, "Yield"] = _colour_for_metric(metric, raw_val)

            if ("Tonnes" in data.columns) and ((metric, "Tonnes") in raw_lookup):
                raw_val = raw_lookup[(metric, "Tonnes")]
                styles.loc[i, "Tonnes"] = _colour_for_metric(metric, raw_val)

        styles.loc[:, "Metric"] = "font-weight: bold;"
        return styles

    return display_df.style.apply(style_fn, axis=None)


def render_macro_overview_tab(sim_results: Dict[str, Any]) -> None:
    """
    Macro post-processing renderer (updated).

    Keeps:
      - 0) Map of all locations
      - 1) Yield plot
      - 3) Tonnes plot
      - 4) Correlation plots
      - 5) Validation statistics table

    Removes:
      - Area plot
      - Age-density plot
      - Debug tables
      - Validation interpretation table
    """
    st.subheader("Macro Overview")

    if not isinstance(sim_results, dict):
        st.error("Macro results missing/invalid (sim_results not a dict).")
        return

    county_points = _get_df(sim_results, "county_points")
    macro_yearly = _get_df(sim_results, "macro_yearly")
    defra = _get_df(sim_results, "defra")

    # --------------------------------------------------------
    # 0) Map of all locations
    # --------------------------------------------------------
    st.markdown("### Location Sample Map")

    m = _merge_map_points(county_points)
    if m.empty:
        st.info("No valid macro lat/lon points found from county_points or uk_orchards_600.csv.")
    else:
        st.map(m)

    # --------------------------------------------------------
    # Validate macro_yearly
    # --------------------------------------------------------
    if macro_yearly.empty or "season_year" not in macro_yearly.columns:
        st.error("macro_yearly missing or empty. Macro plots cannot be rendered.")
        return

    macro_yearly = macro_yearly.copy()
    macro_yearly["season_year"] = pd.to_numeric(macro_yearly["season_year"], errors="coerce").astype("Int64")
    macro_yearly = macro_yearly.dropna(subset=["season_year"]).copy()
    macro_yearly["season_year"] = macro_yearly["season_year"].astype(int)
    macro_yearly = macro_yearly.sort_values("season_year").reset_index(drop=True)

    years = macro_yearly["season_year"].to_numpy(dtype=int)

    # DEFRA normalize
    if not defra.empty and "season_year" in defra.columns:
        defra = defra.copy()
        defra["season_year"] = pd.to_numeric(defra["season_year"], errors="coerce").astype("Int64")
        defra = defra.dropna(subset=["season_year"]).copy()
        defra["season_year"] = defra["season_year"].astype(int)
        defra = defra.sort_values("season_year").reset_index(drop=True)
        defra_years = defra["season_year"].to_numpy(dtype=int)
    else:
        defra_years = None



    # --------------------------------------------------------
    # 1) Yield plot
    # --------------------------------------------------------
    st.markdown("### Macro yield (t/ha) — Model vs DEFRA")

    y_med = _num_series(macro_yearly, "yield_median").to_numpy(dtype=float)
    y_lo = _num_series(macro_yearly, "yield_min").to_numpy(dtype=float)
    y_hi = _num_series(macro_yearly, "yield_max").to_numpy(dtype=float)

    defra_y = None
    if defra_years is not None and "defra_yield_t_ha" in defra.columns:
        defra_y = _num_series(defra, "defra_yield_t_ha").to_numpy(dtype=float)

    fig_yield = _macro_band_figure(
        years=years,
        med=y_med,
        lo=y_lo,
        hi=y_hi,
        title="Yield (t/ha): Model vs DEFRA",
        y_label="Yield (t/ha)",
        defra_years=defra_years,
        defra_vals=defra_y,
        defra_name="DEFRA yield",
        model_name="Model median yield",
    )
    st.plotly_chart(
        fig_yield,
        use_container_width=True,
        config={"displayModeBar": False},
        key="macro_yield_plot",
    )

    # --------------------------------------------------------
    # 3) Tonnes plot
    # --------------------------------------------------------
    st.markdown("### Macro production (tonnes) — Model vs DEFRA")

    t_med = _num_series(macro_yearly, "tonnes_median").to_numpy(dtype=float)
    t_lo = _num_series(macro_yearly, "tonnes_min").to_numpy(dtype=float)
    t_hi = _num_series(macro_yearly, "tonnes_max").to_numpy(dtype=float)

    defra_t = None
    if defra_years is not None and "defra_tonnes" in defra.columns:
        defra_t = _num_series(defra, "defra_tonnes").to_numpy(dtype=float)

    fig_tonnes = _macro_band_figure(
        years=years,
        med=t_med,
        lo=t_lo,
        hi=t_hi,
        title="Tonnes: Model vs DEFRA",
        y_label="Tonnes",
        defra_years=defra_years,
        defra_vals=defra_t,
        defra_name="DEFRA tonnes",
        model_name="Model median tonnes",
    )
    st.plotly_chart(
        fig_tonnes,
        use_container_width=True,
        config={"displayModeBar": False},
        key="macro_tonnes_plot",
    )

    # --------------------------------------------------------
    # Correlation plots + stats source table
    # --------------------------------------------------------
    corr_df = macro_yearly[["season_year", "yield_median", "tonnes_median"]].copy()

    if not defra.empty and "season_year" in defra.columns:
        defra_keep = ["season_year"]
        if "defra_yield_t_ha" in defra.columns:
            defra_keep.append("defra_yield_t_ha")
        if "defra_tonnes" in defra.columns:
            defra_keep.append("defra_tonnes")

        corr_df = corr_df.merge(
            defra[defra_keep].copy(),
            on="season_year",
            how="inner",
        )

    # --------------------------------------------------------
    # 4) Correlation plots
    # --------------------------------------------------------
    st.markdown("### Model vs Real-World Correlation")

    c1, c2 = st.columns(2)

    yield_stats = {
        "r": np.nan,
        "r2": np.nan,
        "rmse": np.nan,
        "mae": np.nan,
        "mape": np.nan,
        "bias": np.nan,
    }
    tonnes_stats = {
        "r": np.nan,
        "r2": np.nan,
        "rmse": np.nan,
        "mae": np.nan,
        "mape": np.nan,
        "bias": np.nan,
    }

    with c1:
        if ("yield_median" in corr_df.columns) and ("defra_yield_t_ha" in corr_df.columns) and (not corr_df.empty):
            y_model = pd.to_numeric(corr_df["yield_median"], errors="coerce").to_numpy(dtype=float)
            y_obs = pd.to_numeric(corr_df["defra_yield_t_ha"], errors="coerce").to_numpy(dtype=float)
            y_years = pd.to_numeric(corr_df["season_year"], errors="coerce").to_numpy(dtype=int)

            yield_stats = _safe_metrics(y_model, y_obs)

            fig_corr_y = _correlation_figure(
                model_vals=y_model,
                observed_vals=y_obs,
                years=y_years,
                title="Yield correlation",
                x_label="DEFRA yield (t/ha)",
                y_label="Model median yield (t/ha)",
            )
            st.plotly_chart(
                fig_corr_y,
                use_container_width=True,
                config={"displayModeBar": False},
                key="macro_yield_correlation_plot",
            )
        else:
            st.info("Yield correlation unavailable.")

    with c2:
        if ("tonnes_median" in corr_df.columns) and ("defra_tonnes" in corr_df.columns) and (not corr_df.empty):
            t_model = pd.to_numeric(corr_df["tonnes_median"], errors="coerce").to_numpy(dtype=float)
            t_obs = pd.to_numeric(corr_df["defra_tonnes"], errors="coerce").to_numpy(dtype=float)
            t_years = pd.to_numeric(corr_df["season_year"], errors="coerce").to_numpy(dtype=int)

            tonnes_stats = _safe_metrics(t_model, t_obs)

            fig_corr_t = _correlation_figure(
                model_vals=t_model,
                observed_vals=t_obs,
                years=t_years,
                title="Tonnes correlation",
                x_label="DEFRA tonnes",
                y_label="Model median tonnes",
            )
            st.plotly_chart(
                fig_corr_t,
                use_container_width=True,
                config={"displayModeBar": False},
                key="macro_tonnes_correlation_plot",
            )
        else:
            st.info("Tonnes correlation unavailable.")

    # --------------------------------------------------------
    # 5) Validation statistics table
    # --------------------------------------------------------
    st.markdown("### Validation")

    stats_table = _build_validation_display_table(yield_stats, tonnes_stats)
    st.dataframe(
        _style_validation_table(stats_table),
        use_container_width=True,
        hide_index=True,
    )