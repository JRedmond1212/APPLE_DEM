from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from DES.DES_runner import run_des_one_seed


# ============================================================
# DEFRA SERIES
# ============================================================

DEFRA_YEARS = list(range(1985, 2025))
DEFRA_AREA_HA = [
    12771, 12635, 12888, 12632, 12209, 11787, 11344, 11064, 10699, 10103,
    8849, 8252, 8276, 8059, 7695, 7662, 6630, 5628, 5153, 6127,
    5505, 5203, 4873, 4935, 4965, 5077, 5207, 5307, 5363, 5295,
    5413, 5893, 6002, 6203, 6315, 6372, 6429, 6167, 5934, 5779
]
DEFRA_TONNES = [
    188760,184920,177360,192240,159000,242880,214800,177960,197640,200520,211080,
    166200,126480,115200,117360,160680,121560,125280,100800,82800,130640,
    141600,155160,127440,142080,146040,149880,153480,139440,156960,177480,
    192480,205320,177400,228040,249360,240840,206920,262800,203160

]
DEFRA_YIELD_T_HA = [
    15.146080632,14.479680528,14.037198263999999,14.91620112,12.587080428,19.893521172,18.223466532,15.687588156,17.863340567999998,
    18.741938496,20.89280412,18.781783248,15.327193403999999,13.919768004,14.562600816,20.881091616000003,15.86530932,
    18.895927596,17.910447756,16.06830972,18.057777048,25.722070847999998,29.821256964,26.152267596,28.790273556,
    29.413897283999997,29.521370892,29.47570578,26.274731484,29.267201196,33.518413596,35.558839836,34.84133718,
    32.889036984,39.98710302,39.486935867999996,37.796610168,35.296313579999996,42.613912764,34.236602628

]

# ============================================================
# VISUAL OUTPUT START YEAR
#   The macro model can still run the full internal range,
#   but outputs used by Post_processing will start from here.
# ============================================================

MACRO_VISUAL_START_YEAR = 2005


# ============================================================
# AGE BINS / INITIAL 2025 DISTRIBUTION
# ============================================================

AGE_BINS = ["0-5", "6-10", "11-15", "16-20", "21-30"]
AGE_BIN_EDGES = {
    "0-5": (0, 5),
    "6-10": (6, 10),
    "11-15": (11, 15),
    "16-20": (16, 20),
    "21-30": (21, 30),
}

TOTAL_HA_BY_AGE_RANGE = {
    "0-5": 1065.81,
    "6-10": 959.94,
    "11-15": 929.28,
    "16-20": 595.58,
    "21-30": 472.33,
}

DEFAULT_ANNUAL_DEATH_RATE = 0.035


def _age_range_weights() -> Dict[str, float]:
    total = float(sum(TOTAL_HA_BY_AGE_RANGE.values())) or 1.0
    return {k: float(v) / total for k, v in TOTAL_HA_BY_AGE_RANGE.items()}


def _uniform_age_distribution_for_bin(bin_name: str) -> Dict[int, float]:
    a0, a1 = AGE_BIN_EDGES[bin_name]
    ages = list(range(a0, a1 + 1))
    w = 1.0 / float(len(ages))
    return {a: w for a in ages}


def _initial_age_ha_vector(total_area_ha: float) -> np.ndarray:
    age_ha = np.zeros(31, dtype=float)
    weights = _age_range_weights()

    for bin_name in AGE_BINS:
        bin_area = float(total_area_ha) * float(weights.get(bin_name, 0.0))
        split = _uniform_age_distribution_for_bin(bin_name)
        for age, frac in split.items():
            if 0 <= age <= 30:
                age_ha[age] += float(bin_area) * float(frac)

    return np.maximum(0.0, age_ha)


def age_ranges_from_age_vector(age_area_by_age: np.ndarray) -> Dict[str, float]:
    d = {k: 0.0 for k in AGE_BINS}
    for bin_name, (a0, a1) in AGE_BIN_EDGES.items():
        d[bin_name] = float(np.sum(age_area_by_age[a0:a1 + 1]))
    return d


def evolve_age_distribution_one_year(
    age_ha: np.ndarray,
    *,
    target_total_area: float,
    annual_death_rate: float,
) -> np.ndarray:
    prev = np.maximum(0.0, np.asarray(age_ha, dtype=float))
    if prev.size != 31:
        prev = np.zeros(31, dtype=float)

    aged = np.zeros(31, dtype=float)
    aged[1:30] = prev[0:29]
    aged[30] = float(prev[29] + prev[30])

    death_rate = float(np.clip(annual_death_rate, 0.0, 0.95))
    survived = aged * (1.0 - death_rate)
    survived = np.maximum(0.0, survived)

    tgt = float(max(0.0, target_total_area))
    cur_total = float(np.sum(survived))

    replant_ha = max(0.0, tgt - cur_total)
    survived[0] += replant_ha

    cur_total2 = float(np.sum(survived))
    if cur_total2 > 1e-9 and cur_total2 > tgt:
        survived *= (tgt / cur_total2)

    return np.maximum(0.0, survived)


# ============================================================
# COUNTY AREA BY YEAR FROM TEMPLATE POINTS
# ============================================================

def build_county_area_by_year_from_points(points_df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(points_df, pd.DataFrame) or points_df.empty:
        raise RuntimeError("points_df empty. Macro needs templates in WeatherTemplates/macro/.")

    df = points_df.copy()
    for c in ["county", "lat", "lon"]:
        if c not in df.columns:
            raise RuntimeError(f"points_df missing column: {c}")

    if "ha_2025" not in df.columns:
        df["ha_2025"] = np.nan

    ha = pd.to_numeric(df["ha_2025"], errors="coerce")
    if ha.notna().any() and float(ha.dropna().sum()) > 1e-9:
        s = float(ha.dropna().sum())
        df["share_2025"] = ha.fillna(0.0) / s
    else:
        n = int(len(df))
        df["share_2025"] = 1.0 / max(1, n)

    out = []
    for y, total_area in zip(DEFRA_YEARS, DEFRA_AREA_HA):
        for _, r in df.iterrows():
            out.append({
                "season_year": int(y),
                "county": str(r["county"]),
                "lat": float(r["lat"]),
                "lon": float(r["lon"]),
                "county_area_ha": float(float(r["share_2025"]) * float(total_area)),
            })
    return pd.DataFrame(out)


def build_county_age_density_by_year(
    county_area_by_year: pd.DataFrame,
    *,
    start_year: int,
    end_year: int,
    annual_death_rate: float = DEFAULT_ANNUAL_DEATH_RATE,
) -> pd.DataFrame:
    df = county_area_by_year.copy()
    df["season_year"] = pd.to_numeric(df["season_year"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["season_year"]).copy()
    df["season_year"] = df["season_year"].astype(int)

    df = df[(df["season_year"] >= int(start_year)) & (df["season_year"] <= int(end_year))].copy()
    if df.empty:
        return pd.DataFrame(columns=[
            "season_year", "county", "age", "age_area_ha", "age_frac",
            "annual_death_rate", "dynamic_replant_ha"
        ])

    out = []

    for county, g in df.groupby("county", sort=False):
        g = g.sort_values("season_year").copy()
        years = g["season_year"].astype(int).tolist()
        areas = g["county_area_ha"].astype(float).tolist()

        age_vec = _initial_age_ha_vector(total_area_ha=float(areas[0]))

        for i, y in enumerate(years):
            tgt_area = float(areas[i])

            if i == 0:
                dynamic_replant_ha = 0.0
            else:
                prev = np.maximum(0.0, np.asarray(age_vec, dtype=float))

                aged = np.zeros(31, dtype=float)
                aged[1:30] = prev[0:29]
                aged[30] = float(prev[29] + prev[30])

                death_rate = float(np.clip(annual_death_rate, 0.0, 0.95))
                survived = aged * (1.0 - death_rate)
                survived = np.maximum(0.0, survived)

                dynamic_replant_ha = max(0.0, float(tgt_area - float(np.sum(survived))))
                survived[0] += dynamic_replant_ha

                cur_total2 = float(np.sum(survived))
                if cur_total2 > 1e-9 and cur_total2 > tgt_area:
                    survived *= (float(tgt_area) / cur_total2)

                age_vec = np.maximum(0.0, survived)

            total = float(np.sum(age_vec)) or 1.0
            for age in range(31):
                out.append({
                    "season_year": int(y),
                    "county": str(county),
                    "age": int(age),
                    "age_area_ha": float(age_vec[age]),
                    "age_frac": float(age_vec[age] / total),
                    "annual_death_rate": float(annual_death_rate),
                    "dynamic_replant_ha": float(dynamic_replant_ha),
                })

    return pd.DataFrame(out)


# ============================================================
# GROWTH-ONLY MODE
# ============================================================

def run_growth_only_for_county(
    *,
    micro_like_config: Dict[str, Any],
    weather_cache: Dict[str, Any],
    seed: int,
) -> pd.DataFrame:
    cfg = dict(micro_like_config)
    cfg["detail_run"] = False
    cfg["run_storage"] = False
    cfg["store_storage_quality_hist"] = False
    cfg.setdefault("run_harvest", False)
    cfg.setdefault("run_harvest_and_grading", False)
    cfg.setdefault("run_grading", False)

    des_out = run_des_one_seed(config=cfg, weather_cache=weather_cache, seed=int(seed))

    growth_df = des_out.get("growth_df", pd.DataFrame())
    if not isinstance(growth_df, pd.DataFrame) or growth_df.empty:
        return pd.DataFrame()

    g = growth_df.copy()
    if "season_year" not in g.columns or "yield_t_ha" not in g.columns:
        return pd.DataFrame()

    g["season_year"] = pd.to_numeric(g["season_year"], errors="coerce").astype("Int64")
    g["yield_t_ha"] = pd.to_numeric(g["yield_t_ha"], errors="coerce")
    g = g.dropna(subset=["season_year", "yield_t_ha"]).copy()
    g["season_year"] = g["season_year"].astype(int)

    return g[["season_year", "yield_t_ha"]].copy()


# ============================================================
# MACRO DRIVER
# ============================================================

def run_macro_growth(
    *,
    base_micro_defaults: Dict[str, Any],
    county_area_by_year: pd.DataFrame,
    county_age_by_year: pd.DataFrame,
    weather_cache_by_county: Dict[str, Dict[str, Any]],
    mc_runs: int,
    base_seed: int,
    progress_callback: Optional[Callable[[str, int, int, int, int, str], None]] = None,
) -> Dict[str, Any]:
    area_df = county_area_by_year.copy()
    area_df["season_year"] = pd.to_numeric(area_df["season_year"], errors="coerce").astype("Int64")
    area_df = area_df.dropna(subset=["season_year"]).copy()
    area_df["season_year"] = area_df["season_year"].astype(int)

    counties = sorted(area_df["county"].unique().tolist())
    if not counties:
        raise RuntimeError("No counties found in county_area_by_year.")

    county_points = (
        area_df.sort_values(["county", "season_year"])
        .groupby("county", as_index=False)
        .first()[["county", "lat", "lon"]]
        .copy()
    )

    years = sorted(area_df["season_year"].unique().tolist())
    if not years:
        raise RuntimeError("No years found in county_area_by_year.")

    macro_area = (
        area_df.groupby("season_year")["county_area_ha"]
        .sum()
        .reset_index()
        .rename(columns={"county_area_ha": "macro_area_ha"})
        .sort_values("season_year")
        .reset_index(drop=True)
    )

    total_orchard_runs = max(1, int(mc_runs) * len(counties))
    orchard_run_counter = 0

    mc_rows = []
    for i in range(int(mc_runs)):
        seed = int(base_seed) + i
        tonnes_by_year = {y: 0.0 for y in years}

        for county in counties:
            orchard_run_counter += 1
            if progress_callback is not None:
                progress_callback(
                    "Macro: running orchard simulations",
                    orchard_run_counter,
                    total_orchard_runs,
                    i + 1,
                    int(mc_runs),
                    str(county),
                )

            wc = weather_cache_by_county.get(county)
            if not isinstance(wc, dict):
                raise RuntimeError(f"Missing weather cache for county: {county}")

            cfg = dict(base_micro_defaults)
            cfg["sim_mode"] = "Macro (Multiple Orchards)"
            cfg["county"] = county
            cfg["orchard_area"] = 1.0

            row0 = county_points[county_points["county"] == county].iloc[0]
            cfg["lat"] = float(row0["lat"])
            cfg["lon"] = float(row0["lon"])

            # Start macro orchards already mature
            sim_start_year = int(min(years))
            sim_end_year = int(max(years))
            cfg["planting_year"] = int(sim_start_year - 40)
            cfg["years_to_sim"] = int(sim_end_year - cfg["planting_year"] + 1)

            g = run_growth_only_for_county(
                micro_like_config=cfg,
                weather_cache=wc,
                seed=seed,
            )
            if g.empty:
                continue

            a = area_df[area_df["county"] == county][["season_year", "county_area_ha"]].copy()
            merged = g.merge(a, on="season_year", how="left")
            merged["county_area_ha"] = pd.to_numeric(merged["county_area_ha"], errors="coerce").fillna(0.0)
            merged["tonnes"] = merged["yield_t_ha"] * merged["county_area_ha"]

            for _, rr in merged.iterrows():
                y = int(rr["season_year"])
                tonnes_by_year[y] += float(rr["tonnes"])

        for y in years:
            area_y = float(macro_area.loc[macro_area["season_year"] == y, "macro_area_ha"].iloc[0])
            tonnes_y = float(tonnes_by_year.get(y, 0.0))
            yield_y = float(tonnes_y / area_y) if area_y > 1e-9 else 0.0
            mc_rows.append({
                "mc_run": int(i),
                "season_year": int(y),
                "macro_area_ha": area_y,
                "macro_tonnes": tonnes_y,
                "macro_yield_t_ha": yield_y,
            })

    mc_macro = pd.DataFrame(mc_rows)
    if mc_macro.empty:
        raise RuntimeError("Macro MC produced no rows (check growth_df outputs).")

    agg = (
        mc_macro.groupby("season_year")
        .agg(
            yield_min=("macro_yield_t_ha", "min"),
            yield_median=("macro_yield_t_ha", "median"),
            yield_max=("macro_yield_t_ha", "max"),
            tonnes_min=("macro_tonnes", "min"),
            tonnes_median=("macro_tonnes", "median"),
            tonnes_max=("macro_tonnes", "max"),
            area_median=("macro_area_ha", "median"),
        )
        .reset_index()
        .sort_values("season_year")
        .reset_index(drop=True)
    )

    agg["area_min"] = agg["area_median"]
    agg["area_max"] = agg["area_median"]

    defra = pd.DataFrame({
        "season_year": DEFRA_YEARS,
        "defra_area_ha": DEFRA_AREA_HA,
        "defra_tonnes": DEFRA_TONNES,
        "defra_yield_t_ha": DEFRA_YIELD_T_HA,
    })

    cad = county_age_by_year.copy()
    if cad.empty:
        age_ranges = pd.DataFrame(columns=["season_year"] + AGE_BINS)
    else:
        sums = cad.groupby(["season_year", "age"])["age_area_ha"].sum().reset_index()
        rows = []
        for y, gy in sums.groupby("season_year"):
            vec = np.zeros(31, dtype=float)
            for _, rr in gy.iterrows():
                a = int(rr["age"])
                if 0 <= a <= 30:
                    vec[a] = float(rr["age_area_ha"])
            bins = age_ranges_from_age_vector(vec)
            tot = float(sum(bins.values())) or 1.0
            rows.append({"season_year": int(y), **{k: float(v) / tot for k, v in bins.items()}})
        age_ranges = pd.DataFrame(rows).sort_values("season_year").reset_index(drop=True)

    # ========================================================
    # FILTER OUTPUTS FOR VISUALS ONLY
    #   Keep internal modelling full-range,
    #   but show 2005 onward in macro visuals.
    # ========================================================
    visual_start_year = int(MACRO_VISUAL_START_YEAR)

    mc_macro_vis = mc_macro[mc_macro["season_year"] >= visual_start_year].copy()
    agg_vis = agg[agg["season_year"] >= visual_start_year].copy()
    defra_vis = defra[defra["season_year"] >= visual_start_year].copy()
    age_ranges_vis = age_ranges[age_ranges["season_year"] >= visual_start_year].copy()
    county_area_by_year_vis = county_area_by_year[county_area_by_year["season_year"] >= visual_start_year].copy()
    county_age_by_year_vis = county_age_by_year[county_age_by_year["season_year"] >= visual_start_year].copy()

    return {
        "county_points": county_points,
        "county_area_by_year": county_area_by_year_vis,
        "county_age_by_year": county_age_by_year_vis,
        "mc_macro_yearly": mc_macro_vis,
        "macro_yearly": agg_vis,
        "defra": defra_vis,
        "macro_age_range_density": age_ranges_vis,
        "macro_visual_start_year": visual_start_year,
    }