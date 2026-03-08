from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


ROOTSTOCK_TABLE = {
    "M27":   {"k": 0.5, "start_fruiting": 2, "full_bearing": 5,  "leaf_area_guide": 1.5},
    "M9":    {"k": 0.5, "start_fruiting": 2, "full_bearing": 5,  "leaf_area_guide": 3.0},
    "M26":   {"k": 0.5, "start_fruiting": 3, "full_bearing": 7,  "leaf_area_guide": 12.5},
    "MM106": {"k": 0.5, "start_fruiting": 4, "full_bearing": 10, "leaf_area_guide": 7.5},
    "MM111": {"k": 0.8, "start_fruiting": 5, "full_bearing": 12, "leaf_area_guide": 12.0},
    "M25":   {"k": 1.0, "start_fruiting": 5, "full_bearing": 12, "leaf_area_guide": 30.0},
}

VARIETY_TABLE = {
    "Bramley Seedling":    {"D_low": 86.0,  "D_high": 95.0,  "L_opt": 30.0},
    "Discovery":           {"D_low": 44.5,  "D_high": 57.0,  "L_opt": 175.0},
    "Cox's Orange Pippin": {"D_low": 50.0,  "D_high": 57.0,  "L_opt": 150.0},
    "Egremont Russet":     {"D_low": 44.5,  "D_high": 57.0,  "L_opt": 175.0},
    "Elstar":              {"D_low": 50.0,  "D_high": 57.0,  "L_opt": 150.0},

    "Braeburn":            {"D_low": 55.0,  "D_high": 68.0,  "L_opt": 140.0},
    "Gala":                {"D_low": 55.0,  "D_high": 65.0,  "L_opt": 160.0},
    "Jazz":                {"D_low": 55.0,  "D_high": 67.0,  "L_opt": 145.0},
    "Pink Lady":           {"D_low": 60.0,  "D_high": 75.0,  "L_opt": 130.0},
    "Cameo":               {"D_low": 55.0,  "D_high": 68.0,  "L_opt": 145.0},
    "Magic Star":          {"D_low": 55.0,  "D_high": 68.0,  "L_opt": 145.0},
    "Red/Early Windsor":   {"D_low": 50.0,  "D_high": 60.0,  "L_opt": 165.0},
    "Jonagold/Jonagored":  {"D_low": 65.0,  "D_high": 80.0,  "L_opt": 120.0},
    "Junami":              {"D_low": 60.0,  "D_high": 72.0,  "L_opt": 135.0},
    "Kissabel":            {"D_low": 50.0,  "D_high": 62.0,  "L_opt": 155.0},
}


@dataclass(frozen=True)
class SampledThresholds:
    chill_required: float
    forcing_required: float
    gdd1_required: float
    gdd2_required: float
    gdd3_required: float
    gdd_base: float = 3.5


@dataclass(frozen=True)
class OptClimate:
    precip_opt: float
    temp_opt: float
    sun_opt_hours: float


def _u(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(rng.uniform(lo, hi))


def _smoothstep(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


def leaf_area_potential(rootstock: str, age: int) -> float:
    rs = ROOTSTOCK_TABLE[rootstock]
    la_max = float(rs["leaf_area_guide"])
    k = float(rs["k"])
    t = max(0.0, float(age))
    return float(la_max * (1.0 - np.exp(-k * (t / 3.0))))


def maturity_factor(rootstock: str, age: int) -> float:
    rs = ROOTSTOCK_TABLE[rootstock]
    start = float(rs["start_fruiting"])
    full = float(rs["full_bearing"])
    if age <= start:
        return 0.05
    if age >= full:
        return 1.0
    x = (age - start) / max(1e-9, (full - start))
    return float(_smoothstep(x))


def apply_pruning_to_target(la_pot: float, la_target: float) -> Tuple[float, float]:
    # EFFECTIVE canopy is the minimum of potential and pruning target
    la_eff = float(min(float(la_pot), float(la_target)))
    if la_pot <= 1e-9:
        return la_eff, 0.0
    pruning_pct = 100.0 * (1.0 - la_eff / float(la_pot))
    return la_eff, float(pruning_pct)


def _count_prefix(ps: np.ndarray, a: int, b: int) -> int:
    if b <= a:
        return 0
    return int(ps[b - 1] - (ps[a - 1] if a > 0 else 0))


def _idx_clamp(i: int, n: int) -> int:
    return int(max(0, min(n - 1, i)))


def _searchsorted_safe(arr: np.ndarray, x: float) -> Tuple[int, bool]:
    n = arr.shape[0]
    if n == 0:
        return 0, False
    i = int(np.searchsorted(arr, x, side="left"))
    if i >= n:
        return n - 1, False
    return i, True


def _date_from_idx(date_list: np.ndarray, idx: int) -> pd.Timestamp:
    idx = _idx_clamp(idx, len(date_list))
    return pd.to_datetime(date_list[idx])


def frost_multiplier(frost1: int, frost2: int, window_len: int) -> float:
    if window_len <= 0:
        return 1.0
    dmg = (0.3 * frost1 + 1.0 * frost2) / max(1.0, float(window_len))
    return float(np.exp(-3.0 * dmg))


def heat_multiplier(heat1: int, heat2: int, window_len: int) -> float:
    if window_len <= 0:
        return 1.0
    dmg = (0.2 * heat1 + 0.7 * heat2) / max(1.0, float(window_len))
    return float(np.exp(-2.0 * dmg))


def pollination_multiplier(bee_ok_days: int) -> float:
    return float(1.0 - np.exp(-bee_ok_days / 1.5))


def mass_from_diameter(D_mm: float, c: float = 4.27e-7) -> float:
    return float(c * (float(D_mm) ** 3))


def water_multiplier(precip_ratio: float) -> float:
    r = float(np.clip(precip_ratio, 0.0, 10.0))
    if r <= 1.0:
        return float(0.55 + 0.45 * r)
    return float(min(1.10, 1.0 + 0.10 * (r - 1.0)))


def mass_weather_components(precip_ratio: float, temp_ratio: float, sun_ratio: float) -> Tuple[float, float, float, float]:
    pr = float(np.clip(precip_ratio, 0.0, 3.0))
    tr = float(np.clip(temp_ratio, 0.0, 3.0))
    sr = float(np.clip(sun_ratio, 0.0, 3.0))

    precip_mod = float(np.clip(0.70 + 0.30 * pr, 0.40, 1.20))
    temp_mod   = float(np.clip(0.75 + 0.25 * tr, 0.50, 1.15))
    sun_mod    = float(np.clip(0.80 + 0.20 * sr, 0.60, 1.10))

    combined = float(np.clip((precip_mod * temp_mod * sun_mod) ** (1.0 / 3.0), 0.30, 1.30))
    return precip_mod, temp_mod, sun_mod, combined


def sample_thresholds(rng: np.random.Generator, arrays: dict) -> Tuple[SampledThresholds, OptClimate]:
    chill_max = float(arrays["chill"][-1]) if len(arrays.get("chill", [])) else 0.0
    forcing_max = float(arrays["forcing"][-1]) if len(arrays.get("forcing", [])) else 0.0
    gdd_max = float(arrays["gdd_ps"][-1]) if len(arrays.get("gdd_ps", [])) else 0.0

    for _ in range(50):
        chill_req = _u(rng, 800, 1000)
        if chill_req <= max(1.0, chill_max):
            break
    else:
        chill_req = min(chill_max, 1000.0)

    for _ in range(50):
        force_req = _u(rng, 300, 400)
        if force_req <= max(1.0, forcing_max):
            break
    else:
        force_req = min(forcing_max, 400.0)

    gdd1 = _u(rng, 120, 220)
    gdd2 = _u(rng, 600, 800)
    gdd3 = _u(rng, 1000, 1100)

    if gdd_max > 10:
        gdd1 = min(gdd1, 0.35 * gdd_max)
        gdd2 = min(gdd2, 0.25 * gdd_max)
        gdd3 = min(gdd3, 0.95 * gdd_max)

        gdd1 = max(gdd1, 10.0)
        gdd2 = max(gdd2, 10.0)
        gdd3 = max(gdd3, 50.0)

    sampled = SampledThresholds(
        chill_required=float(chill_req),
        forcing_required=float(force_req),
        gdd1_required=float(gdd1),
        gdd2_required=float(gdd2),
        gdd3_required=float(gdd3),
        gdd_base=3.5,
    )

    clim = OptClimate(
        precip_opt=_u(rng, 500, 800),
        temp_opt=_u(rng, 2800, 3600),
        sun_opt_hours=2500.0,
    )
    return sampled, clim


def _yearly_risk_bumps_removed() -> Dict[str, int]:
    return {"frost1": 0, "frost2": 0, "poll_bad": 0, "heat1": 0, "heat2": 0, "water": 0}


def run_growth_years(config: Dict[str, Any], weather_cache: Dict[str, Any], rng: np.random.Generator) -> pd.DataFrame:
    year_arrays = weather_cache.get("year_arrays", {})
    if not year_arrays:
        raise RuntimeError("weather_cache is missing year_arrays. Ensure preprocessing built them.")

    variety = config["variety"]
    rootstock = config["rootstock"]

    planting_year = int(config["planting_year"])
    years_to_sim = int(config["years_to_sim"])
    first_year = planting_year
    last_year = planting_year + years_to_sim - 1

    thinning_pct = float(config.get("thinning_target", 30))
    tree_density = float(config.get("tree_density", 1500))
    orchard_area = float(config.get("orchard_area", 1.0))
    la_target = float(config.get("pruning_target", 20.0))

    var = VARIETY_TABLE[variety]
    D_low = float(var["D_low"])
    D_high = float(var["D_high"])
    L_opt = float(var["L_opt"])

    rows = []

    for season_year in range(first_year, last_year + 1):
        arrays = year_arrays.get(int(season_year))
        if arrays is None:
            rows.append({"season_year": int(season_year), "missing_weather": 1})
            continue

        date_list = arrays.get("date_list", None)
        if date_list is None or len(date_list) == 0:
            rows.append({"season_year": int(season_year), "missing_weather": 1})
            continue

        n = len(date_list)

        is_syn_day = arrays.get("is_synthetic_day", None)
        if is_syn_day is None:
            is_syn_day = np.zeros(n, dtype=np.int32)
        else:
            is_syn_day = np.asarray(is_syn_day, dtype=np.int32)
            if is_syn_day.shape[0] != n:
                is_syn_day = np.zeros(n, dtype=np.int32)

        sampled, clim = sample_thresholds(rng, arrays)

        start_idx = 0

        chill_slice = arrays["chill"][start_idx:]
        i_rel, chill_reached = _searchsorted_safe(chill_slice, sampled.chill_required)
        idx_chill = _idx_clamp(start_idx + i_rel, n)

        forcing_at_chill = float(arrays["forcing"][idx_chill])
        forcing_target = forcing_at_chill + sampled.forcing_required

        forcing_slice = arrays["forcing"][idx_chill:]
        i_rel2, forcing_reached = _searchsorted_safe(forcing_slice, forcing_target)
        idx_budbreak = _idx_clamp(idx_chill + i_rel2, n)

        gdd_ps = arrays["gdd_ps"]

        base_bud = float(gdd_ps[idx_budbreak - 1]) if idx_budbreak > 0 else 0.0
        target_bloss = base_bud + sampled.gdd1_required
        idx_bloss, gdd1_reached = _searchsorted_safe(gdd_ps, target_bloss)

        base_bloss = float(gdd_ps[idx_bloss - 1]) if idx_bloss > 0 else 0.0
        target_set = base_bloss + sampled.gdd2_required
        idx_set, gdd2_reached = _searchsorted_safe(gdd_ps, target_set)

        base_set = float(gdd_ps[idx_set - 1]) if idx_set > 0 else 0.0
        target_harv = base_set + sampled.gdd3_required
        idx_harv, gdd3_reached = _searchsorted_safe(gdd_ps, target_harv)

        idx_bloss = max(int(idx_bloss), int(idx_budbreak))
        idx_set = max(int(idx_set), int(idx_bloss))
        idx_harv = max(int(idx_harv), int(idx_set))

        idx_bloss = _idx_clamp(idx_bloss, n)
        idx_set = _idx_clamp(idx_set, n)
        idx_harv = _idx_clamp(idx_harv, n)

        bumps = _yearly_risk_bumps_removed()

        frost1 = _count_prefix(arrays["frost1_ps"], idx_budbreak, idx_bloss) + bumps["frost1"]
        frost2 = _count_prefix(arrays["frost2_ps"], idx_budbreak, idx_bloss) + bumps["frost2"]
        wlen1 = max(0, idx_bloss - idx_budbreak)
        frost_m = frost_multiplier(frost1, frost2, wlen1)

        bee_ok = _count_prefix(arrays["bee_ok_ps"], idx_bloss, idx_set)
        bee_ok = max(0, int(bee_ok) - int(bumps["poll_bad"]))
        poll_m = pollination_multiplier(bee_ok)

        heat1 = _count_prefix(arrays["heat1_ps"], idx_set, idx_harv + 1) + bumps["heat1"]
        heat2 = _count_prefix(arrays["heat2_ps"], idx_set, idx_harv + 1) + bumps["heat2"]
        wlen3 = max(0, (idx_harv + 1) - idx_set)
        heat_m = heat_multiplier(heat1, heat2, wlen3)

        water_stress_bump = int(bumps["water"])

        P = float(arrays["precip"][idx_harv])
        T = float(arrays["temp"][idx_harv])
        S = float(arrays["sun"][idx_harv])

        precip_ratio = P / max(1e-9, clim.precip_opt)
        temp_ratio = T / max(1e-9, clim.temp_opt)
        sun_ratio = S / max(1e-9, clim.sun_opt_hours)

        M_mass = float((precip_ratio + temp_ratio + sun_ratio) / 3.0)
        mass_precip_mod, mass_temp_mod, mass_sun_mod, M_mass_components = mass_weather_components(
            precip_ratio=precip_ratio,
            temp_ratio=temp_ratio,
            sun_ratio=sun_ratio,
        )

        water_m = water_multiplier(precip_ratio)
        M_risk = float(frost_m * poll_m * heat_m)

        # ----------------------------
        # Leaf area (FIXED + correctly output)
        # ----------------------------
        age = int(season_year - planting_year)
        la_pot = leaf_area_potential(rootstock, age)
        maturity = maturity_factor(rootstock, age)
        la_eff, pruning_pct_equiv = apply_pruning_to_target(la_pot, la_target)

        thin_factor = float(1.0 - thinning_pct / 100.0)

        # Fruit number model (unchanged)
        N_pre_thin = float(L_opt * la_eff * maturity * M_risk)
        N = max(0.0, N_pre_thin * thin_factor)

        fruit_number_modifier_total = float(maturity * frost_m * poll_m * heat_m * water_m * thin_factor)
        fruit_number_potential = float(L_opt * la_eff)

        # ----------------------------
        # Fruit mass vs fruit number (FIXED competition linkage)
        # Key change: size depends on N relative to canopy capacity, not on N/la_eff which cancels.
        # ----------------------------
        N_cap = float(max(1e-9, L_opt * la_eff))          # canopy-supported "capacity"
        load_ratio = float(np.clip(N / N_cap, 0.0, 10.0))  # 1 = around optimal

        # Map load_ratio -> diameter smoothly between [D_high (low load), D_low (high load)]
        # - load_ratio=0  => D_high
        # - load_ratio=1  => halfway-ish
        # - load_ratio>>1 => approaches D_low
        D = float(D_high - (D_high - D_low) * (load_ratio / (load_ratio + 1.0)))

        m_pot = mass_from_diameter(D)

        m_at_Dmax = mass_from_diameter(D_high)
        mass_number_modifier = float(np.clip(m_pot / max(1e-12, m_at_Dmax), 0.0, 2.0))

        m = float(m_pot * M_mass)

        fruit_mass_weather_modifier_total = float(M_mass_components)
        fruit_mass_modifier_total = float(fruit_mass_weather_modifier_total * mass_number_modifier)

        fruit_number_modifier = float(fruit_number_modifier_total)
        fruit_mass_modifier = float(fruit_mass_modifier_total)

        yield_t_ha = (tree_density * N * m) / 1000.0
        yield_tonnes_total = float(yield_t_ha * orchard_area)
        yield_total_kg = float(yield_tonnes_total * 1000.0)

        chill_date = _date_from_idx(date_list, idx_chill)
        budbreak_date = _date_from_idx(date_list, idx_budbreak)
        blossom_date = _date_from_idx(date_list, idx_bloss)
        fruitset_date = _date_from_idx(date_list, idx_set)
        harvest_date = _date_from_idx(date_list, idx_harv)

        harvest_is_synthetic = int(is_syn_day[idx_harv]) if n > 0 else 0
        season_has_synthetic = int(bool(np.any(is_syn_day == 1)))

        rows.append(
            {
                "season_year": int(season_year),
                "missing_weather": 0,

                "chill_required": float(sampled.chill_required),
                "forcing_required": float(sampled.forcing_required),
                "gdd1_required": float(sampled.gdd1_required),
                "gdd2_required": float(sampled.gdd2_required),
                "gdd3_required": float(sampled.gdd3_required),
                "gdd_base": float(sampled.gdd_base),

                "precip_opt": float(clim.precip_opt),
                "temp_opt": float(clim.temp_opt),
                "sun_opt_hours": float(clim.sun_opt_hours),

                "chill_complete_date": chill_date,
                "budbreak_date": budbreak_date,
                "blossom_date": blossom_date,
                "fruitset_date": fruitset_date,
                "harvest_date": harvest_date,

                "chill_reached": int(chill_reached),
                "forcing_reached": int(forcing_reached),
                "gdd1_reached": int(gdd1_reached),
                "gdd2_reached": int(gdd2_reached),
                "gdd3_reached": int(gdd3_reached),

                "frost_multiplier": float(frost_m),
                "pollination_multiplier": float(poll_m),
                "heat_multiplier": float(heat_m),

                "frost_days_mild": int(frost1),
                "frost_days_severe": int(frost2),
                "bee_ok_days": int(bee_ok),
                "heat_days_mild": int(heat1),
                "heat_days_severe": int(heat2),

                "risk_bump_frost_minor": 0,
                "risk_bump_frost_major": 0,
                "risk_bump_pollination": 0,
                "risk_bump_heat_minor": 0,
                "risk_bump_heat_major": 0,
                "risk_bump_water_stress": int(water_stress_bump),

                "precip_ratio": float(precip_ratio),
                "temp_ratio": float(temp_ratio),
                "sun_ratio": float(sun_ratio),

                "mass_weather_multiplier": float(M_mass),

                "mass_weather_precip_modifier": float(mass_precip_mod),
                "mass_weather_temp_modifier": float(mass_temp_mod),
                "mass_weather_sun_modifier": float(mass_sun_mod),
                "fruit_mass_weather_modifier_total": float(fruit_mass_weather_modifier_total),

                "season_has_synthetic_weather": int(season_has_synthetic),
                "harvest_day_is_synthetic": int(harvest_is_synthetic),

                "tree_age_years": int(age),

                # LEAF AREA OUTPUTS (FIX 1)
                "leaf_area_potential": float(la_pot),
                "leaf_area_target_m2": float(la_target),
                "leaf_area_eff_m2": float(la_eff),

                "pruning_pct_equiv": float(pruning_pct_equiv),
                "maturity_factor": float(maturity),

                "fruit_number_potential": float(fruit_number_potential),
                "fruit_number_pre_thin": float(N_pre_thin),
                "fruit_number_tree": float(N),
                "thin_factor": float(thin_factor),

                # Useful debug for the new linkage
                "fruit_number_capacity": float(N_cap),
                "fruit_load_ratio": float(load_ratio),

                "crop_load_fruit_per_m2": float(N / max(1e-9, la_eff)),
                "fruit_diameter_mm": float(D),
                "fruit_mass_kg": float(m),

                "risk_water_modifier": float(water_m),
                "fruit_number_modifier_total": float(fruit_number_modifier_total),

                "mass_number_modifier": float(mass_number_modifier),
                "fruit_mass_modifier_total": float(fruit_mass_modifier_total),

                "fruit_number_modifier": float(fruit_number_modifier),
                "fruit_mass_modifier": float(fruit_mass_modifier),

                "yield_t_ha": float(yield_t_ha),
                "yield_tonnes_total": float(yield_tonnes_total),
                "yield_total_kg": float(yield_total_kg),

                "idx_chill": int(idx_chill),
                "idx_budbreak": int(idx_budbreak),
                "idx_blossom": int(idx_bloss),
                "idx_fruitset": int(idx_set),
                "idx_harvest": int(idx_harv),
            }
        )

    return pd.DataFrame(rows)