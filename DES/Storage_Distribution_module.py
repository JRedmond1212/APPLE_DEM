# ============================================================
# DES/Storage_Distribution_module.py (FULL) — UPDATED:
#   ✅ Weekly inflow supported (inventory "flows" in over weeks)
#      - uses harvest_year_out["cold_inflow_by_week"] if present
#      - otherwise falls back to all-at-week0
#   ✅ Inventory_by_week recorded at START of week
#      -> if 100 bins arrive in week0, inventory shows 0 at week0, 100 at week1
#   ✅ total_waste_bins remains "waste in storage at end of year"
#      - waste_scalar carried in (if any)
#      - + waste-band mass in inv_q at end
#
#   ✅ POLICY CHANGES YOU ASKED FOR (storage stage):
#      1) "FIFO" within-grade-band means *proportional draw across quality bins*
#         (e.g., 20 Extras removed as 10@0.8 + 5@0.9 + 5@1.0 if that’s the mix).
#      2) "HQFO" within-grade-band means *highest-quality-first* within that grade band.
#      3) "FEFO" within-grade-band means *lowest-quality-first* within that grade band.
#
#      Implemented by calling:
#         remove_from_histogram_by_policy(inv_q, mask, amount, policy_storage)
#
#      Grade-bucket order is still handled by grade_priority_order(policy_storage)
#      (Extra-first vs low-grade-first), but within-band pull is now truly different.
# ============================================================

from __future__ import annotations

from typing import Any, Dict, Tuple, Optional
import numpy as np

from ABS.Policy_module import (
    grade_priority_order,
    normalize_policy_name,
    remove_from_histogram_by_policy,  # ✅ uses FIFO/FEFO/HQFO within-band logic
)

GRADES = ["Extra", "Class1", "Class2", "Processor"]
ALL_GRADES = ["Extra", "Class1", "Class2", "Processor", "Waste"]

GRADE_Q_BANDS = {
    "Extra": (0.90, 1.00),
    "Class1": (0.70, 0.90),
    "Class2": (0.50, 0.70),
    "Processor": (0.10, 0.50),
    "Waste": (0.00, 0.10),
}


def _grade_band_mask(centers: np.ndarray, grade: str) -> np.ndarray:
    lo, hi = GRADE_Q_BANDS.get(grade, (0.0, 0.0))
    lo = float(lo)
    hi = float(hi)
    if grade == "Extra":
        return (centers >= lo) & (centers <= hi)
    return (centers >= lo) & (centers < hi)


def _hist_from_samples(samples: np.ndarray, edges: np.ndarray) -> np.ndarray:
    s = np.asarray(samples, dtype=float)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return np.ones(len(edges) - 1, dtype=float)
    s = np.clip(s, 0.0, 1.0)
    counts, _ = np.histogram(s, bins=edges)
    counts = counts.astype(float)
    if float(np.sum(counts)) <= 0.0:
        return np.ones(len(edges) - 1, dtype=float)
    return counts


def _get_decay_strength_bins_per_day(config: Dict[str, Any]) -> float:
    decay = config.get("decay_constants", {})
    if isinstance(decay, dict) and "LongTermStorage" in decay and isinstance(decay["LongTermStorage"], dict):
        d = decay["LongTermStorage"]
        ks = []
        for g in GRADES:
            if g in d:
                try:
                    ks.append(float(d[g]))
                except Exception:
                    pass
        k = float(np.mean(ks)) if ks else float(config.get("cold_decay_k", 0.0025))
    else:
        k = float(config.get("cold_decay_k", 0.0025))

    factor = float(config.get("quality_shift_factor", 3.0))
    return float(max(0.0, k * factor))


def _shift_left_mass(frac: np.ndarray, shift_bins: float) -> np.ndarray:
    frac = np.asarray(frac, dtype=float)
    frac = np.maximum(0.0, frac)
    s = float(np.sum(frac))
    if s <= 0.0:
        return frac
    frac = frac / s

    nb = frac.size
    if nb <= 1 or shift_bins <= 1e-12:
        return frac

    x = np.arange(nb, dtype=float)
    xp = x + shift_bins
    xp0 = np.clip(np.floor(xp).astype(int), 0, nb - 1)
    xp1 = np.clip(xp0 + 1, 0, nb - 1)
    w = xp - xp0
    shifted = (1.0 - w) * frac[xp0] + w * frac[xp1]
    shifted = np.maximum(0.0, shifted)
    ss = float(np.sum(shifted))
    if ss > 0.0:
        shifted = shifted / ss
    return shifted


def _get_weekly_demand_from_cache(
    config: Dict[str, Any],
    season_year: int,
    weather_cache: Optional[Dict[str, Any]],
) -> Tuple[np.ndarray, Dict[str, float]]:
    demand_factor = float(config.get("what_if_demand_factor", 1.0))

    if isinstance(weather_cache, dict):
        dw = weather_cache.get("demand_weekly", None)
        if isinstance(dw, dict):
            wt = dw.get("weekly_total_bins", None)
            mix = dw.get("grade_mix", None)
            if wt is not None and mix is not None:
                total = np.asarray(wt, dtype=float) * float(max(0.0, demand_factor))
                ms = float(sum(float(mix.get(g, 0.0)) for g in GRADES)) or 1.0
                mix2 = {g: float(mix.get(g, 0.0)) / ms for g in GRADES}
                if total.size != 52:
                    out = np.zeros(52, dtype=float)
                    n = min(52, int(total.size))
                    out[:n] = total[:n]
                    total = out
                return total, mix2

    grade_mix = config.get(
        "demand_grade_mix",
        {"Extra": 0.25, "Class1": 0.40, "Class2": 0.25, "Processor": 0.10},
    )
    ms = float(sum(float(grade_mix.get(g, 0.0)) for g in GRADES)) or 1.0
    mix2 = {g: float(grade_mix.get(g, 0.0)) / ms for g in GRADES}
    return np.zeros(52, dtype=float), mix2


def _safe_weekly_inflow(
    harvest_year_out: Dict[str, Any],
    cold_store: Dict[str, Any],
    weeks: int,
) -> Dict[str, np.ndarray]:
    """
    Preferred: harvest_year_out["cold_inflow_by_week"][grade] -> length 52 list
    Fallback: all cold_store grade bins dumped into week0.
    """
    inflow = {g: np.zeros(weeks, dtype=float) for g in GRADES}

    cbyw = harvest_year_out.get("cold_inflow_by_week", None)
    if isinstance(cbyw, dict):
        ok_any = False
        for g in GRADES:
            arr = cbyw.get(g, None)
            if isinstance(arr, (list, np.ndarray)):
                a = np.asarray(arr, dtype=float)
                out = np.zeros(weeks, dtype=float)
                n = min(weeks, int(a.size))
                if n > 0:
                    out[:n] = np.maximum(0.0, a[:n])
                inflow[g] = out
                ok_any = True
        if ok_any:
            return inflow

    # fallback: dump cold_store into week0
    for g in GRADES:
        inflow[g][0] = float(max(0.0, float(cold_store.get(g, 0.0) or 0.0)))
    return inflow


def _band_fraction_from_hist(h: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Turn global hist h into a within-band fraction vector (same length as h),
    nonzero only in mask, sums to 1 within band if possible.
    """
    h = np.asarray(h, dtype=float)
    m = np.asarray(mask, dtype=bool)
    out = np.zeros_like(h, dtype=float)
    if h.size == 0 or not np.any(m):
        return out
    w = np.maximum(0.0, h[m])
    s = float(np.sum(w))
    if s <= 1e-12:
        out[m] = 1.0 / float(np.sum(m))
        return out
    out[m] = w / s
    return out


def run_storage_and_distribution(
    config: Dict[str, Any],
    rng: np.random.Generator,
    season_year: int,
    cold_store: Dict[str, Any],
    harvest_year_out: Dict[str, Any],
    weather_cache: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:

    if not bool(config.get("run_storage", True)):
        policy_storage = normalize_policy_name(config.get("policy_storage", "FEFO"))
        ending_inventory = float(sum(float(cold_store.get(g, 0.0)) for g in GRADES))
        waste_scalar = float(cold_store.get("Waste", 0.0) or 0.0)
        return {
            "season_year": int(season_year),
            "policy_storage": policy_storage,
            "weeks": 52,
            "total_demand_bins": 0.0,
            "total_fulfilled_bins": 0.0,
            "fill_rate_overall": 1.0,
            "total_waste_bins": float(waste_scalar),
            "ending_inventory_bins": ending_inventory,
        }

    detail_run = bool(config.get("detail_run", False))
    store_quality_hist = bool(config.get("store_storage_quality_hist", False)) and detail_run

    weeks = 52
    weekly_total_bins, mix = _get_weekly_demand_from_cache(config, int(season_year), weather_cache)

    storage_factor = float(config.get("what_if_storage_factor", 1.0))
    long_term_capacity = float(config.get("long_term_capacity", 2000))
    effective_capacity = max(0.0, long_term_capacity * storage_factor)

    # carried-in waste (ideally 0 if harvest removes waste before cold)
    waste_scalar = float(cold_store.get("Waste", 0.0) or 0.0)

    # weekly inflow (saleable grades only)
    inflow = _safe_weekly_inflow(harvest_year_out, cold_store, weeks)

    # ✅ policy for storage release (both grade-order + within-band rule)
    policy_storage = normalize_policy_name(config.get("policy_storage", "FEFO"))
    fulfill_order = [g for g in grade_priority_order(policy_storage) if g in GRADES]

    q_bins = int(config.get("q_bins", 40))
    edges = np.linspace(0.0, 1.0, q_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    grade_masks = {g: _grade_band_mask(centers, g) for g in ALL_GRADES}

    # base quality shape from harvest samples (used to "inject" inflow into inv_q)
    base_samples = harvest_year_out.get("graded_quality_samples", [])
    if not isinstance(base_samples, list):
        base_samples = []

    if len(base_samples) > 0:
        h = _hist_from_samples(np.asarray(base_samples, dtype=float), edges)
    else:
        h = np.ones(q_bins, dtype=float)

    hs = float(np.sum(h))
    if hs <= 0.0:
        h = np.ones(q_bins, dtype=float)

    # per-grade injection distributions (only within that grade band)
    inj = {g: _band_fraction_from_hist(h, grade_masks[g]) for g in GRADES}

    # inventory distribution starts EMPTY (flow in over weeks)
    inv_q = np.zeros(q_bins, dtype=float)

    demand_by_week = {g: [0.0] * weeks for g in GRADES} if detail_run else None
    fulfilled_by_week = {g: [0.0] * weeks for g in GRADES} if detail_run else None
    fill_rate_by_week = {g: [1.0] * weeks for g in GRADES} if detail_run else None
    inventory_by_week = {g: [0.0] * weeks for g in ALL_GRADES} if detail_run else None
    inventory_quality_hist_by_week = [] if store_quality_hist else None

    total_demand = 0.0
    total_fulfilled = 0.0

    shift_per_day = _get_decay_strength_bins_per_day(config)
    shift_per_week = float(shift_per_day) * 7.0

    for w in range(weeks):

        # -------------------------------------------------
        # (A) Record inventory at START of week
        # -------------------------------------------------
        if detail_run and inventory_by_week is not None:
            for g in ALL_GRADES:
                if g == "Waste":
                    inventory_by_week[g][w] = float(waste_scalar + np.sum(inv_q[grade_masks["Waste"]]))
                else:
                    inventory_by_week[g][w] = float(np.sum(inv_q[grade_masks[g]]))

        # -------------------------------------------------
        # (B) Fulfill demand from current inventory
        #     ✅ NOW uses within-band policy (FIFO/FEFO/HQFO)
        # -------------------------------------------------
        weekly_total = float(max(0.0, weekly_total_bins[w])) if w < weekly_total_bins.size else 0.0
        dem_req = {g: max(0.0, weekly_total * float(mix[g])) for g in GRADES}
        ful = {g: 0.0 for g in GRADES}

        for g in fulfill_order:
            need = float(dem_req[g])
            if need <= 0.0:
                continue

            # ✅ Key change:
            #    - FIFO: proportional across bins inside grade band
            #    - FEFO: lowest-quality first inside grade band
            #    - HQFO: highest-quality first inside grade band
            took = remove_from_histogram_by_policy(
                inv_q,
                grade_masks[g],
                need,
                policy_storage,
                rng=rng,  # not used by deterministic FIFO, but safe to pass
            )
            ful[g] += float(took)

        for g in GRADES:
            total_demand += float(dem_req[g])
            total_fulfilled += float(ful[g])

            if detail_run and demand_by_week is not None:
                demand_by_week[g][w] = float(dem_req[g])
            if detail_run and fulfilled_by_week is not None:
                fulfilled_by_week[g][w] = float(ful[g])
            if detail_run and fill_rate_by_week is not None:
                fill_rate_by_week[g][w] = float(ful[g] / dem_req[g]) if dem_req[g] > 1e-9 else 1.0

        # -------------------------------------------------
        # (C) Apply decay shift AFTER demand (weekly)
        # -------------------------------------------------
        if shift_per_week > 0.0 and w > 0:
            inv_total = float(np.sum(inv_q))
            if inv_total > 1e-12:
                frac = inv_q / inv_total
                inv_q = _shift_left_mass(frac, shift_bins=shift_per_week) * inv_total

        # -------------------------------------------------
        # (D) Add arrivals for THIS week to inventory
        #     (they become visible next week because we recorded start-of-week)
        # -------------------------------------------------
        for g in GRADES:
            a = float(inflow[g][w]) if (g in inflow and w < inflow[g].size) else 0.0
            if a <= 0.0:
                continue
            inv_q += inj[g] * a

        # -------------------------------------------------
        # Capacity check (saleable only, overflow -> waste_scalar)
        #    Overflow removal is NOT your "order release"; it's a forced disposal.
        #    Keep deterministic low-quality-first behavior for overflow.
        # -------------------------------------------------
        saleable_now = float(
            np.sum(inv_q[grade_masks["Extra"]])
            + np.sum(inv_q[grade_masks["Class1"]])
            + np.sum(inv_q[grade_masks["Class2"]])
            + np.sum(inv_q[grade_masks["Processor"]])
        )
        overflow = max(0.0, saleable_now - float(effective_capacity))
        if overflow > 1e-9:
            # remove overflow from lowest quality first (Processor then upward)
            for g in ["Processor", "Class2", "Class1", "Extra"]:
                if overflow <= 0.0:
                    break
                # overflow should be disposed from low quality first, not “policy”
                # so we force FEFO-like within band by passing "FEFO".
                took = remove_from_histogram_by_policy(
                    inv_q, grade_masks[g], overflow, "FEFO", rng=rng
                )
                overflow -= float(took)
                waste_scalar += float(took)

        if store_quality_hist and inventory_quality_hist_by_week is not None:
            inventory_quality_hist_by_week.append(inv_q.astype(np.float32).tolist())

    ending_inventory = float(
        np.sum(inv_q[grade_masks["Extra"]])
        + np.sum(inv_q[grade_masks["Class1"]])
        + np.sum(inv_q[grade_masks["Class2"]])
        + np.sum(inv_q[grade_masks["Processor"]])
    )

    # end-of-year waste in storage = carried waste + waste-band mass
    total_waste = float(waste_scalar + np.sum(inv_q[grade_masks["Waste"]]))

    out: Dict[str, Any] = {
        "season_year": int(season_year),
        "policy_storage": policy_storage,
        "weeks": int(weeks),
        "total_demand_bins": float(total_demand),
        "total_fulfilled_bins": float(total_fulfilled),
        "fill_rate_overall": float(total_fulfilled / total_demand) if total_demand > 1e-9 else 1.0,
        "total_waste_bins": float(total_waste),
        "ending_inventory_bins": float(ending_inventory),
    }

    if detail_run:
        out.update(
            {
                "inventory_by_week": inventory_by_week,
                "demand_by_week": demand_by_week,
                "fulfilled_by_week": fulfilled_by_week,
                "fill_rate_by_week": fill_rate_by_week,
            }
        )
        if store_quality_hist:
            out["inventory_quality_hist_by_week"] = inventory_quality_hist_by_week

    return out