from __future__ import annotations

from typing import Any, Dict, Callable, Optional, List
import time

import numpy as np
import pandas as pd

from DES.DES_runner import run_des_one_seed


GRADES = ["Extra", "Class1", "Class2", "Processor"]


def _score_run(growth_df: pd.DataFrame) -> float:
    """
    Score a run so we can select a "median" run for the detailed rerun.
    Current choice: mean(yield_t_ha) over simulated years.
    """
    if growth_df is None or growth_df.empty or "yield_t_ha" not in growth_df.columns:
        return float("nan")
    s = pd.to_numeric(growth_df["yield_t_ha"], errors="coerce").dropna()
    return float(s.mean()) if not s.empty else float("nan")


def _safe_arr_1d(x: Any, n: int) -> Optional[np.ndarray]:
    if x is None:
        return None
    try:
        a = np.asarray(x, dtype=np.float32).ravel()
    except Exception:
        return None
    if a.size == 0:
        return None
    out = np.zeros(int(n), dtype=np.float32)
    k = min(int(n), int(a.size))
    if k > 0:
        out[:k] = a[:k]
    out[~np.isfinite(out)] = 0.0
    out = np.maximum(out, 0.0)
    return out


def _safe_arr_2d(x: Any) -> Optional[np.ndarray]:
    if x is None:
        return None
    try:
        a = np.asarray(x, dtype=np.float32)
    except Exception:
        return None
    if a.ndim != 2 or a.size == 0:
        return None
    a = a.copy()
    a[~np.isfinite(a)] = 0.0
    a = np.maximum(a, 0.0)
    return a


def _weekly_dict_from_row(row: Dict[str, Any], key: str, weeks: int, grades: List[str]) -> Optional[Dict[str, np.ndarray]]:
    d = row.get(key, None)
    if not isinstance(d, dict):
        return None
    out: Dict[str, np.ndarray] = {}
    ok = False
    for g in grades:
        arr = _safe_arr_1d(d.get(g, None), weeks)
        if arr is None:
            continue
        out[g] = arr
        ok = True
    return out if ok else None


def _update_storage_collectors(
    collectors: Dict[int, Dict[str, Any]],
    storage_row: Dict[str, Any],
) -> None:
    """
    Collect true storage-stage outputs across ALL MC runs,
    then later aggregate true min/median/max by week/bin.
    """
    try:
        season_year = int(storage_row.get("season_year"))
    except Exception:
        return

    weeks = int(storage_row.get("weeks", 52) or 52)
    weeks = max(1, min(weeks, 200))

    rec = collectors.setdefault(
        season_year,
        {
            "weeks": weeks,
            "inventory_quality_hist_runs": [],
            "fill_rate_runs": {g: [] for g in GRADES},
            "demand_by_week_example": None,
        },
    )

    invq = _safe_arr_2d(storage_row.get("inventory_quality_hist_by_week", None))
    if invq is not None:
        rec["inventory_quality_hist_runs"].append(invq.astype(np.float32, copy=False))

    fr = _weekly_dict_from_row(storage_row, "fill_rate_by_week", weeks, GRADES)
    if fr is not None:
        for g in GRADES:
            if g in fr:
                rec["fill_rate_runs"][g].append(fr[g].astype(np.float32, copy=False))

    # demand is generally the same across runs for a given config/year,
    # so we only need one representative copy for plotting
    if rec["demand_by_week_example"] is None:
        dem = _weekly_dict_from_row(storage_row, "demand_by_week", weeks, GRADES)
        if dem is not None:
            rec["demand_by_week_example"] = {g: dem[g].astype(np.float32, copy=False) for g in GRADES}


def _aggregate_storage_uncertainty_by_year(
    collectors: Dict[int, Dict[str, Any]],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for season_year in sorted(collectors.keys()):
        rec = collectors[season_year]
        weeks = int(rec.get("weeks", 52) or 52)

        row: Dict[str, Any] = {
            "season_year": int(season_year),
            "weeks": int(weeks),
        }

        invq_runs = rec.get("inventory_quality_hist_runs", [])
        if invq_runs:
            stack = np.stack(invq_runs, axis=0).astype(np.float32, copy=False)
            row["inventory_quality_hist_min_by_week"] = np.min(stack, axis=0).astype(np.float32).tolist()
            row["inventory_quality_hist_median_by_week"] = np.median(stack, axis=0).astype(np.float32).tolist()
            row["inventory_quality_hist_max_by_week"] = np.max(stack, axis=0).astype(np.float32).tolist()

        fr_min: Dict[str, List[float]] = {}
        fr_med: Dict[str, List[float]] = {}
        fr_max: Dict[str, List[float]] = {}
        for g in GRADES:
            runs_g = rec.get("fill_rate_runs", {}).get(g, [])
            if runs_g:
                stack_g = np.stack(runs_g, axis=0).astype(np.float32, copy=False)
                fr_min[g] = np.min(stack_g, axis=0).astype(np.float32).tolist()
                fr_med[g] = np.median(stack_g, axis=0).astype(np.float32).tolist()
                fr_max[g] = np.max(stack_g, axis=0).astype(np.float32).tolist()

        if fr_min:
            row["fill_rate_min_by_week"] = fr_min
        if fr_med:
            row["fill_rate_median_by_week"] = fr_med
        if fr_max:
            row["fill_rate_max_by_week"] = fr_max

        dem_example = rec.get("demand_by_week_example", None)
        if isinstance(dem_example, dict):
            row["demand_by_week"] = {
                g: np.asarray(dem_example[g], dtype=np.float32).tolist()
                for g in dem_example.keys()
            }

        rows.append(row)

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def run_monte_carlo(
    config: Dict[str, Any],
    weather_cache: Dict[str, Any],
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> Dict[str, Any]:
    """
    FULL STORAGE-UNCERTAINTY PIPELINE:
      - MC loop: Growth + Harvest + Storage for ALL MC runs
      - Storage uncertainty is aggregated from ACTUAL storage outputs
      - Median run is still re-run once for downstream tabs that expect des_out

    Returns dict with keys expected by app.py + Post_processing:
      - mc_yearly
      - summary_by_year
      - harvest_util_by_year
      - storage_by_year                  <-- all scalar storage rows across MC runs
      - storage_uncertainty_by_year      <-- true storage min/median/max by week/bin
      - median_detail
      - pipeline_timings
      - timing_stats
    """
    if not isinstance(config, dict):
        raise TypeError("run_monte_carlo: config must be a dict.")
    if weather_cache is None:
        weather_cache = {}
    if not isinstance(weather_cache, dict):
        raise TypeError("run_monte_carlo: weather_cache must be a dict.")

    mc_runs = int(config.get("mc_runs", 100))
    base_seed = int(config.get("base_seed", 12345))

    if mc_runs <= 0:
        raise ValueError("run_monte_carlo: mc_runs must be > 0.")

    t0 = time.perf_counter()

    mc_growth_rows: List[pd.DataFrame] = []
    harvest_rows: List[pd.DataFrame] = []
    storage_rows: List[pd.DataFrame] = []

    timings: List[Dict[str, float]] = []
    run_scores: List[float] = []
    seeds: List[int] = []

    growth_total = 0.0
    harvest_total = 0.0
    storage_total = 0.0

    # ---------------------------------------------------------
    # ALL MC runs now include storage + storage detail outputs
    # so storage uncertainty is based on REAL storage runs
    # ---------------------------------------------------------
    cfg_all = dict(config)
    cfg_all["detail_run"] = True
    cfg_all["run_storage"] = True
    cfg_all["store_storage_quality_hist"] = True

    storage_collectors: Dict[int, Dict[str, Any]] = {}

    for i in range(mc_runs):
        if progress_callback:
            progress_callback("Monte Carlo + DES", i + 1, mc_runs)

        seed = base_seed + i
        seeds.append(seed)

        des_out = run_des_one_seed(config=cfg_all, weather_cache=weather_cache, seed=seed)

        t = des_out.get("timings", {}) or {}
        g_s = float(t.get("growth_s", 0.0) or 0.0)
        h_s = float(t.get("harvest_s", 0.0) or 0.0)
        s_s = float(t.get("storage_s", 0.0) or 0.0)

        growth_total += g_s
        harvest_total += h_s
        storage_total += s_s
        timings.append(t)

        # Growth
        growth_df = des_out.get("growth_df", pd.DataFrame())
        if not isinstance(growth_df, pd.DataFrame):
            growth_df = pd.DataFrame()

        run_scores.append(_score_run(growth_df))

        if not growth_df.empty:
            g2 = growth_df.copy()
            g2["mc_run"] = int(i)
            mc_growth_rows.append(g2)

        # Harvest
        harvest_yearly = des_out.get("harvest_yearly", pd.DataFrame())
        if not isinstance(harvest_yearly, pd.DataFrame):
            harvest_yearly = pd.DataFrame()

        if not harvest_yearly.empty:
            h2 = harvest_yearly.copy()
            if "season_year" in h2.columns:
                h2["season_year"] = pd.to_numeric(h2["season_year"], errors="coerce")
            h2["mc_run"] = int(i)
            harvest_rows.append(h2)

        # Storage
        storage_df = des_out.get("storage_by_year", pd.DataFrame())
        if not isinstance(storage_df, pd.DataFrame):
            storage_df = pd.DataFrame()

        if not storage_df.empty:
            s2 = storage_df.copy()
            if "season_year" in s2.columns:
                s2["season_year"] = pd.to_numeric(s2["season_year"], errors="coerce")
            s2["mc_run"] = int(i)
            storage_rows.append(s2)

            for _, rr in s2.iterrows():
                _update_storage_collectors(storage_collectors, rr.to_dict())

    # ---------------------------------------------------------
    # Choose median run (by score)
    # ---------------------------------------------------------
    scores_arr = np.asarray(run_scores, dtype=float)
    finite = np.isfinite(scores_arr)

    if finite.any():
        med = float(np.nanmedian(scores_arr[finite]))
        idx_med = int(np.nanargmin(np.abs(scores_arr - med)))
    else:
        idx_med = int(mc_runs // 2)

    median_seed = int(seeds[idx_med])

    # ---------------------------------------------------------
    # Median rerun retained for downstream tabs expecting des_out
    # ---------------------------------------------------------
    cfg_med = dict(config)
    cfg_med["detail_run"] = True
    cfg_med["run_storage"] = True
    cfg_med["store_storage_quality_hist"] = True

    t_med0 = time.perf_counter()
    des_median = run_des_one_seed(config=cfg_med, weather_cache=weather_cache, seed=median_seed)
    median_detail_s = float(time.perf_counter() - t_med0)

    # ---------------------------------------------------------
    # Post processing outputs
    # ---------------------------------------------------------
    t_post0 = time.perf_counter()

    mc_yearly = pd.concat(mc_growth_rows, ignore_index=True) if mc_growth_rows else pd.DataFrame()
    harvest_util_by_year = pd.concat(harvest_rows, ignore_index=True) if harvest_rows else pd.DataFrame()
    storage_by_year = pd.concat(storage_rows, ignore_index=True) if storage_rows else pd.DataFrame()
    storage_uncertainty_by_year = _aggregate_storage_uncertainty_by_year(storage_collectors)

    summary_by_year = pd.DataFrame()
    if (
        not mc_yearly.empty
        and "season_year" in mc_yearly.columns
        and "yield_t_ha" in mc_yearly.columns
    ):
        g = mc_yearly.groupby("season_year")["yield_t_ha"]
        summary_by_year = g.agg(
            yield_t_ha_min="min",
            yield_t_ha_max="max",
            yield_t_ha_mean="mean",
            yield_t_ha_median="median",
        ).reset_index()
        summary_by_year["season_year"] = pd.to_numeric(summary_by_year["season_year"], errors="coerce").astype(int)
        summary_by_year = summary_by_year.sort_values("season_year").reset_index(drop=True)

    post_s = float(time.perf_counter() - t_post0)

    def _nanmed(vals: List[float]) -> float:
        a = np.asarray(vals, dtype=float)
        return float(np.nanmedian(a)) if a.size else 0.0

    growth_list = [float(t.get("growth_s", np.nan)) for t in timings]
    harvest_list = [float(t.get("harvest_s", np.nan)) for t in timings]
    storage_list = [float(t.get("storage_s", np.nan)) for t in timings]
    des_total_list = [float(t.get("des_total_s", np.nan)) for t in timings]

    t_total = float(time.perf_counter() - t0)

    return {
        "mc_yearly": mc_yearly,
        "summary_by_year": summary_by_year,
        "harvest_util_by_year": harvest_util_by_year,
        "storage_by_year": storage_by_year,
        "storage_uncertainty_by_year": storage_uncertainty_by_year,
        "median_detail": {
            "mc_run_index": int(idx_med),
            "seed": int(median_seed),
            "des_out": des_median,
            "detail_runtime_s": float(median_detail_s),
        },
        "pipeline_timings": {
            "total_s": float(t_total),
            "post_processing_s": float(post_s),
        },
        "timing_stats": {
            "growth_total_s": float(growth_total),
            "harvest_total_s": float(harvest_total),
            "storage_total_s": float(storage_total),
            "growth_median_s": float(_nanmed(growth_list)),
            "harvest_median_s": float(_nanmed(harvest_list)),
            "storage_median_s": float(_nanmed(storage_list)),
            "des_median_s": float(_nanmed(des_total_list)),
        },
    }