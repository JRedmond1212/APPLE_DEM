# MCS/MonteCarlo_module.py
from __future__ import annotations

from typing import Any, Dict, Callable, Optional, List
import time

import numpy as np
import pandas as pd

from DES.DES_runner import run_des_one_seed


def _score_run(growth_df: pd.DataFrame) -> float:
    """
    Score a run so we can select a "median" run for the detailed rerun.
    Current choice: mean(yield_t_ha) over simulated years.
    """
    if growth_df is None or growth_df.empty or "yield_t_ha" not in growth_df.columns:
        return float("nan")
    s = pd.to_numeric(growth_df["yield_t_ha"], errors="coerce").dropna()
    return float(s.mean()) if not s.empty else float("nan")


def run_monte_carlo(
    config: Dict[str, Any],
    weather_cache: Dict[str, Any],
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> Dict[str, Any]:
    """
    FAST PIPELINE:
      - MC loop: Growth + Harvest ONLY
      - Choose median run (by mean yield)
      - Re-run median seed in detail mode with storage ON

    Returns dict with keys expected by app.py + Post_processing:
      - mc_yearly
      - summary_by_year
      - harvest_util_by_year
      - storage_by_year
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

    timings: List[Dict[str, float]] = []
    run_scores: List[float] = []
    seeds: List[int] = []

    growth_total = 0.0
    harvest_total = 0.0

    # ---------------------------------------------------------
    # MC loop MUST be "fast": no storage, no detail outputs
    # ---------------------------------------------------------
    cfg_fast = dict(config)
    cfg_fast["detail_run"] = False
    cfg_fast["run_storage"] = False
    cfg_fast["store_storage_quality_hist"] = False

    for i in range(mc_runs):
        if progress_callback:
            # stage label should match what app.py expects
            progress_callback("Monte Carlo + DES", i + 1, mc_runs)

        seed = base_seed + i
        seeds.append(seed)

        des_out = run_des_one_seed(config=cfg_fast, weather_cache=weather_cache, seed=seed)

        t = des_out.get("timings", {}) or {}
        g_s = float(t.get("growth_s", 0.0) or 0.0)
        h_s = float(t.get("harvest_s", 0.0) or 0.0)
        s_s = float(t.get("storage_s", 0.0) or 0.0)

        # Guardrail: storage must not run in MC loop
        if s_s > 1e-6:
            raise RuntimeError(
                f"Storage ran during MC loop (mc_run={i}, storage_s={s_s:.6f}). "
                f"Check run_storage=False is reaching DES_runner and storage module."
            )

        growth_total += g_s
        harvest_total += h_s
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
    # Median rerun: detail + storage ON
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

    storage_by_year = des_median.get("storage_by_year", pd.DataFrame())
    if not isinstance(storage_by_year, pd.DataFrame):
        storage_by_year = pd.DataFrame()
    if not storage_by_year.empty:
        storage_by_year = storage_by_year.copy()
        storage_by_year["mc_run"] = int(idx_med)

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
    des_total_list = [float(t.get("des_total_s", np.nan)) for t in timings]

    t_total = float(time.perf_counter() - t0)

    return {
        "mc_yearly": mc_yearly,
        "summary_by_year": summary_by_year,
        "harvest_util_by_year": harvest_util_by_year,
        "storage_by_year": storage_by_year,
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
            "storage_total_s": 0.0,  # storage not run in MC loop
            "growth_median_s": float(_nanmed(growth_list)),
            "harvest_median_s": float(_nanmed(harvest_list)),
            "storage_median_s": 0.0,
            "des_median_s": float(_nanmed(des_total_list)),
        },
    }