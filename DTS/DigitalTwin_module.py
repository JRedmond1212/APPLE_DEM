## Purpose:
##   Digital Twin Simulation (DTS) module for APPLE_DEM.
##   - Runs a Micro-style DES for the CURRENT season only
##   - Uses a hybrid weather timeline:
##       historic (to today) + forecast (UKMO) + climatology (post-forecast)
##   - Reuses the same Growth → Harvest → Storage pipeline
##   - Designed for continuous updating as new weather/observations arrive
##
## Inputs:
##   - config (dict):
##       same as Micro, but years_to_sim is forced to 1 (current season)
##   - weather_cache (dict):
##       built by Pre_processing with Digital Twin mode enabled
##   - seed (int): RNG seed
##
## Outputs:
##   - dt_result (dict):
##       {
##         "season_year": int,
##         "growth_df": pd.DataFrame (1 row),
##         "harvest_logs": dict[str, pd.DataFrame],
##         "distribution": dict[str, Any],
##         "kpis": dict
##       }
##
## Notes / Assumptions:
##   - This is intentionally thin: it delegates almost everything to DES_runner.
##   - The *difference* between Micro and Digital Twin lives in Pre_processing:
##       how weather_cache["weather_array"] is constructed.
##   - You can extend this later with:
##       - data assimilation (overwrite fruit counts / diameters mid-season)
##       - rolling re-runs (e.g. nightly update)
##       - alerting (risk thresholds exceeded)

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from DES.DES_runner import run_des_one_seed


def run_digital_twin(
    config: Dict[str, Any],
    weather_cache: Dict[str, Any],
    seed: int = 12345,
) -> Dict[str, Any]:
    """
    Run a single-season Digital Twin simulation.
    """

    # Enforce single-year simulation
    cfg = dict(config)
    cfg["years_to_sim"] = 1

    res = run_des_one_seed(
        config=cfg,
        weather_cache=weather_cache,
        seed=int(seed),
    )

    growth_df: pd.DataFrame = res["growth_df"]
    if growth_df.empty:
        raise RuntimeError("Digital Twin growth_df is empty.")

    season_year = int(growth_df.iloc[0]["season_year"])

    return {
        "season_year": season_year,
        "growth_df": growth_df,
        "harvest_logs": res["harvest_logs_by_year"].get(season_year, {}),
        "distribution": res["distribution_by_year"].get(season_year, {}),
        "kpis": res["kpis_by_year"].iloc[0].to_dict()
        if not res["kpis_by_year"].empty
        else {},
        "seed": int(seed),
    }
