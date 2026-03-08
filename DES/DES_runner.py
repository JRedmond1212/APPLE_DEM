# DES/DES_runner.py
from __future__ import annotations

from typing import Any, Dict, List, Tuple
import time

import numpy as np
import pandas as pd

from DES.Growth_module import run_growth_years
from DES.Harvest_Grading_module import run_harvest_and_grading
from DES.Storage_Distribution_module import run_storage_and_distribution


def _as_dict_row(x: Any) -> Dict[str, Any]:
    if isinstance(x, dict):
        return dict(x)
    if isinstance(x, pd.Series):
        return x.to_dict()
    return {}


def _normalize_harvest_return(
    harvest_ret: Any,
    season_year: int,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Returns: (harvest_yearly_row_dict, cold_store_dict, harvest_year_out_dict)

    Supports:
      - (cold_store_dict, year_out_dict)  <-- common: Harvest module returns tuple
      - dict payloads with keys
      - dataframe/series (best-effort)
    """
    y = int(season_year)

    # Most common: (cold, out)
    if isinstance(harvest_ret, tuple) and len(harvest_ret) == 2:
        cold, out = harvest_ret
        cold = cold if isinstance(cold, dict) else {}
        out = out if isinstance(out, dict) else {}

        row = dict(out)
        row["season_year"] = y
        if "harvest_anchor_date" in row:
            try:
                row["harvest_anchor_date"] = pd.to_datetime(row["harvest_anchor_date"], errors="coerce")
            except Exception:
                pass

        return row, cold, out

    # dict wrapper payload
    if isinstance(harvest_ret, dict):
        hy = harvest_ret.get("harvest_yearly", None)
        if isinstance(hy, pd.DataFrame) and not hy.empty:
            row = hy.iloc[0].to_dict()
        elif isinstance(hy, pd.Series):
            row = hy.to_dict()
        elif isinstance(hy, dict):
            row = dict(hy)
        else:
            yo = harvest_ret.get("harvest_year_out", None) or harvest_ret.get("year_out", None)
            row = dict(yo) if isinstance(yo, dict) else {}

        row["season_year"] = y

        cold = harvest_ret.get("cold_store", None) or harvest_ret.get("cold_store_out", None)
        cold = cold if isinstance(cold, dict) else {}

        yo = harvest_ret.get("harvest_year_out", None) or harvest_ret.get("year_out", None)
        yo = yo if isinstance(yo, dict) else {}

        return row, cold, yo

    # Series as row
    if isinstance(harvest_ret, pd.Series):
        row = harvest_ret.to_dict()
        row["season_year"] = y
        return row, {}, {}

    # DataFrame as row
    if isinstance(harvest_ret, pd.DataFrame) and not harvest_ret.empty:
        row = harvest_ret.iloc[0].to_dict()
        row["season_year"] = y
        return row, {}, {}

    return {"season_year": y}, {}, {}


def run_des_one_seed(
    config: Dict[str, Any],
    weather_cache: Dict[str, Any],
    seed: int,
) -> Dict[str, Any]:
    """
    Runs one DES seed.

    Controls:
      - config["run_storage"]=False => skip storage (used inside MC loop / macro)
      - config["detail_run"]=True   => storage emits weekly curves + quality hist

      - config["run_harvest"]=False OR config["run_harvest_and_grading"]=False => skip harvest entirely
        (growth-only mode, for macro)
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(int(seed))

    # ----------------------------
    # 1) Growth (ALL years)
    # ----------------------------
    t_g0 = time.perf_counter()
    growth_df = run_growth_years(config=config, weather_cache=weather_cache, rng=rng)
    growth_s = float(time.perf_counter() - t_g0)

    if not isinstance(growth_df, pd.DataFrame):
        growth_df = pd.DataFrame()

    if growth_df.empty:
        des_total_s = float(time.perf_counter() - t0)
        return {
            "growth_df": growth_df,
            "harvest_yearly": pd.DataFrame(),
            "storage_by_year": pd.DataFrame(),
            "timings": {
                "growth_s": growth_s,
                "harvest_s": 0.0,
                "storage_s": 0.0,
                "des_total_s": des_total_s,
            },
            "growth_out": {"growth_df": growth_df},
            "harvest_out": {
                "harvest_yearly": pd.DataFrame(),
                "cold_store_by_year": {},
                "harvest_year_out_by_year": {},
            },
        }

    if "season_year" in growth_df.columns:
        growth_df = growth_df.sort_values("season_year").reset_index(drop=True)

    # ----------------------------
    # OPTIONAL: Growth-only (skip harvest/storage)
    # ----------------------------
    run_harvest = bool(config.get("run_harvest", True))
    run_harvest_and_grading_flag = bool(config.get("run_harvest_and_grading", True))

    if (not run_harvest) or (not run_harvest_and_grading_flag):
        des_total_s = float(time.perf_counter() - t0)
        return {
            "growth_df": growth_df,
            "harvest_yearly": pd.DataFrame(),
            "storage_by_year": pd.DataFrame(),
            "timings": {
                "growth_s": growth_s,
                "harvest_s": 0.0,
                "storage_s": 0.0,
                "des_total_s": des_total_s,
            },
            "growth_out": {"growth_df": growth_df},
            "harvest_out": {
                "harvest_yearly": pd.DataFrame(),
                "cold_store_by_year": {},
                "harvest_year_out_by_year": {},
            },
        }

    # ----------------------------
    # 2) Harvest & Grading (per year)
    # ----------------------------
    t_h0 = time.perf_counter()

    harvest_yearly_rows: List[Dict[str, Any]] = []
    cold_store_by_year: Dict[int, Dict[str, Any]] = {}
    harvest_year_out_by_year: Dict[int, Dict[str, Any]] = {}

    for _, gr in growth_df.iterrows():
        g_row = _as_dict_row(gr)
        y = int(g_row.get("season_year", -1))
        if y < 0:
            continue

        ret = run_harvest_and_grading(config=config, rng=rng, growth_row=g_row)

        hy_row, cold, year_out = _normalize_harvest_return(ret, season_year=y)

        hy_row["season_year"] = int(y)
        harvest_yearly_rows.append(hy_row)

        if isinstance(cold, dict) and cold:
            cold_store_by_year[int(y)] = cold
        if isinstance(year_out, dict) and year_out:
            harvest_year_out_by_year[int(y)] = year_out

    harvest_yearly = pd.DataFrame(harvest_yearly_rows) if harvest_yearly_rows else pd.DataFrame()
    harvest_s = float(time.perf_counter() - t_h0)

    harvest_out: Dict[str, Any] = {
        "harvest_yearly": harvest_yearly,
        "cold_store_by_year": cold_store_by_year,
        "harvest_year_out_by_year": harvest_year_out_by_year,
    }

    # ----------------------------
    # 3) Storage & Distribution (optional)
    # ----------------------------
    run_storage = bool(config.get("run_storage", True))
    storage_by_year = pd.DataFrame()
    storage_s = 0.0

    if run_storage:
        t_s0 = time.perf_counter()

        rows: List[Dict[str, Any]] = []
        years = sorted(set(cold_store_by_year.keys()) & set(harvest_year_out_by_year.keys()))
        for y in years:
            row = run_storage_and_distribution(
                config=config,
                rng=rng,
                season_year=int(y),
                cold_store=cold_store_by_year[int(y)],
                harvest_year_out=harvest_year_out_by_year[int(y)],
                weather_cache=weather_cache,
            )
            if isinstance(row, dict):
                rows.append(row)

        storage_by_year = pd.DataFrame(rows) if rows else pd.DataFrame()
        storage_s = float(time.perf_counter() - t_s0)

    des_total_s = float(time.perf_counter() - t0)

    return {
        "growth_df": growth_df,
        "harvest_yearly": harvest_yearly,
        "storage_by_year": storage_by_year,
        "timings": {
            "growth_s": growth_s,
            "harvest_s": harvest_s,
            "storage_s": storage_s,
            "des_total_s": des_total_s,
        },
        "growth_out": {"growth_df": growth_df},
        "harvest_out": harvest_out,
    }