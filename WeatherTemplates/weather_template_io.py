# WeatherTemplates/weather_template_io.py
from __future__ import annotations

import io
import json
import os
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

TEMPLATE_FOLDER_DEFAULT = os.path.join("WeatherTemplates", "templates")


# ============================================================
# Robust conversions
# ============================================================
def _to_datetime64D_array(x: Any) -> np.ndarray:
    """
    Robustly convert anything date-like into numpy datetime64[D] array.

    Fixes:
      TypeError: Cannot cast DatetimeIndex to dtype datetime64[D]
    by avoiding DatetimeIndex.astype("datetime64[D]").
    """
    if x is None:
        return np.array([], dtype="datetime64[D]")

    if isinstance(x, np.ndarray) and np.issubdtype(x.dtype, np.datetime64):
        return x.astype("datetime64[D]")

    dt = pd.to_datetime(x, errors="coerce", utc=False)

    if isinstance(dt, pd.DatetimeIndex):
        return dt.normalize().to_numpy(dtype="datetime64[D]")

    if isinstance(dt, pd.Series):
        return dt.dt.normalize().to_numpy(dtype="datetime64[D]")

    arr = np.asarray(dt)
    if arr.size == 0:
        return np.array([], dtype="datetime64[D]")
    return arr.astype("datetime64[D]")


def _ensure_np_float32(a: Any) -> np.ndarray:
    return np.asarray(a, dtype=np.float32)


def _ensure_np_int32(a: Any) -> np.ndarray:
    return np.asarray(a, dtype=np.int32)


# ============================================================
# Template pack/unpack
# ============================================================
def weather_cache_to_template_bytes(weather_cache: Dict[str, Any]) -> Tuple[bytes, bytes, bytes]:
    """
    Export ONLY:
      - daily weather (daily_df)  [optional]
      - year_arrays (accumulations + risk prefix sums + date_list) [required]
      - meta [optional]

    Returns:
      daily_csv_bytes, arrays_npz_bytes, meta_json_bytes
    """
    daily = weather_cache.get("daily_df", None)
    year_arrays = weather_cache.get("year_arrays", None)
    meta = weather_cache.get("meta", {}) or {}

    if year_arrays is None or not isinstance(year_arrays, dict) or not year_arrays:
        raise ValueError("weather_cache missing year_arrays; cannot export template.")

    daily_csv_bytes = b""
    if isinstance(daily, pd.DataFrame) and not daily.empty:
        daily_csv_bytes = daily.to_csv(index=False).encode("utf-8")

    buf = io.BytesIO()
    npz_dict: Dict[str, np.ndarray] = {}

    # Store arrays in flat keys: f"{year}__{field}"
    for year, arrs in year_arrays.items():
        y = int(year)
        arrs = arrs or {}
        for k, v in arrs.items():
            key = f"{y}__{k}"
            if k == "date_list":
                npz_dict[key] = _to_datetime64D_array(v)
            elif k in ("date_ord",):
                npz_dict[key] = _ensure_np_int32(v)
            elif k.endswith("_ps") or k in ("frost1_ps", "frost2_ps", "heat1_ps", "heat2_ps", "bee_ok_ps"):
                npz_dict[key] = _ensure_np_int32(v)
            else:
                if k in ("chill", "forcing", "precip", "temp", "sun", "gdd_ps"):
                    npz_dict[key] = _ensure_np_float32(v)
                else:
                    npz_dict[key] = np.asarray(v)

    np.savez_compressed(buf, **npz_dict)
    arrays_npz_bytes = buf.getvalue()

    meta_json_bytes = json.dumps(meta, indent=2, sort_keys=True, default=str).encode("utf-8")
    return daily_csv_bytes, arrays_npz_bytes, meta_json_bytes


def template_bytes_to_weather_cache(
    daily_csv_bytes: bytes,
    arrays_npz_bytes: bytes,
    meta_json_bytes: bytes,
) -> Dict[str, Any]:
    """
    Rebuild a weather_cache that Growth needs:
      - daily_df (optional)
      - year_arrays (required)
      - meta
    """
    daily_df: pd.DataFrame = pd.DataFrame()
    if daily_csv_bytes:
        try:
            daily_df = pd.read_csv(io.BytesIO(daily_csv_bytes))
            if "date" in daily_df.columns:
                daily_df["date"] = pd.to_datetime(daily_df["date"], errors="coerce").dt.date
        except Exception:
            daily_df = pd.DataFrame()

    meta: Dict[str, Any] = {}
    if meta_json_bytes:
        try:
            meta = json.loads(meta_json_bytes.decode("utf-8"))
        except Exception:
            meta = {}

    year_arrays: Dict[int, Dict[str, np.ndarray]] = {}
    with np.load(io.BytesIO(arrays_npz_bytes), allow_pickle=False) as z:
        for full_key in z.files:
            if "__" not in full_key:
                continue
            y_str, field = full_key.split("__", 1)
            try:
                y = int(y_str)
            except Exception:
                continue
            year_arrays.setdefault(y, {})
            arr = z[full_key]
            if field == "date_list":
                arr = np.asarray(arr).astype("datetime64[D]")
            year_arrays[y][field] = arr

    return {
        "daily_df": daily_df,
        "weather_array": daily_df,
        "year_arrays": year_arrays,
        "meta": meta,
    }


# ============================================================
# Folder-based template IO (Micro mode)
# ============================================================
def list_templates(folder: str = TEMPLATE_FOLDER_DEFAULT) -> List[str]:
    if not os.path.isdir(folder):
        return []
    out: List[str] = []
    for name in os.listdir(folder):
        if name.endswith(".npz"):
            out.append(name[:-4])
    out.sort()
    return out


def load_template_from_folder(template_name: str, folder: str = TEMPLATE_FOLDER_DEFAULT) -> Dict[str, Any]:
    """
    Expects files:
      - <name>.npz (required)
      - <name>.csv (optional)
      - <name>.json (optional)
    """
    npz_path = os.path.join(folder, f"{template_name}.npz")
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"Template NPZ not found: {npz_path}")

    csv_path = os.path.join(folder, f"{template_name}.csv")
    json_path = os.path.join(folder, f"{template_name}.json")

    arrays_npz_bytes = open(npz_path, "rb").read()
    daily_csv_bytes = open(csv_path, "rb").read() if os.path.isfile(csv_path) else b""
    meta_json_bytes = open(json_path, "rb").read() if os.path.isfile(json_path) else b"{}"

    return template_bytes_to_weather_cache(daily_csv_bytes, arrays_npz_bytes, meta_json_bytes)


def save_template_to_folder(
    *,
    folder: str,
    template_name: str,
    daily_csv_bytes: bytes,
    arrays_npz_bytes: bytes,
    meta_json_bytes: bytes,
) -> Tuple[str, str, str]:
    os.makedirs(folder, exist_ok=True)

    npz_path = os.path.join(folder, f"{template_name}.npz")
    csv_path = os.path.join(folder, f"{template_name}.csv")
    json_path = os.path.join(folder, f"{template_name}.json")

    with open(npz_path, "wb") as f:
        f.write(arrays_npz_bytes)

    if daily_csv_bytes:
        with open(csv_path, "wb") as f:
            f.write(daily_csv_bytes)
    else:
        if os.path.isfile(csv_path):
            os.remove(csv_path)

    if meta_json_bytes:
        with open(json_path, "wb") as f:
            f.write(meta_json_bytes)

    return csv_path, npz_path, json_path
