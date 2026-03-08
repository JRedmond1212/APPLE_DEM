# DES/Scenario_module.py
from __future__ import annotations

from typing import Any, Dict
import copy
import numpy as np


def apply_capped_resources(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns a config copy with resource caps applied (if enabled).
    This is used by harvest/storage modules.
    """
    cfg = copy.deepcopy(config)

    if not cfg.get("scenarios_enabled", False):
        return cfg
    if not cfg.get("scen_capped_resources_enabled", False):
        return cfg

    # Capacity caps
    if "cap_long_term_capacity" in cfg:
        cfg["long_term_capacity"] = int(cfg["cap_long_term_capacity"])
    if "cap_field_capacity" in cfg:
        cfg["field_capacity"] = int(cfg["cap_field_capacity"])
    if "cap_pregrading_capacity" in cfg:
        cfg["pregrading_capacity"] = int(cfg["cap_pregrading_capacity"])

    # Bin availability cap (downstream should respect)
    cfg["bin_availability_total"] = int(cfg.get("cap_bin_availability_total", cfg.get("bin_availability_total", 0)))

    # Scale worker capacity mu by multipliers (keeps sigma behavior)
    workers = cfg.get("workers", {})
    if isinstance(workers, dict):
        def scale(role: str, mult: float):
            r = workers.get(role)
            if isinstance(r, dict):
                r["mu"] = float(r.get("mu", 0.0)) * float(mult)
                workers[role] = r

        scale("Harvesters", float(cfg.get("cap_labour_multiplier_harvesters", 1.0)))
        scale("Graders", float(cfg.get("cap_labour_multiplier_graders", 1.0)))
        scale("EmptyBinShuttle", float(cfg.get("cap_labour_multiplier_empty_shuttle", 1.0)))
        scale("FilledBinShuttle", float(cfg.get("cap_labour_multiplier_filled_shuttle", 1.0)))

        cfg["workers"] = workers

    return cfg


def sample_environment_shocks(config: Dict[str, Any], rng: np.random.Generator) -> Dict[str, Any]:
    """
    Returns a dict of environmental shock draws for THIS MC seed.
    These can be applied inside Growth (recommended) without touching preprocessing arrays.
    """
    if not config.get("scenarios_enabled", False) or not config.get("scen_env_enabled", False):
        return {"frost_snap": False, "drought": False, "temp_trend_c_per_year": 0.0}

    frost_p = float(config.get("env_frost_snap_prob", 0.0))
    drought_p = float(config.get("env_drought_prob", 0.0))
    return {
        "frost_snap": bool(rng.random() < frost_p),
        "drought": bool(rng.random() < drought_p),
        "temp_trend_c_per_year": float(config.get("env_temp_trend_c_per_year", 0.0)),
    }


def extract_policy_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract policy selections into a clean dict (used by harvest/storage modules).
    """
    if not config.get("scenarios_enabled", False) or not config.get("scen_policy_enabled", False):
        return {
            "enabled": False,
            "policy_harvest": "Highest Grade First",
            "policy_grading": "FIFO",
            "policy_storage_shipping": "FEFO",
            "lookahead_days": int(config.get("policy_lookahead_days", 14)),
        }

    return {
        "enabled": True,
        "policy_harvest": str(config.get("policy_harvest", "Highest Grade First")),
        "policy_grading": str(config.get("policy_grading", "FIFO")),
        "policy_storage_shipping": str(config.get("policy_storage_shipping", "FEFO")),
        "lookahead_days": int(config.get("policy_lookahead_days", 14)),
    }
