# ============================================================
# HARVEST & GRADING MODULE (FULL) — UPDATED FOR FIFO MIXTURE:
#   ✅ FIFO harvest picking = weighted round-robin grade mixture (latent mix)
#   ✅ FIFO grading flow = weighted round-robin grade mixture (latent mix)
#   ✅ FEFO / Highest Grade First behaviour unchanged
#   ✅ Waste handling + utilisation logic unchanged
# ============================================================

from __future__ import annotations

from typing import Any, Dict, Tuple, List
import numpy as np
import pandas as pd

from ABS.Policy_module import (
    grade_priority_order,
    normalize_policy_name,
    take_weighted_round_robin,
)

GRADES = ["Extra", "Class1", "Class2", "Processor"]
ALL_GRADES = ["Extra", "Class1", "Class2", "Processor", "Waste"]

GRADE_Q_BANDS = {
    "Extra": (0.85, 1.00),
    "Class1": (0.70, 0.85),
    "Class2": (0.50, 0.70),
    "Processor": (0.25, 0.50),
    "Waste": (0.00, 0.25),
}


def _draw_capacity(rng: np.random.Generator, workers: Dict[str, Any], role: str) -> float:
    """
    Fractional workers allowed.
    Sum of n_eff i.i.d Normal(mu,sigma) approx Normal(n_eff*mu, sqrt(n_eff)*sigma)
    """
    r = workers.get(role, {"n": 0.0, "mu": 0.0, "sigma": 0.0})

    n_eff = float(r.get("n", 0.0))
    if not np.isfinite(n_eff) or n_eff <= 0.0:
        return 0.0

    mu = float(r.get("mu", 0.0))
    sigma = float(r.get("sigma", 0.0))

    cap = rng.normal(loc=n_eff * mu, scale=np.sqrt(n_eff) * sigma)
    return float(max(0.0, cap))


def _decay_quality(q: float, k: float) -> float:
    return float(max(0.0, q * np.exp(-k)))


def _grade_from_quality(q: float) -> str:
    for g, (lo, hi) in GRADE_Q_BANDS.items():
        if lo <= q <= hi:
            return g
    return "Waste"


def _grade_mid_q(grade: str) -> float:
    lo, hi = GRADE_Q_BANDS.get(grade, (0.0, 1.0))
    return float((lo + hi) / 2.0)


def _append_capped_samples(
    rng: np.random.Generator,
    samples: List[float],
    new_samples: np.ndarray,
    cap: int,
) -> None:
    if cap <= 0:
        return
    samples.extend([float(x) for x in new_samples])
    if len(samples) > cap:
        idx = rng.choice(len(samples), size=cap, replace=False)
        idx.sort()
        kept = [samples[i] for i in idx]
        samples[:] = kept


def run_harvest_and_grading(
    config: Dict[str, Any],
    growth_row: Dict[str, Any],
    rng: np.random.Generator,
    start_day: int = 0,
    log_level: str = "none",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:

    # ---------------- inputs ----------------
    kg_per_bin = float(config.get("kg_per_bin", 350.0))
    latent_mix = config.get("latent_grade_mix", {"Extra": 30, "Class1": 40, "Class2": 20, "Processor": 10})
    workers = config.get("workers", {})
    decay = config.get("decay_constants", {})

    bins_factor = float(config.get("what_if_bins_factor", 1.0))  # limits empties released (not yield)

    detail_run = bool(config.get("detail_run", False))
    samples_cap = int(config.get("quality_samples_cap", 6000))

    policy_harvest = normalize_policy_name(config.get("policy_harvest", "FIFO"))
    policy_grading = normalize_policy_name(config.get("policy_grading", "FIFO"))

    # legacy grade orders (used when policy != FIFO mixed)
    pick_order = grade_priority_order(policy_harvest)
    flow_order = grade_priority_order(policy_grading)

    # FIFO-mix parameters (tunable)
    cycle_size = int(config.get("fifo_cycle_size", 10))         # 10 => 30/30/20/10 -> 3/3/2/1
    step_bins = float(config.get("fifo_step_bins", 1.0))        # 1 bin steps yields the alternating pattern

    # ---------------- growth outputs ----------------
    total_kg = float(growth_row.get("yield_total_kg", growth_row.get("yield_tonnes_total", 0.0) * 1000.0))
    harvest_anchor = pd.to_datetime(growth_row.get("harvest_date", pd.Timestamp("2000-01-01")), errors="coerce")
    if pd.isna(harvest_anchor):
        harvest_anchor = pd.Timestamp("2000-01-01")
    harvest_anchor = harvest_anchor.normalize()

    if not np.isfinite(total_kg) or total_kg <= 0:
        cold = {g: 0.0 for g in ALL_GRADES}
        out = {
            "harvest_anchor_date": harvest_anchor,
            "harvest_window_len": 1,
            "date_series": [harvest_anchor],
            "trees_series": [0.0],
            "field_empty_series": [0.0],
            "field_filled_series": [0.0],
            "sts_filled_series": [0.0],

            # utilisation (raw + active)
            "util_empty": 0.0, "util_pick": 0.0, "util_shuttle": 0.0, "util_grade": 0.0,
            "util_empty_active": 0.0, "util_pick_active": 0.0, "util_shuttle_active": 0.0, "util_grade_active": 0.0,

            "graded_quality_samples": [],
            "initial_bins_on_trees": 0.0,
            "waste_bins_removed_total": 0.0,
            "ending_flow_bins": 0.0,
            "cold_total_nonwaste": 0.0,
            "mass_balance_error": 0.0,

            **{f"cold_{g}": 0.0 for g in ALL_GRADES},
        }
        return cold, out

    # total yield bins (true amount of fruit-bins represented by yield)
    total_bins = float(total_kg / max(1e-12, kg_per_bin))

    # empties available is a separate constraint
    total_bins_available = float(total_bins * max(0.0, bins_factor))

    mix_sum = float(sum(latent_mix.get(g, 0) for g in GRADES)) or 1.0
    bins_on_trees = {g: total_bins * (float(latent_mix.get(g, 0)) / mix_sum) for g in GRADES}

    initial_bins_on_trees = float(sum(bins_on_trees.values()))

    field_capacity = int(float(config.get("field_capacity", 500)))
    pregrading_capacity = int(float(config.get("pregrading_capacity", 300)))
    long_term_capacity = int(float(config.get("long_term_capacity", 2000)))

    field_empty = 0.0
    field_filled = {g: 0.0 for g in ALL_GRADES}
    sts = {g: 0.0 for g in ALL_GRADES}
    cold = {g: 0.0 for g in ALL_GRADES}

    # representative quality per bucket
    q_trees = {g: _grade_mid_q(g) for g in ALL_GRADES}
    q_field = {g: _grade_mid_q(g) for g in ALL_GRADES}
    q_sts = {g: _grade_mid_q(g) for g in ALL_GRADES}

    # series
    date_series: List[pd.Timestamp] = []
    trees_series: List[float] = []
    field_empty_series: List[float] = []
    field_filled_series: List[float] = []
    sts_filled_series: List[float] = []

    # ---------------- utilisation bookkeeping ----------------
    cap_empty_sum = cap_pick_sum = cap_shut_sum = cap_grade_sum = 0.0
    used_empty_sum = used_pick_sum = used_shut_sum = used_grade_sum = 0.0

    cap_empty_active_sum = cap_pick_active_sum = cap_shut_active_sum = cap_grade_active_sum = 0.0
    used_empty_active_sum = used_pick_active_sum = used_shut_active_sum = used_grade_active_sum = 0.0

    graded_quality_samples: List[float] = []
    released_empty_total = 0.0

    # ---------------- waste bookkeeping (removed from system) ----------------
    waste_bins_removed_trees = 0.0
    waste_bins_removed_field = 0.0
    waste_bins_removed_sts = 0.0
    waste_bins_removed_pre_cold = 0.0

    def stage_k(stage: str, grade: str) -> float:
        d = decay.get(stage, {})
        if not isinstance(d, dict):
            return 0.0
        if grade in d:
            return float(d.get(grade, 0.0))
        # fallback: treat waste like Processor constant
        if grade == "Waste" and "Processor" in d:
            return float(d.get("Processor", 0.0))
        return 0.0

    def _remove_any_waste_bucket(src: Dict[str, float], which: str) -> None:
        """Safety: if Waste exists in any in-flow dict, remove it immediately."""
        nonlocal waste_bins_removed_trees, waste_bins_removed_field, waste_bins_removed_sts
        w = float(src.get("Waste", 0.0))
        if w > 0.0:
            src["Waste"] = 0.0
            if which == "trees":
                waste_bins_removed_trees += w
            elif which == "field":
                waste_bins_removed_field += w
            elif which == "sts":
                waste_bins_removed_sts += w

    def _apply_decay_and_remove_waste(
        src: Dict[str, float],
        qsrc: Dict[str, float],
        stage_name: str,
        which: str,
    ) -> None:
        """
        Weekly:
          - decay each grade bucket’s representative quality
          - if it crosses to Waste: REMOVE bins (count waste)
          - if it crosses to another grade: shift bucket
        """
        nonlocal waste_bins_removed_trees, waste_bins_removed_field, waste_bins_removed_sts

        # first remove any waste that might already exist (prevents clog)
        _remove_any_waste_bucket(src, which)

        for g in list(GRADES):
            amt = float(src.get(g, 0.0))
            if amt <= 0.0:
                continue

            k = stage_k(stage_name, g)
            if k <= 0.0:
                continue

            qnew = _decay_quality(float(qsrc.get(g, _grade_mid_q(g))), k)
            newg = _grade_from_quality(qnew)
            qsrc[g] = qnew

            if newg == "Waste":
                src[g] = 0.0
                if which == "trees":
                    waste_bins_removed_trees += amt
                elif which == "field":
                    waste_bins_removed_field += amt
                elif which == "sts":
                    waste_bins_removed_sts += amt
                continue

            if newg != g:
                src[g] = 0.0
                src[newg] = float(src.get(newg, 0.0)) + amt
                qsrc[newg] = min(float(qsrc.get(newg, _grade_mid_q(newg))), qnew)

    # ---------------- simulation ----------------
    max_days = 365
    for day in range(max_days):
        cur_date = harvest_anchor + pd.Timedelta(days=int(day))
        date_series.append(cur_date)

        # weekly decay
        if day % 7 == 0:
            _apply_decay_and_remove_waste(bins_on_trees, q_trees, "BinsOnTrees", which="trees")
            _apply_decay_and_remove_waste(field_filled, q_field, "FieldBins", which="field")
            _apply_decay_and_remove_waste(sts, q_sts, "PreGrading", which="sts")

        # ----------------------------------------------------
        # Compute backlogs & spaces
        # ----------------------------------------------------
        trees_total = float(sum(bins_on_trees.get(g, 0.0) for g in GRADES))
        field_filled_total = float(sum(field_filled.get(g, 0.0) for g in GRADES))
        sts_total = float(sum(sts.get(g, 0.0) for g in GRADES))
        cold_total = float(sum(cold.get(g, 0.0) for g in GRADES))

        field_used = float(field_empty + field_filled_total)
        field_space = max(0.0, float(field_capacity) - field_used)

        sts_space = max(0.0, float(pregrading_capacity) - sts_total)
        cold_space = max(0.0, float(long_term_capacity) - cold_total)

        remaining_release = max(0.0, total_bins_available - released_empty_total)

        # ----------------------------------------------------
        # EMPTY BIN SHUTTLE
        # ----------------------------------------------------
        empty_forward_active = (trees_total > 1e-9) and (field_space > 1e-9) and (remaining_release > 1e-9)
        empty_reverse_active = (trees_total <= 1e-9) and (field_empty > 1e-9)
        empty_active = empty_forward_active or empty_reverse_active

        cap_empty = _draw_capacity(rng, workers, "EmptyBinShuttle") if empty_active else 0.0
        cap_empty_sum += cap_empty
        cap_empty_active_sum += cap_empty

        moved_empty = 0.0
        if empty_forward_active:
            moved_empty = min(field_space, cap_empty, remaining_release)
            field_empty += moved_empty
            released_empty_total += moved_empty
        elif empty_reverse_active:
            moved_empty = min(field_empty, cap_empty)
            field_empty -= moved_empty

        used_empty_sum += moved_empty
        used_empty_active_sum += moved_empty

        # ----------------------------------------------------
        # HARVESTERS: Trees -> Field filled
        #   FIFO => weighted round-robin mix (latent mix)
        #   else => grade_priority_order (existing)
        # ----------------------------------------------------
        pick_active = (trees_total > 1e-9) and (field_empty > 1e-9)
        cap_pick = _draw_capacity(rng, workers, "Harvesters") if pick_active else 0.0
        cap_pick_sum += cap_pick
        cap_pick_active_sum += cap_pick

        can_fill = min(field_empty, cap_pick)
        filled = 0.0

        if can_fill > 0.0:
            if policy_harvest == "FIFO":
                # mixed-grade FIFO: take in a repeating weighted pattern
                takes = take_weighted_round_robin(
                    bins_on_trees, GRADES, can_fill, latent_mix,
                    cycle_size=cycle_size, step_bins=step_bins
                )
                for g in GRADES:
                    t = float(takes.get(g, 0.0))
                    if t > 0.0:
                        field_filled[g] = float(field_filled.get(g, 0.0)) + t
                        filled += t
            else:
                # legacy behavior (HGF or FEFO by grade order)
                for g in pick_order:
                    if filled >= can_fill:
                        break
                    take = min(float(bins_on_trees.get(g, 0.0)), float(can_fill - filled))
                    if take <= 0.0:
                        continue
                    bins_on_trees[g] = float(bins_on_trees.get(g, 0.0)) - take
                    field_filled[g] = float(field_filled.get(g, 0.0)) + take
                    filled += take

            field_empty -= filled

        used_pick_sum += filled
        used_pick_active_sum += filled

        # ----------------------------------------------------
        # FILLED BIN SHUTTLE: Field filled -> STS
        #   FIFO grading => mixed-grade FIFO
        # ----------------------------------------------------
        field_filled_total = float(sum(field_filled.get(g, 0.0) for g in GRADES))
        sts_total = float(sum(sts.get(g, 0.0) for g in GRADES))
        sts_space = max(0.0, float(pregrading_capacity) - sts_total)

        shuttle_active = (field_filled_total > 1e-9) and (sts_space > 1e-9)
        cap_shut = _draw_capacity(rng, workers, "FilledBinShuttle") if shuttle_active else 0.0
        cap_shut_sum += cap_shut
        cap_shut_active_sum += cap_shut

        can_move = min(cap_shut, sts_space)
        moved = 0.0

        if can_move > 0.0:
            if policy_grading == "FIFO":
                takes = take_weighted_round_robin(
                    field_filled, GRADES, can_move, latent_mix,
                    cycle_size=cycle_size, step_bins=step_bins
                )
                for g in GRADES:
                    t = float(takes.get(g, 0.0))
                    if t > 0.0:
                        sts[g] = float(sts.get(g, 0.0)) + t
                        moved += t
            else:
                for g in flow_order:
                    if moved >= can_move:
                        break
                    take = min(float(field_filled.get(g, 0.0)), float(can_move - moved))
                    if take <= 0.0:
                        continue
                    field_filled[g] = float(field_filled.get(g, 0.0)) - take
                    sts[g] = float(sts.get(g, 0.0)) + take
                    moved += take

        _remove_any_waste_bucket(field_filled, "field")
        _remove_any_waste_bucket(sts, "sts")

        used_shut_sum += moved
        used_shut_active_sum += moved

        # ----------------------------------------------------
        # GRADERS: STS -> Cold (after cold-entry decay)
        #   FIFO grading => mixed-grade FIFO order of processing
        # ----------------------------------------------------
        sts_total = float(sum(sts.get(g, 0.0) for g in GRADES))
        cold_total = float(sum(cold.get(g, 0.0) for g in GRADES))
        cold_space = max(0.0, float(long_term_capacity) - cold_total)

        grade_active = (sts_total > 1e-9) and (cold_space > 1e-9)
        cap_grade = _draw_capacity(rng, workers, "Graders") if grade_active else 0.0
        cap_grade_sum += cap_grade
        cap_grade_active_sum += cap_grade

        can_grade = min(cap_grade, cold_space)
        graded = 0.0

        if can_grade > 0.0:
            if policy_grading == "FIFO":
                # We "schedule" which grades to process in FIFO mixed order,
                # then process that amount per grade (still respecting availability).
                takes = take_weighted_round_robin(
                    sts, GRADES, can_grade, latent_mix,
                    cycle_size=cycle_size, step_bins=step_bins
                )
                # apply cold-entry decay and re-grading for each grade's moved quantity
                for g in GRADES:
                    take = float(takes.get(g, 0.0))
                    if take <= 0.0:
                        continue

                    # cold-entry decay then grade
                    k_cold = stage_k("LongTermStorage", g)
                    q_in = float(q_sts.get(g, _grade_mid_q(g)))
                    if k_cold > 0.0:
                        q_in = _decay_quality(q_in, k_cold)

                    newg = _grade_from_quality(q_in)

                    if newg == "Waste":
                        waste_bins_removed_pre_cold += take
                    else:
                        cold[newg] = float(cold.get(newg, 0.0)) + take

                    graded += take

                    if detail_run:
                        n_samp = int(min(120, max(1, round(take))))
                        samp = rng.normal(q_in, 0.03, size=n_samp)
                        samp = np.clip(samp, 0.0, 1.0)
                        _append_capped_samples(rng, graded_quality_samples, samp, samples_cap)
            else:
                for g in flow_order:
                    if graded >= can_grade:
                        break
                    take = min(float(sts.get(g, 0.0)), float(can_grade - graded))
                    if take <= 0.0:
                        continue

                    sts[g] = float(sts.get(g, 0.0)) - take

                    k_cold = stage_k("LongTermStorage", g)
                    q_in = float(q_sts.get(g, _grade_mid_q(g)))
                    if k_cold > 0.0:
                        q_in = _decay_quality(q_in, k_cold)

                    newg = _grade_from_quality(q_in)

                    if newg == "Waste":
                        waste_bins_removed_pre_cold += take
                    else:
                        cold[newg] = float(cold.get(newg, 0.0)) + take

                    graded += take

                    if detail_run:
                        n_samp = int(min(120, max(1, round(take))))
                        samp = rng.normal(q_in, 0.03, size=n_samp)
                        samp = np.clip(samp, 0.0, 1.0)
                        _append_capped_samples(rng, graded_quality_samples, samp, samples_cap)

        _remove_any_waste_bucket(sts, "sts")

        used_grade_sum += graded
        used_grade_active_sum += graded

        # ----------------------------------------------------
        # SERIES
        # ----------------------------------------------------
        trees_series.append(float(sum(bins_on_trees.get(g, 0.0) for g in GRADES)))
        field_empty_series.append(float(max(0.0, field_empty)))
        field_filled_series.append(float(max(0.0, sum(field_filled.get(g, 0.0) for g in GRADES))))
        sts_filled_series.append(float(max(0.0, sum(sts.get(g, 0.0) for g in GRADES))))

        total_left = (
            float(sum(bins_on_trees.get(g, 0.0) for g in GRADES))
            + float(field_empty)
            + float(sum(field_filled.get(g, 0.0) for g in GRADES))
            + float(sum(sts.get(g, 0.0) for g in GRADES))
        )
        if total_left <= 1e-6:
            break

    # ---------------- utilisation ----------------
    util_empty = float(used_empty_sum / cap_empty_sum) if cap_empty_sum > 1e-9 else 0.0
    util_pick = float(used_pick_sum / cap_pick_sum) if cap_pick_sum > 1e-9 else 0.0
    util_shuttle = float(used_shut_sum / cap_shut_sum) if cap_shut_sum > 1e-9 else 0.0
    util_grade = float(used_grade_sum / cap_grade_sum) if cap_grade_sum > 1e-9 else 0.0

    util_empty_active = float(used_empty_active_sum / cap_empty_active_sum) if cap_empty_active_sum > 1e-9 else 0.0
    util_pick_active = float(used_pick_active_sum / cap_pick_active_sum) if cap_pick_active_sum > 1e-9 else 0.0
    util_shuttle_active = float(used_shut_active_sum / cap_shut_active_sum) if cap_shut_active_sum > 1e-9 else 0.0
    util_grade_active = float(used_grade_active_sum / cap_grade_active_sum) if cap_grade_active_sum > 1e-9 else 0.0

    harvest_window_len = int(max(1, len(trees_series)))

    # ---------------- mass balance ----------------
    waste_bins_removed_total = (
        float(waste_bins_removed_trees)
        + float(waste_bins_removed_field)
        + float(waste_bins_removed_sts)
        + float(waste_bins_removed_pre_cold)
    )

    ending_flow_bins = (
        float(sum(bins_on_trees.get(g, 0.0) for g in GRADES))
        + float(sum(field_filled.get(g, 0.0) for g in GRADES))
        + float(sum(sts.get(g, 0.0) for g in GRADES))
    )

    cold_total_nonwaste = float(sum(cold.get(g, 0.0) for g in GRADES))

    mass_balance_error = float(initial_bins_on_trees - (cold_total_nonwaste + waste_bins_removed_total + ending_flow_bins))

    out = {
        "harvest_anchor_date": harvest_anchor,
        "harvest_window_len": harvest_window_len,
        "date_series": [d.to_pydatetime() for d in date_series] if date_series else [harvest_anchor.to_pydatetime()],
        "trees_series": trees_series if trees_series else [0.0],
        "field_empty_series": field_empty_series if field_empty_series else [0.0],
        "field_filled_series": field_filled_series if field_filled_series else [0.0],
        "sts_filled_series": sts_filled_series if sts_filled_series else [0.0],

        # raw utilisation (kept)
        "util_empty": float(np.clip(util_empty, 0.0, 1.0)),
        "util_pick": float(np.clip(util_pick, 0.0, 1.0)),
        "util_shuttle": float(np.clip(util_shuttle, 0.0, 1.0)),
        "util_grade": float(np.clip(util_grade, 0.0, 1.0)),

        # fixed utilisation (active denominators)
        "util_empty_active": float(np.clip(util_empty_active, 0.0, 1.0)),
        "util_pick_active": float(np.clip(util_pick_active, 0.0, 1.0)),
        "util_shuttle_active": float(np.clip(util_shuttle_active, 0.0, 1.0)),
        "util_grade_active": float(np.clip(util_grade_active, 0.0, 1.0)),

        "graded_quality_samples": graded_quality_samples if detail_run else [],
        "policy_harvest": policy_harvest,
        "policy_grading": policy_grading,

        # mass balance outputs
        "initial_bins_on_trees": float(initial_bins_on_trees),
        "waste_bins_removed_total": float(waste_bins_removed_total),
        "waste_bins_removed_trees": float(waste_bins_removed_trees),
        "waste_bins_removed_field": float(waste_bins_removed_field),
        "waste_bins_removed_sts": float(waste_bins_removed_sts),
        "waste_bins_removed_pre_cold": float(waste_bins_removed_pre_cold),
        "ending_flow_bins": float(ending_flow_bins),
        "cold_total_nonwaste": float(cold_total_nonwaste),
        "mass_balance_error": float(mass_balance_error),

        **{f"cold_{g}": float(cold.get(g, 0.0)) for g in ALL_GRADES},
    }

    return cold, out