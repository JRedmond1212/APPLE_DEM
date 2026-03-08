from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Iterable, Tuple
import numpy as np


@dataclass
class BinLot:
    grade: str
    qty_bins: float
    q: float = 1.0
    age_days: int = 0


GRADES = ["Extra", "Class1", "Class2", "Processor"]
HIGH_TO_LOW = ["Extra", "Class1", "Class2", "Processor"]
LOW_TO_HIGH = list(reversed(HIGH_TO_LOW))


# -----------------------------
# Policy name normalization
# -----------------------------
def normalize_policy_name(policy_name: str) -> str:
    """
    Normalizes policy labels used across the model.

    Grade-order policies (across grade buckets):
      - "Highest Grade First"   (Extra -> ... -> Processor)
      - "FEFO"                  (lowest grade first in your current design)
      - "FIFO"                  (default)

    Within-band (quality-histogram) policies:
      - "HQFO"  : remove highest-quality first within the selected band
      - "FEFO"  : remove lowest-quality first within the selected band
      - "FIFO"  : remove proportionally across the selected band
    """
    p = str(policy_name or "").strip().lower()
    if not p:
        return "FIFO"

    # Explicit within-band policies
    if p in {"hqfo", "highest quality first", "high quality first"}:
        return "HQFO"
    if p in {"fefo", "first expire first out"}:
        return "FEFO"
    if p in {"fifo", "first in first out"}:
        return "FIFO"

    # Existing fuzzy matches
    if "highest" in p and ("grade" in p):
        return "Highest Grade First"
    if "highest" in p and ("quality" in p):
        return "HQFO"

    if "order" in p:
        return "Order-Driven"

    if "fefo" in p:
        return "FEFO"
    if "fifo" in p:
        return "FIFO"
    if "highest" in p:
        # ambiguous "highest": default to grade-first (your old behavior)
        return "Highest Grade First"

    return "FIFO"


# -----------------------------
# Grade priority order (bucket-level)
# -----------------------------
def grade_priority_order(policy_name: str) -> List[str]:
    """
    Decides which grade buckets are attempted first.

    NOTE:
      This is *not* the within-band (histogram) policy.
      This only controls the order of grade buckets, e.g. Extra before Class1.

    IMPORTANT:
      FIFO here still maps to HIGH_TO_LOW (legacy behavior).
      If you want FIFO "mixed grade" movement, use take_weighted_round_robin()
      in the calling module (harvest/grading), which we do.
    """
    p = normalize_policy_name(policy_name)
    if p == "Highest Grade First":
        return HIGH_TO_LOW

    # Your existing behavior mapped FEFO -> LOW_TO_HIGH
    if p == "FEFO":
        return LOW_TO_HIGH

    return HIGH_TO_LOW


# -----------------------------
# Within-band policy (quality histogram)
# -----------------------------
def within_band_policy(policy_name: str) -> str:
    """
    Returns how to remove *within* a grade band when using a quality histogram.

      - HQFO: highest-quality first (rightmost bins)
      - FEFO: lowest-quality first  (leftmost bins)
      - FIFO: proportional across band (deterministic mixture pull)
    """
    p = normalize_policy_name(policy_name)
    if p in {"HQFO", "FEFO", "FIFO"}:
        return p

    # If someone passes "Highest Grade First" to a histogram remover,
    # default to FIFO within the band (safe + neutral).
    return "FIFO"


def remove_from_histogram_by_policy(
    inv_q: np.ndarray,
    mask: np.ndarray,
    amount: float,
    policy_name: str,
    *,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """
    Removes `amount` bins from inv_q where mask==True, using within-band policy.

    Policies:
      - HQFO: take highest-quality first within mask
      - FEFO: take lowest-quality first within mask
      - FIFO: take proportional across mask (deterministic; RNG not used)

    Returns: bins actually removed.
    """
    if amount <= 0.0:
        return 0.0

    inv_q = np.asarray(inv_q, dtype=float)
    inv_q[~np.isfinite(inv_q)] = 0.0
    inv_q[:] = np.maximum(0.0, inv_q)

    mask = np.asarray(mask, dtype=bool)
    idx = np.where(mask)[0]
    if idx.size == 0:
        return 0.0

    avail = inv_q[idx]
    total_avail = float(np.sum(avail))
    if total_avail <= 1e-12:
        return 0.0

    take_total = min(float(amount), total_avail)

    p = within_band_policy(policy_name)

    # FIFO = proportional across the band (this matches your example exactly)
    if p == "FIFO":
        frac = avail / total_avail
        take = frac * take_total
        inv_q[idx] = np.maximum(0.0, inv_q[idx] - take)
        return float(take_total)

    # FEFO/HQFO = directional draw
    idx2 = idx[::-1] if p == "HQFO" else idx  # HQFO high->low, FEFO low->high
    remaining = take_total

    for j in idx2:
        if remaining <= 1e-12:
            break
        a = float(inv_q[j])
        if a <= 0.0:
            continue
        t = min(a, remaining)
        inv_q[j] = a - t
        remaining -= t

    return float(take_total - remaining)


# -----------------------------
# FIFO "mixed-grade" helpers (NEW)
# -----------------------------
def _normalize_weights(
    weights: Dict[str, float],
    grades: List[str],
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    s = 0.0
    for g in grades:
        try:
            w = float(weights.get(g, 0.0))
        except Exception:
            w = 0.0
        w = max(0.0, w)
        out[g] = w
        s += w
    if s <= 1e-12:
        # fallback uniform
        return {g: 1.0 / float(len(grades)) for g in grades}
    return {g: out[g] / s for g in grades}


def _build_weighted_cycle(
    weights_norm: Dict[str, float],
    grades: List[str],
    *,
    cycle_size: int = 10,
) -> List[str]:
    """
    Build a deterministic weighted "pattern" list of grades for FIFO mixed flow.

    Example: weights (0.3,0.3,0.2,0.1), cycle_size=10 -> [E,E,E,C1,C1,C1,C2,C2,P]
    Then repeated.

    We use largest-remainder allocation so totals match cycle_size.
    """
    cycle_size = int(max(1, cycle_size))
    w = np.array([float(weights_norm.get(g, 0.0)) for g in grades], dtype=float)
    w = np.maximum(0.0, w)
    s = float(w.sum())
    if s <= 1e-12:
        # uniform
        w[:] = 1.0 / float(len(grades))
    else:
        w /= s

    raw = w * float(cycle_size)
    base = np.floor(raw).astype(int)
    rem = raw - base

    # ensure at least 1 slot for any grade with nonzero weight, if possible
    nonzero = np.where(w > 1e-12)[0]
    if nonzero.size > 0:
        for i in nonzero:
            if base[i] == 0 and base.sum() < cycle_size:
                base[i] = 1

    # distribute remaining slots by largest remainder
    deficit = cycle_size - int(base.sum())
    if deficit > 0:
        order = np.argsort(-rem)  # descending remainder
        k = 0
        while deficit > 0 and k < order.size:
            base[order[k]] += 1
            deficit -= 1
            k += 1
        # if still deficit (all rem same), just cycle
        k = 0
        while deficit > 0:
            base[k % len(grades)] += 1
            deficit -= 1
            k += 1

    # build cycle in grade order blocks (the repetition creates the “even” mixture)
    cycle: List[str] = []
    for g, n in zip(grades, base.tolist()):
        cycle.extend([g] * int(max(0, n)))

    # safety
    if not cycle:
        cycle = grades[:]  # fallback

    return cycle


def take_weighted_round_robin(
    src: Dict[str, float],
    grades: List[str],
    amount: float,
    weights: Dict[str, float],
    *,
    cycle_size: int = 10,
    step_bins: float = 1.0,
) -> Dict[str, float]:
    """
    Deterministic 'mixed grade' FIFO mover.

    - Builds a weighted cycle from weights (latent mix) with cycle_size bins per cycle.
    - Takes from src in the order of that cycle, repeating.
    - Skips grades with zero available.
    - Takes in steps of `step_bins` (default 1 bin) to preserve the “3-3-2-1 then repeat” feel.

    Returns dict of amounts taken per grade. Mutates src in-place.
    """
    out = {g: 0.0 for g in grades}
    remaining = float(max(0.0, amount))
    if remaining <= 1e-12:
        return out

    # normalize weights across the provided grades
    wn = _normalize_weights(weights, grades)
    cycle = _build_weighted_cycle(wn, grades, cycle_size=int(cycle_size))

    # clamp step
    step = float(max(1e-6, step_bins))

    i = 0
    guard = 0
    # guard prevents infinite loops if everything is empty
    while remaining > 1e-12 and guard < 10_000_000:
        guard += 1
        g = cycle[i % len(cycle)]
        i += 1

        a = float(src.get(g, 0.0) or 0.0)
        if a <= 1e-12:
            # skip empty grade
            # quick exit if all empty
            if all(float(src.get(gg, 0.0) or 0.0) <= 1e-12 for gg in grades):
                break
            continue

        t = min(a, remaining, step)
        if t <= 1e-12:
            continue

        src[g] = a - t
        out[g] += t
        remaining -= t

    return out


# -----------------------------
# Existing helper
# -----------------------------
def summarize_inventory(inv: Dict[str, List[BinLot]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for g, lots in inv.items():
        out[g] = float(sum(float(l.qty_bins) for l in lots))
    return out