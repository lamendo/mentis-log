"""Structural interpretation layer over an existing segmentation.

This module does **not** assign semantic labels (no "incident",
"recovery", "root_cause"). It describes each segment's *change
structure* only, using three labels:

    stable      — low mean / low std / low upper-tail on the
                  structural-change curve inside the segment
    transition  — intermediate elevation and/or strong association
                  with a nearby boundary
    volatile    — elevated mean AND std, or strong internal
                  fluctuation / upper tail

The rules are deterministic and take exactly the data that the rest of
the pipeline already produces: the instability / divergence curve and
the list of detected boundary indices. No model, no training, no LLM.

Usage::

    from interpretation import interpret_segments
    out = interpret_segments(curve, boundaries, schema="structural_v1")
    # out = {"segments": [...], "interpretation": {...}}

This layer is strictly optional. Callers that do not request
interpretation see the original segmentation output unchanged.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


STRUCTURAL_LABELS = ("stable", "transition", "volatile")
SCHEMA_V1 = "structural_v1"

_DISCLAIMER = (
    "Profiles describe structural change only, not semantic incident classes."
)


# ── internal helpers ─────────────────────────────────────────────

def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _build_segment_ranges(
    curve_len: int,
    boundaries: Sequence[int],
) -> List[Tuple[int, int]]:
    """Convert a boundary list into [start, end) ranges covering [0, N).

    Boundaries at or outside (0, N) are filtered. Duplicate / unsorted
    inputs are handled. Empty / no-op boundary lists yield a single
    full-range segment.
    """
    n = int(curve_len)
    if n <= 0:
        return []
    b_clean = sorted({int(b) for b in boundaries if 0 < int(b) < n})
    ranges: List[Tuple[int, int]] = []
    prev = 0
    for b in b_clean:
        if b > prev:
            ranges.append((prev, b))
            prev = b
    if prev < n:
        ranges.append((prev, n))
    return ranges


def _segment_stats(slice_: np.ndarray) -> Dict[str, float]:
    if slice_.size == 0:
        return {"mean": 0.0, "std": 0.0, "max": 0.0, "p90": 0.0}
    return {
        "mean": float(slice_.mean()),
        "std":  float(slice_.std()),
        "max":  float(slice_.max()),
        "p90":  float(np.percentile(slice_, 90)),
    }


def _global_stats(curve_arr: np.ndarray) -> Tuple[float, float, float]:
    if curve_arr.size == 0:
        return 0.0, 0.0, 0.0
    return (
        float(curve_arr.mean()),
        float(curve_arr.std()),
        float(curve_arr.max()),
    )


# ── label rules ──────────────────────────────────────────────────
#
# Intent of each rule block:
#
#   stable
#       The segment looks quiet vs the rest of the log. Relative mean,
#       relative std and upper-tail (p90) are all below the
#       "baseline-ish" thresholds.
#
#   volatile
#       The segment is internally busy. Either mean AND std are both
#       well above global, or the upper tail reaches high absolute
#       levels relative to the global max, or max + std together signal
#       erratic behaviour.
#
#   transition
#       The segment is structurally marked by at least one strong
#       adjacent boundary or has moderate elevation — a plausible
#       phase change rather than a quiet or chaotic interior.
#
# Fallback is stable (conservative — we prefer to under-call change
# than over-call).

def _classify(
    seg_mean: float, seg_std: float, seg_max: float, seg_p90: float,
    bl: Optional[float], br: Optional[float],
    gm: float, gs: float, gmax: float,
) -> Tuple[str, float]:
    # Degenerate-curve early-outs: if the whole curve is flat (no global
    # variation) or effectively zero, there is no structural change to
    # classify — every segment is stable at full confidence.
    if gmax < 1e-12 or gs < 1e-9:
        return "stable", 1.0

    eps = 1e-9
    m_rel = seg_mean / max(gm, eps)
    sd_rel = seg_std / max(gs, eps)
    mx_rel = seg_max / max(gmax, eps)
    p90_rel = seg_p90 / max(gmax, eps)

    bl_val = 0.0 if bl is None else float(bl)
    br_val = 0.0 if br is None else float(br)
    b_rel = max(bl_val, br_val) / max(gmax, eps)

    # Rule 1: stable
    if m_rel < 0.9 and sd_rel < 0.9 and p90_rel < 0.45:
        return "stable", _confidence_stable(m_rel, sd_rel, p90_rel)

    # Rule 2: volatile
    if (
        (m_rel >= 1.15 and sd_rel >= 1.15)
        or p90_rel >= 0.65
        or (mx_rel >= 0.85 and sd_rel >= 1.0)
    ):
        return "volatile", _confidence_volatile(m_rel, sd_rel, p90_rel)

    # Rule 3: transition
    if b_rel >= 0.5 or m_rel >= 1.0 or p90_rel >= 0.5:
        return "transition", _confidence_transition(b_rel, m_rel, p90_rel)

    # Fallback: stable (conservative)
    return "stable", _confidence_stable(m_rel, sd_rel, p90_rel)


def _confidence_stable(m_rel: float, sd_rel: float, p90_rel: float) -> float:
    """Distance of segment from the `stable` ideal (inverted)."""
    combined = (
        0.5 * min(1.0, m_rel / 1.2)
        + 0.3 * min(1.0, sd_rel / 1.2)
        + 0.2 * min(1.0, p90_rel / 0.6)
    )
    return _clamp01(1.0 - _clamp01(combined))


def _confidence_volatile(m_rel: float, sd_rel: float, p90_rel: float) -> float:
    combined = (
        0.4 * min(1.0, m_rel / 1.5)
        + 0.3 * min(1.0, sd_rel / 1.5)
        + 0.3 * min(1.0, p90_rel / 0.8)
    )
    return _clamp01(combined)


def _confidence_transition(b_rel: float, m_rel: float, p90_rel: float) -> float:
    combined = (
        0.5 * min(1.0, b_rel / 0.8)
        + 0.3 * min(1.0, m_rel / 1.2)
        + 0.2 * min(1.0, p90_rel / 0.6)
    )
    return _clamp01(combined)


# ── human-readable summaries ────────────────────────────────────

_SUMMARY_TEMPLATES = {
    "stable":     "Mostly stable behavior with low internal change.",
    "transition": "Elevated change level; likely transition phase.",
    "volatile":   "High internal variability; structurally volatile segment.",
}


def _summary_for(profile: str, idx: int, n_total: int) -> str:
    if profile == "stable" and idx == 0 and n_total > 1:
        return "Stable opening segment."
    if profile == "stable" and idx == n_total - 1 and n_total > 1:
        return "Stable closing segment."
    return _SUMMARY_TEMPLATES.get(profile, _SUMMARY_TEMPLATES["stable"])


# ── public API ──────────────────────────────────────────────────

def interpret_segments(
    curve: Sequence[float],
    boundaries: Sequence[int],
    *,
    schema: str = SCHEMA_V1,
) -> Dict[str, Any]:
    """Assign structural profiles to segments of a curve.

    Parameters
    ----------
    curve :
        The structural-change curve (one value per position in whatever
        unit the caller is working in — chars or lines).
    boundaries :
        Positions (in the same unit as the curve) where a new segment
        begins. Positions at 0 or outside the curve are filtered.
    schema :
        Only ``"structural_v1"`` is defined. Kept as an argument so
        future schemas can coexist without changing the call site.

    Returns
    -------
    dict
        ``{"segments": [...], "interpretation": {...}}``.
        Each segment dict has:
            id, start, end, length,
            mean_score, std_score, max_score, p90_score,
            boundary_strength_left, boundary_strength_right,
            profile, profile_confidence, summary
    """
    if schema != SCHEMA_V1:
        raise ValueError(
            f"Unknown interpretation schema {schema!r}; expected {SCHEMA_V1!r}"
        )

    curve_arr = np.asarray(curve, dtype=np.float64)
    n = int(curve_arr.size)
    ranges = _build_segment_ranges(n, boundaries)
    gm, gs, gmax = _global_stats(curve_arr)

    out_segments: List[Dict[str, Any]] = []
    for i, (start, end) in enumerate(ranges):
        slice_ = curve_arr[start:end]
        s = _segment_stats(slice_)

        # Boundary strength = curve value at the segment's boundary edge.
        # First segment has no left boundary; last has no right boundary.
        bl: Optional[float] = None
        br: Optional[float] = None
        if i > 0 and 0 <= start < n:
            bl = float(curve_arr[start])
        if i < len(ranges) - 1 and 0 <= end < n:
            br = float(curve_arr[end])

        profile, confidence = _classify(
            s["mean"], s["std"], s["max"], s["p90"],
            bl, br, gm, gs, gmax,
        )
        summary = _summary_for(profile, i, len(ranges))

        out_segments.append({
            "id": i,
            "start": int(start),
            "end": int(end),
            "length": int(end - start),
            "mean_score": round(s["mean"], 6),
            "std_score": round(s["std"], 6),
            "max_score": round(s["max"], 6),
            "p90_score": round(s["p90"], 6),
            "boundary_strength_left": (
                round(bl, 6) if bl is not None else None
            ),
            "boundary_strength_right": (
                round(br, 6) if br is not None else None
            ),
            "profile": profile,
            "profile_confidence": round(confidence, 2),
            "summary": summary,
        })

    return {
        "segments": out_segments,
        "interpretation": {
            "enabled": True,
            "label_schema": schema,
            "disclaimer": _DISCLAIMER,
        },
    }


__all__ = ["interpret_segments", "STRUCTURAL_LABELS", "SCHEMA_V1"]
