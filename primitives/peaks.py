"""Quantile peak selection over an instability curve.

Extracted bit-exact from mentis_ui/node_adapters/regime_adapters.py
(lines 1587-1688, exec_quantile_peak_select). Pure numpy.

Algorithm:
  1. tau = np.quantile(curve, q)
  2. Candidates: local maxima where curve[i] >= tau
  3. Prominence filter: keep if curve[i] - max(left_min, right_min) >= min_prominence
  4. Non-maximum suppression within nms_radius
  5. Min-distance greedy selection
  6. Consolidation of nearby peaks within consolidation_radius (keep max)
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


def quantile_peak_select(
    curve: List[float],
    *,
    quantile: float = 0.80,
    min_distance: int = 100,
    min_prominence: float = 0.01,
    prominence_k: int = 30,
    nms_radius: int = 50,
    consolidation_radius: int = 100,
) -> List[Dict[str, Any]]:
    """Select breakpoints using signal-relative quantile threshold.

    Parameters match the defaults passed by regime_full_stack_v1 in
    mentis_lab/_pipelines.py:2069-2073 (quantile=0.80, min_distance=100,
    min_prominence=0.01, nms_radius=50, consolidation_radius=100).

    Returns a list of dicts, one per curve position:
        {"position": int, "combined_score": float, "is_boundary": bool}
    """
    if curve is None:
        return []
    arr = np.asarray(curve, dtype=np.float64).ravel()
    N = len(arr)
    if N < 3:
        return []

    quantile = float(quantile)
    min_distance = int(min_distance)
    min_prominence = float(min_prominence)
    prominence_k = int(prominence_k)
    nms_radius = int(nms_radius)
    consolidation_radius = int(consolidation_radius)

    tau = float(np.quantile(arr, quantile))

    # ── Local maxima above tau ──────────────────────────────────
    candidates: List = []
    for i in range(1, N - 1):
        if arr[i] >= tau and arr[i] >= arr[i - 1] and arr[i] >= arr[i + 1]:
            excess = arr[i] - tau
            candidates.append((i, excess))
    if not candidates:
        return []

    # ── Prominence filtering ────────────────────────────────────
    pk = max(3, min(prominence_k, N // 4))
    filtered: List = []
    for pos, excess in candidates:
        lo, hi = max(0, pos - pk), min(N, pos + pk + 1)
        left = arr[lo:pos]
        right = arr[pos + 1:hi]
        left_min = left.min() if len(left) > 0 else arr[pos]
        right_min = right.min() if len(right) > 0 else arr[pos]
        prom = arr[pos] - max(left_min, right_min)
        if prom >= min_prominence:
            filtered.append((pos, excess * prom))
    if not filtered:
        return []
    candidates = filtered

    # ── Non-maximum suppression ─────────────────────────────────
    eff_radius = max(3, min(nms_radius, N // 4))
    sorted_c = sorted(candidates, key=lambda x: x[1], reverse=True)
    suppressed = set()
    nms_result: List = []
    for pos, score in sorted_c:
        if pos in suppressed:
            continue
        nms_result.append((pos, score))
        for other_pos, _ in sorted_c:
            if other_pos != pos and abs(other_pos - pos) < eff_radius:
                suppressed.add(other_pos)
    candidates = nms_result

    # ── Min-distance greedy selection ───────────────────────────
    candidates.sort(key=lambda x: x[1], reverse=True)
    selected: List[int] = []
    for pos, score in candidates:
        if all(abs(pos - s) >= min_distance for s in selected):
            selected.append(pos)

    # ── Consolidation ───────────────────────────────────────────
    if len(selected) > 1:
        selected.sort()
        clusters: List[List[int]] = []
        current = [selected[0]]
        for p in selected[1:]:
            if p - current[-1] <= consolidation_radius:
                current.append(p)
            else:
                clusters.append(current)
                current = [p]
        clusters.append(current)
        selected = sorted(max(cl, key=lambda x: arr[x]) for cl in clusters)

    selected_set = set(selected)
    boundaries = []
    for i in range(N):
        is_bp = i in selected_set
        boundaries.append({
            "position": i,
            "combined_score": round(float(arr[i]), 4),
            "is_boundary": is_bp,
        })

    return boundaries
