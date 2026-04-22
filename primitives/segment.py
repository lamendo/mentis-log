"""Simple segmentation between detected boundaries.

Extracted bit-exact from mentis_ui/node_dispatch.py:685-714
(exec_regime_segment). Stdlib only.
"""
from __future__ import annotations

from typing import Any, Dict, List


def regime_segment(
    boundaries: List[Dict[str, Any]],
    *,
    min_windows: int = 4,
) -> List[Dict[str, Any]]:
    """Split the signal at is_boundary positions.

    Segments shorter than min_windows are dropped (their range is folded
    into the next accepted segment's starting point).

    Each segment dict has:
        start_idx, end_idx, n_windows, label, display_title,
        lead_phrases, category_label, coherence_score
    — matching the shape emitted by the in-repo regime_segment node.
    """
    if boundaries is None:
        return []

    min_win = int(min_windows)
    boundary_positions = [
        b["position"] for b in boundaries if b.get("is_boundary")
    ]
    boundary_positions.sort()
    n = len(boundaries)

    segments: List[Dict[str, Any]] = []
    prev = 0
    for bp in boundary_positions + [n]:
        if bp - prev >= min_win:
            segments.append({
                "start_idx": prev,
                "end_idx": bp,
                "n_windows": bp - prev,
                "label": f"Segment {len(segments)}",
                "display_title": f"Segment {len(segments)}",
                "lead_phrases": [],
                "category_label": "",
                "coherence_score": 0.5,
            })
        prev = bp
    return segments
