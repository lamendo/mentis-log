"""Local multiscale refinement of coarse line-level boundaries.

The coarse detector (`runtime.segment`) finds the right transition
*area* for each regime change, but the exact line can be off by a
handful of lines. This module does a second, purely local pass:

  1. For each coarse boundary b, take a local line window
     ``[b - radius_lines, b + radius_lines)``.
  2. Recompute a fine-grained per-character structural signal inside
     that window (char-classes: alpha / digit / whitespace / slash /
     colon / brackets / quote / punct / other).
  3. Compute a rolling left-vs-right class-histogram JSD at several
     scales and aggregate. The refined position is the argmax of the
     aggregated score.
  4. Map the char offset back to a global line index.
  5. Deduplicate refined boundaries and return.

Why this is cleaner than ``local argmax on the coarse curve``:
  - the coarse curve is token-based and downsampled to line granularity;
    its peaks localise to the coarse line, not inside it.
  - the fine signal here runs at character granularity on a locally
    recomputed, mode-independent structural encoding — it pins the
    transition to where the character-level pattern actually changes.
  - the multiscale aggregation prefers positions that are strong at
    several scales, not a single sharp per-scale spike (which may be
    an artifact of any one window width).

No new dependencies. Deterministic. O(radius · n_scales) per boundary.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ── Character-class encoding ─────────────────────────────────────

_CLASS_ALPHA = 0
_CLASS_DIGIT = 1
_CLASS_WS = 2
_CLASS_SLASH = 3
_CLASS_COLON = 4
_CLASS_BRACKETS = 5
_CLASS_QUOTE = 6
_CLASS_PUNCT = 7
_CLASS_OTHER = 8
_NUM_CLASSES = 9

_BRACKETS = frozenset("{}[]()<>")
_QUOTES = frozenset("\"'`")
_PUNCT = frozenset("!@#$%^&*-+=.,;?|~\\_")


def _char_class_of(ch: str) -> int:
    if ch.isalpha():
        return _CLASS_ALPHA
    if ch.isdigit():
        return _CLASS_DIGIT
    if ch.isspace():
        return _CLASS_WS
    if ch == "/":
        return _CLASS_SLASH
    if ch == ":":
        return _CLASS_COLON
    if ch in _BRACKETS:
        return _CLASS_BRACKETS
    if ch in _QUOTES:
        return _CLASS_QUOTE
    if ch in _PUNCT:
        return _CLASS_PUNCT
    return _CLASS_OTHER


def _encode_classes(text: str) -> np.ndarray:
    """Per-character class ids as an int32 array."""
    n = len(text)
    out = np.empty(n, dtype=np.int32)
    for i, ch in enumerate(text):
        out[i] = _char_class_of(ch)
    return out


# ── Rolling JSD over char-class histograms at one scale ─────────

def _jsd_nine(left: np.ndarray, right: np.ndarray, eps: float) -> float:
    """Jensen-Shannon divergence on 9-bin count arrays (numpy float64)."""
    sum_l = float(left.sum()) + _NUM_CLASSES * eps
    sum_r = float(right.sum()) + _NUM_CLASSES * eps
    if sum_l <= 0 or sum_r <= 0:
        return 0.0
    p = (left + eps) / sum_l
    q = (right + eps) / sum_r
    m = 0.5 * (p + q)
    return float(
        0.5 * np.sum(p * np.log(p / m))
        + 0.5 * np.sum(q * np.log(q / m))
    )


def _scale_score(
    class_seq: np.ndarray, scale: int, eps: float = 1e-9,
) -> np.ndarray:
    """Rolling JSD(left_window, right_window) over char-classes at one scale.

    Windows: left = class_seq[i-scale : i], right = class_seq[i : i+scale].
    Returns a score array of the same length; positions where a window
    is truncated emit 0.0.

    Vectorised: one-hot + cumsum(axis=0) gives every windowed histogram
    as a slice difference, and the JSD reduces to pure numpy on an
    ``(n_positions, 9)`` batch. ~100× faster than the per-position
    rolling-dict implementation.
    """
    n = class_seq.size
    score = np.zeros(n, dtype=np.float64)
    s = int(scale)
    K = _NUM_CLASSES
    if s < 1 or n < 2 * s:
        return score

    # One-hot encoding → (n, K) int64
    oh = np.zeros((n, K), dtype=np.int64)
    oh[np.arange(n), class_seq] = 1
    # Cumulative per-class counts → (n+1, K). cum[i] = counts in [0, i).
    cum = np.empty((n + 1, K), dtype=np.int64)
    cum[0] = 0
    cum[1:] = oh.cumsum(axis=0)

    lo = s
    hi = n - s + 1  # exclusive end of valid centre positions
    idx = np.arange(lo, hi)
    left = (cum[idx] - cum[idx - s]).astype(np.float64)
    right = (cum[idx + s] - cum[idx]).astype(np.float64)

    sum_l = left.sum(axis=1, keepdims=True) + K * eps
    sum_r = right.sum(axis=1, keepdims=True) + K * eps
    p = (left + eps) / sum_l
    q = (right + eps) / sum_r
    m = 0.5 * (p + q)
    jsd = (
        0.5 * (p * np.log(p / m)).sum(axis=1)
        + 0.5 * (q * np.log(q / m)).sum(axis=1)
    )
    score[lo:hi] = jsd
    return score


def _multiscale_score(
    class_seq: np.ndarray,
    scales: Sequence[int],
    eps: float = 1e-9,
) -> Tuple[np.ndarray, int]:
    """Sum of per-scale normalized scores divided by number of usable
    scales. Returns (agg, n_used_scales). If no scale is usable the
    returned array is all-zero."""
    n = class_seq.size
    agg = np.zeros(n, dtype=np.float64)
    n_used = 0
    for s in scales:
        if s < 1 or n < 2 * int(s):
            continue
        sc = _scale_score(class_seq, int(s), eps=eps)
        mx = float(sc.max())
        if mx > 0.0:
            sc = sc / mx
        agg += sc
        n_used += 1
    if n_used > 0:
        agg /= float(n_used)
    return agg, n_used


# ── Onset extraction ────────────────────────────────────────────

def _find_onset_index(
    agg: np.ndarray,
    sep_idx: int,
    *,
    alpha: float,
    persistence: int,
) -> int:
    """Find the earliest sustained-elevation index at or before
    ``sep_idx``.

    Algorithm (deterministic):
      threshold = alpha * agg[sep_idx]
      walk left from sep_idx while agg[i-1] >= threshold
      onset = left edge of that run
      if the run is shorter than ``persistence`` → fall back to
      sep_idx (no distinct onset)

    The onset is always ``<= sep_idx`` and always within ``agg``.
    """
    n = int(agg.size)
    if n == 0:
        return 0
    sep = int(max(0, min(n - 1, sep_idx)))
    peak = float(agg[sep])
    if peak <= 0.0:
        return sep
    threshold = float(alpha) * peak

    i = sep
    while i > 0 and float(agg[i - 1]) >= threshold:
        i -= 1
    run_start = i
    run_len = sep - run_start + 1
    if run_len < int(persistence):
        return sep
    return run_start


# ── Public API ──────────────────────────────────────────────────

_METHOD_NAME = "local_multiscale_char_classes_v1"
_DEFAULT_SCALES: Tuple[int, ...] = (4, 16, 64)
_DEFAULT_ONSET_ALPHA = 0.6
_DEFAULT_ONSET_PERSISTENCE = 8


def refine_boundaries_local_multiscale(
    text: str,
    raw_boundaries: Sequence[int],
    *,
    radius_lines: int = 256,
    fine_signal: str = "char_classes",
    scales: Optional[Sequence[int]] = None,
    drop_edge_artifacts: bool = False,
    onset_alpha: float = _DEFAULT_ONSET_ALPHA,
    onset_persistence: int = _DEFAULT_ONSET_PERSISTENCE,
) -> Dict[str, Any]:
    """Refine each coarse line-level boundary using a local multiscale
    character-class structural signal.

    Parameters
    ----------
    text : str
        The full input text whose lines are the coarse segmentation
        unit.
    raw_boundaries : Sequence[int]
        Line indices produced by the coarse detector.
    radius_lines : int
        Half-window size (in lines) for the local refinement. The local
        window for a boundary ``b`` is ``lines[b-radius : b+radius)``.
    fine_signal : str
        Currently only ``"char_classes"`` is supported.
    scales : Sequence[int] or None
        Character-scale window widths for the multiscale JSD. Defaults
        to ``(4, 16, 64)`` — roughly "word / short phrase / line".
    drop_edge_artifacts : bool
        When True, boundaries whose local refinement is degenerate
        (flat score, window too narrow, near file edge) are dropped
        instead of kept. Default False — we prefer to keep the raw
        boundary as the safe fallback.

    Returns
    -------
    dict with keys:
        refined_boundaries  list[int], sorted, deduped separator line
                            indices (public default).
        boundary_details    per-raw-boundary dict with keys:
                              - raw
                              - refined         (alias for separator,
                                                 kept for backward compat)
                              - separator       (point of maximal
                                                 structural split; argmax
                                                 of the local multiscale
                                                 score)
                              - onset           (earliest sustained
                                                 elevation leading into
                                                 the separator; always
                                                 <= separator)
                              - status          one of:
                                - "refined", "refined_same"
                                - "kept_raw_too_narrow"
                                - "kept_raw_degenerate"
                                - "kept_raw_flat"
                                - "dropped_edge_artifact"
        refinement          {enabled, method, radius_lines, scales,
                             fine_signal, public_boundary_semantics,
                             onset_alpha, onset_persistence}
    """
    if fine_signal != "char_classes":
        raise ValueError(
            f"Unknown fine_signal {fine_signal!r}; "
            f"only 'char_classes' is supported in v1."
        )
    if scales is None:
        scales = _DEFAULT_SCALES
    scales = tuple(int(s) for s in scales if int(s) >= 1)
    if not scales:
        scales = _DEFAULT_SCALES

    metadata = {
        "enabled": True,
        "method": _METHOD_NAME,
        "radius_lines": int(radius_lines),
        "scales": list(scales),
        "fine_signal": fine_signal,
        "public_boundary_semantics": "separator",
        "onset_alpha": float(onset_alpha),
        "onset_persistence": int(onset_persistence),
    }

    if not raw_boundaries:
        return {
            "refined_boundaries": [],
            "boundary_details": [],
            "refinement": metadata,
        }

    lines = text.splitlines()
    n_lines = len(lines)
    if n_lines < 2:
        # Nothing meaningful to refine.
        return {
            "refined_boundaries": [int(b) for b in raw_boundaries
                                   if 0 < int(b) < n_lines],
            "boundary_details": [
                {
                    "raw": int(b),
                    "refined": int(b),
                    "separator": int(b),
                    "onset": int(b),
                    "status": "kept_raw_degenerate",
                }
                for b in raw_boundaries
            ],
            "refinement": metadata,
        }

    details: List[Dict[str, Any]] = []
    refined: List[int] = []

    for raw in raw_boundaries:
        raw_b = int(raw)
        lo = max(0, raw_b - int(radius_lines))
        hi = min(n_lines, raw_b + int(radius_lines))

        if hi - lo < 4:
            _append_kept(refined, details, raw_b, "kept_raw_too_narrow",
                         drop_edge_artifacts)
            continue

        local_lines = lines[lo:hi]
        # Local text and per-line char starts inside this local window.
        local_line_starts = [0]
        pos = 0
        for ln in local_lines[:-1]:
            pos += len(ln) + 1  # +1 for '\n' join
            local_line_starts.append(pos)
        local_text = "\n".join(local_lines)
        if not local_text:
            _append_kept(refined, details, raw_b, "kept_raw_degenerate",
                         drop_edge_artifacts)
            continue

        class_seq = _encode_classes(local_text)
        agg, n_used = _multiscale_score(class_seq, scales)
        if n_used == 0 or float(agg.max()) < 1e-12:
            _append_kept(refined, details, raw_b, "kept_raw_flat",
                         drop_edge_artifacts)
            continue

        separator_char_offset = int(np.argmax(agg))
        onset_char_offset = _find_onset_index(
            agg, separator_char_offset,
            alpha=onset_alpha, persistence=onset_persistence,
        )

        # Map char offsets → local line indices via searchsorted.
        starts_arr = np.asarray(local_line_starts, dtype=np.int64)

        def _offset_to_line(off: int) -> int:
            idx = int(
                np.searchsorted(starts_arr, int(off), side="right")
            ) - 1
            idx = max(0, min(len(local_lines) - 1, idx))
            return lo + idx

        separator_line = _offset_to_line(separator_char_offset)
        onset_line = _offset_to_line(onset_char_offset)
        # Onset cannot exceed separator (it might map to the same
        # line when the onset falls within the separator's line).
        if onset_line > separator_line:
            onset_line = separator_line

        status = (
            "refined" if separator_line != raw_b else "refined_same"
        )
        refined.append(separator_line)
        details.append({
            "raw": raw_b,
            "refined": separator_line,   # kept for backward compat
            "separator": separator_line,
            "onset": onset_line,
            "status": status,
        })

    # Deduplicate + sort (boundaries that collapsed to the same line
    # after refinement become one).
    seen: set = set()
    dedup: List[int] = []
    for b in refined:
        if 0 < b < n_lines and b not in seen:
            seen.add(b)
            dedup.append(int(b))
    dedup.sort()

    return {
        "refined_boundaries": dedup,
        "boundary_details": details,
        "refinement": metadata,
    }


def _append_kept(
    refined_list: List[int],
    details_list: List[Dict[str, Any]],
    raw_b: int,
    status: str,
    drop_edge: bool,
) -> None:
    """Helper: either drop or keep-as-raw, with a status string.

    In the keep-raw case both ``separator`` and ``onset`` equal
    ``raw`` — the refinement could not improve on the coarse
    boundary, so both views collapse to the fallback.
    """
    if drop_edge:
        details_list.append({
            "raw": raw_b,
            "refined": None,
            "separator": None,
            "onset": None,
            "status": "dropped_edge_artifact",
        })
        return
    refined_list.append(raw_b)
    details_list.append({
        "raw": raw_b,
        "refined": raw_b,
        "separator": raw_b,
        "onset": raw_b,
        "status": status,
    })


__all__ = [
    "refine_boundaries_local_multiscale",
]
