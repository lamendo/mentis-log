"""End-to-end orchestration: text → route + boundaries + segments.

Two operating modes:
  - mode="line" (DEFAULT): one step per log line. Fast, semantic, and
    used by the ``strategy="auto"`` recommended path.
  - mode="char": one step per character. Bit-exact vs
    mentis_lab.regime_full_stack_v1 ``seg`` output — kept for parity
    testing and small-text research.

Strategy abstraction (user-facing):
  - strategy="auto" (DEFAULT)
      mode="line"  → q_jsd + tokens + rolling + min_token_freq=2
                     (best configuration found on log benchmarks)
      mode="char"  → heuristic (coverage-routed policy, bit-exact)
  - strategy="heuristic"
      Coverage-routed divergence policy regardless of mode.
  - strategy="qalign"
      Experimental Q-aligned path. Users typically set --signal /
      --q-mode / --scoring explicitly in this mode.

Advanced kwargs (scoring, signal_type, q_mode, min_token_freq) override
strategy defaults when passed explicitly.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from primitives.policy import (
    policy_select_projection,
    policy_select_projection_line,
)
from primitives.peaks import quantile_peak_select
from primitives.segment import regime_segment
from primitives.qalign import (
    compute_qalign_curve,
    compute_qalign_curve_tokens,
    char_byte_signal,
    line_length_signal,
)
from primitives.linemode import _line_tokenize


# Defaults for char mode — match regime_full_stack_v1.
DEFAULTS_CHAR = {
    "window": 100,
    "quantile": 0.80,
    "min_distance": 100,
    "min_prominence": 0.01,
    "nms_radius": 50,
    "consolidation_radius": 100,
    "min_segment_windows": 4,
}

# Defaults for line mode — coarse regime detector, not event detector.
DEFAULTS_LINE = {
    "window": 25,
    "quantile": 0.995,
    "min_distance": 500,
    "min_prominence": 0.03,
    "nms_radius": 250,
    "consolidation_radius": 500,
    "min_segment_windows": 20,
}

# Back-compat alias.
DEFAULTS = DEFAULTS_CHAR

VALID_STRATEGIES = ("auto", "heuristic", "qalign")


def _resolve_strategy(
    strategy: str,
    mode: str,
    scoring: Optional[str],
    signal_type: Optional[str],
    q_mode: Optional[str],
    min_token_freq: Optional[int],
) -> Dict[str, Any]:
    """Resolve strategy into concrete kwargs. Explicit user values win."""
    if strategy == "auto":
        if mode == "char":
            # Bit-exact heuristic path — preserves reference parity.
            defaults = {
                "scoring": "heuristic",
                "signal_type": "line_length",
                "q_mode": "global",
                "min_token_freq": 1,
            }
        else:
            # The configuration that beat heuristic on log benchmarks:
            #   q_jsd + tokens + rolling + min_token_freq=2
            defaults = {
                "scoring": "q_jsd",
                "signal_type": "tokens",
                "q_mode": "rolling",
                "min_token_freq": 2,
            }
    elif strategy == "heuristic":
        defaults = {
            "scoring": "heuristic",
            "signal_type": "line_length",
            "q_mode": "global",
            "min_token_freq": 1,
        }
    elif strategy == "qalign":
        # Q-alignment with sensible defaults for experimentation.
        defaults = {
            "scoring": "q_jsd",
            "signal_type": "tokens" if mode == "line" else "line_length",
            "q_mode": "rolling",
            "min_token_freq": 2 if mode == "line" else 1,
        }
    else:
        raise ValueError(
            f"Unknown strategy {strategy!r}; "
            f"expected one of {VALID_STRATEGIES}"
        )

    # Explicit user kwargs override strategy-picked defaults.
    return {
        "scoring": scoring if scoring is not None else defaults["scoring"],
        "signal_type": (
            signal_type if signal_type is not None else defaults["signal_type"]
        ),
        "q_mode": q_mode if q_mode is not None else defaults["q_mode"],
        "min_token_freq": (
            min_token_freq if min_token_freq is not None
            else defaults["min_token_freq"]
        ),
    }


def _empty_result(mode: str, sample_rate: int) -> Dict[str, Any]:
    base = {
        "mode": mode,
        "sample_rate": sample_rate,
        "route": {"selected": "none", "coverage": 0, "reason": "empty input"},
        "n_boundaries": 0,
        "boundaries": [],
        "n_segments": 0,
        "segments": [],
    }
    if mode == "char":
        base["n_chars"] = 0
    else:
        base["n_lines"] = 0
    return base


def _line_meta_for_qalign(text: str, signal_len: int) -> Dict[str, Any]:
    _, offsets, _, _, _ = _line_tokenize(text)
    return {
        "n_lines": int(signal_len),
        "line_char_offsets": offsets.tolist(),
    }


def _edge_cleanup_threshold(n_lines: int) -> int:
    """Unified threshold for dropping trivially short edge segments.

    ``max(32, int(0.001 · n_lines))`` — a flat 32-line floor for small
    logs plus a 0.1% scale for very large ones.
    """
    return max(32, int(0.001 * int(n_lines)))


def _cleanup_edge_boundaries(
    boundary_positions,
    n_lines: int,
) -> tuple:
    """Drop boundaries that produce trivially short edge segments.

    Returns ``(cleaned_boundaries, dropped_info_list)``. The two edge
    checks are independent: a log may drop the first boundary, the
    last boundary, both, or neither.
    """
    threshold = _edge_cleanup_threshold(n_lines)
    cleaned = [int(b) for b in boundary_positions]
    dropped = []
    # First segment span is [0, boundaries[0]); if shorter than
    # threshold, drop boundaries[0].
    if cleaned and cleaned[0] < threshold:
        dropped.append({
            "side": "first",
            "boundary": cleaned[0],
            "segment_length": cleaned[0],
            "threshold_lines": threshold,
            "reason": (
                f"first segment length {cleaned[0]} < "
                f"{threshold} (edge_cleanup threshold)"
            ),
        })
        cleaned = cleaned[1:]
    # Last segment span is [boundaries[-1], n_lines); if shorter than
    # threshold, drop boundaries[-1].
    if cleaned and (int(n_lines) - cleaned[-1]) < threshold:
        tail = int(n_lines) - cleaned[-1]
        dropped.append({
            "side": "last",
            "boundary": cleaned[-1],
            "segment_length": tail,
            "threshold_lines": threshold,
            "reason": (
                f"last segment length {tail} < "
                f"{threshold} (edge_cleanup threshold)"
            ),
        })
        cleaned = cleaned[:-1]
    return cleaned, dropped


def _line_segments_from_boundaries(
    boundary_positions,
    n_lines: int,
    line_char_offsets,
    *,
    min_windows: int,
) -> list:
    """Build line-mode segment dicts from a sorted boundary list.

    Used both by the main pipeline and by the refinement step to
    rebuild segments after boundaries shift.
    """
    bs = sorted(set(int(b) for b in boundary_positions if 0 < int(b) < n_lines))
    splits = [0] + bs + [int(n_lines)]

    def _char_at(line_idx: int) -> int:
        if not line_char_offsets:
            return 0
        lo = min(max(0, line_idx), len(line_char_offsets) - 1)
        return int(line_char_offsets[lo])

    out_segments = []
    for i in range(len(splits) - 1):
        start = splits[i]
        end = splits[i + 1]
        if end - start < int(min_windows):
            continue
        out_segments.append({
            "start_line": start,
            "end_line": end,
            "n_lines": end - start,
            "char_start_approx": _char_at(start),
            "char_end_approx": _char_at(end),
            "label": f"Segment {len(out_segments)}",
        })
    return out_segments


def _effective_quantile(
    curve_arr: np.ndarray,
    *,
    threshold_mode: str,
    fallback_quantile: float,
    mean_std_k: float,
    topk_n: int,
) -> float:
    if threshold_mode == "quantile" or curve_arr.size == 0:
        return float(fallback_quantile)
    if threshold_mode == "mean_std":
        tau = float(curve_arr.mean() + mean_std_k * curve_arr.std())
        q = float((curve_arr <= tau).sum()) / float(curve_arr.size)
        return float(min(0.9999, max(0.0, q)))
    if threshold_mode == "topk":
        k = int(max(1, min(topk_n, curve_arr.size)))
        tau = float(np.sort(curve_arr)[-k])
        q = float((curve_arr < tau).sum()) / float(curve_arr.size)
        return float(min(0.9999, max(0.0, q)))
    raise ValueError(
        f"Unknown threshold_mode {threshold_mode!r}; "
        f"expected 'quantile'|'mean_std'|'topk'"
    )


def segment(
    text: str,
    *,
    # Product-level knobs
    strategy: str = "auto",
    mode: str = "line",
    sample_rate: int = 1,
    include_curve: bool = False,
    interpret: bool = False,
    refine: bool = True,
    # Runtime default is 64 — a conservative local nudge, not a large
    # swing. The refine module's API default is 256 so callers who
    # explicitly invoke the refinement function can opt into a wider
    # search window.
    refine_radius_lines: int = 64,
    edge_cleanup: bool = True,
    # Expert overrides — None means "let strategy decide"
    scoring: Optional[str] = None,
    signal_type: Optional[str] = None,
    q_mode: Optional[str] = None,
    min_token_freq: Optional[int] = None,
    # Non-strategy knobs (always user-settable)
    dist_bins: int = 32,
    threshold_mode: str = "quantile",
    mean_std_k: float = 2.0,
    topk_n: int = 10,
    **overrides: Any,
) -> Dict[str, Any]:
    """Run the segmentation pipeline.

    Default invocation uses the production-grade strategy:
        segment(text)  ==  mode='line' + q_jsd + tokens + rolling +
                           min_token_freq=2

    For bit-exact parity vs the in-repo reference pipeline, pass
    ``mode='char'`` (which routes strategy='auto' to the heuristic path).

    For the old heuristic line-mode behaviour, pass
    ``strategy='heuristic'``.
    """
    if mode not in ("char", "line"):
        raise ValueError(f"Unknown mode {mode!r}; expected 'char' or 'line'")
    if strategy not in VALID_STRATEGIES:
        raise ValueError(
            f"Unknown strategy {strategy!r}; expected {VALID_STRATEGIES}"
        )

    resolved = _resolve_strategy(
        strategy, mode, scoring, signal_type, q_mode, min_token_freq,
    )
    scoring = resolved["scoring"]
    signal_type = resolved["signal_type"]
    q_mode = resolved["q_mode"]
    min_token_freq = resolved["min_token_freq"]

    if scoring not in ("heuristic", "q_jsd", "q_kl"):
        raise ValueError(
            f"Unknown scoring {scoring!r}; "
            f"expected 'heuristic'|'q_jsd'|'q_kl'"
        )
    if signal_type not in ("line_length", "tokens"):
        raise ValueError(
            f"Unknown signal_type {signal_type!r}; "
            f"expected 'line_length' or 'tokens'"
        )
    if signal_type == "tokens" and mode != "line":
        raise ValueError(
            "signal_type='tokens' is only valid with mode='line' "
            "(tokens are a per-line concept)."
        )
    s = max(1, int(sample_rate))

    if not isinstance(text, str) or not text:
        return _empty_result(mode, s)

    defaults = DEFAULTS_CHAR if mode == "char" else DEFAULTS_LINE
    params = {**defaults, **overrides}

    # ── 1) Build the instability curve ─────────────────────────────
    meta: Dict[str, Any] = {}
    token_meta: Dict[str, Any] = {}

    if scoring == "heuristic":
        if mode == "char":
            pol = policy_select_projection(
                text, window=params["window"], sample_rate=s,
            )
        else:
            pol = policy_select_projection_line(
                text, window=params["window"], sample_rate=s,
            )
            meta = pol.get("meta_out", {}) or {}
        curve = pol["curve_out"]
        route = pol["route_out"]
    else:
        divergence = "jsd" if scoring == "q_jsd" else "kl"
        if signal_type == "tokens":
            lines = text.splitlines()
            meta = _line_meta_for_qalign(text, len(lines))
            curve_arr, token_meta = compute_qalign_curve_tokens(
                lines,
                window=params["window"],
                q_mode=q_mode,
                divergence=divergence,
                sample_rate=s,
                min_token_freq=min_token_freq,
            )
            reason_suffix = (
                f"signal=tokens, q_mode={q_mode}, "
                f"vocab_size={token_meta.get('vocab_size', 0)}"
            )
        else:
            if mode == "char":
                signal = char_byte_signal(text)
            else:
                signal = line_length_signal(text)
                meta = _line_meta_for_qalign(text, signal.size)
            curve_arr = compute_qalign_curve(
                signal,
                window=params["window"],
                bins=dist_bins,
                q_mode=q_mode,
                divergence=divergence,
                sample_rate=s,
            )
            reason_suffix = (
                f"signal=line_length, q_mode={q_mode}, bins={dist_bins}"
            )
        curve = curve_arr.tolist()
        route = {
            "selected": f"Q_{divergence.upper()}",
            "coverage": None,
            "reason": (
                f"Q-aligned {divergence.upper()} divergence ({reason_suffix})"
            ),
        }

    if not curve:
        out = _empty_result(mode, s)
        out["route"] = route
        if meta and mode == "line":
            out["n_lines"] = meta.get("n_lines", 0)
        return out

    # ── 2) Peak selection — threshold mapped to equivalent quantile ─
    def _scale(v: int) -> int:
        return max(1, int(v) // s) if s > 1 else int(v)

    curve_arr_np = np.asarray(curve, dtype=np.float64)
    effective_q = _effective_quantile(
        curve_arr_np,
        threshold_mode=threshold_mode,
        fallback_quantile=params["quantile"],
        mean_std_k=mean_std_k,
        topk_n=topk_n,
    )

    boundary_list = quantile_peak_select(
        curve,
        quantile=effective_q,
        min_distance=_scale(params["min_distance"]),
        min_prominence=params["min_prominence"],
        nms_radius=_scale(params["nms_radius"]),
        consolidation_radius=_scale(params["consolidation_radius"]),
    )

    # ── 3) Segment assembly (in sampled-curve space) ───────────────
    segments = regime_segment(
        boundary_list,
        min_windows=_scale(params["min_segment_windows"]),
    )

    # ── 4) Rescale to user-unit (char or line) ─────────────────────
    if s > 1:
        for seg in segments:
            seg["start_idx"] = int(seg["start_idx"] * s)
            seg["end_idx"] = int(seg["end_idx"] * s)
            seg["n_windows"] = seg["end_idx"] - seg["start_idx"]

    # ── 5) Shape output per mode ───────────────────────────────────
    if mode == "char":
        boundary_positions = [
            int(seg["start_idx"]) for seg in segments
            if int(seg["start_idx"]) > 0
        ]
        out: Dict[str, Any] = {
            "mode": "char",
            "sample_rate": s,
            "n_chars": len(text),
            "route": route,
            "n_boundaries": len(boundary_positions),
            "boundaries": boundary_positions,
            "n_segments": len(segments),
            "segments": segments,
        }
    else:
        line_char_offsets = (meta or {}).get("line_char_offsets", [])
        n_lines = (meta or {}).get("n_lines", 0)

        def _char_at(line_idx: int) -> int:
            if not line_char_offsets:
                return 0
            lo = min(max(0, line_idx), len(line_char_offsets) - 1)
            return int(line_char_offsets[lo])

        line_segments = []
        for seg in segments:
            sl = int(seg["start_idx"])
            el = int(seg["end_idx"])
            line_segments.append({
                "start_line": sl,
                "end_line": el,
                "n_lines": el - sl,
                "char_start_approx": _char_at(sl),
                "char_end_approx": _char_at(el),
                "label": seg.get("label", ""),
            })
        boundary_positions = [
            int(x["start_line"]) for x in line_segments
            if int(x["start_line"]) > 0
        ]

        # ── 5.5) Local multiscale boundary refinement (line mode only) ──
        # Coarse line-level boundaries can land in the right transition
        # zone but a few lines early / late. The refinement below
        # recomputes a character-class JSD at several scales inside a
        # local window around each coarse boundary and moves the
        # boundary to the multiscale argmax. Raw boundaries remain
        # available under ``raw_boundaries``.
        boundary_details = None
        refinement_meta = None
        raw_boundary_positions = list(boundary_positions)
        if refine and boundary_positions:
            from refine import refine_boundaries_local_multiscale
            refinement = refine_boundaries_local_multiscale(
                text, boundary_positions,
                radius_lines=int(refine_radius_lines),
            )
            new_boundaries = refinement["refined_boundaries"]
            if new_boundaries:
                # Rebuild line_segments from refined boundaries so
                # segments and boundaries stay consistent.
                line_segments = _line_segments_from_boundaries(
                    new_boundaries, n_lines, line_char_offsets,
                    min_windows=int(params["min_segment_windows"]),
                )
                boundary_positions = [
                    int(x["start_line"]) for x in line_segments
                    if int(x["start_line"]) > 0
                ]
            boundary_details = refinement["boundary_details"]
            refinement_meta = refinement["refinement"]

        # ── 5.7) Edge-segment cleanup ───────────────────────────────
        # After any refinement, drop boundaries that produce trivially
        # short edge segments (first or last). Conservative: only
        # touches edges; never collapses an interior boundary.
        edge_cleanup_meta = None
        if edge_cleanup and boundary_positions and n_lines > 0:
            cleaned, dropped = _cleanup_edge_boundaries(
                boundary_positions, n_lines,
            )
            if dropped:
                boundary_positions = cleaned
                line_segments = _line_segments_from_boundaries(
                    cleaned, n_lines, line_char_offsets,
                    min_windows=int(params["min_segment_windows"]),
                )
            edge_cleanup_meta = {
                "enabled": True,
                "threshold_lines": _edge_cleanup_threshold(n_lines),
                "dropped_boundaries": dropped,
            }

        out = {
            "mode": "line",
            "sample_rate": s,
            "n_lines": n_lines,
            "route": route,
            "n_boundaries": len(boundary_positions),
            "boundaries": boundary_positions,
            "n_segments": len(line_segments),
            "segments": line_segments,
        }
        if refinement_meta is not None:
            out["raw_boundaries"] = raw_boundary_positions
            out["boundary_details"] = boundary_details
            out["boundary_refinement"] = refinement_meta
        if edge_cleanup_meta is not None:
            out["edge_cleanup"] = edge_cleanup_meta

    # ── 6) Diagnostics (additive) ──────────────────────────────────
    out["strategy"] = strategy
    out["scoring"] = scoring
    out["threshold_mode"] = threshold_mode
    if scoring != "heuristic":
        out["q_mode"] = q_mode
        out["divergence"] = "jsd" if scoring == "q_jsd" else "kl"
        out["signal_type"] = signal_type
        if signal_type == "tokens":
            out["vocab_size"] = int(token_meta.get("vocab_size", 0))
            out["avg_tokens_per_line"] = float(
                token_meta.get("avg_tokens_per_line", 0.0)
            )
            out["sparsity"] = float(token_meta.get("sparsity", 1.0))
            out["min_token_freq"] = int(min_token_freq)
        else:
            out["dist_bins"] = int(dist_bins)
    out["effective_quantile"] = round(float(effective_q), 6)
    if curve_arr_np.size:
        out["curve_stats"] = {
            "mean": float(curve_arr_np.mean()),
            "std": float(curve_arr_np.std()),
            "min": float(curve_arr_np.min()),
            "max": float(curve_arr_np.max()),
        }

    if include_curve:
        out["curve"] = curve

    # ── 7) Optional structural interpretation ─────────────────────
    # Additive: when ``interpret=False`` (default), existing output
    # shape is preserved bit-identically. When enabled, ``segments``
    # is replaced with the richer structural schema and an
    # ``interpretation`` metadata block is added.
    if interpret:
        from interpretation import interpret_segments
        # boundaries are in user-unit space (char or line); curve is in
        # sampled-curve space. Map boundaries into curve-index space for
        # slicing, then scale back before returning.
        curve_boundaries = [
            int(b) // s for b in boundary_positions
        ]
        interp = interpret_segments(curve, curve_boundaries)
        if s > 1:
            for seg in interp["segments"]:
                seg["start"] = int(seg["start"]) * s
                seg["end"] = int(seg["end"]) * s
                seg["length"] = int(seg["length"]) * s
        out["segments"] = interp["segments"]
        out["n_segments"] = len(interp["segments"])
        out["interpretation"] = interp["interpretation"]

    return out
