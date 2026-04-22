"""Experimental Q-alignment scoring for segmentation.

Conceptual model
----------------
For each evaluation position i:
    P_i = local distribution over a window of the input signal
    Q   = reference distribution (global / rolling / prefix)
    d_i = divergence(P_i, Q)

The divergence curve is then passed into the existing peak-selection
machinery unchanged — this module only produces a curve.

This is an additive experimental path. The default (``scoring =
"heuristic"``) is preserved exactly.

All functions reuse the numerically-stable JSD / KL from
``primitives.similarity`` so Q-alignment shares the same eps-smoothing
behaviour as the reference pipeline.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .similarity import jsd as _jsd_core, kl_divergence as _kl_core


# ── Token-based signal path ───────────────────────────────────────
#
# The numeric (line-length) path above is a mathematically valid but
# semantically weak signal on logs (log-level shifts don't show up in
# length). The token path below compares the *vocabulary distribution*
# of a window to a reference vocabulary distribution Q, which is where
# the actual regime signal lives for logs.
#
# No scipy / sklearn / heavy NLP deps — pure regex + numpy.


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize_line(line: str) -> List[str]:
    """Deterministic tokenizer: extract ``[A-Za-z0-9_]+`` runs, lowercase.

    Case-folding makes ``INFO`` and ``info`` collide, which is the point —
    severity prefixes should contribute to the same token slot. Empty
    lines return ``[]``.
    """
    if not line:
        return []
    return [t.lower() for t in _TOKEN_RE.findall(line)]


def build_vocab(
    lines: List[str],
    *,
    min_freq: int = 1,
) -> Dict[str, int]:
    """Build a stable sorted vocab from a list of lines.

    ``min_freq`` prunes rare tokens — set to 2 to drop hapax legomena
    (tokens occurring exactly once), which is common for logs with
    unique request-IDs / trace-IDs.
    """
    counts: Counter = Counter()
    for line in lines:
        counts.update(tokenize_line(line))
    tokens = sorted(t for t, c in counts.items() if c >= max(1, int(min_freq)))
    return {t: i for i, t in enumerate(tokens)}


def line_to_vector(line: str, vocab: Dict[str, int]) -> np.ndarray:
    """Bag-of-words count vector over ``vocab``. Shape ``(|vocab|,)``, float32."""
    V = len(vocab)
    vec = np.zeros(V, dtype=np.float32)
    if V == 0:
        return vec
    for t in tokenize_line(line):
        idx = vocab.get(t)
        if idx is not None:
            vec[idx] += 1.0
    return vec


# Memory-safety threshold: lines_to_matrix refuses to allocate beyond
# this (float32 bytes). compute_qalign_curve_tokens does NOT use it —
# this threshold only guards the explicit dense helper.
_DENSE_MEM_THRESHOLD_BYTES = 2 * (1 << 30)  # 2 GiB


def _estimate_dense_bytes(
    n_lines: int,
    vocab_size: int,
    dtype_bytes: int = 4,
) -> int:
    """Rough cost of an ``(n_lines, vocab_size)`` dense matrix."""
    return int(n_lines) * int(vocab_size) * int(dtype_bytes)


def lines_to_matrix(
    lines: List[str],
    vocab: Dict[str, int],
) -> np.ndarray:
    """Stack per-line vectors into an ``(n_lines, |vocab|)`` float32 matrix.

    Raises :class:`MemoryError` when the estimated allocation exceeds
    :data:`_DENSE_MEM_THRESHOLD_BYTES`. Large logs must go through
    :func:`compute_qalign_curve_tokens`, which uses a sparse streaming
    implementation.
    """
    V = len(vocab)
    N = len(lines)
    est = _estimate_dense_bytes(N, V)
    if est > _DENSE_MEM_THRESHOLD_BYTES:
        raise MemoryError(
            f"lines_to_matrix: a dense (N={N}, V={V}) float32 matrix "
            f"would need ~{est / (1 << 30):.2f} GiB, exceeding the "
            f"{_DENSE_MEM_THRESHOLD_BYTES / (1 << 30):.1f} GiB safety "
            f"threshold. compute_qalign_curve_tokens uses a sparse "
            f"streaming implementation that does not require this "
            f"allocation."
        )
    M = np.zeros((N, V), dtype=np.float32)
    if V == 0:
        return M
    for i, line in enumerate(lines):
        for t in tokenize_line(line):
            idx = vocab.get(t)
            if idx is not None:
                M[i, idx] += 1.0
    return M


# ── Sparse CSR builder + rolling-window helpers ─────────────────────
#
# compute_qalign_curve_tokens operates on per-line token counts in a
# compact sparse CSR-like layout:
#   data   : int32[nnz]   vocab indices
#   counts : int32[nnz]   occurrences of that vocab index in the line
#   indptr : int64[N+1]   start offsets into data / counts per line
# plus three V-dimensional int64 running-count buffers (P, outer,
# prefix) updated incrementally via np.add.at / np.subtract.at. This
# keeps memory at O(nnz + V) instead of O(N·V).

def _build_csr_counts(
    lines: List[str], vocab: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Build (data, counts, indptr, total_tokens) sparse layout."""
    n = len(lines)
    from collections import Counter as _Counter
    per_line: List[Counter] = []
    total_entries = 0
    total_tokens = 0
    for line in lines:
        c: _Counter = _Counter()
        for t in tokenize_line(line):
            idx = vocab.get(t)
            if idx is not None:
                c[idx] += 1
        per_line.append(c)
        total_entries += len(c)
        total_tokens += sum(c.values())

    data = np.empty(total_entries, dtype=np.int32)
    counts = np.empty(total_entries, dtype=np.int32)
    indptr = np.zeros(n + 1, dtype=np.int64)
    cursor = 0
    for i, c in enumerate(per_line):
        for k, v in c.items():
            data[cursor] = k
            counts[cursor] = v
            cursor += 1
        indptr[i + 1] = cursor
    return data, counts, indptr, total_tokens


def _add_line_to_buf(
    buf: np.ndarray,
    line_idx: int,
    data_arr: np.ndarray,
    counts_arr: np.ndarray,
    indptr_arr: np.ndarray,
) -> None:
    """Increment ``buf`` (int64 V-dim) by the counts of line ``line_idx``."""
    lo = int(indptr_arr[line_idx])
    hi = int(indptr_arr[line_idx + 1])
    if lo < hi:
        # Vocab indices within one line are unique (Counter dedup), so
        # fancy-indexed += is safe here. np.add.at is equivalent but
        # ~3-5x slower; we use the direct form.
        buf[data_arr[lo:hi]] += counts_arr[lo:hi]


def _remove_line_from_buf(
    buf: np.ndarray,
    line_idx: int,
    data_arr: np.ndarray,
    counts_arr: np.ndarray,
    indptr_arr: np.ndarray,
) -> None:
    lo = int(indptr_arr[line_idx])
    hi = int(indptr_arr[line_idx + 1])
    if lo < hi:
        buf[data_arr[lo:hi]] -= counts_arr[lo:hi]


def _jsd_dense(P_buf: np.ndarray, Q_buf: np.ndarray, V: int,
               eps: float) -> float:
    sum_p = float(P_buf.sum()) + V * eps
    sum_q = float(Q_buf.sum()) + V * eps
    if sum_p <= 0.0 or sum_q <= 0.0:
        return 0.0
    p = (P_buf.astype(np.float64) + eps) / sum_p
    q = (Q_buf.astype(np.float64) + eps) / sum_q
    m = 0.5 * (p + q)
    return float(
        0.5 * np.sum(p * np.log(p / m))
        + 0.5 * np.sum(q * np.log(q / m))
    )


def _kl_dense(P_buf: np.ndarray, Q_buf: np.ndarray, V: int,
              eps: float) -> float:
    sum_p = float(P_buf.sum()) + V * eps
    sum_q = float(Q_buf.sum()) + V * eps
    if sum_p <= 0.0 or sum_q <= 0.0:
        return 0.0
    p = (P_buf.astype(np.float64) + eps) / sum_p
    q = (Q_buf.astype(np.float64) + eps) / sum_q
    return float(np.sum(p * np.log(p / q)))


def compute_qalign_curve_tokens(
    lines: List[str],
    *,
    window: int = 10,
    q_mode: str = "global",
    divergence: str = "jsd",
    sample_rate: int = 1,
    min_token_freq: int = 1,
    rolling_extra: int = 3,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Compute Q-aligned divergence curve on token distributions per line.

    Sparse streaming implementation. Memory footprint is
    ``O(nnz + V)`` — it never builds an ``(N, V)`` dense matrix.

    Returns ``(curve, meta)`` where ``meta`` contains
    ``vocab_size``, ``avg_tokens_per_line``, ``sparsity``,
    and ``dense_matrix_avoided_bytes`` (for telemetry).
    """
    if divergence not in ("jsd", "kl"):
        raise ValueError(
            f"Unknown divergence {divergence!r}; expected 'jsd' or 'kl'"
        )
    if q_mode not in ("global", "rolling", "prefix"):
        raise ValueError(
            f"Unknown q_mode {q_mode!r}; "
            f"expected 'global'|'rolling'|'prefix'"
        )

    N = len(lines)
    s = max(1, int(sample_rate))
    n_eval = len(range(0, N, s))

    vocab = build_vocab(lines, min_freq=min_token_freq)
    V = len(vocab)

    if V == 0 or N < 2:
        return (
            np.zeros(max(1, n_eval), dtype=np.float64),
            {
                "vocab_size": V,
                "avg_tokens_per_line": 0.0,
                "sparsity": 1.0,
                "dense_matrix_avoided_bytes": _estimate_dense_bytes(N, V),
            },
        )

    data, counts_arr, indptr, total_tokens = _build_csr_counts(lines, vocab)
    meta: Dict[str, Any] = {
        "vocab_size": V,
        "avg_tokens_per_line": total_tokens / float(N) if N else 0.0,
        "sparsity": 1.0 - (int(data.size) / float(max(1, N * V))),
        "dense_matrix_avoided_bytes": _estimate_dense_bytes(N, V),
    }

    # ── Rolling V-dim buffers (int64 counts, reused each step) ──────
    P_buf = np.zeros(V, dtype=np.int64)
    inner_left = 0
    inner_right = 0  # exclusive

    # Global Q once (used for q_mode="global" or as degenerate fallback).
    # np.bincount with int weights returns float64 — cast to int64.
    Q_global = np.bincount(
        data, weights=counts_arr, minlength=V,
    ).astype(np.int64)

    outer_buf: Optional[np.ndarray] = None
    outer_left = outer_right = 0
    Q_rolling_buf: Optional[np.ndarray] = None
    prefix_buf: Optional[np.ndarray] = None
    prefix_end = 0

    if q_mode == "rolling":
        outer_buf = np.zeros(V, dtype=np.int64)
        Q_rolling_buf = np.zeros(V, dtype=np.int64)
    elif q_mode == "prefix":
        prefix_buf = np.zeros(V, dtype=np.int64)

    div_fn = _jsd_dense if divergence == "jsd" else _kl_dense

    curve = np.zeros(n_eval, dtype=np.float64)
    for k, i in enumerate(range(0, N, s)):
        if i < 1 or i >= N - 1:
            continue

        # ── Advance P (inner) window to [i-w, i+w) ────────────────
        target_left = max(0, i - window)
        target_right = min(N, i + window)
        while inner_right < target_right:
            _add_line_to_buf(
                P_buf, inner_right, data, counts_arr, indptr,
            )
            inner_right += 1
        while inner_right > target_right:
            inner_right -= 1
            _remove_line_from_buf(
                P_buf, inner_right, data, counts_arr, indptr,
            )
        while inner_left < target_left:
            _remove_line_from_buf(
                P_buf, inner_left, data, counts_arr, indptr,
            )
            inner_left += 1
        while inner_left > target_left:
            inner_left -= 1
            _add_line_to_buf(
                P_buf, inner_left, data, counts_arr, indptr,
            )

        # ── Pick Q according to mode ──────────────────────────────
        if q_mode == "global":
            Q = Q_global
        elif q_mode == "rolling":
            # Outer window [i - w*rolling_extra, i + w*rolling_extra).
            outer_size = max(1, window) * max(1, rolling_extra)
            out_target_left = max(0, i - outer_size)
            out_target_right = min(N, i + outer_size)
            while outer_right < out_target_right:
                _add_line_to_buf(
                    outer_buf, outer_right, data, counts_arr, indptr,
                )
                outer_right += 1
            while outer_right > out_target_right:
                outer_right -= 1
                _remove_line_from_buf(
                    outer_buf, outer_right, data, counts_arr, indptr,
                )
            while outer_left < out_target_left:
                _remove_line_from_buf(
                    outer_buf, outer_left, data, counts_arr, indptr,
                )
                outer_left += 1
            while outer_left > out_target_left:
                outer_left -= 1
                _add_line_to_buf(
                    outer_buf, outer_left, data, counts_arr, indptr,
                )
            # Q_rolling = outer - inner, in-place reuse of Q_rolling_buf
            np.subtract(outer_buf, P_buf, out=Q_rolling_buf)
            if int(Q_rolling_buf.sum()) <= 0:
                Q = Q_global
            else:
                Q = Q_rolling_buf
        else:  # prefix
            # prefix_buf accumulates lines [0, i) monotonically
            while prefix_end < i:
                _add_line_to_buf(
                    prefix_buf, prefix_end, data, counts_arr, indptr,
                )
                prefix_end += 1
            if i <= 1 or int(prefix_buf.sum()) <= 0:
                Q = Q_global
            else:
                Q = prefix_buf

        curve[k] = div_fn(P_buf, Q, V, eps)

    return curve, meta


# ── Distribution construction ─────────────────────────────────────

def row_to_distribution(
    values,
    *,
    method: str = "hist",
    bins: int = 32,
    eps: float = 1e-9,
    value_range: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """Convert a 1-D numeric array into a normalised probability distribution.

    Parameters
    ----------
    values :
        1-D numeric array or anything numpy can coerce. Empty → uniform.
    method :
        Only ``"hist"`` is implemented. Kept as a keyword to leave room
        for future methods (kde, softmax) without re-shaping the API.
    bins :
        Histogram bin count (default 32).
    eps :
        Smoothing constant added to every bin so the output has no
        exact zeros (safe for downstream log / KL / JSD).
    value_range :
        Optional ``(lo, hi)`` fixing the histogram range. When Q and
        multiple P_i are compared, they MUST share the same range —
        otherwise bins are not aligned. The caller passes the
        shared range.

    Returns
    -------
    ndarray of shape (bins,) summing to 1.

    Constant input: the numeric range collapses to zero width. We
    synthesise a symmetric range of width 1 around the value so the
    histogram still works and returns a valid degenerate-smoothed
    distribution (does not crash).
    """
    if method != "hist":
        raise ValueError(f"Unknown method {method!r}; only 'hist' supported")

    arr = np.asarray(values, dtype=np.float64).ravel()
    if arr.size == 0:
        return np.full(bins, 1.0 / bins, dtype=np.float64)

    if value_range is None:
        lo = float(arr.min())
        hi = float(arr.max())
    else:
        lo, hi = float(value_range[0]), float(value_range[1])

    if hi - lo < 1e-12:
        # Constant / near-constant — widen range symmetrically so
        # np.histogram has a positive-width range and produces a
        # single-populated bin that still normalises cleanly.
        lo -= 0.5
        hi += 0.5

    hist, _ = np.histogram(arr, bins=bins, range=(lo, hi))
    p = hist.astype(np.float64) + eps
    return p / p.sum()


# ── Divergence helpers ────────────────────────────────────────────

def js_divergence(p, q, eps: float = 1e-9) -> float:
    """Jensen-Shannon divergence JSD(P || Q).

    Thin wrapper over :func:`primitives.similarity.jsd` so this module
    is self-contained for readers, but numerics are shared.
    """
    return _jsd_core(np.asarray(p), np.asarray(q), eps=eps)


def kl_divergence(p, q, eps: float = 1e-9) -> float:
    """Kullback-Leibler divergence D_KL(P || Q)."""
    return _kl_core(np.asarray(p), np.asarray(q), eps=eps)


# ── Q construction strategies ─────────────────────────────────────

def build_q(
    signal: np.ndarray,
    *,
    q_mode: str = "global",
    i: int = 0,
    window: int = 100,
    rolling_extra: int = 3,
    bins: int = 32,
    value_range: Optional[Tuple[float, float]] = None,
    eps: float = 1e-9,
) -> np.ndarray:
    """Construct the reference distribution Q.

    q_mode
    ------
    ``"global"`` :
        Q from the full signal. Position-independent. Cache-friendly.
    ``"rolling"`` :
        Q from a neighbourhood of size ``window * rolling_extra``
        centred on ``i``, EXCLUDING the local core ``[i-window, i+window]``
        so P_i is not compared to itself.
    ``"prefix"`` :
        Q from ``signal[:i]`` — content before the current position.
        Useful for change-point interpretation: "is the tail different
        from what came before?"
    """
    N = signal.shape[0]

    if q_mode == "global":
        return row_to_distribution(
            signal, bins=bins, value_range=value_range, eps=eps,
        )

    if q_mode == "rolling":
        outer = int(max(1, window) * max(1, rolling_extra))
        lo = max(0, i - outer)
        hi = min(N, i + outer)
        core_lo = max(0, i - window)
        core_hi = min(N, i + window)
        parts = []
        if lo < core_lo:
            parts.append(signal[lo:core_lo])
        if core_hi < hi:
            parts.append(signal[core_hi:hi])
        neighborhood = (
            np.concatenate(parts) if parts else signal[lo:hi]
        )
        if neighborhood.size == 0:
            return row_to_distribution(
                signal, bins=bins, value_range=value_range, eps=eps,
            )
        return row_to_distribution(
            neighborhood, bins=bins, value_range=value_range, eps=eps,
        )

    if q_mode == "prefix":
        if i <= 1:
            return row_to_distribution(
                signal, bins=bins, value_range=value_range, eps=eps,
            )
        return row_to_distribution(
            signal[:i], bins=bins, value_range=value_range, eps=eps,
        )

    raise ValueError(
        f"Unknown q_mode {q_mode!r}; expected 'global'|'rolling'|'prefix'"
    )


# ── Full curve ────────────────────────────────────────────────────

def compute_qalign_curve(
    signal,
    *,
    window: int = 100,
    bins: int = 32,
    q_mode: str = "global",
    divergence: str = "jsd",
    sample_rate: int = 1,
    rolling_extra: int = 3,
    eps: float = 1e-9,
) -> np.ndarray:
    """Divergence curve d(i) = divergence(P_i, Q) over the signal.

    The curve is length ``ceil(N / sample_rate)``. Positions i with
    i < 1 or i >= N-1 emit 0.0 (no meaningful window).

    ``window`` is in the same unit as ``signal`` (chars or lines).
    ``bins`` is shared across all histograms; all histograms share the
    same (signal.min, signal.max) value-range so P_i and Q are directly
    comparable.
    """
    sig = np.asarray(signal, dtype=np.float64).ravel()
    N = sig.size
    s = max(1, int(sample_rate))

    n_eval = len(range(0, N, s))
    if N < 2:
        return np.zeros(max(1, n_eval), dtype=np.float64)

    lo = float(sig.min())
    hi = float(sig.max())
    if hi - lo < 1e-12:
        lo -= 0.5
        hi += 0.5
    value_range = (lo, hi)

    # Precompute Q_global once — also used as a fallback for degenerate
    # "rolling" / "prefix" edges.
    Q_global = row_to_distribution(
        sig, bins=bins, value_range=value_range, eps=eps,
    )
    if divergence == "jsd":
        div_fn = js_divergence
    elif divergence == "kl":
        div_fn = kl_divergence
    else:
        raise ValueError(
            f"Unknown divergence {divergence!r}; expected 'jsd' or 'kl'"
        )

    curve = np.zeros(n_eval, dtype=np.float64)
    eval_positions = range(0, N, s)
    for k, i in enumerate(eval_positions):
        if i < 1 or i >= N - 1:
            continue
        lo_i = max(0, i - window)
        hi_i = min(N, i + window)
        P_i = row_to_distribution(
            sig[lo_i:hi_i], bins=bins, value_range=value_range, eps=eps,
        )
        if q_mode == "global":
            Q = Q_global
        else:
            Q = build_q(
                sig, q_mode=q_mode, i=i, window=window,
                rolling_extra=rolling_extra, bins=bins,
                value_range=value_range, eps=eps,
            )
        curve[k] = div_fn(P_i, Q, eps=eps)

    return curve


# ── Signal builders (text → 1-D numeric) ──────────────────────────

def char_byte_signal(text: str) -> np.ndarray:
    """Per-character byte value in [0, 1]. Mirror of plot._byte_signal."""
    if not text:
        return np.zeros(0, dtype=np.float64)
    arr = np.empty(len(text), dtype=np.float64)
    for i, ch in enumerate(text):
        b = ch.encode("utf-8", errors="replace")
        arr[i] = b[0] / 255.0 if b else 0.0
    return arr


def line_length_signal(text: str) -> np.ndarray:
    """Per-line length in [0, 1]. Mirror of plot._line_length_signal."""
    if not text:
        return np.zeros(0, dtype=np.float64)
    lines = text.splitlines()
    if not lines:
        return np.zeros(0, dtype=np.float64)
    lengths = np.asarray([len(line) for line in lines], dtype=np.float64)
    m = float(lengths.max()) if lengths.size else 1.0
    if m < 1.0:
        m = 1.0
    return lengths / m


__all__ = [
    "row_to_distribution",
    "js_divergence",
    "kl_divergence",
    "build_q",
    "compute_qalign_curve",
    "char_byte_signal",
    "line_length_signal",
    # token-based path
    "tokenize_line",
    "build_vocab",
    "line_to_vector",
    "lines_to_matrix",
    "compute_qalign_curve_tokens",
]
