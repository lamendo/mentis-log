"""Line-level analysis unit for large logs.

Same conceptual pipeline as the char-level path — local distributions,
divergence, policy, peaks, segmentation — but the base unit is one
log line instead of one character position.

For a 16 MB log (~160k lines × ~100 chars/line), char mode is
O(N_chars · window) ≈ 10^10 ops; line mode is O(N_lines · window) with
window in line units (default 10) ≈ 10^6 ops — roughly four orders of
magnitude less work at the cost of coarser resolution.

This is a *different operating mode*, not a parity-preserving
optimisation. Boundary positions in line mode are line indices, not
character offsets.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np


def _line_tokenize(text: str):
    """Split text into lines; build CSR-like (data, indptr) of word ids.

    Returns
    -------
    lines : list[str]
        The physical lines (no trailing newline).
    line_char_offsets : np.ndarray, shape (n_lines + 1,)
        Cumulative character offsets — line i covers chars
        [line_char_offsets[i], line_char_offsets[i+1]).
        Useful for reconstructing approximate char boundaries.
    data : np.ndarray, shape (total_unique_tokens,)
        Concatenated per-line *unique* vocab indices.
    indptr : np.ndarray, shape (n_lines + 1,)
        Start offsets into `data` — line i's unique vocab ids are
        data[indptr[i]:indptr[i+1]].
    V : int
        Vocabulary size (number of unique words across all lines).
    """
    # splitlines drops trailing newlines; we also need char offsets for mapping
    raw_lines = text.splitlines()
    n_lines = len(raw_lines)

    # Compute char offsets — assume one \n per line (standard for text logs).
    # We walk the original text to be correct regardless of \r\n etc.
    line_char_offsets = np.zeros(n_lines + 1, dtype=np.int64)
    if n_lines > 0:
        pos = 0
        for i, line in enumerate(raw_lines):
            line_char_offsets[i] = pos
            pos += len(line)
            # Advance past the newline character(s), if any, in the source
            # text. We rely on text[pos] == '\n' or '\r\n' sequences.
            if pos < len(text) and text[pos] == "\r":
                pos += 1
            if pos < len(text) and text[pos] == "\n":
                pos += 1
        line_char_offsets[n_lines] = pos
    # else leaves [0]

    # Two-pass vocab build
    vocab_idx: dict = {}
    per_line_ids: List[np.ndarray] = []
    for line in raw_lines:
        words = line.lower().split()
        ids = []
        seen = set()
        for w in words:
            if w in seen:
                continue
            seen.add(w)
            wi = vocab_idx.get(w)
            if wi is None:
                wi = len(vocab_idx)
                vocab_idx[w] = wi
            ids.append(wi)
        per_line_ids.append(np.asarray(ids, dtype=np.int64))

    V = len(vocab_idx)

    # Build CSR-like (data, indptr)
    total = int(sum(a.size for a in per_line_ids))
    data = np.empty(total, dtype=np.int64)
    indptr = np.zeros(n_lines + 1, dtype=np.int64)
    cursor = 0
    for i, arr in enumerate(per_line_ids):
        data[cursor:cursor + arr.size] = arr
        cursor += arr.size
        indptr[i + 1] = cursor

    return raw_lines, line_char_offsets, data, indptr, V


def _window_dist_line(
    a: int,
    b: int,
    data: np.ndarray,
    indptr: np.ndarray,
    V: int,
    n_lines: int,
    eps: float = 1e-12,
) -> np.ndarray:
    """Probability distribution over vocab for lines in [a, b).

    Counts each vocab id once per line it appears in. A word that occurs
    in multiple lines within the window contributes its count = number
    of lines it appears in.
    """
    lo, hi = max(0, a), min(n_lines, b)
    if lo >= hi or V == 0:
        counts = np.full(V if V else 1, eps, dtype=np.float64)
        return counts / counts.sum()

    slice_ids = data[indptr[lo]:indptr[hi]]
    if slice_ids.size == 0:
        counts = np.full(V, eps, dtype=np.float64)
        return counts / counts.sum()

    counts = np.bincount(slice_ids, minlength=V).astype(np.float64)
    counts += eps
    return counts / counts.sum()


def _jsd_bit_exact(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """Matches mentis_kalem.similarity.jsd: re-applies _as_prob to both."""
    p_ = p + eps
    p_ = p_ / p_.sum()
    q_ = q + eps
    q_ = q_ / q_.sum()
    m = 0.5 * (p_ + q_)
    return float(
        0.5 * np.sum(p_ * np.log(p_ / m)) + 0.5 * np.sum(q_ * np.log(q_ / m))
    )


def compute_lexical_jk_line(
    text: str,
    *,
    window: int = 10,
    sample_rate: int = 1,
) -> Tuple[List[float], List[float], dict]:
    """Compute line-level J (JSD) and K (forward KL) curves.

    Parameters
    ----------
    text : str
        Raw log/text.
    window : int
        Window size in LINES (default 10). Left window is
        lines [i-window, i), right window is lines [i, i+window).
    sample_rate : int
        Evaluate curve every `sample_rate`-th line. Default 1
        (every line).

    Returns
    -------
    J_list : list[float]  length = ceil(n_lines / sample_rate)
    K_list : list[float]  length = same as J
    meta : dict
        {"n_lines": int, "line_char_offsets": list[int],
         "sample_rate": int}

    Positions 0 and n_lines-1 of the logical curve are 0.0 by
    convention (no left or right window).
    """
    if not isinstance(text, str) or not text:
        return [], [], {
            "n_lines": 0, "line_char_offsets": [0], "sample_rate": sample_rate,
        }

    lines, line_char_offsets, data, indptr, V = _line_tokenize(text)
    n_lines = len(lines)
    if V == 0 or n_lines < 2:
        return (
            [0.0] * n_lines, [0.0] * n_lines,
            {"n_lines": n_lines,
             "line_char_offsets": line_char_offsets.tolist(),
             "sample_rate": sample_rate},
        )

    w = int(window)
    s = max(1, int(sample_rate))

    # We evaluate the curve at positions i = 0, s, 2s, ... up to n_lines-1.
    eval_positions = list(range(0, n_lines, s))
    n_eval = len(eval_positions)

    def _kl(p: np.ndarray, q: np.ndarray) -> float:
        return float(np.sum(p * np.log(p / q)))

    J = np.zeros(n_eval, dtype=np.float64)
    K = np.zeros(n_eval, dtype=np.float64)
    for k, i in enumerate(eval_positions):
        if i < 1 or i >= n_lines - 1:
            continue
        left = _window_dist_line(i - w, i, data, indptr, V, n_lines)
        right = _window_dist_line(i, i + w, data, indptr, V, n_lines)
        J[k] = _jsd_bit_exact(left, right)
        K[k] = _kl(right, left)

    return (
        J.tolist(), K.tolist(),
        {"n_lines": n_lines,
         "line_char_offsets": line_char_offsets.tolist(),
         "sample_rate": s,
         "eval_positions_len": n_eval},
    )
