"""Lexical instability signals: JSD and predictive KL on token-window
distributions, one value per character position.

Extracted bit-exact from mentis_ui/node_adapters/kalem_adapters.py
(lines 1967-2106). Pure numpy.

Optimisations applied (bit-exact preserving):
 - _window_dist replaced by np.unique + np.bincount
 - tokenisation shared between JSD and predictive-KL via
   :func:`compute_lexical_jk`
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .similarity import jsd as _jsd


def _tokenize(text: str):
    """Produce (tokens_list, char_to_tok_arr, tok_to_vocab_arr, V, N).

    char_to_tok_arr[i] = index of the token that char i belongs to.
    tok_to_vocab_arr[t] = vocab index of tokens_list[t].
    Matches the sequential-find scheme from the reference adapter.
    """
    N = len(text)
    lower = text.lower()
    tokens_list = lower.split()

    char_to_tok = [0] * N  # default 0 (= first token ti)
    pos = 0
    for ti, tok in enumerate(tokens_list):
        idx = lower.find(tok, pos)
        if idx < 0:
            idx = pos
        end = min(idx + len(tok), N)
        for c in range(idx, end):
            char_to_tok[c] = ti
        pos = idx + len(tok)

    vocab = sorted(set(tokens_list))
    vocab_idx = {t: i for i, t in enumerate(vocab)}
    V = len(vocab)

    char_to_tok_arr = np.asarray(char_to_tok, dtype=np.int64)
    if tokens_list:
        tok_to_vocab_arr = np.asarray(
            [vocab_idx[t] for t in tokens_list], dtype=np.int64
        )
    else:
        tok_to_vocab_arr = np.zeros(0, dtype=np.int64)

    return tokens_list, char_to_tok_arr, tok_to_vocab_arr, V, N


def _window_dist_vec(
    start: int,
    end: int,
    char_to_tok_arr: np.ndarray,
    tok_to_vocab_arr: np.ndarray,
    V: int,
    N: int,
    eps: float = 1e-12,
) -> np.ndarray:
    """Probability distribution over vocab for chars in [start, end).

    Bit-exact replacement for the reference set+loop version:
      - Unique token positions in window (set behaviour).
      - Each distinct ti contributes +1 at vocab_idx[tokens_list[ti]].
        (Note: distinct ti's may map to the same vocab index, in which
        case counts is summed — matches the reference.)
    """
    lo, hi = max(0, start), min(N, end)
    if lo >= hi or V == 0:
        counts = np.full(V if V else 1, eps, dtype=np.float64)
        return counts / counts.sum()

    unique_ti = np.unique(char_to_tok_arr[lo:hi])
    L = tok_to_vocab_arr.shape[0]
    valid_ti = unique_ti[(unique_ti >= 0) & (unique_ti < L)]
    if valid_ti.size == 0:
        counts = np.full(V, eps, dtype=np.float64)
        return counts / counts.sum()

    vocab_ids = tok_to_vocab_arr[valid_ti]
    counts = np.bincount(vocab_ids, minlength=V).astype(np.float64)
    counts += eps
    return counts / counts.sum()


def _jsd_bit_exact(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """Bit-exact JSD matching mentis_kalem.similarity.jsd.

    The reference applies :func:`_as_prob` to each input before the JSD
    computation — it re-adds eps and re-normalises even if the caller
    already passed a valid probability distribution. We replicate that
    here inline (faster than two function calls) so the curve matches
    the reference to the last float bit.
    """
    p_ = p + eps
    p_ = p_ / p_.sum()
    q_ = q + eps
    q_ = q_ / q_.sum()
    m = 0.5 * (p_ + q_)
    return float(
        0.5 * np.sum(p_ * np.log(p_ / m)) + 0.5 * np.sum(q_ * np.log(q_ / m))
    )


def lexical_jsd(text: str, *, window: int = 100) -> List[float]:
    """Per-character JSD between left-window and right-window token distributions.

    Returns a list of length N (len(text)). Positions 0 and N-1 are 0.0.
    """
    if not isinstance(text, str) or not text:
        return []

    tokens_list, c2t, t2v, V, N = _tokenize(text)
    if V == 0:
        return [0.0] * N

    w = int(window)
    out = np.zeros(N, dtype=np.float64)
    for i in range(1, N - 1):
        left = _window_dist_vec(i - w, i, c2t, t2v, V, N)
        right = _window_dist_vec(i, i + w, c2t, t2v, V, N)
        out[i] = _jsd_bit_exact(left, right)
    return out.tolist()


def lexical_predictive_kl(
    text: str,
    *,
    window: int = 100,
    mode: str = "forward",
) -> List[float]:
    """Per-character predictive KL between left/right windows.

    mode = "forward":  KL(p_right || p_left)
         = "backward": KL(p_left  || p_right)
         = "bidirectional": 0.5 * (forward + backward)
    """
    if not isinstance(text, str) or not text:
        return []

    tokens_list, c2t, t2v, V, N = _tokenize(text)
    if V == 0:
        return [0.0] * N

    w = int(window)

    def _kl(p: np.ndarray, q: np.ndarray) -> float:
        return float(np.sum(p * np.log(p / q)))

    out = np.zeros(N, dtype=np.float64)
    for i in range(1, N - 1):
        p_left = _window_dist_vec(i - w, i, c2t, t2v, V, N)
        p_right = _window_dist_vec(i, i + w, c2t, t2v, V, N)
        if mode == "forward":
            out[i] = _kl(p_right, p_left)
        elif mode == "backward":
            out[i] = _kl(p_left, p_right)
        else:
            out[i] = 0.5 * _kl(p_right, p_left) + 0.5 * _kl(p_left, p_right)
    return out.tolist()


def compute_lexical_jk(
    text: str,
    *,
    window: int = 100,
    kl_mode: str = "forward",
    sample_rate: int = 1,
) -> Tuple[List[float], List[float]]:
    """Compute JSD curve and predictive-KL curve sharing tokenisation.

    Parameters
    ----------
    text : str
    window : int
        Window size in characters (each side).
    kl_mode : str
        "forward" (default), "backward", or "bidirectional".
    sample_rate : int
        Evaluate the curve only at every N-th character position
        (default 1 = every position, matches reference). Values > 1
        reduce runtime to O(N / sample_rate) at the cost of resolution.

    Returns
    -------
    (J, K) : each a list of length ceil(N / sample_rate) when
    sample_rate > 1, otherwise length N. Positions corresponding to
    i == 0 or i == N-1 in the original text are 0.0.

    When sample_rate == 1 the output is bit-exact vs the reference
    pipeline.
    """
    if not isinstance(text, str) or not text:
        return [], []

    tokens_list, c2t, t2v, V, N = _tokenize(text)
    if V == 0:
        return [0.0] * N, [0.0] * N

    w = int(window)
    s = max(1, int(sample_rate))

    def _kl(p: np.ndarray, q: np.ndarray) -> float:
        return float(np.sum(p * np.log(p / q)))

    eval_positions = list(range(0, N, s))
    n_eval = len(eval_positions)
    J = np.zeros(n_eval, dtype=np.float64)
    K = np.zeros(n_eval, dtype=np.float64)

    for k, i in enumerate(eval_positions):
        if i < 1 or i >= N - 1:
            continue
        p_left = _window_dist_vec(i - w, i, c2t, t2v, V, N)
        p_right = _window_dist_vec(i, i + w, c2t, t2v, V, N)
        J[k] = _jsd_bit_exact(p_left, p_right)
        if kl_mode == "forward":
            K[k] = _kl(p_right, p_left)
        elif kl_mode == "backward":
            K[k] = _kl(p_left, p_right)
        else:
            K[k] = 0.5 * _kl(p_right, p_left) + 0.5 * _kl(p_left, p_right)

    return J.tolist(), K.tolist()
