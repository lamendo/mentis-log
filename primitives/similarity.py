"""Information-theoretic primitives.

Extracted bit-exact from mentis_kalem/src/mentis_kalem/similarity.py
(lines 1-113). Pure numpy.
"""
from __future__ import annotations

import numpy as np


def _as_prob(arr: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalise a non-negative array to a valid probability distribution.

    Adds *eps* to every element before normalising so that zero entries
    do not cause log(0) or division-by-zero.
    """
    a = np.asarray(arr, dtype=np.float64).ravel()
    if a.size == 0:
        raise ValueError("Input array must be non-empty")
    if np.any(a < 0):
        raise ValueError("Probability values must be non-negative")
    a = a + eps
    return a / a.sum()


def kl_divergence(p: np.ndarray, q: np.ndarray, *, eps: float = 1e-12) -> float:
    """Kullback-Leibler divergence D_KL(P || Q).

    Returns 0.0 when P == Q. Automatically normalises inputs via _as_prob.
    """
    p_ = _as_prob(p, eps=eps)
    q_ = _as_prob(q, eps=eps)
    if p_.shape != q_.shape:
        raise ValueError(f"Shape mismatch: p has {p_.shape}, q has {q_.shape}")
    return float(np.sum(p_ * np.log(p_ / q_)))


def jsd(p: np.ndarray, q: np.ndarray, *, eps: float = 1e-12) -> float:
    """Jensen-Shannon divergence. Symmetric, bounded in [0, ln 2]."""
    p_ = _as_prob(p, eps=eps)
    q_ = _as_prob(q, eps=eps)
    if p_.shape != q_.shape:
        raise ValueError(f"Shape mismatch: p has {p_.shape}, q has {q_.shape}")
    m = 0.5 * (p_ + q_)
    return float(
        0.5 * np.sum(p_ * np.log(p_ / m)) + 0.5 * np.sum(q_ * np.log(q_ / m))
    )
