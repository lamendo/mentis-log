"""Coverage-based policy router over three lexical divergence candidates.

Extracted bit-exact from mentis_ui/node_adapters/kalem_adapters.py
(lines 2109-2295, specifically exec_kalem_instability_gated_kl,
exec_kalem_instability_hard_gated_kl, exec_kalem_policy_select_projection).

Core law:
    Direction is activated by structural sparsity.
    Coverage of the GKM signal picks the candidate.

    coverage < 0.05  → HGB (hard directional KL, top-k gated)
    coverage < 0.15  → GKM (soft directional KL, JSD-gated)
    else             → LXM (symmetric JSD)

Thresholds (0.05, 0.15) are global constants, NOT tuned per input.

Optimisations applied (bit-exact preserving):
 - J and K are computed once per policy call and reused across all
   three candidate branches via :func:`lexical.compute_lexical_jk`.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .lexical import (
    lexical_jsd,
    lexical_predictive_kl,
    compute_lexical_jk,
)
from .linemode import compute_lexical_jk_line


def _mm(curve: np.ndarray) -> np.ndarray:
    """Min-max normalise to [0, 1]. Returns zeros if degenerate."""
    lo, hi = float(curve.min()), float(curve.max())
    rng = hi - lo
    return (curve - lo) / rng if rng > 1e-12 else np.zeros_like(curve)


def _combine_gated_kl(
    J_arr: np.ndarray,
    K_arr: np.ndarray,
    *,
    alpha: float = 0.7,
) -> np.ndarray:
    """S = mm(J) * (1 + alpha * mm(K)). Operates on already-computed curves."""
    n = min(len(J_arr), len(K_arr))
    J = J_arr[:n]
    K = K_arr[:n]
    return _mm(J) * (1.0 + alpha * _mm(K))


def _combine_hard_gated_kl(
    J_arr: np.ndarray,
    K_arr: np.ndarray,
    *,
    mode: str = "threshold",
    tau_pctile: float = 75.0,
    topk: int = 8,
) -> np.ndarray:
    """Gate K by J > percentile(J[J>0], tau_pctile). Optional top-k mask."""
    n = min(len(J_arr), len(K_arr))
    J = J_arr[:n]
    K = K_arr[:n]
    j_pos = J[J > 0]
    if len(j_pos) == 0:
        return np.zeros(n, dtype=np.float64)
    tau = float(np.percentile(j_pos, tau_pctile))
    mask = J > tau
    S = np.where(mask, K, 0.0)
    if mode == "topk" and int(mask.sum()) > topk:
        masked_vals = S.copy()
        masked_vals[~mask] = -1.0
        threshold_val = float(np.sort(masked_vals)[-topk])
        S = np.where(S >= threshold_val, S, 0.0)
    return S


def gated_kl(
    text: str,
    *,
    alpha: float = 0.7,
    window: int = 100,
    _J: Optional[List[float]] = None,
    _K: Optional[List[float]] = None,
) -> List[float]:
    """Soft-gated KL (GKM) on text.

    S(i) = mm(J(i)) * (1 + alpha * mm(K(i)))

    Accepts precomputed J/K via the _J/_K kwargs to skip redundant work
    when called from :func:`policy_select_projection`.
    """
    if not isinstance(text, str) or not text:
        return []
    if _J is None:
        _J = lexical_jsd(text, window=window)
    if _K is None:
        _K = lexical_predictive_kl(text, window=window, mode="forward")
    J = np.asarray(_J, dtype=np.float64)
    K = np.asarray(_K, dtype=np.float64)
    return _combine_gated_kl(J, K, alpha=alpha).tolist()


def hard_gated_kl(
    text: str,
    *,
    mode: str = "threshold",
    tau_pctile: float = 75.0,
    topk: int = 8,
    window: int = 100,
    _J: Optional[List[float]] = None,
    _K: Optional[List[float]] = None,
) -> List[float]:
    """Hard-gated KL (HGB) on text. Accepts precomputed _J/_K kwargs."""
    if not isinstance(text, str) or not text:
        return []
    if _J is None:
        _J = lexical_jsd(text, window=window)
    if _K is None:
        _K = lexical_predictive_kl(text, window=window, mode="forward")
    J = np.asarray(_J, dtype=np.float64)
    K = np.asarray(_K, dtype=np.float64)
    return _combine_hard_gated_kl(
        J, K, mode=mode, tau_pctile=tau_pctile, topk=topk
    ).tolist()


def policy_select_projection(
    text: str,
    *,
    window: int = 100,
    sample_rate: int = 1,
) -> Dict[str, Any]:
    """Select one of three instability candidates based on GKM coverage.

    Pipeline (matches the text-path of regime_full_stack_v1):
      1. Compute lexical J (JSD) and K (forward KL) ONCE — shared
         tokenisation. Compose GKM = mm(J) * (1 + 0.7 * mm(K)).
      2. coverage = |{i : GKM[i] > 0.5 * max(GKM)}| / len(GKM)
      3. Route:
           coverage < 0.05  → HGB (mode=topk, tau_pctile=75, topk=8)
           coverage < 0.15  → GKM (already computed at step 1)
           else             → LXM (the J curve from step 1)
      4. Emit selected curve + route metadata.

    All three branches reuse the same J, K — no redundant tokenisation
    or divergence passes. Bit-exact equivalent to the reference adapter.

    Returns a dict:
        {"curve_out": list[float], "route_out": {selected, coverage, reason}}
    """
    if not isinstance(text, str) or not text:
        return {
            "curve_out": [],
            "route_out": {
                "selected": "none",
                "coverage": 0,
                "reason": "empty input",
            },
        }

    # Step 1 — one tokenisation, one J, one K, one GKM
    J_list, K_list = compute_lexical_jk(
        text, window=window, kl_mode="forward", sample_rate=sample_rate,
    )
    J = np.asarray(J_list, dtype=np.float64)
    K = np.asarray(K_list, dtype=np.float64)
    gkm_curve = _combine_gated_kl(J, K, alpha=0.7)

    # Step 2 — coverage on the GKM curve
    if gkm_curve.size == 0 or gkm_curve.max() < 1e-12:
        coverage = 0.0
    else:
        coverage = float(
            np.count_nonzero(gkm_curve > 0.5 * gkm_curve.max())
        ) / len(gkm_curve)

    # Step 3 — route
    if coverage < 0.05:
        selected = "HGB"
        reason = "sparse (coverage < 0.05) — event-like — hard directional KL"
        curve = _combine_hard_gated_kl(
            J, K, mode="topk", tau_pctile=75, topk=8
        ).tolist()
    elif coverage < 0.15:
        selected = "GKM"
        reason = "structured (0.05 <= coverage < 0.15) — soft directional KL"
        curve = gkm_curve.tolist()
    else:
        selected = "LXM"
        reason = "dense (coverage >= 0.15) — symmetric JSD"
        curve = J_list  # original unnormalised J

    return {
        "curve_out": curve,
        "route_out": {
            "selected": selected,
            "coverage": round(coverage, 6),
            "reason": reason,
        },
    }


def policy_select_projection_line(
    text: str,
    *,
    window: int = 10,
    sample_rate: int = 1,
) -> Dict[str, Any]:
    """Line-level analogue of :func:`policy_select_projection`.

    Uses :func:`linemode.compute_lexical_jk_line` to produce J (JSD) and
    K (forward KL) at line granularity, then the same three-candidate
    coverage router picks GKM / LXM / HGB.

    Returns a dict:
        curve_out  — selected curve (length ceil(n_lines / sample_rate))
        route_out  — {selected, coverage, reason}
        meta_out   — {mode, n_lines, line_char_offsets, sample_rate}
    """
    if not isinstance(text, str) or not text:
        return {
            "curve_out": [],
            "route_out": {
                "selected": "none", "coverage": 0, "reason": "empty input",
            },
            "meta_out": {
                "mode": "line", "n_lines": 0,
                "line_char_offsets": [0], "sample_rate": sample_rate,
            },
        }

    J_list, K_list, meta = compute_lexical_jk_line(
        text, window=window, sample_rate=sample_rate,
    )
    if not J_list:
        return {
            "curve_out": [],
            "route_out": {
                "selected": "none", "coverage": 0,
                "reason": "no evaluable positions",
            },
            "meta_out": {"mode": "line", **meta},
        }

    J = np.asarray(J_list, dtype=np.float64)
    K = np.asarray(K_list, dtype=np.float64)
    gkm_curve = _combine_gated_kl(J, K, alpha=0.7)

    if gkm_curve.size == 0 or gkm_curve.max() < 1e-12:
        coverage = 0.0
    else:
        coverage = float(
            np.count_nonzero(gkm_curve > 0.5 * gkm_curve.max())
        ) / len(gkm_curve)

    if coverage < 0.05:
        selected = "HGB"
        reason = "sparse (coverage < 0.05) — event-like — hard directional KL"
        curve = _combine_hard_gated_kl(
            J, K, mode="topk", tau_pctile=75, topk=8
        ).tolist()
    elif coverage < 0.15:
        selected = "GKM"
        reason = "structured (0.05 <= coverage < 0.15) — soft directional KL"
        curve = gkm_curve.tolist()
    else:
        selected = "LXM"
        reason = "dense (coverage >= 0.15) — symmetric JSD"
        curve = J_list

    return {
        "curve_out": curve,
        "route_out": {
            "selected": selected,
            "coverage": round(coverage, 6),
            "reason": reason,
        },
        "meta_out": {"mode": "line", **meta},
    }
