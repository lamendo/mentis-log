"""Unit tests for each primitive — hand-computed ground truth.

Runs without any mentis_ai dependency. Only numpy + pytest.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

_here = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_here))

from primitives.similarity import jsd, kl_divergence
from primitives.lexical import lexical_jsd, lexical_predictive_kl
from primitives.policy import gated_kl, hard_gated_kl, policy_select_projection
from primitives.peaks import quantile_peak_select
from primitives.segment import regime_segment


# ── Similarity ─────────────────────────────────────────────────────────

def test_jsd_identical_distributions_is_zero():
    p = np.array([0.5, 0.5])
    assert jsd(p, p) < 1e-10


def test_jsd_bounded_by_ln2():
    p = np.array([1.0, 0.0, 0.0])
    q = np.array([0.0, 0.0, 1.0])
    val = jsd(p, q)
    assert 0.0 <= val <= math.log(2) + 1e-6


def test_jsd_symmetric():
    p = np.array([0.1, 0.4, 0.5])
    q = np.array([0.6, 0.2, 0.2])
    assert abs(jsd(p, q) - jsd(q, p)) < 1e-12


def test_kl_identical_is_zero():
    p = np.array([0.25, 0.25, 0.5])
    assert kl_divergence(p, p) < 1e-10


def test_kl_nonnegative():
    p = np.array([0.1, 0.9])
    q = np.array([0.9, 0.1])
    assert kl_divergence(p, q) > 0.0


# ── Lexical ────────────────────────────────────────────────────────────

def test_lexical_jsd_returns_length_n_list():
    text = "alpha beta gamma delta epsilon zeta"
    out = lexical_jsd(text, window=5)
    assert isinstance(out, list)
    assert len(out) == len(text)


def test_lexical_jsd_endpoints_are_zero():
    text = "alpha beta gamma delta epsilon zeta"
    out = lexical_jsd(text, window=5)
    assert out[0] == 0.0
    assert out[-1] == 0.0


def test_lexical_jsd_empty_text():
    assert lexical_jsd("", window=10) == []


def test_lexical_predictive_kl_forward_vs_backward_differ():
    """Forward and backward KL should generally differ for asymmetric windows."""
    text = "one two three four five " * 20 + "six seven eight nine ten " * 20
    fwd = lexical_predictive_kl(text, window=30, mode="forward")
    bwd = lexical_predictive_kl(text, window=30, mode="backward")
    # Some position should differ
    diff = sum(abs(f - b) for f, b in zip(fwd, bwd))
    assert diff > 0.0


# ── Policy ─────────────────────────────────────────────────────────────

def test_policy_empty_text_returns_none_route():
    r = policy_select_projection("", window=10)
    assert r["route_out"]["selected"] == "none"
    assert r["curve_out"] == []


def test_policy_dense_text_routes_lxm():
    """Long uniform prose has high coverage → LXM."""
    text = (
        "The quick brown fox jumps over the lazy dog. Pack my box with "
        "five dozen liquor jugs. How vexingly quick daft zebras jump. "
    ) * 20
    r = policy_select_projection(text, window=50)
    assert r["route_out"]["selected"] == "LXM"
    assert r["route_out"]["coverage"] >= 0.15


def test_gated_kl_bounded_and_nonnegative():
    text = "alpha beta " * 50 + "gamma delta " * 50
    curve = gated_kl(text, window=20)
    assert all(v >= 0.0 for v in curve)


def test_hard_gated_kl_sparse():
    text = "alpha beta " * 50 + "gamma delta " * 50
    curve = hard_gated_kl(text, mode="topk", topk=5, window=20)
    nonzero = sum(1 for v in curve if v > 0)
    # hard gate produces very few nonzero values
    assert nonzero <= len(curve) // 5


# ── Peaks ──────────────────────────────────────────────────────────────

def test_quantile_peak_select_empty_curve():
    assert quantile_peak_select([]) == []


def test_quantile_peak_select_finds_expected_peaks():
    """Curve with three clear peaks well separated by >min_distance."""
    curve = [0.0] * 500
    # Three well-separated peaks
    for pos in (100, 250, 400):
        curve[pos] = 1.0
        curve[pos - 1] = 0.5
        curve[pos + 1] = 0.5
    # Add tiny baseline noise to make quantile/prominence sensible
    curve = [v + 0.01 if v == 0 else v for v in curve]

    boundaries = quantile_peak_select(
        curve, quantile=0.9, min_distance=50,
        min_prominence=0.1, nms_radius=20, consolidation_radius=20,
    )
    detected = sorted(b["position"] for b in boundaries if b["is_boundary"])
    assert detected == [100, 250, 400]


def test_quantile_peak_select_output_length_matches_curve():
    curve = [0.1, 0.2, 0.5, 0.9, 0.5, 0.2, 0.1] * 20
    boundaries = quantile_peak_select(
        curve, quantile=0.8, min_distance=5,
        min_prominence=0.01, nms_radius=3, consolidation_radius=3,
    )
    assert len(boundaries) == len(curve)


# ── Segment ────────────────────────────────────────────────────────────

def test_regime_segment_empty():
    assert regime_segment([]) == []


def test_regime_segment_basic():
    # Boundary positions: 5 and 12. Total length 20. min_windows=4.
    N = 20
    boundaries = [
        {"position": i, "combined_score": 0.0, "is_boundary": i in (5, 12)}
        for i in range(N)
    ]
    segs = regime_segment(boundaries, min_windows=4)
    # Expected segments: [0,5), [5,12), [12,20)
    assert len(segs) == 3
    assert segs[0]["start_idx"] == 0 and segs[0]["end_idx"] == 5
    assert segs[1]["start_idx"] == 5 and segs[1]["end_idx"] == 12
    assert segs[2]["start_idx"] == 12 and segs[2]["end_idx"] == 20


def test_regime_segment_min_windows_drops_short():
    N = 20
    # Boundaries at 2 and 15. First segment [0,2) has only 2 chars < min_windows=4
    boundaries = [
        {"position": i, "combined_score": 0.0, "is_boundary": i in (2, 15)}
        for i in range(N)
    ]
    segs = regime_segment(boundaries, min_windows=4)
    # [0,2) dropped; [2,15) kept; [15,20) kept
    assert len(segs) == 2
    assert segs[0]["start_idx"] == 2 and segs[0]["end_idx"] == 15
    assert segs[1]["start_idx"] == 15 and segs[1]["end_idx"] == 20


if __name__ == "__main__":
    import subprocess
    subprocess.run(["pytest", __file__, "-v"])
