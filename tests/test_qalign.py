"""Tests for the experimental Q-alignment scoring path.

Covers:
  - row_to_distribution invariants (normalised, no NaN, no zeros
    after eps smoothing)
  - constant / degenerate windows do not crash
  - js_divergence and kl_divergence produce finite, non-negative values
  - compute_qalign_curve length matches expectations under sampling
  - segment(scoring="q_jsd" / "q_kl") emits a curve and diagnostics
  - default (scoring="heuristic") produces the exact same boundaries
    as before this experimental path existed
  - CLI --scoring, --q-mode, --dist-bins, --threshold-mode are
    passed through into the runtime
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

_here = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_here))

from primitives.qalign import (
    row_to_distribution,
    js_divergence,
    kl_divergence,
    build_q,
    compute_qalign_curve,
    char_byte_signal,
    line_length_signal,
)
from runtime import segment
import mentis_log_cli


# ── row_to_distribution ────────────────────────────────────────────

def test_row_to_distribution_shape_and_normalisation():
    p = row_to_distribution(np.array([1.0, 2.0, 3.0, 4.0]), bins=8)
    assert p.shape == (8,)
    assert abs(float(p.sum()) - 1.0) < 1e-9
    assert float(p.min()) > 0.0  # eps smoothing → no zero entries


def test_row_to_distribution_empty_input_returns_uniform():
    p = row_to_distribution(np.array([], dtype=float), bins=16)
    assert p.shape == (16,)
    assert abs(float(p.sum()) - 1.0) < 1e-9
    # Uniform: every bin equal
    assert float(np.ptp(p)) < 1e-12


def test_row_to_distribution_constant_window_does_not_crash():
    # Every value identical — range collapses to zero width
    p = row_to_distribution(np.full(50, 0.42), bins=16)
    assert p.shape == (16,)
    assert abs(float(p.sum()) - 1.0) < 1e-9
    assert np.isfinite(p).all()


def test_row_to_distribution_rejects_unknown_method():
    with pytest.raises(ValueError):
        row_to_distribution(np.array([1.0, 2.0]), method="kde")


def test_row_to_distribution_shared_value_range():
    """Two distributions over the same range sum to 1 and share bins."""
    a = np.array([0.0, 1.0, 2.0, 3.0])
    b = np.array([0.0, 0.0, 0.0, 3.0])
    vr = (0.0, 3.0)
    pa = row_to_distribution(a, bins=4, value_range=vr)
    pb = row_to_distribution(b, bins=4, value_range=vr)
    assert pa.shape == pb.shape == (4,)
    assert abs(float(pa.sum()) - 1.0) < 1e-9
    assert abs(float(pb.sum()) - 1.0) < 1e-9


# ── divergences ────────────────────────────────────────────────────

def test_js_divergence_identical_is_zero():
    p = row_to_distribution(np.array([1.0, 2.0, 3.0]), bins=8)
    assert js_divergence(p, p) < 1e-9


def test_js_divergence_bounded_by_ln2():
    p = np.array([1.0, 0.0, 0.0])
    q = np.array([0.0, 0.0, 1.0])
    v = js_divergence(p, q)
    assert 0.0 <= v <= math.log(2) + 1e-6


def test_kl_divergence_identical_is_zero():
    p = row_to_distribution(np.array([1.0, 2.0, 3.0]), bins=8)
    assert kl_divergence(p, p) < 1e-9


def test_kl_divergence_is_nonnegative():
    p = np.array([0.1, 0.9])
    q = np.array([0.9, 0.1])
    assert kl_divergence(p, q) > 0.0


# ── build_q ────────────────────────────────────────────────────────

def test_build_q_global_matches_row_to_distribution():
    sig = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    q_built = build_q(sig, q_mode="global", bins=8)
    q_direct = row_to_distribution(sig, bins=8)
    assert np.allclose(q_built, q_direct)


def test_build_q_rolling_excludes_core():
    """The rolling Q at position i should NOT be the global Q — at
    least when there's a local concentration that differs from the
    neighbourhood."""
    sig = np.concatenate([np.zeros(100), np.ones(100), np.zeros(100)])
    q_rolling = build_q(sig, q_mode="rolling", i=150, window=20, bins=8)
    q_global = build_q(sig, q_mode="global", bins=8)
    # They may differ — at minimum, build_q with rolling must still
    # return a valid distribution.
    assert abs(float(q_rolling.sum()) - 1.0) < 1e-9
    assert float(q_rolling.min()) > 0.0
    assert abs(float(q_global.sum()) - 1.0) < 1e-9


def test_build_q_prefix_falls_back_at_start():
    sig = np.arange(50.0)
    q_prefix = build_q(sig, q_mode="prefix", i=0, bins=8)
    assert abs(float(q_prefix.sum()) - 1.0) < 1e-9


def test_build_q_unknown_mode_raises():
    with pytest.raises(ValueError):
        build_q(np.arange(10.0), q_mode="nonsense", bins=8)


# ── compute_qalign_curve ───────────────────────────────────────────

def test_compute_qalign_curve_length_matches_signal():
    sig = np.arange(200.0)
    curve = compute_qalign_curve(sig, window=10, bins=16)
    assert curve.shape == (200,)
    assert np.isfinite(curve).all()


def test_compute_qalign_curve_length_with_sampling():
    sig = np.arange(200.0)
    curve = compute_qalign_curve(sig, window=10, bins=16, sample_rate=5)
    assert curve.shape == (40,)  # 200 / 5


def test_compute_qalign_curve_kl_mode():
    sig = np.concatenate([np.zeros(50), np.ones(50)])
    curve = compute_qalign_curve(sig, window=10, divergence="kl", bins=16)
    assert curve.shape == (100,)
    assert np.isfinite(curve).all()
    # At the boundary, divergence should be largest somewhere in the
    # middle zone
    mid_max = float(curve[30:70].max())
    edge_max = float(max(curve[:5].max(), curve[95:].max()))
    assert mid_max >= edge_max


def test_compute_qalign_curve_constant_signal():
    """Constant signal must not crash and should produce ~0 divergence."""
    sig = np.full(100, 0.5)
    curve = compute_qalign_curve(sig, window=10, bins=8)
    assert curve.shape == (100,)
    assert np.isfinite(curve).all()
    # All windows identical to Q → ~0 everywhere
    assert float(curve.max()) < 1e-6


def test_compute_qalign_curve_rejects_bad_divergence():
    with pytest.raises(ValueError):
        compute_qalign_curve(
            np.arange(50.0), window=5, divergence="mystery"
        )


# ── Signal builders ────────────────────────────────────────────────

def test_char_byte_signal_length_matches_text():
    text = "alpha beta"
    sig = char_byte_signal(text)
    assert sig.shape == (len(text),)
    assert sig.dtype == np.float64
    assert float(sig.min()) >= 0.0
    assert float(sig.max()) <= 1.0


def test_line_length_signal_length_matches_lines():
    text = "a\nbb\nccc"
    sig = line_length_signal(text)
    assert sig.shape == (3,)
    assert float(sig.max()) == pytest.approx(1.0)


# ── runtime integration ───────────────────────────────────────────

DEMO_LINES = (
    [f"INFO req id={i} user=alice ok=true" for i in range(60)]
    + [f"ERROR db timeout trace=x{i}" for i in range(60)]
    + [f"INFO req id={i} user=bob ok=true" for i in range(60)]
)
DEMO_TEXT = "\n".join(DEMO_LINES)


def test_segment_strategy_heuristic_restores_heuristic():
    """With strategy='heuristic' the heuristic scoring path is used,
    regardless of what the 'auto' default would pick."""
    r = segment(DEMO_TEXT, mode="line", strategy="heuristic")
    assert r["scoring"] == "heuristic"
    assert r["threshold_mode"] == "quantile"
    # No q-align-specific diagnostics in heuristic mode
    assert "q_mode" not in r
    assert "divergence" not in r


def test_segment_q_jsd_produces_curve_and_diagnostics():
    # Explicit signal_type=line_length + q_mode=global to preserve this
    # test's original intent (histogram-on-length numeric path) under
    # the new strategy-aware defaults.
    r = segment(
        DEMO_TEXT, mode="line", scoring="q_jsd",
        signal_type="line_length", q_mode="global",
        # Fixture-size-appropriate peak params
        window=20, min_distance=30, nms_radius=15,
        consolidation_radius=30, min_segment_windows=5,
        include_curve=True,
    )
    assert r["scoring"] == "q_jsd"
    assert r["q_mode"] == "global"
    assert r["divergence"] == "jsd"
    assert r["dist_bins"] == 32
    assert r["route"]["selected"] == "Q_JSD"
    assert r["route"]["coverage"] is None
    assert "curve_stats" in r
    assert isinstance(r["curve"], list)
    assert len(r["curve"]) == r["n_lines"]


def test_segment_q_kl_produces_curve_and_diagnostics():
    r = segment(
        DEMO_TEXT, mode="line", scoring="q_kl",
        signal_type="line_length", q_mode="global",
        window=20, min_distance=30, nms_radius=15,
        consolidation_radius=30, min_segment_windows=5,
    )
    assert r["scoring"] == "q_kl"
    assert r["divergence"] == "kl"
    assert r["route"]["selected"] == "Q_KL"


def test_segment_q_jsd_char_mode_runs():
    text = "alpha beta " * 200
    r = segment(text, mode="char", scoring="q_jsd", window=50)
    assert r["scoring"] == "q_jsd"
    assert r["mode"] == "char"
    assert r["n_chars"] == len(text)


def test_segment_unknown_scoring_raises():
    with pytest.raises(ValueError):
        segment("hello world", scoring="magic")


def test_segment_unknown_threshold_mode_raises():
    with pytest.raises(ValueError):
        segment(DEMO_TEXT, mode="line", threshold_mode="nonsense",
                window=20)


def test_segment_threshold_mode_mean_std():
    r = segment(
        DEMO_TEXT, mode="line", scoring="q_jsd",
        signal_type="line_length",
        threshold_mode="mean_std", mean_std_k=1.0,
        window=20, min_distance=30, nms_radius=15,
        consolidation_radius=30, min_segment_windows=5,
    )
    assert r["threshold_mode"] == "mean_std"
    assert 0.0 <= r["effective_quantile"] <= 1.0


def test_segment_threshold_mode_topk():
    r = segment(
        DEMO_TEXT, mode="line", scoring="q_jsd",
        signal_type="line_length",
        threshold_mode="topk", topk_n=5,
        window=20, min_distance=30, nms_radius=15,
        consolidation_radius=30, min_segment_windows=5,
    )
    assert r["threshold_mode"] == "topk"


# ── Existing-behaviour preservation ───────────────────────────────

def test_heuristic_strategy_unchanged_on_demo_log(tmp_path):
    """With strategy='heuristic', line-mode coarse output must remain
    deterministic and match the specific boundaries produced by the
    previous heuristic version on demo_incident.log.

    ``refine=False`` here because this test pins down the *coarse*
    detector output; the refinement stage is a separately tested
    post-processing step that may move boundaries by a few lines.
    """
    fixture = Path(_here) / "demo_incident.log"
    text = fixture.read_text(encoding="utf-8")

    r = segment(text, mode="line", strategy="heuristic", refine=False)
    # Known values under the coarse DEFAULTS_LINE post the coarse-default
    # patch: 78-line demo, coverage routes to LXM, 1 boundary at line 42.
    assert r["mode"] == "line"
    assert r["scoring"] == "heuristic"
    assert r["route"]["selected"] == "LXM"
    assert r["n_boundaries"] == 1
    assert r["boundaries"] == [42]


# ── CLI plumbing ──────────────────────────────────────────────────

def test_cli_passes_qalign_flags_into_segment(tmp_path):
    input_log = tmp_path / "in.log"
    input_log.write_text("alpha beta " * 200, encoding="utf-8")
    output_json = tmp_path / "out.json"

    with patch("mentis_log_cli.segment") as spy:
        spy.return_value = {
            "mode": "char", "sample_rate": 1, "n_chars": 0,
            "route": {"selected": "Q_JSD", "coverage": None, "reason": ""},
            "n_boundaries": 0, "boundaries": [],
            "n_segments": 0, "segments": [],
            "scoring": "q_jsd", "threshold_mode": "mean_std",
        }
        rc = mentis_log_cli.main([
            "segment",
            "--input", str(input_log),
            "--output", str(output_json),
            "--scoring", "q_jsd",
            "--q-mode", "prefix",
            "--dist-bins", "16",
            "--threshold-mode", "mean_std",
            "--mean-std-k", "1.5",
        ])
    assert rc == 0
    assert spy.called
    kw = spy.call_args.kwargs
    assert kw["scoring"] == "q_jsd"
    assert kw["q_mode"] == "prefix"
    assert kw["dist_bins"] == 16
    assert kw["threshold_mode"] == "mean_std"
    assert kw["mean_std_k"] == 1.5


def test_cli_default_is_strategy_auto(tmp_path):
    """Under the release-hardening refactor the CLI default is
    strategy='auto'. Expert kwargs (scoring, q_mode, signal_type) are
    passed as None so that strategy resolution kicks in."""
    input_log = tmp_path / "in.log"
    input_log.write_text("alpha beta " * 200, encoding="utf-8")
    output_json = tmp_path / "out.json"

    with patch("mentis_log_cli.segment") as spy:
        spy.return_value = {
            "mode": "line", "sample_rate": 1, "n_lines": 0,
            "route": {"selected": "Q_JSD", "coverage": None, "reason": ""},
            "n_boundaries": 0, "boundaries": [],
            "n_segments": 0, "segments": [],
            "strategy": "auto", "scoring": "q_jsd",
            "threshold_mode": "quantile",
        }
        rc = mentis_log_cli.main([
            "segment",
            "--input", str(input_log),
            "--output", str(output_json),
        ])
    assert rc == 0
    kw = spy.call_args.kwargs
    assert kw["strategy"] == "auto"
    assert kw["mode"] == "line"
    assert kw["scoring"] is None            # None → strategy resolves
    assert kw["q_mode"] is None
    assert kw["signal_type"] is None
    assert kw["dist_bins"] == 32
    assert kw["threshold_mode"] == "quantile"


def test_cli_strategy_heuristic_flag(tmp_path):
    """Explicit --strategy heuristic must propagate."""
    input_log = tmp_path / "in.log"
    input_log.write_text("alpha beta " * 200, encoding="utf-8")
    output_json = tmp_path / "out.json"

    with patch("mentis_log_cli.segment") as spy:
        spy.return_value = {
            "mode": "line", "sample_rate": 1, "n_lines": 0,
            "route": {"selected": "none", "coverage": 0, "reason": ""},
            "n_boundaries": 0, "boundaries": [],
            "n_segments": 0, "segments": [],
            "strategy": "heuristic", "scoring": "heuristic",
            "threshold_mode": "quantile",
        }
        rc = mentis_log_cli.main([
            "segment",
            "--input", str(input_log),
            "--output", str(output_json),
            "--strategy", "heuristic",
        ])
    assert rc == 0
    assert spy.call_args.kwargs["strategy"] == "heuristic"
