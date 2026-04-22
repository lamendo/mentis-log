"""Tests for the token-based Q-alignment path.

Covers tasks #10.1 – #10.7:
  1. tokenization deterministic
  2. vocab stable across runs
  3. line_to_vector shape correct
  4. distributions sum to 1
  5. compute_qalign_curve_tokens returns correct length
  6. segment(..., scoring=q_jsd, signal=tokens) runs end-to-end
  7. existing tests must all stay green  (covered by the full suite)
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

_here = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_here))

from primitives.qalign import (
    tokenize_line,
    build_vocab,
    line_to_vector,
    lines_to_matrix,
    compute_qalign_curve_tokens,
    js_divergence,
)
from runtime import segment
import mentis_log_cli


# ── tokenize_line ─────────────────────────────────────────────────

def test_tokenize_line_deterministic():
    s = "INFO req id=7a1 user=alice ok=true"
    assert tokenize_line(s) == tokenize_line(s)
    # Deterministic across calls; order preserved
    assert tokenize_line(s) == [
        "info", "req", "id", "7a1", "user", "alice", "ok", "true",
    ]


def test_tokenize_line_lowercase_folding():
    assert tokenize_line("INFO info Info") == ["info", "info", "info"]


def test_tokenize_line_punctuation_split():
    # = and spaces split tokens; _ and digits stay inside one token
    assert tokenize_line("a=b.c_d 42") == ["a", "b", "c_d", "42"]


def test_tokenize_line_empty_and_whitespace():
    assert tokenize_line("") == []
    assert tokenize_line("   \t  ") == []


# ── build_vocab ───────────────────────────────────────────────────

def test_build_vocab_sorted_and_stable():
    lines = ["foo bar", "bar baz", "foo"]
    v1 = build_vocab(lines)
    v2 = build_vocab(lines)
    assert v1 == v2
    # Sorted keys → token → index
    assert list(v1.keys()) == ["bar", "baz", "foo"]
    assert v1 == {"bar": 0, "baz": 1, "foo": 2}


def test_build_vocab_min_freq_prunes():
    lines = ["common common common", "rare"]
    v = build_vocab(lines, min_freq=2)
    assert "common" in v and "rare" not in v


def test_build_vocab_empty_lines():
    assert build_vocab([]) == {}
    assert build_vocab(["", "", ""]) == {}


# ── line_to_vector / lines_to_matrix ──────────────────────────────

def test_line_to_vector_shape_and_counts():
    vocab = {"foo": 0, "bar": 1, "baz": 2}
    v = line_to_vector("foo foo bar", vocab)
    assert v.shape == (3,)
    assert v.dtype == np.float32
    assert list(v) == [2.0, 1.0, 0.0]


def test_line_to_vector_ignores_out_of_vocab():
    vocab = {"foo": 0}
    v = line_to_vector("foo unknown other", vocab)
    assert float(v.sum()) == 1.0


def test_lines_to_matrix_shape():
    lines = ["foo bar", "baz", ""]
    vocab = build_vocab(lines)
    M = lines_to_matrix(lines, vocab)
    assert M.shape == (3, len(vocab))
    assert M.dtype == np.float32
    # Row sums equal token counts per line
    assert float(M[0].sum()) == 2.0
    assert float(M[1].sum()) == 1.0
    assert float(M[2].sum()) == 0.0


def test_lines_to_matrix_empty_vocab():
    M = lines_to_matrix(["foo"], {})
    assert M.shape == (1, 0)


# ── compute_qalign_curve_tokens ───────────────────────────────────

def _build_regime_fixture(n_each: int = 40):
    """Three regimes with clearly distinct vocabulary."""
    a = [f"INFO req id={i} user=alice ok=true" for i in range(n_each)]
    b = [f"ERROR db timeout trace=x{i}" for i in range(n_each)]
    c = [f"FATAL out_of_memory heap=99 node=node_{i:02d}" for i in range(n_each)]
    return a + b + c


def test_compute_qalign_curve_tokens_length_matches_lines():
    lines = _build_regime_fixture(40)
    curve, meta = compute_qalign_curve_tokens(lines, window=10, q_mode="global")
    assert isinstance(curve, np.ndarray)
    assert curve.shape == (len(lines),)
    assert np.isfinite(curve).all()


def test_compute_qalign_curve_tokens_sample_rate_shortens():
    lines = _build_regime_fixture(50)
    curve, meta = compute_qalign_curve_tokens(
        lines, window=10, sample_rate=5,
    )
    assert curve.shape == (len(lines) // 5,)


def test_compute_qalign_curve_tokens_meta_populated():
    lines = _build_regime_fixture(30)
    _, meta = compute_qalign_curve_tokens(lines, window=10)
    assert meta["vocab_size"] > 0
    assert meta["avg_tokens_per_line"] > 0
    assert 0.0 <= meta["sparsity"] <= 1.0


def test_compute_qalign_curve_tokens_differentiates_regimes():
    """On a multi-regime fixture the curve must be measurably non-flat
    under every Q mode. (Peak *location* depends on Q mode — global
    highlights content farthest from the mean, prefix highlights
    newly-introduced regimes — so we only assert spread, not exact
    peak positions.)"""
    lines = _build_regime_fixture(40)
    for q_mode in ("global", "rolling", "prefix"):
        curve, _ = compute_qalign_curve_tokens(
            lines, window=10, q_mode=q_mode,
        )
        spread = float(curve.max() - curve.min())
        assert spread > 0.01, (
            f"q_mode={q_mode}: curve is flat (spread={spread:.4f})"
        )


def test_compute_qalign_curve_tokens_kl_mode():
    lines = _build_regime_fixture(20)
    curve, _ = compute_qalign_curve_tokens(
        lines, window=8, divergence="kl",
    )
    assert curve.shape == (len(lines),)
    assert np.isfinite(curve).all()
    assert float(curve.min()) >= 0.0  # KL is non-negative


def test_compute_qalign_curve_tokens_empty_vocab_is_safe():
    # Lines with no alphanumeric tokens → vocab empty → zero curve
    lines = ["!!! ???", "... ~~~"] * 10
    curve, meta = compute_qalign_curve_tokens(lines, window=5)
    assert meta["vocab_size"] == 0
    assert float(curve.max()) == 0.0


def test_compute_qalign_curve_tokens_min_freq_pruning():
    # Without pruning: every unique id (7a_0, 7a_1, ...) is in vocab.
    # With min_freq=2: only tokens appearing in >=2 lines remain.
    lines = [f"INFO id=uniq_{i}" for i in range(20)]
    _, meta_nofreq = compute_qalign_curve_tokens(
        lines, window=5, min_token_freq=1,
    )
    _, meta_freq2 = compute_qalign_curve_tokens(
        lines, window=5, min_token_freq=2,
    )
    # Pruned vocab must be strictly smaller
    assert meta_freq2["vocab_size"] < meta_nofreq["vocab_size"]
    # The per-line uniq_* tokens should be pruned out at min_freq=2
    assert meta_freq2["vocab_size"] <= 3  # INFO, id only survive


def test_compute_qalign_curve_tokens_rejects_bad_divergence():
    lines = _build_regime_fixture(10)
    with pytest.raises(ValueError):
        compute_qalign_curve_tokens(lines, window=3, divergence="mystery")


def test_compute_qalign_curve_tokens_rejects_bad_q_mode():
    lines = _build_regime_fixture(10)
    with pytest.raises(ValueError):
        compute_qalign_curve_tokens(lines, window=3, q_mode="nonsense")


# ── segment() end-to-end with tokens ──────────────────────────────

FIXTURE_TEXT = "\n".join(_build_regime_fixture(40))


def test_segment_tokens_end_to_end():
    r = segment(
        FIXTURE_TEXT, mode="line", scoring="q_jsd", signal_type="tokens",
        window=10, quantile=0.90,  # fixture-size-appropriate threshold
        min_distance=20, nms_radius=10,
        consolidation_radius=20, min_segment_windows=5,
    )
    assert r["mode"] == "line"
    assert r["scoring"] == "q_jsd"
    assert r["signal_type"] == "tokens"
    assert r["vocab_size"] > 0
    assert r["divergence"] == "jsd"
    # Should find at least one boundary in this clear 3-regime fixture
    assert r["n_boundaries"] >= 1


def test_segment_tokens_kl_mode():
    r = segment(
        FIXTURE_TEXT, mode="line", scoring="q_kl", signal_type="tokens",
        window=10, quantile=0.90,
        min_distance=20, nms_radius=10,
        consolidation_radius=20, min_segment_windows=5,
    )
    assert r["signal_type"] == "tokens"
    assert r["divergence"] == "kl"
    assert r["route"]["selected"] == "Q_KL"


def test_segment_tokens_requires_line_mode():
    with pytest.raises(ValueError):
        segment(
            "alpha beta", mode="char", scoring="q_jsd", signal_type="tokens",
        )


def test_segment_default_signal_is_tokens_in_line_mode():
    """Under strategy='auto' (the new default), line mode uses the
    token-based Q-align path by design. Callers that want the numeric
    histogram path can pass signal_type='line_length' explicitly."""
    r = segment(
        FIXTURE_TEXT, mode="line",
        window=10, quantile=0.90,
        min_distance=20, nms_radius=10,
        consolidation_radius=20, min_segment_windows=5,
    )
    assert r["strategy"] == "auto"
    assert r["signal_type"] == "tokens"
    assert "vocab_size" in r


def test_segment_line_length_still_available():
    """Expert override: scoring=q_jsd + signal_type=line_length is
    honoured even under strategy='auto'."""
    r = segment(
        FIXTURE_TEXT, mode="line", scoring="q_jsd",
        signal_type="line_length",
        window=10, quantile=0.90,
        min_distance=20, nms_radius=10,
        consolidation_radius=20, min_segment_windows=5,
    )
    assert r["signal_type"] == "line_length"
    assert "dist_bins" in r
    assert "vocab_size" not in r


def test_segment_strategy_heuristic_no_signal_diagnostics():
    """Heuristic strategy must not emit signal_type diagnostics."""
    r = segment(FIXTURE_TEXT, mode="line", strategy="heuristic")
    assert r["scoring"] == "heuristic"
    assert "signal_type" not in r
    assert "vocab_size" not in r


# ── CLI plumbing ──────────────────────────────────────────────────

def test_cli_passes_signal_and_min_token_freq(tmp_path):
    input_log = tmp_path / "in.log"
    input_log.write_text(FIXTURE_TEXT, encoding="utf-8")
    output_json = tmp_path / "out.json"

    with patch("mentis_log_cli.segment") as spy:
        spy.return_value = {
            "mode": "line", "sample_rate": 1, "n_lines": 0,
            "route": {"selected": "Q_JSD", "coverage": None, "reason": ""},
            "n_boundaries": 0, "boundaries": [],
            "n_segments": 0, "segments": [],
            "scoring": "q_jsd", "threshold_mode": "quantile",
            "signal_type": "tokens", "vocab_size": 50,
        }
        rc = mentis_log_cli.main([
            "segment",
            "--mode", "line",
            "--input", str(input_log),
            "--output", str(output_json),
            "--scoring", "q_jsd",
            "--signal", "tokens",
            "--min-token-freq", "2",
        ])
    assert rc == 0
    kw = spy.call_args.kwargs
    assert kw["signal_type"] == "tokens"
    assert kw["min_token_freq"] == 2


def test_cli_default_signal_is_resolved_by_strategy(tmp_path):
    """Under strategy='auto' the CLI does not pass signal_type nor
    min_token_freq — they stay None so the runtime resolves them via
    strategy (tokens + min_freq=2 for line mode)."""
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
    assert kw["signal_type"] is None
    assert kw["min_token_freq"] is None


def test_cli_explicit_signal_overrides_auto(tmp_path):
    """Explicit --signal still wins under strategy='auto'."""
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
            "--signal", "line_length",
            "--min-token-freq", "3",
        ])
    assert rc == 0
    kw = spy.call_args.kwargs
    assert kw["signal_type"] == "line_length"
    assert kw["min_token_freq"] == 3
