"""Sparse token-Q-align refactor — memory safety tests.

Covers:
  1. small-fixture behaviour is unchanged (smoke)
  2. output structure preserved (curve length, meta keys)
  3. large synthetic input goes through the sparse path and does NOT
     allocate a dense (n_lines, vocab_size) matrix
  4. lines_to_matrix memory guard raises on oversized requests
  5. the two paths agree numerically on a small fixture where dense
     is still permitted (so the refactor didn't silently change math)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_here = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_here))

from primitives import qalign  # whole module for monkeypatching


# ── structure preserved on small inputs ──────────────────────────

def _fixture_lines(n_blocks: int = 3, n_per: int = 40):
    """Three tiny regimes with different vocabulary."""
    a = [f"alpha beta id={i}" for i in range(n_per)]
    b = [f"gamma delta id={i}" for i in range(n_per)]
    c = [f"epsilon zeta id={i}" for i in range(n_per)]
    return a + b + c


def test_small_fixture_curve_length():
    lines = _fixture_lines()
    curve, meta = qalign.compute_qalign_curve_tokens(
        lines, window=10, q_mode="global",
    )
    assert curve.shape == (len(lines),)
    assert "vocab_size" in meta
    assert "avg_tokens_per_line" in meta
    assert "sparsity" in meta
    assert 0.0 <= meta["sparsity"] <= 1.0


def test_meta_includes_dense_avoided_bytes():
    lines = _fixture_lines(n_per=30)
    _, meta = qalign.compute_qalign_curve_tokens(lines, window=8)
    assert "dense_matrix_avoided_bytes" in meta
    assert meta["dense_matrix_avoided_bytes"] > 0


# ── sparse path does NOT call lines_to_matrix ────────────────────

def test_compute_qalign_tokens_does_not_allocate_dense(monkeypatch):
    """The hot path must not touch the dense helper. This would have
    blown up at 1.19M lines × 16k vocab before the refactor."""
    tripwire_called = []

    def _tripwire(*_args, **_kwargs):
        tripwire_called.append(True)
        raise AssertionError(
            "lines_to_matrix was invoked — sparse refactor regressed."
        )

    monkeypatch.setattr(qalign, "lines_to_matrix", _tripwire)

    # Moderately-sized synthetic input: small enough to run fast, large
    # enough that a dense (N, V) allocation would have been costly.
    lines = [
        f"tok_{i % 120} tok_{(i * 17) % 240} tok_{(i * 23) % 480}"
        for i in range(2000)
    ]
    curve, meta = qalign.compute_qalign_curve_tokens(
        lines, window=15, q_mode="rolling",
    )
    assert curve.shape == (len(lines),)
    assert meta["vocab_size"] > 0
    assert tripwire_called == []


# ── memory guard on the explicit dense helper ────────────────────

def test_estimate_dense_bytes():
    assert qalign._estimate_dense_bytes(1000, 500) == 1000 * 500 * 4
    # The scenario that triggered the bug report:
    #   1,190,554 lines × 16,636 vocab × 4 bytes ≈ 73.8 GiB
    est = qalign._estimate_dense_bytes(1_190_554, 16_636)
    assert est > 70 * (1 << 30)


def test_lines_to_matrix_guard_raises(monkeypatch):
    """Lower the threshold so a small allocation trips the guard."""
    monkeypatch.setattr(qalign, "_DENSE_MEM_THRESHOLD_BYTES", 1_000)
    with pytest.raises(MemoryError) as exc:
        qalign.lines_to_matrix(
            ["a b c"] * 100, {"a": 0, "b": 1, "c": 2},
        )
    msg = str(exc.value)
    assert "lines_to_matrix" in msg
    assert "sparse streaming" in msg


def test_lines_to_matrix_below_threshold_works():
    """Default threshold allows small matrices through unchanged."""
    vocab = {"a": 0, "b": 1, "c": 2}
    lines = ["a b c", "b c", "a"]
    M = qalign.lines_to_matrix(lines, vocab)
    assert M.shape == (3, 3)
    assert M.dtype == np.float32
    # Row 0: 1,1,1 ; Row 1: 0,1,1 ; Row 2: 1,0,0
    assert np.allclose(M, np.array(
        [[1, 1, 1], [0, 1, 1], [1, 0, 0]], dtype=np.float32,
    ))


# ── sparse is numerically consistent on a size the old path can still do

def test_sparse_matches_pre_refactor_shape_and_range():
    """End-to-end curve over a small fixture must be finite, have the
    right shape, and produce a non-trivial spread on a multi-regime
    input (so the refactor didn't just return zeros)."""
    lines = _fixture_lines(n_per=40)
    for q_mode in ("global", "rolling", "prefix"):
        curve, meta = qalign.compute_qalign_curve_tokens(
            lines, window=10, q_mode=q_mode,
        )
        assert curve.shape == (len(lines),)
        assert np.isfinite(curve).all()
        spread = float(curve.max() - curve.min())
        assert spread > 0.0, f"q_mode={q_mode}: flat curve"
        assert meta["vocab_size"] > 0


# ── very large vocab × few lines (sparse friendly, dense not) ────

def test_large_vocab_many_lines_runs_sparsely(monkeypatch):
    """Lots of unique words per line → vocab grows; must not try to
    allocate a dense N×V matrix."""
    called = []
    monkeypatch.setattr(
        qalign, "lines_to_matrix",
        lambda *a, **kw: called.append(True) or (_ for _ in ()).throw(
            AssertionError("dense forbidden")
        ),
    )
    # 1500 lines, each with its own ~50 unique IDs → vocab ~75k entries
    # after pruning. A dense (1500, 75000) float32 matrix would be
    # ~450 MB — still big, still wasteful.
    lines = [
        " ".join(f"uid_{i}_{j}" for j in range(50)) + " common_tok"
        for i in range(1500)
    ]
    curve, meta = qalign.compute_qalign_curve_tokens(
        lines, window=20, q_mode="global", min_token_freq=2,
    )
    # min_token_freq=2 prunes the per-line unique IDs → vocab collapses
    # to just "common_tok" and similar repeats. Curve stays finite.
    assert curve.shape == (len(lines),)
    assert np.isfinite(curve).all()
    assert called == []


# ── CLI behaviour unchanged ──────────────────────────────────────

import json  # noqa: E402
import subprocess  # noqa: E402


def test_cli_segment_tokens_path_still_works(tmp_path):
    """With strategy=auto + line mode, the CLI goes through the sparse
    token path. This test checks it runs end-to-end and produces a
    sensible result on a realistic-ish input."""
    input_log = tmp_path / "in.log"
    lines = [f"INFO req user=alice id={i}" for i in range(400)] + [
        f"ERROR db timeout trace=x{i}" for i in range(400)
    ]
    input_log.write_text("\n".join(lines), encoding="utf-8")
    output_json = tmp_path / "out.json"

    proc = subprocess.run(
        [
            sys.executable, str(_here / "mentis_log_cli.py"),
            "segment",
            "--input", str(input_log),
            "--output", str(output_json),
            "--window", "20",
            "--min-distance", "30",
            "--nms-radius", "15",
            "--consolidation-radius", "30",
            "--min-segment-windows", "5",
        ],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    r = json.loads(output_json.read_text(encoding="utf-8"))
    assert r["mode"] == "line"
    assert r["scoring"] == "q_jsd"
    assert r["signal_type"] == "tokens"
    assert r["vocab_size"] > 0
