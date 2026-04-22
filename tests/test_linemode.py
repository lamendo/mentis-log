"""Line-mode + sampling smoke tests.

No reference-parity expectation (line mode is a different unit),
but structural invariants must hold.
"""
from __future__ import annotations

import sys
from pathlib import Path

_here = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_here))

from runtime import segment
from primitives.linemode import _line_tokenize, compute_lexical_jk_line


# ── Line tokenisation ──────────────────────────────────────────────

def test_line_tokenize_basic():
    text = "alpha beta\ngamma delta\nepsilon"
    lines, offsets, data, indptr, V = _line_tokenize(text)
    assert lines == ["alpha beta", "gamma delta", "epsilon"]
    assert V == 5  # alpha, beta, gamma, delta, epsilon
    assert offsets.shape[0] == 4  # n_lines + 1
    # line 0 covers chars [0, 10), line 1 [11, 22), line 2 [23, 30)
    assert offsets[0] == 0
    # Per-line word id counts via indptr
    assert indptr[1] - indptr[0] == 2  # alpha, beta
    assert indptr[2] - indptr[1] == 2  # gamma, delta
    assert indptr[3] - indptr[2] == 1  # epsilon
    # data holds ids (not vocabulary — positional)
    assert len(data) == 5


def test_line_tokenize_empty():
    lines, offsets, data, indptr, V = _line_tokenize("")
    assert lines == []
    assert V == 0


# ── compute_lexical_jk_line shape ─────────────────────────────────

def test_jk_line_length_matches_eval_positions():
    text = "\n".join(["alpha beta gamma"] * 30 + ["delta epsilon zeta"] * 30)
    J, K, meta = compute_lexical_jk_line(text, window=5)
    assert meta["n_lines"] == 60
    assert len(J) == 60
    assert len(K) == 60
    # Endpoints are 0 (no left / right window)
    assert J[0] == 0.0 and J[-1] == 0.0


def test_jk_line_sample_rate_shortens():
    text = "\n".join([f"line{i} word_{i % 5}" for i in range(100)])
    J_full, K_full, _ = compute_lexical_jk_line(text, window=5)
    J_5, K_5, meta = compute_lexical_jk_line(text, window=5, sample_rate=5)
    assert len(J_full) == 100
    assert len(J_5) == 20  # ceil(100 / 5)
    assert meta["sample_rate"] == 5


# ── End-to-end segment() in line mode ─────────────────────────────

def test_segment_line_mode_shape():
    # A synthetic log with a clear topic shift at line 50.
    # The new coarse DEFAULTS_LINE (min_distance=500, nms_radius=250,
    # consolidation_radius=500, min_segment_windows=20) are tuned for
    # real multi-thousand-line logs and would suppress every boundary
    # on a 100-line fixture. We therefore override the peak-selection
    # params to values appropriate for this fixture size.
    top = [f"INFO request id={i} user=alice ok=true" for i in range(50)]
    bot = [f"ERROR database timeout trace=x{i}" for i in range(50)]
    text = "\n".join(top + bot)

    r = segment(
        text, mode="line",
        window=10,
        min_distance=20, nms_radius=10,
        consolidation_radius=20, min_segment_windows=5,
    )
    assert r["mode"] == "line"
    assert r["n_lines"] == 100
    assert r["sample_rate"] == 1
    # Must detect the shift
    assert r["n_boundaries"] >= 1
    # Some boundary should land near line 50 (± window)
    assert any(40 <= b <= 60 for b in r["boundaries"])
    # Segments have start_line / end_line / n_lines
    for s in r["segments"]:
        assert "start_line" in s and "end_line" in s
        assert s["end_line"] > s["start_line"]
        assert s["n_lines"] == s["end_line"] - s["start_line"]
        # Approximate char offsets derived from line offsets
        assert "char_start_approx" in s
        assert "char_end_approx" in s


def test_segment_line_mode_empty():
    r = segment("", mode="line")
    assert r["mode"] == "line"
    assert r["n_lines"] == 0
    assert r["n_boundaries"] == 0
    assert r["segments"] == []


# ── Sampling (char + line) ────────────────────────────────────────

def test_segment_char_sample_rate_scales_output():
    text = "alpha beta gamma delta " * 500  # ~12k chars
    r1 = segment(text, mode="char")
    r5 = segment(text, mode="char", sample_rate=5)
    # Both cover the full char range
    assert r1["n_chars"] == r5["n_chars"]
    assert r1["mode"] == r5["mode"] == "char"
    assert r1["sample_rate"] == 1
    assert r5["sample_rate"] == 5
    # Sampled segments are in char-space (after rescale by sample_rate)
    for s in r5["segments"]:
        assert s["start_idx"] <= r5["n_chars"]
        assert s["end_idx"] <= r5["n_chars"]


def test_segment_line_sample_rate_scales_output():
    top = [f"INFO a={i} b={i % 3}" for i in range(200)]
    bot = [f"ERROR x=y{i}" for i in range(200)]
    text = "\n".join(top + bot)

    # Fixture-size-appropriate peak params (see test_segment_line_mode_shape).
    peak_kwargs = dict(
        min_distance=40, nms_radius=20,
        consolidation_radius=40, min_segment_windows=5,
    )
    r1 = segment(text, mode="line", window=10, **peak_kwargs)
    r4 = segment(text, mode="line", window=10, sample_rate=4, **peak_kwargs)
    assert r1["n_lines"] == r4["n_lines"]
    assert r4["sample_rate"] == 4
    # Segment boundaries land in original line space
    for s in r4["segments"]:
        assert s["start_line"] <= r4["n_lines"]
        assert s["end_line"] <= r4["n_lines"]


# ── Char mode unchanged ───────────────────────────────────────────

def test_segment_char_mode_when_explicit():
    """Char mode must still produce a char-shaped result when explicitly
    requested. (The library default changed to mode='line' in the release
    hardening pass; char mode is the explicit parity path.)"""
    text = "The quick brown fox. " * 50
    r = segment(text, mode="char")
    assert r["mode"] == "char"
    assert r["sample_rate"] == 1
    assert r["n_chars"] == len(text)
    # Preserved keys from pre-mode schema
    assert "route" in r
    assert "boundaries" in r
    assert "segments" in r


# ── DEFAULTS_LINE coarse semantics ─────────────────────────────────

def test_defaults_line_coarse_values():
    """DEFAULTS_LINE must be the coarse regime-detector settings.

    These defaults are intentionally very conservative so that line
    mode surfaces *phase* shifts (e.g. normal → incident → recovery)
    on real multi-thousand-line logs rather than event-level noise.
    """
    from runtime import DEFAULTS_LINE
    assert DEFAULTS_LINE["window"] == 25
    assert DEFAULTS_LINE["quantile"] == 0.995
    assert DEFAULTS_LINE["min_distance"] == 500
    assert DEFAULTS_LINE["min_prominence"] == 0.03
    assert DEFAULTS_LINE["nms_radius"] == 250
    assert DEFAULTS_LINE["consolidation_radius"] == 500
    assert DEFAULTS_LINE["min_segment_windows"] == 20
