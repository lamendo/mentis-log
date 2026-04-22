"""Tests for the structural interpretation layer.

Covers:
  1. no boundaries -> one segment
  2. multiple boundaries -> correct segment ranges
  3. only ``structural_v1`` labels appear and are in-set
  4. profile_confidence is in [0, 1]
  5. summary strings are present on every segment
  6. CLI ``--interpret`` adds ``interpretation`` + enriched segments
  7. ``--interpret`` unset leaves existing output unchanged
  8. clearly-low-change input → stable; clearly-high-change → volatile
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_here = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_here))

from interpretation import (
    STRUCTURAL_LABELS,
    SCHEMA_V1,
    interpret_segments,
)
from runtime import segment


# ── module-level helpers ─────────────────────────────────────────

def _run_cli(*args, **kw) -> subprocess.CompletedProcess:
    py = sys.executable
    cmd = [py, str(_here / "mentis_log_cli.py"), *args]
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


# ── direct interpret_segments() ──────────────────────────────────

def test_no_boundaries_gives_one_segment():
    curve = [0.1, 0.2, 0.1, 0.15, 0.1]
    out = interpret_segments(curve, boundaries=[])
    segs = out["segments"]
    assert len(segs) == 1
    assert segs[0]["start"] == 0
    assert segs[0]["end"] == len(curve)
    assert segs[0]["length"] == len(curve)


def test_multiple_boundaries_correct_ranges():
    curve = list(np.linspace(0.0, 1.0, 50))
    out = interpret_segments(curve, boundaries=[10, 30])
    segs = out["segments"]
    assert [s["start"] for s in segs] == [0, 10, 30]
    assert [s["end"] for s in segs] == [10, 30, 50]
    assert [s["length"] for s in segs] == [10, 20, 20]
    assert [s["id"] for s in segs] == [0, 1, 2]


def test_boundaries_deduped_and_sorted():
    curve = [0.0] * 100
    out = interpret_segments(curve, boundaries=[60, 30, 60, 10])
    assert [s["start"] for s in out["segments"]] == [0, 10, 30, 60]


def test_boundaries_outside_curve_are_ignored():
    curve = [0.0] * 20
    out = interpret_segments(curve, boundaries=[0, 10, 20, 50])
    # 0 and 20 are at the edges and must NOT create an empty segment;
    # 50 is past the end.
    starts = [s["start"] for s in out["segments"]]
    assert starts == [0, 10]


def test_all_labels_valid():
    rng = np.random.default_rng(42)
    curve = rng.random(300).tolist()
    out = interpret_segments(curve, boundaries=[50, 150, 250])
    for seg in out["segments"]:
        assert seg["profile"] in STRUCTURAL_LABELS


def test_confidence_in_unit_interval():
    rng = np.random.default_rng(7)
    curve = rng.random(500).tolist()
    out = interpret_segments(curve, boundaries=[100, 200, 350, 450])
    for seg in out["segments"]:
        c = seg["profile_confidence"]
        assert 0.0 <= c <= 1.0


def test_summary_present_on_every_segment():
    curve = [0.0] * 100
    out = interpret_segments(curve, boundaries=[40])
    for seg in out["segments"]:
        assert isinstance(seg["summary"], str)
        assert seg["summary"].strip() != ""


def test_interpretation_metadata():
    curve = [0.0] * 20
    out = interpret_segments(curve, boundaries=[])
    meta = out["interpretation"]
    assert meta["enabled"] is True
    assert meta["label_schema"] == SCHEMA_V1
    assert "structural change only" in meta["disclaimer"].lower()


def test_unknown_schema_raises():
    with pytest.raises(ValueError):
        interpret_segments([0.0] * 10, boundaries=[], schema="semantic_v2")


# ── structural behaviour ────────────────────────────────────────

def test_low_change_segment_labelled_stable():
    """A dead-flat curve must contain only stable segments."""
    curve = [0.01] * 500
    out = interpret_segments(curve, boundaries=[])
    assert out["segments"][0]["profile"] == "stable"


def test_high_variability_segment_labelled_volatile():
    """Segment with large internal variation is volatile."""
    # Build a curve where one middle region has wildly higher values
    # relative to the rest.
    low = [0.02] * 400
    busy = [0.8, 0.1, 0.7, 0.05, 0.75, 0.2, 0.9, 0.05] * 25  # 200
    curve = low + busy + low  # 1000 total
    out = interpret_segments(curve, boundaries=[400, 600])
    middle = out["segments"][1]  # [400, 600)
    assert middle["profile"] in ("volatile", "transition")
    # Flanks remain stable
    assert out["segments"][0]["profile"] == "stable"
    assert out["segments"][2]["profile"] == "stable"


def test_boundary_strength_endpoints_are_none():
    curve = list(np.linspace(0.1, 0.9, 40))
    out = interpret_segments(curve, boundaries=[10, 25])
    assert out["segments"][0]["boundary_strength_left"] is None
    assert out["segments"][-1]["boundary_strength_right"] is None
    # Middle segment has both
    mid = out["segments"][1]
    assert mid["boundary_strength_left"] is not None
    assert mid["boundary_strength_right"] is not None


# ── runtime integration ─────────────────────────────────────────

DEMO_TEXT = "\n".join(
    [f"INFO req id={i} user=alice" for i in range(60)]
    + [f"ERROR db timeout trace=x{i}" for i in range(60)]
    + [f"INFO req id={i} user=bob" for i in range(60)]
)


def test_segment_interpret_flag_adds_fields():
    r = segment(
        DEMO_TEXT, mode="line", strategy="heuristic", interpret=True,
        window=10, min_distance=20, nms_radius=10,
        consolidation_radius=20, min_segment_windows=5,
    )
    assert "interpretation" in r
    assert r["interpretation"]["label_schema"] == SCHEMA_V1
    for seg in r["segments"]:
        assert "profile" in seg
        assert seg["profile"] in STRUCTURAL_LABELS
        assert 0.0 <= seg["profile_confidence"] <= 1.0
        assert seg["summary"]
        assert "start" in seg and "end" in seg and "length" in seg


def test_segment_without_interpret_unchanged():
    """Without --interpret the existing output shape is preserved."""
    r = segment(
        DEMO_TEXT, mode="line", strategy="heuristic",
        window=10, min_distance=20, nms_radius=10,
        consolidation_radius=20, min_segment_windows=5,
    )
    # No interpretation block
    assert "interpretation" not in r
    # Segments still carry the original line-mode fields
    if r["segments"]:
        seg = r["segments"][0]
        assert "start_line" in seg
        assert "profile" not in seg


# ── CLI plumbing ────────────────────────────────────────────────

def test_cli_interpret_flag(tmp_path):
    input_log = tmp_path / "in.log"
    input_log.write_text(DEMO_TEXT, encoding="utf-8")
    output_json = tmp_path / "out.json"

    proc = _run_cli(
        "segment",
        "--input", str(input_log),
        "--output", str(output_json),
        "--strategy", "heuristic",
        "--interpret",
        # Peak params fit the fixture size
        "--window", "10",
        "--min-distance", "20",
        "--nms-radius", "10",
        "--consolidation-radius", "20",
        "--min-segment-windows", "5",
    )
    assert proc.returncode == 0, proc.stderr
    r = json.loads(output_json.read_text(encoding="utf-8"))
    assert "interpretation" in r
    assert r["interpretation"]["label_schema"] == SCHEMA_V1
    for seg in r["segments"]:
        assert seg["profile"] in STRUCTURAL_LABELS


def test_cli_without_interpret_flag_preserves_schema(tmp_path):
    input_log = tmp_path / "in.log"
    input_log.write_text(DEMO_TEXT, encoding="utf-8")
    output_json = tmp_path / "out.json"

    proc = _run_cli(
        "segment",
        "--input", str(input_log),
        "--output", str(output_json),
        "--strategy", "heuristic",
        "--window", "10",
        "--min-distance", "20",
        "--nms-radius", "10",
        "--consolidation-radius", "20",
        "--min-segment-windows", "5",
    )
    assert proc.returncode == 0, proc.stderr
    r = json.loads(output_json.read_text(encoding="utf-8"))
    assert "interpretation" not in r
    # Original line-mode segment fields still present
    if r["segments"]:
        assert "start_line" in r["segments"][0]
        assert "profile" not in r["segments"][0]


def test_cli_annotate_segments_implies_interpret(tmp_path):
    """--annotate-segments should also enable interpretation so the
    plot has profile data to render."""
    input_log = tmp_path / "in.log"
    input_log.write_text(DEMO_TEXT, encoding="utf-8")
    output_json = tmp_path / "out.json"
    output_png = tmp_path / "plot.png"

    proc = _run_cli(
        "segment",
        "--input", str(input_log),
        "--output", str(output_json),
        "--strategy", "heuristic",
        "--plot", str(output_png),
        "--annotate-segments",
        "--window", "10",
        "--min-distance", "20",
        "--nms-radius", "10",
        "--consolidation-radius", "20",
        "--min-segment-windows", "5",
    )
    assert proc.returncode == 0, proc.stderr
    assert output_png.exists()
    r = json.loads(output_json.read_text(encoding="utf-8"))
    assert "interpretation" in r
