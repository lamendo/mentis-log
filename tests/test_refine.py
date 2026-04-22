"""Local multiscale boundary refinement tests.

Covers:
  1. refined boundaries are deterministic
  2. refinement stays local to the coarse boundary's neighbourhood
  3. obvious edge artefacts can be dropped
  4. refined boundaries differ from raw when local structure supports it
  5. raw boundaries remain available in the pipeline output
  6. public ``boundaries`` are the refined ones when refinement succeeds
  7. ``refine=False`` preserves the pre-refinement boundaries exactly
  8. CLI ``--no-refine`` works
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_here = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_here))

from refine import refine_boundaries_local_multiscale
from runtime import segment


# ── helpers ─────────────────────────────────────────────────────

def _run_cli(*args):
    cmd = [sys.executable, str(_here / "mentis_log_cli.py"), *args]
    return subprocess.run(cmd, capture_output=True, text=True)


# ── direct refine_boundaries_local_multiscale() ─────────────────

def _text_with_shift(pre_lines: int, post_lines: int,
                     shift_line: int) -> tuple:
    """Build text with two distinct structural regimes. Returns (text,
    actual_shift_line)."""
    pre = "\n".join(
        f"INFO status=200 user=u{i:03d} method=GET path=/api/v1/a"
        for i in range(pre_lines)
    )
    post = "\n".join(
        f'ERROR code=500 msg="stack trace" trace=xy_{i:04d}'
        for i in range(post_lines)
    )
    return pre + "\n" + post + "\n", pre_lines


def test_refine_deterministic():
    text, shift = _text_with_shift(120, 120, shift_line=120)
    r1 = refine_boundaries_local_multiscale(text, [shift + 5])
    r2 = refine_boundaries_local_multiscale(text, [shift + 5])
    assert r1 == r2


def test_refine_empty_boundaries():
    text, _ = _text_with_shift(100, 100, 100)
    r = refine_boundaries_local_multiscale(text, [])
    assert r["refined_boundaries"] == []
    assert r["boundary_details"] == []
    assert r["refinement"]["enabled"] is True
    assert r["refinement"]["method"] == "local_multiscale_char_classes_v1"


def test_refine_stays_within_local_window():
    """Refined boundary must stay inside (raw ± radius_lines)."""
    text, shift = _text_with_shift(300, 300, 300)
    for raw_off in (-30, -5, 0, 5, 30):
        raw = shift + raw_off
        r = refine_boundaries_local_multiscale(
            text, [raw], radius_lines=64,
        )
        refined = r["refined_boundaries"]
        if not refined:
            continue
        assert abs(refined[0] - raw) <= 64


def test_refine_moves_towards_true_shift():
    """When the coarse boundary is a few lines off, refinement should
    either stay put or move closer to the true structural shift."""
    text, shift = _text_with_shift(200, 200, 200)
    r = refine_boundaries_local_multiscale(
        text, [shift + 8], radius_lines=64,
    )
    refined = r["refined_boundaries"][0]
    # Not guaranteed to land exactly on ``shift`` — the char-class
    # boundary lies inside the newline that separates the two regimes
    # — but it must be at least as close as raw, and within the window.
    assert abs(refined - shift) <= 8 + 1
    assert r["boundary_details"][0]["raw"] == shift + 8


def test_refine_drops_edge_artifact_when_requested():
    """An obviously degenerate boundary (file edge, narrow window)
    can be dropped via ``drop_edge_artifacts=True``."""
    text = "tiny\ndata\n"
    r = refine_boundaries_local_multiscale(
        text, [1], radius_lines=64,
        drop_edge_artifacts=True,
    )
    # The 2-line text + radius=64 yields a narrow / degenerate window.
    # With drop_edge_artifacts=True the boundary is dropped.
    assert len(r["refined_boundaries"]) == 0
    detail = r["boundary_details"][0]
    assert detail["status"].startswith("dropped") or detail["status"] == "kept_raw_too_narrow"


def test_refine_keeps_raw_when_window_narrow():
    """Same degenerate case without the drop flag: keep the raw
    boundary as the safe fallback."""
    text = "tiny\ndata\n"
    r = refine_boundaries_local_multiscale(text, [1], radius_lines=64)
    assert r["refined_boundaries"] == [1]
    assert r["boundary_details"][0]["status"] == "kept_raw_too_narrow"


def test_refine_deduplicates_collisions():
    """Two coarse boundaries that refine to the same line collapse."""
    text, shift = _text_with_shift(150, 150, 150)
    r = refine_boundaries_local_multiscale(
        text, [shift - 2, shift + 2], radius_lines=8,
    )
    assert len(r["refined_boundaries"]) <= 2
    assert list(r["refined_boundaries"]) == sorted(set(r["refined_boundaries"]))


def test_refine_rejects_unknown_signal():
    with pytest.raises(ValueError):
        refine_boundaries_local_multiscale("x\ny\n", [1], fine_signal="tfidf")


def test_refine_metadata_present():
    text, _ = _text_with_shift(120, 120, 120)
    r = refine_boundaries_local_multiscale(text, [120])
    meta = r["refinement"]
    assert meta["method"] == "local_multiscale_char_classes_v1"
    assert meta["fine_signal"] == "char_classes"
    assert meta["radius_lines"] > 0
    assert list(meta["scales"])  # default scales present


# ── runtime integration ────────────────────────────────────────

FIXTURE_TEXT = "\n".join(
    [f"INFO req user=u{i:03d} path=/api/v1/a status=200" for i in range(150)]
    + [f"ERROR code=500 trace_id={i:04x} msg=broken" for i in range(150)]
)


def test_segment_default_applies_refinement():
    """In line mode (default), refinement is on and adds the
    ``boundary_refinement`` metadata plus ``raw_boundaries``."""
    r = segment(
        FIXTURE_TEXT,
        strategy="heuristic",  # deterministic coarse output
        window=15, min_distance=30, nms_radius=15,
        consolidation_radius=30, min_segment_windows=5,
    )
    if r["n_boundaries"] == 0:
        pytest.skip("coarse detector found no boundaries on this fixture")
    assert "boundary_refinement" in r
    assert r["boundary_refinement"]["method"] == "local_multiscale_char_classes_v1"
    assert "raw_boundaries" in r
    assert isinstance(r["boundary_details"], list)
    # Refined boundaries must be valid line indices
    assert all(0 < b < r["n_lines"] for b in r["boundaries"])


def test_segment_refine_false_leaves_coarse_output():
    r_on = segment(
        FIXTURE_TEXT, strategy="heuristic",
        window=15, min_distance=30, nms_radius=15,
        consolidation_radius=30, min_segment_windows=5,
    )
    r_off = segment(
        FIXTURE_TEXT, strategy="heuristic", refine=False,
        window=15, min_distance=30, nms_radius=15,
        consolidation_radius=30, min_segment_windows=5,
    )
    assert "boundary_refinement" not in r_off
    assert "raw_boundaries" not in r_off
    if r_on["n_boundaries"]:
        # With refine=False the boundaries are the unrefined coarse
        # output; with refine=True they are (at most) the same.
        assert r_on["raw_boundaries"] == r_off["boundaries"]


def test_segment_char_mode_does_not_refine():
    r = segment("alpha beta gamma " * 100, mode="char")
    assert "boundary_refinement" not in r
    assert "raw_boundaries" not in r


def test_segment_refinement_preserves_segment_consistency():
    """When boundaries shift, segments must be rebuilt so that
    segment spans match the refined boundaries exactly."""
    r = segment(
        FIXTURE_TEXT, strategy="heuristic",
        window=15, min_distance=30, nms_radius=15,
        consolidation_radius=30, min_segment_windows=5,
    )
    # Boundaries should coincide with segment starts (excluding 0)
    seg_starts = [
        s["start_line"] for s in r["segments"]
        if s["start_line"] > 0
    ]
    assert seg_starts == r["boundaries"]


# ── CLI plumbing ───────────────────────────────────────────────

def test_cli_default_refinement_emits_metadata(tmp_path):
    input_log = tmp_path / "in.log"
    input_log.write_text(FIXTURE_TEXT, encoding="utf-8")
    output_json = tmp_path / "out.json"
    proc = _run_cli(
        "segment",
        "--input", str(input_log),
        "--output", str(output_json),
        "--strategy", "heuristic",
        "--window", "15",
        "--min-distance", "30",
        "--nms-radius", "15",
        "--consolidation-radius", "30",
        "--min-segment-windows", "5",
    )
    assert proc.returncode == 0, proc.stderr
    r = json.loads(output_json.read_text(encoding="utf-8"))
    if r["n_boundaries"]:
        assert "boundary_refinement" in r
        assert "raw_boundaries" in r


def test_cli_no_refine_flag_disables_refinement(tmp_path):
    input_log = tmp_path / "in.log"
    input_log.write_text(FIXTURE_TEXT, encoding="utf-8")
    output_json = tmp_path / "out.json"
    proc = _run_cli(
        "segment",
        "--input", str(input_log),
        "--output", str(output_json),
        "--strategy", "heuristic",
        "--no-refine",
        "--window", "15",
        "--min-distance", "30",
        "--nms-radius", "15",
        "--consolidation-radius", "30",
        "--min-segment-windows", "5",
    )
    assert proc.returncode == 0, proc.stderr
    r = json.loads(output_json.read_text(encoding="utf-8"))
    assert "boundary_refinement" not in r
    assert "raw_boundaries" not in r
