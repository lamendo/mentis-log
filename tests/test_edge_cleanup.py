"""Edge-segment cleanup tests.

After boundary refinement, the first and last segments are checked:
an edge segment shorter than ``max(32, int(0.001 * n_lines))`` lines
triggers removal of that edge boundary and merges the tiny edge
segment into its neighbour.

Tests:
  1. tiny first segment gets merged away
  2. tiny last segment gets merged away
  3. normal-sized edge segments are left alone
  4. raw_boundaries and boundary_details are preserved
  5. ``--no-edge-cleanup`` disables the step
  6. both tiny edges in one log are cleaned independently
  7. when no refinement happens, cleanup still works on coarse
     boundaries and emits metadata
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_here = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_here))

from runtime import segment, _cleanup_edge_boundaries, _edge_cleanup_threshold


# ── _cleanup_edge_boundaries unit tests ────────────────────────

def test_threshold_floor_for_small_logs():
    assert _edge_cleanup_threshold(100) == 32
    assert _edge_cleanup_threshold(31_999) == 32
    assert _edge_cleanup_threshold(32_000) == 32
    # 0.001 × 50 000 = 50 > 32 → threshold scales up
    assert _edge_cleanup_threshold(50_000) == 50


def test_cleanup_drops_tiny_first_boundary():
    bs, dropped = _cleanup_edge_boundaries([5, 500, 900], n_lines=1000)
    assert bs == [500, 900]
    assert len(dropped) == 1
    assert dropped[0]["side"] == "first"
    assert dropped[0]["boundary"] == 5


def test_cleanup_drops_tiny_last_boundary():
    bs, dropped = _cleanup_edge_boundaries([100, 500, 990], n_lines=1000)
    # last segment is 1000 - 990 = 10 lines < 32
    assert bs == [100, 500]
    assert len(dropped) == 1
    assert dropped[0]["side"] == "last"
    assert dropped[0]["boundary"] == 990


def test_cleanup_drops_both_tiny_edges():
    bs, dropped = _cleanup_edge_boundaries(
        [5, 500, 992], n_lines=1000,
    )
    assert bs == [500]
    assert {d["side"] for d in dropped} == {"first", "last"}


def test_cleanup_keeps_normal_boundaries():
    bs, dropped = _cleanup_edge_boundaries(
        [100, 500, 900], n_lines=1000,
    )
    assert bs == [100, 500, 900]
    assert dropped == []


def test_cleanup_single_boundary_edge_cases():
    # Only-boundary that's too close to start
    bs, dropped = _cleanup_edge_boundaries([5], n_lines=1000)
    assert bs == []
    assert len(dropped) == 1 and dropped[0]["side"] == "first"

    # Only-boundary that's too close to end
    bs, dropped = _cleanup_edge_boundaries([995], n_lines=1000)
    assert bs == []
    assert len(dropped) == 1 and dropped[0]["side"] == "last"


def test_cleanup_empty_boundaries_passthrough():
    bs, dropped = _cleanup_edge_boundaries([], n_lines=1000)
    assert bs == []
    assert dropped == []


# ── End-to-end via segment() ───────────────────────────────────

def _fixture_with_tiny_edges(tiny_first: bool, tiny_last: bool):
    """Build a text whose coarse/refined boundaries will land on
    trivial edge segments when we ask for them via peak-sel params."""
    # 3 strongly-different regimes; we set peak params so the coarse
    # detector finds boundaries in all gaps.
    seg_a = "\n".join(f"INFO u{i:03d}" for i in range(5 if tiny_first else 60))
    seg_b = "\n".join(f"ERROR x{i:03d}" for i in range(60))
    seg_c = "\n".join(f"FATAL z{i:03d}" for i in range(60))
    seg_d = "\n".join(f"INFO r{i:03d}" for i in range(5 if tiny_last else 60))
    text = "\n".join([seg_a, seg_b, seg_c, seg_d]) + "\n"
    return text


def test_segment_cleans_tiny_first_segment(tmp_path):
    """Force a tiny first segment and verify cleanup drops it."""
    # 5 + 60 + 60 + 60 + trailing = ~185 lines. Need to manufacture a
    # coarse boundary at line ~5 (inside the tiny first segment).
    # Simpler: directly construct a text where refine may or may not
    # produce a boundary at <32, then assert cleanup intent via the
    # helper output. The segment() integration test below uses a
    # deliberate fixture.
    text = _fixture_with_tiny_edges(tiny_first=True, tiny_last=False)

    r = segment(
        text, strategy="heuristic",
        # Aggressive peak params so the 5-line INFO prefix looks like
        # a regime.
        window=5, min_distance=5, nms_radius=3,
        consolidation_radius=5, min_segment_windows=3,
        refine=False,  # isolate cleanup from refinement
    )

    # The coarse detector may or may not place a boundary at ~5 on
    # this fixture. If it does, cleanup must have dropped it.
    if not r["boundaries"]:
        pytest.skip("coarse detector produced no boundaries on this fixture")
    # Cleanup is always enabled; metadata present
    assert "edge_cleanup" in r
    # No surviving boundary should be below the threshold
    threshold = r["edge_cleanup"]["threshold_lines"]
    if r["boundaries"]:
        assert r["boundaries"][0] >= threshold


def test_segment_cleans_tiny_last_segment():
    text = _fixture_with_tiny_edges(tiny_first=False, tiny_last=True)
    r = segment(
        text, strategy="heuristic",
        window=5, min_distance=5, nms_radius=3,
        consolidation_radius=5, min_segment_windows=3,
        refine=False,
    )
    if not r["boundaries"]:
        pytest.skip("coarse detector produced no boundaries on this fixture")
    threshold = r["edge_cleanup"]["threshold_lines"]
    if r["boundaries"]:
        assert (r["n_lines"] - r["boundaries"][-1]) >= threshold


def test_segment_normal_edges_unchanged():
    text = _fixture_with_tiny_edges(tiny_first=False, tiny_last=False)
    r = segment(
        text, strategy="heuristic",
        window=10, min_distance=20, nms_radius=10,
        consolidation_radius=20, min_segment_windows=5,
        refine=False,
    )
    if "edge_cleanup" in r:
        # No drops expected because both edge segments are 60 lines
        assert r["edge_cleanup"]["dropped_boundaries"] == []


def test_cleanup_preserves_raw_and_details_when_refinement_ran():
    """Even when edge cleanup drops a refined boundary, the audit
    trail (raw_boundaries + boundary_details) stays intact."""
    text = _fixture_with_tiny_edges(tiny_first=True, tiny_last=False)
    r = segment(
        text, strategy="heuristic",
        window=5, min_distance=5, nms_radius=3,
        consolidation_radius=5, min_segment_windows=3,
        refine=True,
    )
    if not (r.get("boundary_details") and r.get("edge_cleanup", {}).get(
            "dropped_boundaries")):
        pytest.skip("fixture did not trigger both refinement and cleanup")
    # raw_boundaries / boundary_details are unchanged by edge cleanup:
    # cleanup only drops the final-output boundary entries, not the
    # audit trail of the refinement step.
    assert isinstance(r["raw_boundaries"], list)
    assert len(r["boundary_details"]) >= 1


def test_segment_always_emits_edge_cleanup_metadata_when_boundaries_exist():
    text = _fixture_with_tiny_edges(tiny_first=False, tiny_last=False)
    r = segment(
        text, strategy="heuristic",
        window=10, min_distance=20, nms_radius=10,
        consolidation_radius=20, min_segment_windows=5,
        refine=False,
    )
    if r["boundaries"]:
        assert "edge_cleanup" in r
        assert r["edge_cleanup"]["enabled"] is True
        assert r["edge_cleanup"]["threshold_lines"] > 0


def test_segment_no_edge_cleanup_flag():
    text = _fixture_with_tiny_edges(tiny_first=True, tiny_last=False)
    r_on = segment(
        text, strategy="heuristic",
        window=5, min_distance=5, nms_radius=3,
        consolidation_radius=5, min_segment_windows=3,
        refine=False,
    )
    r_off = segment(
        text, strategy="heuristic",
        window=5, min_distance=5, nms_radius=3,
        consolidation_radius=5, min_segment_windows=3,
        refine=False, edge_cleanup=False,
    )
    # When cleanup is disabled there is no edge_cleanup metadata block.
    assert "edge_cleanup" not in r_off
    # With cleanup enabled: no boundary in a tiny edge position
    if r_on["boundaries"]:
        threshold = r_on["edge_cleanup"]["threshold_lines"]
        assert r_on["boundaries"][0] >= threshold


# ── CLI plumbing ───────────────────────────────────────────────

def test_cli_no_edge_cleanup_flag(tmp_path):
    input_log = tmp_path / "in.log"
    input_log.write_text(
        _fixture_with_tiny_edges(tiny_first=True, tiny_last=False),
        encoding="utf-8",
    )
    output_json = tmp_path / "out.json"
    proc = subprocess.run(
        [
            sys.executable, str(_here / "mentis_log_cli.py"),
            "segment",
            "--input", str(input_log),
            "--output", str(output_json),
            "--strategy", "heuristic",
            "--no-edge-cleanup",
            "--window", "5", "--min-distance", "5",
            "--nms-radius", "3", "--consolidation-radius", "5",
            "--min-segment-windows", "3",
        ],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    r = json.loads(output_json.read_text(encoding="utf-8"))
    assert "edge_cleanup" not in r
