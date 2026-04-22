"""Onset vs separator tests for the local multiscale refinement.

Covers the new onset extraction added on top of the existing
separator logic:

  1. onset <= separator for every refined boundary
  2. onset stays within the refinement window (same line-radius as
     the separator)
  3. onset is deterministic across repeated calls
  4. existing separator behaviour is unchanged by the onset addition
  5. public ``boundaries`` still use separator values by default
  6. ``public_boundary_semantics`` metadata is present and reads
     ``"separator"``
  7. the onset tuning kwargs (alpha / persistence) are honoured
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_here = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_here))

from refine import refine_boundaries_local_multiscale, _find_onset_index
from runtime import segment


# ── fixtures ───────────────────────────────────────────────────

def _text_with_shift(pre_lines: int, post_lines: int) -> tuple:
    pre = "\n".join(
        f"INFO status=200 user=u{i:03d} method=GET path=/api/v1/a"
        for i in range(pre_lines)
    )
    post = "\n".join(
        f'ERROR code=500 msg="stack trace" trace=xy_{i:04d}'
        for i in range(post_lines)
    )
    return pre + "\n" + post + "\n", pre_lines


# ── _find_onset_index direct unit tests ───────────────────────

def test_onset_index_on_sharp_peak_falls_back_to_sep():
    """A lone spike — no sustained neighbour — onset = sep."""
    agg = np.zeros(100, dtype=np.float64)
    agg[50] = 1.0  # isolated peak; no run
    onset = _find_onset_index(agg, 50, alpha=0.6, persistence=8)
    assert onset == 50


def test_onset_index_on_plateau_moves_left():
    """A long plateau at 0.8 with peak 1.0 at the right end — onset
    sits at the left edge of the plateau."""
    agg = np.zeros(100, dtype=np.float64)
    agg[20:61] = 0.8
    agg[60] = 1.0  # peak is the rightmost plateau cell
    onset = _find_onset_index(agg, 60, alpha=0.6, persistence=8)
    # threshold = 0.6 * 1.0 = 0.6 → plateau values (0.8) are above
    # threshold → onset walks back to the plateau's left edge.
    assert onset == 20


def test_onset_index_stops_at_dip():
    """A dip below threshold interrupts the run."""
    agg = np.zeros(100, dtype=np.float64)
    agg[20:41] = 0.8
    agg[41] = 0.1   # dip
    agg[42:61] = 0.9
    agg[60] = 1.0   # peak
    onset = _find_onset_index(agg, 60, alpha=0.6, persistence=4)
    # threshold = 0.6 → walking left from 60, dip at 41 stops us.
    assert onset == 42


def test_onset_index_persistence_gate():
    """A short run (< persistence) falls back to separator."""
    agg = np.zeros(100, dtype=np.float64)
    agg[58:61] = 0.8
    agg[60] = 1.0
    onset = _find_onset_index(agg, 60, alpha=0.6, persistence=8)
    # Run is only 3 long; persistence=8 → onset = sep
    assert onset == 60


def test_onset_index_zero_peak_returns_sep():
    agg = np.zeros(10, dtype=np.float64)
    assert _find_onset_index(agg, 5, alpha=0.6, persistence=4) == 5


# ── End-to-end onset tests on real text ───────────────────────

def test_onset_le_separator_always():
    text, shift = _text_with_shift(200, 200)
    r = refine_boundaries_local_multiscale(
        text, [shift - 3, shift + 5, shift + 25], radius_lines=80,
    )
    for d in r["boundary_details"]:
        if d.get("onset") is not None and d.get("separator") is not None:
            assert d["onset"] <= d["separator"]


def test_onset_stays_within_refinement_window():
    text, shift = _text_with_shift(200, 200)
    radius = 64
    r = refine_boundaries_local_multiscale(
        text, [shift + 4], radius_lines=radius,
    )
    d = r["boundary_details"][0]
    if d["onset"] is not None:
        assert abs(d["onset"] - d["raw"]) <= radius


def test_onset_deterministic():
    text, shift = _text_with_shift(200, 200)
    r1 = refine_boundaries_local_multiscale(
        text, [shift + 3], radius_lines=64,
    )
    r2 = refine_boundaries_local_multiscale(
        text, [shift + 3], radius_lines=64,
    )
    assert r1["boundary_details"] == r2["boundary_details"]


def test_separator_unchanged_vs_before_onset_addition():
    """The separator value must still equal the argmax of the local
    score — adding onset must not shift separator."""
    text, shift = _text_with_shift(250, 250)
    r = refine_boundaries_local_multiscale(
        text, [shift + 7], radius_lines=64,
    )
    d = r["boundary_details"][0]
    # separator equals the legacy 'refined' field
    assert d["separator"] == d["refined"]
    # Public boundaries list still uses separator values
    assert d["separator"] in r["refined_boundaries"] or d["status"].startswith("dropped")


def test_kept_raw_paths_set_separator_and_onset_to_raw():
    """When refinement cannot improve (narrow window etc.), both
    separator and onset collapse to the raw boundary."""
    text = "tiny\ndata\n"
    r = refine_boundaries_local_multiscale(text, [1], radius_lines=64)
    d = r["boundary_details"][0]
    assert d["raw"] == 1
    assert d["separator"] == 1
    assert d["onset"] == 1
    assert d["status"] == "kept_raw_too_narrow"


def test_metadata_declares_public_boundary_semantics():
    text, shift = _text_with_shift(120, 120)
    r = refine_boundaries_local_multiscale(text, [shift])
    meta = r["refinement"]
    assert meta["public_boundary_semantics"] == "separator"
    assert "onset_alpha" in meta
    assert "onset_persistence" in meta


def test_alpha_and_persistence_affect_onset():
    """Lower alpha or shorter persistence should widen the sustained
    run → onset can move earlier (never later)."""
    text, shift = _text_with_shift(300, 300)
    r_tight = refine_boundaries_local_multiscale(
        text, [shift + 5], radius_lines=80,
        onset_alpha=0.9, onset_persistence=32,
    )
    r_loose = refine_boundaries_local_multiscale(
        text, [shift + 5], radius_lines=80,
        onset_alpha=0.4, onset_persistence=4,
    )
    tight = r_tight["boundary_details"][0]
    loose = r_loose["boundary_details"][0]
    # Separator must be identical under both settings (alpha only
    # affects onset)
    assert tight["separator"] == loose["separator"]
    # Looser thresholds let onset walk further left or stay put
    assert loose["onset"] <= tight["onset"]


# ── runtime integration ──────────────────────────────────────

def test_public_boundaries_use_separator_semantics():
    text, shift = _text_with_shift(400, 400)
    r = segment(
        text, strategy="heuristic",
        window=15, min_distance=30, nms_radius=15,
        consolidation_radius=30, min_segment_windows=5,
    )
    if r["n_boundaries"] == 0:
        pytest.skip("coarse detector found no boundaries here")
    # The public boundaries must correspond to the separator values
    # in boundary_details.
    separators = [
        d["separator"] for d in r["boundary_details"]
        if d.get("separator") is not None
    ]
    # All public boundaries should appear in the separator set
    # (dedup + sorting may remove some, but not introduce new)
    for b in r["boundaries"]:
        assert b in separators


def test_runtime_exposes_public_boundary_semantics():
    text, shift = _text_with_shift(400, 400)
    r = segment(
        text, strategy="heuristic",
        window=15, min_distance=30, nms_radius=15,
        consolidation_radius=30, min_segment_windows=5,
    )
    if r["n_boundaries"] == 0:
        pytest.skip("coarse detector found no boundaries here")
    assert r["boundary_refinement"]["public_boundary_semantics"] == "separator"


def test_runtime_boundary_details_contain_onset():
    text, shift = _text_with_shift(400, 400)
    r = segment(
        text, strategy="heuristic",
        window=15, min_distance=30, nms_radius=15,
        consolidation_radius=30, min_segment_windows=5,
    )
    if r["n_boundaries"] == 0:
        pytest.skip("coarse detector found no boundaries here")
    for d in r["boundary_details"]:
        assert "separator" in d
        assert "onset" in d
        if d["separator"] is not None and d["onset"] is not None:
            assert d["onset"] <= d["separator"]
