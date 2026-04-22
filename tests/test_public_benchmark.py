"""Public-benchmark workflow tests.

Covers:
  - BGL line parsing on inline samples
  - derive_boundaries_from_labels behaviour on label flips
  - tolerance-based boundary matching
  - BGL / HDFS adapter end-to-end on committed smoke fixtures
  - CLI --dataset invocation produces a valid JSON artifact
  - synthetic --input-dir benchmark still works (back-compat)
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_here = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_here))

from benchmarks.adapters import bgl, hdfs, REGISTRY
from benchmarks.adapters.evaluation import (
    match_boundaries,
    merge_nearby_indices,
    derive_boundaries_from_labels,
)


# ── evaluation helpers ───────────────────────────────────────────

def test_match_boundaries_perfect():
    p, r, f = match_boundaries([100, 200], [100, 200], tolerance=5)
    assert (p, r, f) == (1.0, 1.0, 1.0)


def test_match_boundaries_within_tolerance():
    p, r, f = match_boundaries([102, 198], [100, 200], tolerance=5)
    assert p == 1.0 and r == 1.0


def test_match_boundaries_out_of_tolerance():
    p, r, f = match_boundaries([120], [100], tolerance=5)
    assert p == 0.0 and r == 0.0 and f == 0.0


def test_match_boundaries_partial_match():
    p, r, f = match_boundaries([100, 500], [100, 200, 300], tolerance=5)
    # 1 of 2 predicted matches; recall 1/3. Tolerances account for the
    # 4-decimal rounding applied by match_boundaries.
    assert p == pytest.approx(0.5, abs=1e-4)
    assert r == pytest.approx(1 / 3, abs=1e-4)


def test_merge_nearby_indices_collapses_cluster():
    assert merge_nearby_indices([10, 12, 30, 80], merge_window=5) == [10, 30, 80]


def test_merge_nearby_indices_unsorted():
    assert merge_nearby_indices([80, 10, 12], merge_window=5) == [10, 80]


# ── derive_boundaries_from_labels ────────────────────────────────

def test_derive_no_labels():
    assert derive_boundaries_from_labels([]) == []


def test_derive_single_phase():
    labels = ["a"] * 500
    assert derive_boundaries_from_labels(labels, min_run=100) == []


def test_derive_two_phases_clean():
    labels = ["a"] * 300 + ["b"] * 300
    # boundary at index 300
    assert derive_boundaries_from_labels(labels, min_run=100) == [300]


def test_derive_short_run_absorbed():
    # Brief 10-line flip inside a 500-a/500-a run should not create a boundary.
    labels = ["a"] * 500 + ["b"] * 10 + ["a"] * 490
    assert derive_boundaries_from_labels(labels, min_run=100) == []


def test_derive_merges_nearby():
    labels = ["a"] * 200 + ["b"] * 200 + ["c"] * 200
    # Transitions at 200 and 400; merge_window=50 does not collapse
    # since they are 200 apart
    out = derive_boundaries_from_labels(
        labels, min_run=100, merge_window=50,
    )
    assert out == [200, 400]


# ── BGL adapter on inline samples ────────────────────────────────

BGL_SAMPLE_LINES = [
    "- 1 2025.01.01 node 2025-01-01-00.00.00.000000 node RAS KERNEL INFO ok",
    "- 1 2025.01.01 node 2025-01-01-00.00.00.000001 node RAS KERNEL INFO ok",
    "KERNDTLB 1 2025.01.01 node 2025-01-01-00.00.00.000002 node RAS KERNEL FATAL oops",
    "KERNDTLB 1 2025.01.01 node 2025-01-01-00.00.00.000003 node RAS KERNEL FATAL oops",
    "- 1 2025.01.01 node 2025-01-01-00.00.00.000004 node RAS KERNEL INFO ok",
]


def test_bgl_parse_line():
    lab, msg = bgl._parse_line(BGL_SAMPLE_LINES[0])
    assert lab == "-"
    assert msg.endswith("ok")
    lab2, _ = bgl._parse_line(BGL_SAMPLE_LINES[2])
    assert lab2 == "KERNDTLB"


def test_bgl_parse_empty_line():
    assert bgl._parse_line("") == ("-", "")
    assert bgl._parse_line("   ") == ("-", "")


def test_bgl_parse_label_only():
    assert bgl._parse_line("KERNDTLB")[0] == "KERNDTLB"


def test_bgl_load_on_smoke_fixture():
    ds = bgl.load(_here / "benchmarks" / "datasets" / "public" / "bgl")
    assert ds.name == "bgl"
    assert ds.n_lines > 500
    assert ds.target_metadata["target_type"] == "derived_from_label_transitions"
    assert "alert_fraction" in ds.target_metadata
    # smoke fixture: 4000 lines with 4 derived boundaries expected
    assert len(ds.derived_boundaries) == 4
    for b in ds.derived_boundaries:
        assert 0 < b < ds.n_lines


def test_bgl_load_missing_log_raises(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        bgl.load(empty)


def test_bgl_load_max_lines_truncates(tmp_path):
    # Copy a tiny chunk into a temp dir
    src = (_here / "benchmarks" / "datasets" / "public" / "bgl"
           / "BGL_smoke.log").read_text(encoding="utf-8")
    tmp_log = tmp_path / "BGL.log"
    tmp_log.write_text(src, encoding="utf-8")
    ds = bgl.load(tmp_path, max_lines=250, min_run=50)
    assert ds.n_lines == 250


# ── HDFS adapter ────────────────────────────────────────────────

def test_hdfs_load_on_smoke_fixture():
    ds = hdfs.load(_here / "benchmarks" / "datasets" / "public" / "hdfs")
    assert ds.name == "hdfs"
    assert ds.n_lines > 500
    assert ds.target_metadata["target_type"] == "derived_from_severity_transitions"
    assert "disclaimer" in ds.target_metadata
    assert set(ds.labels).issubset({"INFO", "WARN", "WARNING", "ERROR",
                                    "FATAL", "DEBUG", "OTHER"})


def test_registry_contains_bgl_and_hdfs():
    assert "bgl" in REGISTRY
    assert "hdfs" in REGISTRY


# ── CLI --dataset invocation ─────────────────────────────────────

def _run_cli(*args) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(_here / "mentis_log_cli.py"), *args]
    return subprocess.run(cmd, capture_output=True, text=True)


def test_cli_benchmark_dataset_bgl(tmp_path):
    output = tmp_path / "bgl.json"
    proc = _run_cli(
        "benchmark",
        "--dataset", "bgl",
        "--data-dir", str(_here / "benchmarks" / "datasets" / "public" / "bgl"),
        "--output", str(output),
        "--strategies", "auto,heuristic",
    )
    assert proc.returncode == 0, proc.stderr
    assert output.exists()

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["mode"] == "dataset"
    assert report["dataset"] == "bgl"
    assert "target_metadata" in report
    assert report["target_metadata"]["target_type"] == "derived_from_label_transitions"
    assert report["strategies"] == ["auto", "heuristic"]
    assert len(report["per_file"]) == 1
    # Markdown summary is written by default
    assert output.with_suffix(".md").exists()


def test_cli_benchmark_dataset_requires_data_dir(tmp_path):
    output = tmp_path / "bgl.json"
    proc = _run_cli(
        "benchmark",
        "--dataset", "bgl",
        # missing --data-dir
        "--output", str(output),
    )
    # Should fail because data-dir is not provided
    assert proc.returncode != 0


def test_cli_benchmark_dir_still_works(tmp_path):
    output = tmp_path / "syn.json"
    proc = _run_cli(
        "benchmark",
        "--input-dir", str(_here / "benchmarks" / "synthetic"),
        "--output", str(output),
        "--strategies", "auto",
    )
    assert proc.returncode == 0, proc.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["mode"] == "directory"


def test_cli_benchmark_dir_and_dataset_mutually_exclusive(tmp_path):
    proc = _run_cli(
        "benchmark",
        "--dataset", "bgl",
        "--input-dir", str(tmp_path),
        "--output", str(tmp_path / "x.json"),
    )
    assert proc.returncode != 0


def test_cli_benchmark_dataset_no_summary_md(tmp_path):
    output = tmp_path / "bgl.json"
    proc = _run_cli(
        "benchmark",
        "--dataset", "bgl",
        "--data-dir", str(_here / "benchmarks" / "datasets" / "public" / "bgl"),
        "--output", str(output),
        "--no-summary-md",
    )
    assert proc.returncode == 0
    assert not output.with_suffix(".md").exists()


def test_committed_result_artifacts_exist_and_parse():
    """The shipped reference artifacts must be present and parseable."""
    results = _here / "benchmarks" / "results"
    for name in (
        "synthetic_default_vs_heuristic.json",
        "bgl_smoke_default_vs_heuristic.json",
        "hdfs_smoke_default_vs_heuristic.json",
    ):
        path = results / name
        assert path.exists(), f"missing reference artifact: {path}"
        report = json.loads(path.read_text(encoding="utf-8"))
        assert "strategies" in report
        assert "summary" in report
