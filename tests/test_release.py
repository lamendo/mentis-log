"""Release-hardening tests.

Covers:
  1. Default segment() uses the recommended strategy (auto + line +
     tokens + JSD + rolling + min_freq=2).
  2. Old heuristic path still works when explicitly requested.
  3. CLI parses without error for each subcommand's --help.
  4. benchmark subcommand runs on the committed synthetic fixtures
     and produces a sensible JSON report.
  5. Existing behaviour remains stable where explicitly requested.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_here = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_here))

from runtime import segment
import mentis_log_cli


FIXTURE_TEXT = "\n".join(
    [f"INFO req id={i} user=alice ok=true" for i in range(60)]
    + [f"ERROR db timeout trace=x{i}" for i in range(60)]
    + [f"FATAL oom heap=99 node=node_{i:02d}" for i in range(60)]
)


# ── Default strategy ──────────────────────────────────────────────

def test_default_segment_uses_auto_line_tokens():
    """The library default must match the release-hardening spec:
    strategy=auto, mode=line, and in line mode auto resolves to
    q_jsd + tokens + rolling + min_token_freq=2."""
    r = segment(FIXTURE_TEXT)
    assert r["strategy"] == "auto"
    assert r["mode"] == "line"
    assert r["scoring"] == "q_jsd"
    assert r["signal_type"] == "tokens"
    assert r["q_mode"] == "rolling"
    assert r["min_token_freq"] == 2


def test_default_char_mode_routes_to_heuristic():
    """In char mode, strategy='auto' must route to the bit-exact
    heuristic path (so parity with the reference pipeline holds)."""
    r = segment("alpha beta gamma " * 100, mode="char")
    assert r["strategy"] == "auto"
    assert r["scoring"] == "heuristic"


# ── Explicit heuristic still works ────────────────────────────────

def test_explicit_heuristic_strategy():
    r = segment(FIXTURE_TEXT, strategy="heuristic")
    assert r["strategy"] == "heuristic"
    assert r["scoring"] == "heuristic"
    # No q-align diagnostics leaked in
    assert "signal_type" not in r
    assert "q_mode" not in r


def test_explicit_kwargs_override_strategy():
    """User-supplied scoring / q_mode / signal_type override the
    strategy-picked defaults."""
    r = segment(
        FIXTURE_TEXT,
        strategy="auto",
        scoring="q_kl",
        q_mode="global",
        signal_type="line_length",
    )
    assert r["strategy"] == "auto"
    assert r["scoring"] == "q_kl"
    assert r["q_mode"] == "global"
    assert r["signal_type"] == "line_length"


def test_unknown_strategy_raises():
    with pytest.raises(ValueError):
        segment(FIXTURE_TEXT, strategy="magic")


# ── CLI help parses ───────────────────────────────────────────────

def _run_cli(*args, expect_ok: bool = True) -> subprocess.CompletedProcess:
    py = sys.executable
    cmd = [py, str(_here / "mentis_log_cli.py"), *args]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if expect_ok:
        assert proc.returncode == 0, f"cli failed: {proc.stderr}"
    return proc


def test_cli_top_help_parses():
    proc = _run_cli("--help")
    assert "segment" in proc.stdout
    assert "plot" in proc.stdout
    assert "benchmark" in proc.stdout


def test_cli_segment_help_parses():
    proc = _run_cli("segment", "--help")
    # Basic group is visible
    assert "basic" in proc.stdout.lower()
    # Advanced group header present
    assert "advanced" in proc.stdout.lower()
    # Strategy visible at the top level
    assert "--strategy" in proc.stdout


def test_cli_benchmark_help_parses():
    proc = _run_cli("benchmark", "--help")
    assert "--input-dir" in proc.stdout
    assert "--strategies" in proc.stdout


def test_cli_plot_help_parses():
    proc = _run_cli("plot", "--help")
    assert "--comparison" in proc.stdout


# ── Benchmark subcommand runs ─────────────────────────────────────

def test_benchmark_runs_on_synthetic_fixtures(tmp_path):
    """The committed synthetic fixtures must be benchmarkable out of
    the box — the primary product test for the `benchmark` command."""
    fixtures_dir = _here / "benchmarks" / "synthetic"
    assert fixtures_dir.is_dir()
    # At least three fixtures committed
    log_count = len(list(fixtures_dir.glob("*.log")))
    assert log_count >= 3

    output = tmp_path / "bench.json"
    proc = _run_cli(
        "benchmark",
        "--input-dir", str(fixtures_dir),
        "--output", str(output),
        "--strategies", "auto,heuristic",
    )
    assert output.exists()

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["strategies"] == ["auto", "heuristic"]
    assert len(report["per_file"]) == log_count
    # Every file has annotations committed
    assert all(entry["annotated"] for entry in report["per_file"])
    # Summary structure
    for strat in ("auto", "heuristic"):
        s = report["summary"][strat]
        assert s["annotated_files"] == log_count
        assert s["mean_f1"] is not None
        assert 0.0 <= s["mean_f1"] <= 1.0


def test_benchmark_without_annotations(tmp_path):
    """Benchmark should still run when no JSON annotations are present
    (reports runtime + boundary counts, skips F1)."""
    log = tmp_path / "only.log"
    log.write_text("INFO a\n" * 200 + "ERROR b\n" * 200, encoding="utf-8")

    output = tmp_path / "bench.json"
    proc = _run_cli(
        "benchmark",
        "--input-dir", str(tmp_path),
        "--output", str(output),
        "--strategies", "auto",
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["per_file"][0]["annotated"] is False
    # Runtime is still populated, F1 summary is None
    assert report["summary"]["auto"]["total_runtime_seconds"] >= 0
    assert report["summary"]["auto"]["mean_f1"] is None


# ── Stability of explicit behaviours ──────────────────────────────

def test_char_mode_strategy_auto_matches_heuristic_output():
    """In char mode, strategy='auto' must produce the same result as
    strategy='heuristic' (both route through the bit-exact path)."""
    text = "The quick brown fox. " * 50
    a = segment(text, mode="char", strategy="auto")
    h = segment(text, mode="char", strategy="heuristic")
    assert a["route"]["selected"] == h["route"]["selected"]
    assert a["n_segments"] == h["n_segments"]
    for sa, sh in zip(a["segments"], h["segments"]):
        assert sa["start_idx"] == sh["start_idx"]
        assert sa["end_idx"] == sh["end_idx"]
