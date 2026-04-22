"""Plot-level tests for the char/line mode semantics fix.

Verifies:
  1. plot_comparison accepts mode="line" without raising
  2. _line_length_signal returns one value per line (line-level raw)
  3. CLI passes mode=args.mode into plot_comparison
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

_here = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_here))

# Skip the whole module if matplotlib is not installed.
pytest.importorskip("matplotlib")

# Force the non-interactive Agg backend BEFORE any pyplot import.
# Windows' default TkAgg backend leaks Tk resources across tests and
# produces intermittent "invalid argument" failures when many figures
# are created in a single pytest run.
import matplotlib as _mpl  # noqa: E402
_mpl.use("Agg", force=True)

from plot import _line_length_signal, plot_comparison  # noqa: E402
import mentis_log_cli  # noqa: E402


@pytest.fixture(autouse=True)
def _close_matplotlib_figures():
    """Close all figures between tests to avoid state leak on Windows."""
    yield
    import matplotlib.pyplot as plt
    plt.close("all")


# ── _line_length_signal ───────────────────────────────────────────

def test_line_length_signal_shape_matches_line_count():
    text = "short\na longer line here\ntiny\n"
    sig = _line_length_signal(text)
    assert sig.ndim == 1
    # splitlines drops trailing newlines, giving 3 lines
    assert sig.shape == (3,)


def test_line_length_signal_normalised_to_0_1():
    text = "a\nbb\nccc\ndddd"
    sig = _line_length_signal(text)
    assert float(sig.min()) >= 0.0
    assert float(sig.max()) <= 1.0
    # Longest line normalises to 1.0
    assert float(sig.max()) == 1.0


def test_line_length_signal_empty_input():
    assert _line_length_signal("").shape == (0,)
    assert _line_length_signal("\n\n\n").shape == (3,)


# ── plot_comparison(mode="line") ──────────────────────────────────

@pytest.fixture
def _small_line_mode_inputs():
    """A text + curve that use the same (line) unit."""
    lines = [f"INFO req id={i} user=alice ok=true" for i in range(40)] + [
        f"ERROR db timeout trace=x{i}" for i in range(40)
    ]
    text = "\n".join(lines)
    # Fake curve of length == n_lines so the panels are comparable.
    n = text.count("\n") + 1
    curve = np.zeros(n, dtype=np.float64)
    curve[35:45] = np.linspace(0.0, 1.0, 10)  # fake peak around the shift
    boundaries = [40]
    route = {"selected": "GKM", "coverage": 0.1, "reason": "test"}
    return text, curve.tolist(), boundaries, route


def test_plot_comparison_line_mode_returns_figure(_small_line_mode_inputs, tmp_path):
    text, curve, boundaries, route = _small_line_mode_inputs
    out = tmp_path / "cmp_line.png"
    fig = plot_comparison(
        text, curve, boundaries, route,
        mode="line", output_path=str(out),
    )
    assert fig is not None
    assert out.exists() and out.stat().st_size > 0


def test_plot_comparison_char_mode_still_works(_small_line_mode_inputs, tmp_path):
    text, curve, boundaries, route = _small_line_mode_inputs
    out = tmp_path / "cmp_char.png"
    fig = plot_comparison(
        text, curve, boundaries, route,
        mode="char", output_path=str(out),
    )
    assert fig is not None
    assert out.exists() and out.stat().st_size > 0


def test_plot_comparison_rejects_unknown_mode(_small_line_mode_inputs):
    text, curve, boundaries, route = _small_line_mode_inputs
    with pytest.raises(ValueError):
        plot_comparison(text, curve, boundaries, route, mode="token")


def test_plot_comparison_filters_out_of_range_boundaries(
    _small_line_mode_inputs, tmp_path,
):
    """Boundaries beyond N must not break the plot."""
    text, curve, _, route = _small_line_mode_inputs
    # Inject an obviously-invalid boundary
    boundaries = [5, 10**9, 50]
    out = tmp_path / "cmp_filter.png"
    fig = plot_comparison(
        text, curve, boundaries, route,
        mode="line", output_path=str(out),
    )
    assert fig is not None


# ── CLI wires --mode through to plot_comparison ───────────────────

def test_cli_passes_mode_into_plot_comparison(tmp_path):
    input_log = tmp_path / "in.log"
    input_log.write_text(
        "\n".join([f"INFO a={i} u=alice" for i in range(60)] +
                 [f"ERROR b={i} x=broken" for i in range(60)]),
        encoding="utf-8",
    )
    output_png = tmp_path / "out.png"
    output_json = tmp_path / "out.json"

    with patch("plot.plot_comparison") as spy:
        # Return a truthy dummy so the CLI can continue if needed.
        spy.return_value = object()
        rc = mentis_log_cli.main([
            "segment",
            "--mode", "line",
            "--input", str(input_log),
            "--output", str(output_json),
            "--plot-comparison", str(output_png),
            # Fixture-size-appropriate peak params
            "--min-distance", "20",
            "--nms-radius", "10",
            "--consolidation-radius", "20",
            "--min-segment-windows", "5",
        ])
    assert rc == 0
    assert spy.called
    kwargs = spy.call_args.kwargs
    assert kwargs.get("mode") == "line"
    assert kwargs.get("output_path") == str(output_png)


def test_cli_default_mode_is_line_for_plot_comparison(tmp_path):
    """After the release-hardening refactor the CLI default mode is
    'line' — log files are the primary product input."""
    input_log = tmp_path / "in.log"
    input_log.write_text("alpha beta gamma " * 200, encoding="utf-8")
    output_png = tmp_path / "out.png"
    output_json = tmp_path / "out.json"

    with patch("plot.plot_comparison") as spy:
        spy.return_value = object()
        rc = mentis_log_cli.main([
            "segment",
            "--input", str(input_log),
            "--output", str(output_json),
            "--plot-comparison", str(output_png),
        ])
    assert rc == 0
    assert spy.called
    assert spy.call_args.kwargs.get("mode") == "line"


def test_cli_explicit_char_mode_still_works(tmp_path):
    """Explicit --mode char still routes through char-mode plotting."""
    input_log = tmp_path / "in.log"
    input_log.write_text("alpha beta gamma " * 200, encoding="utf-8")
    output_png = tmp_path / "out.png"
    output_json = tmp_path / "out.json"

    with patch("plot.plot_comparison") as spy:
        spy.return_value = object()
        rc = mentis_log_cli.main([
            "segment",
            "--input", str(input_log),
            "--output", str(output_json),
            "--plot-comparison", str(output_png),
            "--mode", "char",
        ])
    assert rc == 0
    assert spy.call_args.kwargs.get("mode") == "char"
