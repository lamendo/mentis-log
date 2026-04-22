"""Demo plots for the Mentis log-segmentation extract.

Two functions:
  - plot_segmentation(...)  single-panel "sales" plot.
  - plot_comparison(...)     two-panel raw-vs-structured "aha" plot.

Design goals:
  1. Make the regime structure obvious in under three seconds.
  2. Keep the plot clean enough to share (HN / Reddit / slide).
  3. Make the policy-router decision visible as a coloured badge.

Optional dependency: matplotlib. Core extract has no plot dependency.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np


# Coverage-branch colour code — the single most important visual cue
_ROUTE_COLORS = {
    "HGB": "#C73E1D",    # sparse / event-like — red-orange
    "GKM": "#F18F01",    # structured — amber
    "LXM": "#2E86AB",    # dense / symmetric — blue
    "none": "#888888",
}

_CURVE_COLOR = "#1f3b5a"   # deep blue-grey for the curve
_PEAK_COLOR = "#C73E1D"    # red-orange for boundary peaks
_SEG_ALT = ("#f8f9fa", "#edf4fb")

# Colour cues for the three structural profiles (subtle, not alarming).
_PROFILE_COLORS = {
    "stable":     "#2E7D32",   # muted green
    "transition": "#B06A00",   # muted amber
    "volatile":   "#B03030",   # muted red
}


def _apply_clean_style(ax) -> None:
    """Tufte-ish minimal style: no top/right spines, light grid, clean ticks."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#888888")
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(axis="both", colors="#555555", labelsize=9, length=3)
    ax.grid(False)


def _byte_signal(text: str) -> np.ndarray:
    """Raw per-character byte value, normalised to [0, 1]."""
    if not text:
        return np.zeros(0, dtype=np.float32)
    arr = np.empty(len(text), dtype=np.float32)
    for i, ch in enumerate(text):
        b = ch.encode("utf-8", errors="replace")
        arr[i] = b[0] / 255.0 if b else 0.0
    return arr


def _line_length_signal(text: str) -> np.ndarray:
    """Raw per-line signal: line length normalised to [0, 1].

    Returns one value per line. Used by the line-mode comparison plot so
    that both panels live in the same (line-index) space — otherwise
    a per-char top panel and per-line bottom panel are not comparable.
    """
    if not text:
        return np.zeros(0, dtype=np.float32)
    lines = text.splitlines()
    if not lines:
        return np.zeros(0, dtype=np.float32)
    lengths = np.asarray([len(line) for line in lines], dtype=np.float32)
    m = float(lengths.max()) if lengths.size else 1.0
    if m < 1.0:
        m = 1.0
    return lengths / m


def _route_badge(route: Dict[str, Any]) -> str:
    """Top-right coloured badge — route only, no coverage.

    Coverage is not meaningful for every route (e.g. Q_JSD emits
    ``coverage=None``), so keeping it would be inconsistent. The badge
    now always shows just the selected route.
    """
    return f"route: {route.get('selected', '?')}"


def _draw_stats_block(ax, route: Dict[str, Any], n_boundaries: int) -> None:
    """Compact monospaced stats strip inside the plot area.

    One line: ``route <name>   boundaries <n>``. Coverage is omitted
    because it is None for the Q-aligned routes.
    """
    sel = route.get("selected", "?")
    text = f"route  {sel}    boundaries  {n_boundaries}"
    ax.text(
        0.005, 0.97, text,
        transform=ax.transAxes, ha="left", va="top",
        fontsize=9, color="#333333",
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.4",
                  facecolor="white", edgecolor="#cccccc", linewidth=0.6),
        zorder=5,
    )


def _draw_segments(ax, curve_len: int, boundaries: Sequence[int]) -> None:
    """Alternating very-subtle segment backgrounds."""
    splits = [0] + sorted(int(b) for b in boundaries) + [curve_len]
    for i in range(len(splits) - 1):
        ax.axvspan(
            splits[i], splits[i + 1],
            facecolor=_SEG_ALT[i % 2], alpha=0.7, linewidth=0, zorder=0,
        )


def _draw_curve(ax, curve: np.ndarray, color: str = _CURVE_COLOR) -> None:
    """Curve with subtle fill-below for visual mass."""
    x = np.arange(len(curve))
    ax.fill_between(x, 0, curve, color=color, alpha=0.12, linewidth=0, zorder=1)
    ax.plot(x, curve, color=color, linewidth=1.8, alpha=0.95, zorder=2)


def _draw_annotations(
    ax,
    curve_len: int,
    annotations: Optional[Sequence[Dict[str, Any]]],
) -> None:
    """Render small profile labels at the midpoint of each annotated
    segment. ``annotations`` is a list of dicts with keys
    ``start`` / ``end`` / ``profile`` (positions in curve-index space).
    """
    if not annotations:
        return
    ymin, ymax = ax.get_ylim()
    y = ymax * 0.96
    for ann in annotations:
        start = int(ann.get("start", 0))
        end = int(ann.get("end", curve_len))
        profile = str(ann.get("profile", ""))
        if end <= start:
            continue
        mid = (start + end) // 2
        if mid < 0 or mid >= curve_len:
            continue
        color = _PROFILE_COLORS.get(profile, "#444444")
        ax.text(
            mid, y, profile,
            ha="center", va="top", fontsize=8.5, color="white",
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.2",
                      facecolor=color, edgecolor="none", alpha=0.85),
            zorder=5,
        )


def _draw_boundaries(ax, curve: np.ndarray, boundaries: Sequence[int]) -> None:
    """Vertical dashed lines + peak dots with white halo."""
    ymin, ymax = ax.get_ylim()
    for b in boundaries:
        ax.axvline(b, linestyle="--", linewidth=1.0,
                   color=_PEAK_COLOR, alpha=0.55, zorder=2)
    if len(boundaries) == 0 or len(curve) == 0:
        return
    idx = np.asarray([int(b) for b in boundaries if 0 <= int(b) < len(curve)])
    if idx.size == 0:
        return
    # white halo
    ax.scatter(idx, curve[idx], s=80, color="white",
               edgecolors="white", linewidths=2.5, zorder=3)
    # coloured peak dot
    ax.scatter(idx, curve[idx], s=36, color=_PEAK_COLOR,
               edgecolors="white", linewidths=1.0, zorder=4,
               label="Detected change")


def plot_segmentation(
    curve: Sequence[float],
    boundaries: Sequence[int],
    route: Optional[Dict[str, Any]] = None,
    *,
    title: str = "Mentis — Structural Change Detection",
    subtitle: Optional[str] = None,
    output_path: Optional[str] = None,
    figsize=(14, 5),
    dpi: int = 150,
    show: bool = False,
    x_label: str = "Sequence position",
    annotations: Optional[Sequence[Dict[str, Any]]] = None,
) -> "object":
    """Single-panel sales plot.

    Returns the matplotlib Figure. Saves to `output_path` if given.
    """
    import matplotlib.pyplot as plt

    c = np.asarray(curve, dtype=np.float64)
    N = len(c)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    _draw_segments(ax, N, boundaries)
    _draw_curve(ax, c)

    # set ylim before boundary markers use them
    ymax = max(1e-6, float(np.nanmax(c))) if N else 1.0
    ax.set_ylim(-0.02 * ymax, ymax * 1.08)
    _draw_boundaries(ax, c, boundaries)

    # Title + coverage-branch badge
    ax.set_title(title, fontsize=15, weight="bold", loc="left",
                 color="#222222", pad=16)
    if route is not None:
        sel = route.get("selected", "?")
        badge_color = _ROUTE_COLORS.get(sel, "#555555")
        badge_text = _route_badge(route)
        ax.text(
            0.995, 1.02, badge_text,
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=10.5, color="white", weight="bold",
            bbox=dict(boxstyle="round,pad=0.42",
                      facecolor=badge_color, edgecolor="none"),
        )

    if subtitle:
        ax.text(0.0, 1.02, subtitle, transform=ax.transAxes,
                ha="left", va="bottom", fontsize=10, color="#555555")

    ax.set_xlabel(x_label, fontsize=10, color="#444444")
    ax.set_ylabel("Structural change signal", fontsize=10, color="#444444")
    ax.set_xlim(0, N - 1 if N else 1)

    # Self-documenting stats block
    if route is not None:
        _draw_stats_block(ax, route, len(list(boundaries)))

    _draw_annotations(ax, N, annotations)

    _apply_clean_style(ax)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight",
                    facecolor="white")
    if show:
        plt.show()
    return fig


def plot_comparison(
    text: str,
    curve: Sequence[float],
    boundaries: Sequence[int],
    route: Optional[Dict[str, Any]] = None,
    *,
    mode: str = "char",
    title: Optional[str] = None,
    output_path: Optional[str] = None,
    figsize=(14, 8),
    dpi: int = 150,
    show: bool = False,
    x_label: str = "Sequence position",
    annotations: Optional[Sequence[Dict[str, Any]]] = None,
) -> "object":
    """Two-panel before/after plot.

    Top: raw view of the input (byte-per-char in ``mode='char'``,
    length-per-line in ``mode='line'``).
    Bottom: policy-routed instability curve with segments highlighted.

    Both panels must share an x-axis unit. In ``mode='char'`` the curve
    has one value per char; in ``mode='line'`` it has one value per
    line. The raw signal is chosen to match.
    """
    import matplotlib.pyplot as plt

    if mode not in ("char", "line"):
        raise ValueError(f"Unknown mode {mode!r}; expected 'char' or 'line'")

    c = np.asarray(curve, dtype=np.float64)
    if mode == "line":
        raw = _line_length_signal(text)
        top_title = "Raw log stream (per line) — no structure extracted"
        default_title = (
            "Same file. Top: raw line view. "
            "Bottom: after structural change measurement."
        )
        raw_ylabel = "Line length"
    else:
        raw = _byte_signal(text)
        top_title = "Raw log stream (as sequence) — no structure extracted"
        default_title = (
            "Same sequence. Top: raw stream view. "
            "Bottom: after structural change measurement."
        )
        raw_ylabel = ""

    if title is None:
        title = default_title

    # Align lengths — both signals now share the same unit
    N = min(len(raw), len(c))
    raw = raw[:N]
    c = c[:N]
    # Filter boundaries to plotted range (avoid out-of-axis markers)
    boundaries = [int(b) for b in boundaries if 0 <= int(b) < N]

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=figsize, dpi=dpi, sharex=True,
        gridspec_kw={"height_ratios": [1, 1.2], "hspace": 0.28},
    )
    fig.patch.set_facecolor("white")

    # ── Top panel: raw stream view (no structural measurement) ──
    ax_top.set_facecolor("white")
    x_raw = np.arange(len(raw))
    ax_top.fill_between(x_raw, 0, raw, color="#888888", alpha=0.18,
                        linewidth=0)
    ax_top.plot(x_raw, raw, color="#555555", linewidth=0.5, alpha=0.65)
    ax_top.set_ylim(0, 1.05)
    ax_top.set_ylabel(raw_ylabel, fontsize=10, color="#444444")
    ax_top.set_title(top_title,
                     fontsize=12, color="#555555", loc="left", pad=8)
    _apply_clean_style(ax_top)

    # ── Bottom panel: policy-routed curve + segments ──
    ax_bot.set_facecolor("white")
    _draw_segments(ax_bot, N, boundaries)
    _draw_curve(ax_bot, c)
    ymax = max(1e-6, float(np.nanmax(c))) if N else 1.0
    ax_bot.set_ylim(-0.02 * ymax, ymax * 1.08)
    _draw_boundaries(ax_bot, c, boundaries)

    ax_bot.set_ylabel("Structural change signal", fontsize=10, color="#444444")
    ax_bot.set_xlabel(x_label, fontsize=10, color="#444444")
    ax_bot.set_title("After structural change measurement — segments and boundaries detected",
                     fontsize=12, color="#222222", weight="bold", loc="left", pad=8)
    _apply_clean_style(ax_bot)

    # Global title + route badge
    fig.suptitle(title, fontsize=15, weight="bold", color="#222222",
                 x=0.02, y=0.985, ha="left")
    if route is not None:
        sel = route.get("selected", "?")
        badge_color = _ROUTE_COLORS.get(sel, "#555555")
        badge_text = _route_badge(route)
        fig.text(
            0.985, 0.985, badge_text,
            ha="right", va="top",
            fontsize=10.5, color="white", weight="bold",
            bbox=dict(boxstyle="round,pad=0.42",
                      facecolor=badge_color, edgecolor="none"),
        )

    if route is not None:
        _draw_stats_block(ax_bot, route, len(list(boundaries)))

    _draw_annotations(ax_bot, N, annotations)

    ax_bot.set_xlim(0, N - 1 if N else 1)

    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight",
                    facecolor="white")
    if show:
        plt.show()
    return fig


__all__ = ["plot_segmentation", "plot_comparison"]
