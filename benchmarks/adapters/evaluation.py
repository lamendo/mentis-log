"""Benchmark-local evaluation helpers.

These are intentionally kept out of the core runtime (``primitives/``)
because they are benchmark-tooling concerns:
  - deriving expected targets from public-dataset labels
  - tolerance-based boundary matching
  - aggregating precision / recall / F1
  - writing a compact markdown summary

Runtime code must never import from this module.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


def match_boundaries(
    predicted: Sequence[int],
    expected: Sequence[int],
    tolerance: int,
) -> Tuple[float, float, float]:
    """Precision / recall / F1 of ``predicted`` vs ``expected`` with a
    one-sided tolerance window. Each expected boundary may match at
    most one predicted boundary (greedy nearest-first)."""
    predicted = sorted(int(p) for p in predicted)
    expected = sorted(int(e) for e in expected)
    if not expected and not predicted:
        return 1.0, 1.0, 1.0
    if not expected:
        return 0.0, 1.0, 0.0
    if not predicted:
        return 1.0, 0.0, 0.0

    matched = [False] * len(expected)
    tp = 0
    for p in predicted:
        best = float("inf")
        bi = -1
        for j, e in enumerate(expected):
            if matched[j]:
                continue
            d = abs(p - e)
            if d <= tolerance and d < best:
                best = d
                bi = j
        if bi >= 0:
            matched[bi] = True
            tp += 1
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(expected) if expected else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )
    return round(precision, 4), round(recall, 4), round(f1, 4)


def merge_nearby_indices(
    indices: Iterable[int],
    merge_window: int,
) -> List[int]:
    """Collapse indices that are within ``merge_window`` of each other
    into the smallest one in the cluster. Deterministic."""
    out: List[int] = []
    for i in sorted(int(x) for x in indices):
        if not out or i - out[-1] > merge_window:
            out.append(i)
    return out


def derive_boundaries_from_labels(
    labels: Sequence[str],
    *,
    min_run: int = 200,
    merge_window: int = 50,
) -> List[int]:
    """Derive phase-level boundary positions from a per-line label stream.

    A *phase* is a maximal run of consecutive identical labels. Runs
    shorter than ``min_run`` are absorbed into the previous run (so a
    brief flip does not emit a spurious boundary). Transitions between
    surviving phases are returned, then merged within ``merge_window``.

    This is an honest derivation heuristic, not a manual segmentation
    ground truth.
    """
    if not labels:
        return []

    # Build run list [(start, end, label)]
    runs: List[Tuple[int, int, str]] = []
    current = labels[0]
    run_start = 0
    for i in range(1, len(labels)):
        if labels[i] != current:
            runs.append((run_start, i, current))
            run_start = i
            current = labels[i]
    runs.append((run_start, len(labels), current))

    # Absorb short runs into previous; then coalesce consecutive
    # same-label runs (which can arise after absorption).
    filtered: List[Tuple[int, int, str]] = []
    for start, end, lab in runs:
        if end - start < min_run and filtered:
            # Absorb short run into the previous run (keep previous label).
            prev_start, _, prev_lab = filtered[-1]
            filtered[-1] = (prev_start, end, prev_lab)
        elif filtered and filtered[-1][2] == lab:
            # Merge consecutive same-label runs.
            prev_start, _, prev_lab = filtered[-1]
            filtered[-1] = (prev_start, end, prev_lab)
        else:
            filtered.append((start, end, lab))

    transitions = [r[0] for r in filtered[1:]]
    return merge_nearby_indices(transitions, merge_window)


def write_summary_md(
    report: Dict[str, Any],
    output_path: Path,
) -> None:
    """Emit a small Markdown summary next to the JSON result."""
    lines: List[str] = []
    dataset = report.get("dataset", "unknown")
    input_dir = report.get("input_dir", "")
    target_meta = report.get("target_metadata", {})

    lines.append(f"# Benchmark report — {dataset}")
    lines.append("")
    if input_dir:
        lines.append(f"- Input: `{input_dir}`")
    if target_meta:
        ttype = target_meta.get("target_type", "unknown")
        source = target_meta.get("source", "")
        lines.append(f"- Target type: `{ttype}`"
                     + (f" (source: {source})" if source else ""))
    strategies = report.get("strategies", [])
    lines.append(f"- Strategies compared: `{', '.join(strategies)}`")
    tol = report.get("tolerance_lines")
    if tol is not None:
        lines.append(f"- Tolerance (lines): `{tol}`")
    lines.append("")

    summary = report.get("summary", {})
    if summary:
        lines.append("## Aggregate metrics")
        lines.append("")
        header = (
            "| strategy | runtime (s) | annotated files | "
            "mean P | mean R | mean F1 |"
        )
        sep = "|---|---:|---:|---:|---:|---:|"
        lines.append(header)
        lines.append(sep)
        for strat, s in summary.items():
            mp = s.get("mean_precision")
            mr = s.get("mean_recall")
            mf = s.get("mean_f1")
            def _fmt(v):
                return f"{v:.3f}" if isinstance(v, (int, float)) else "–"
            lines.append(
                f"| {strat} | {s.get('total_runtime_seconds', 0):.2f} "
                f"| {s.get('annotated_files', 0)} "
                f"| {_fmt(mp)} | {_fmt(mr)} | {_fmt(mf)} |"
            )
        lines.append("")

    per_file = report.get("per_file", [])
    if per_file:
        lines.append("## Per-file results")
        lines.append("")
        for entry in per_file:
            fname = entry.get("file", "?")
            n_lines = entry.get("n_lines", 0)
            exp = entry.get("expected_boundaries", [])
            lines.append(f"### `{fname}`")
            lines.append(
                f"- n_lines: `{n_lines}`"
                f"  | expected: `{len(exp)}`"
                f"  | annotated: `{entry.get('annotated', False)}`"
            )
            for strat, res in (entry.get("results") or {}).items():
                line = (
                    f"  - **{strat}**: "
                    f"{res.get('runtime_seconds', 0):.2f}s, "
                    f"bnds={res.get('n_boundaries', 0)}"
                )
                if "f1" in res:
                    line += (
                        f", P={res['precision']:.3f} "
                        f"R={res['recall']:.3f} F1={res['f1']:.3f}"
                    )
                lines.append(line)
            lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(
        "Target derivation is documented in the corresponding adapter. "
        "This is not a manually-annotated segmentation benchmark; "
        "expected boundaries are derived from label/severity transitions "
        "in the source log."
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def save_report(report: Dict[str, Any], output_path: Path,
                write_md: bool = True) -> None:
    """Save the JSON report and optionally a sibling .md summary."""
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if write_md:
        write_summary_md(report, output_path.with_suffix(".md"))
