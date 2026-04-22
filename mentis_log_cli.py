#!/usr/bin/env python3
"""mentis_log_cli — structural segmentation for text and log files.

Three subcommands:

    segment     segment a single file, emit JSON
    plot        render a PNG showing the structural-change signal
    benchmark   run a directory of logs and report metrics

Quickstart
----------
    python mentis_log_cli.py segment   --input app.log  --output result.json
    python mentis_log_cli.py plot      --input app.log  --output plot.png --comparison
    python mentis_log_cli.py benchmark --input-dir benchmarks/synthetic --output bench.json

Defaults are tuned for logs:
    strategy = "auto"   (recommended)
    mode     = "line"   (per-line analysis)

For char-level parity with the in-repo reference pipeline pass ``--mode char``.
For the old heuristic line-mode behaviour pass ``--strategy heuristic``.
Advanced flags are kept available but grouped under "advanced / experimental".
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from runtime import segment, DEFAULTS_CHAR, DEFAULTS_LINE  # noqa: E402


# ── helpers ──────────────────────────────────────────────────────

_PEAK_KEYS = (
    "window", "quantile", "min_distance", "min_prominence",
    "nms_radius", "consolidation_radius", "min_segment_windows",
)


def _collect_overrides(args: argparse.Namespace) -> dict:
    out = {}
    for k in _PEAK_KEYS:
        v = getattr(args, k, None)
        if v is not None:
            out[k] = v
    return out


def _collect_qalign_kwargs(args: argparse.Namespace) -> dict:
    """Expert Q-alignment controls. None means 'let strategy decide'."""
    return {
        "strategy": args.strategy,
        "scoring": getattr(args, "scoring", None),
        "signal_type": getattr(args, "signal", None),
        "q_mode": getattr(args, "q_mode", None),
        "min_token_freq": getattr(args, "min_token_freq", None),
        "dist_bins": getattr(args, "dist_bins", 32),
        "threshold_mode": getattr(args, "threshold_mode", "quantile"),
        "mean_std_k": getattr(args, "mean_std_k", 2.0),
        "topk_n": getattr(args, "topk_n", 10),
    }


# ── segment ──────────────────────────────────────────────────────

def _cmd_segment(args: argparse.Namespace) -> int:
    if args.input and args.input != "-":
        text = Path(args.input).read_text(encoding="utf-8", errors="replace")
    else:
        text = sys.stdin.read()

    overrides = _collect_overrides(args)
    qalign = _collect_qalign_kwargs(args)
    want_plot = bool(args.plot) or bool(args.plot_comparison)
    want_interp = bool(args.interpret) or bool(
        getattr(args, "annotate_segments", False)
    )
    result = segment(
        text,
        mode=args.mode,
        sample_rate=args.sample_rate,
        include_curve=want_plot or args.include_curve,
        interpret=want_interp,
        refine=not bool(getattr(args, "no_refine", False)),
        refine_radius_lines=int(getattr(args, "refine_radius_lines", 256)),
        edge_cleanup=not bool(getattr(args, "no_edge_cleanup", False)),
        **qalign,
        **overrides,
    )

    if want_plot:
        try:
            from plot import plot_segmentation, plot_comparison
        except ImportError as e:
            sys.stderr.write(
                f"ERROR: matplotlib not installed — run "
                f"`pip install -r requirements-plot.txt` ({e})\n"
            )
            return 3
        unit = "line" if args.mode == "line" else "character"
        unit_count = (
            result.get("n_lines") if args.mode == "line"
            else result.get("n_chars")
        )
        subtitle = (
            f"{unit_count} {unit}s · {result['n_segments']} segments · "
            f"mode={result.get('mode')} · "
            f"{result['route'].get('reason', '')}"
        )

        # Build annotations for --annotate-segments.
        annotations = None
        if args.annotate_segments and result.get("interpretation"):
            s = max(1, int(args.sample_rate))
            annotations = []
            for seg in result["segments"]:
                # seg.start/end are in user-unit space; curve is in
                # sampled-index space. Divide by sample_rate to align.
                annotations.append({
                    "start": int(seg["start"]) // s,
                    "end": int(seg["end"]) // s,
                    "profile": seg.get("profile", ""),
                })

        if args.plot:
            plot_segmentation(
                result["curve"], result["boundaries"], result["route"],
                output_path=args.plot, subtitle=subtitle,
                x_label=f"{unit.capitalize()} position",
                annotations=annotations,
            )
        if args.plot_comparison:
            plot_comparison(
                text, result["curve"], result["boundaries"], result["route"],
                mode=args.mode,
                output_path=args.plot_comparison,
                x_label=f"{unit.capitalize()} position",
                annotations=annotations,
            )

    if not args.include_curve and "curve" in result:
        result.pop("curve", None)

    payload = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output and args.output != "-":
        Path(args.output).write_text(payload, encoding="utf-8")
    elif args.output == "-":
        sys.stdout.write(payload)
        sys.stdout.write("\n")
    return 0


# ── plot ─────────────────────────────────────────────────────────

def _cmd_plot(args: argparse.Namespace) -> int:
    if args.input and args.input != "-":
        text = Path(args.input).read_text(encoding="utf-8", errors="replace")
    else:
        text = sys.stdin.read()

    result = segment(
        text, mode=args.mode, sample_rate=args.sample_rate,
        strategy=args.strategy, include_curve=True,
    )
    try:
        from plot import plot_segmentation, plot_comparison
    except ImportError as e:
        sys.stderr.write(
            f"ERROR: matplotlib not installed — run "
            f"`pip install -r requirements-plot.txt` ({e})\n"
        )
        return 3

    unit = "line" if args.mode == "line" else "character"
    unit_count = (
        result.get("n_lines") if args.mode == "line" else result.get("n_chars")
    )
    x_label = f"{unit.capitalize()} position"
    if args.comparison:
        plot_comparison(
            text, result["curve"], result["boundaries"], result["route"],
            mode=args.mode,
            output_path=args.output, x_label=x_label,
        )
    else:
        plot_segmentation(
            result["curve"], result["boundaries"], result["route"],
            output_path=args.output, x_label=x_label,
            subtitle=(f"{unit_count} {unit}s · "
                      f"{result['n_segments']} segments · "
                      f"mode={result.get('mode')} · "
                      f"{result['route'].get('reason', '')}"),
        )
    sys.stderr.write(
        f"Saved plot to {args.output}  "
        f"(mode={args.mode}, route={result['route'].get('selected')}, "
        f"boundaries={result['n_boundaries']})\n"
    )
    return 0


# ── benchmark ────────────────────────────────────────────────────

# Reuse benchmark-local helpers so the metric logic lives in one place.
from benchmarks.adapters.evaluation import (  # noqa: E402
    match_boundaries as _match_boundaries,
    save_report as _save_report,
)
from benchmarks.adapters import REGISTRY as _DATASET_REGISTRY  # noqa: E402


def _load_annotation(log_path: Path) -> dict:
    """Return annotation dict if a sibling .json exists, else empty."""
    ann = log_path.with_suffix(".json")
    if not ann.exists():
        return {}
    try:
        return json.loads(ann.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _run_one(text: str, strategy: str) -> dict:
    t0 = time.perf_counter()
    r = segment(text, mode="line", strategy=strategy)
    dt = time.perf_counter() - t0
    return {
        "strategy": strategy,
        "runtime_seconds": round(dt, 3),
        "n_lines": r.get("n_lines", 0),
        "n_boundaries": r["n_boundaries"],
        "boundaries": r["boundaries"],
        "route": r["route"].get("selected"),
    }


def _aggregate(totals: dict, strategies: list) -> dict:
    summary = {}
    for strat in strategies:
        t = totals[strat]
        n = len(t["f1"])
        summary[strat] = {
            "total_runtime_seconds": round(t["time"], 3),
            "mean_precision": (round(sum(t["precision"]) / n, 4) if n else None),
            "mean_recall": (round(sum(t["recall"]) / n, 4) if n else None),
            "mean_f1": (round(sum(t["f1"]) / n, 4) if n else None),
            "annotated_files": n,
        }
    return summary


def _print_file_row(log_name: str, annotated: bool,
                    per_strategy: dict, strategies: list) -> None:
    print(f"  {log_name}  (annotated={annotated})")
    for strat in strategies:
        r = per_strategy[strat]
        line = (f"    {strat:<10s} {r['runtime_seconds']:>6.2f}s  "
                f"bnds={r['n_boundaries']:<4d}")
        if "f1" in r:
            line += (f"  P={r['precision']:.3f} "
                     f"R={r['recall']:.3f} F1={r['f1']:.3f}")
        print(line)
    print()


def _print_aggregate(summary: dict) -> None:
    print("Aggregate")
    for strat, s in summary.items():
        line = (f"  {strat:<10s} total={s['total_runtime_seconds']:>6.2f}s"
                f"  annotated={s['annotated_files']}")
        if s["mean_f1"] is not None:
            line += (f"  mean P={s['mean_precision']:.3f}"
                     f"  R={s['mean_recall']:.3f}"
                     f"  F1={s['mean_f1']:.3f}")
        print(line)


def _cmd_benchmark_dir(args: argparse.Namespace) -> int:
    """Directory-walk benchmark (--input-dir)."""
    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        sys.stderr.write(f"ERROR: {input_dir} is not a directory\n")
        return 2

    log_paths = sorted(input_dir.rglob("*.log"))
    if not log_paths:
        sys.stderr.write(f"WARN: no *.log files found under {input_dir}\n")

    strategies = args.strategies.split(",") if args.strategies else ["auto", "heuristic"]
    tol_default = int(args.tolerance) if args.tolerance is not None else 100

    per_file = []
    totals = {s: {"f1": [], "precision": [], "recall": [], "time": 0.0}
              for s in strategies}

    print(f"Benchmark (directory) over {len(log_paths)} log files "
          f"under {input_dir}")
    print(f"Strategies: {strategies}  |  tolerance (lines): {tol_default}")
    print()

    for log_path in log_paths:
        text = log_path.read_text(encoding="utf-8", errors="replace")
        ann = _load_annotation(log_path)
        expected = ann.get("expected_line_boundaries", [])
        tol = int(ann.get("tolerance_lines", tol_default))

        per_strategy = {}
        for strat in strategies:
            res = _run_one(text, strat)
            if expected:
                p, r, f1 = _match_boundaries(res["boundaries"], expected, tol)
                res.update({"precision": p, "recall": r, "f1": f1})
                totals[strat]["precision"].append(p)
                totals[strat]["recall"].append(r)
                totals[strat]["f1"].append(f1)
            totals[strat]["time"] += res["runtime_seconds"]
            per_strategy[strat] = res

        per_file.append({
            "file": str(log_path.relative_to(input_dir)),
            "n_lines": per_strategy[strategies[0]]["n_lines"],
            "expected_boundaries": expected,
            "annotated": bool(expected),
            "results": per_strategy,
        })
        _print_file_row(log_path.name, bool(expected), per_strategy, strategies)

    summary = _aggregate(totals, strategies)
    _print_aggregate(summary)

    report = {
        "mode": "directory",
        "input_dir": str(input_dir),
        "tolerance_lines": tol_default,
        "strategies": strategies,
        "per_file": per_file,
        "summary": summary,
    }
    out_path = Path(args.output)
    _save_report(report, out_path, write_md=not args.no_summary_md)
    print(f"\nSaved: {out_path}")
    return 0


def _cmd_benchmark_dataset(args: argparse.Namespace) -> int:
    """Public-dataset benchmark (--dataset <name> --data-dir <path>)."""
    if args.dataset not in _DATASET_REGISTRY:
        sys.stderr.write(
            f"ERROR: unknown dataset {args.dataset!r}. "
            f"Available: {sorted(_DATASET_REGISTRY)}\n"
        )
        return 2

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        sys.stderr.write(f"ERROR: {data_dir} is not a directory\n")
        return 2

    loader = _DATASET_REGISTRY[args.dataset]
    load_kwargs = {}
    if args.min_run is not None:
        load_kwargs["min_run"] = int(args.min_run)
    if args.merge_window is not None:
        load_kwargs["merge_window"] = int(args.merge_window)
    if args.tolerance is not None:
        load_kwargs["tolerance"] = int(args.tolerance)
    if args.max_lines is not None:
        load_kwargs["max_lines"] = int(args.max_lines)

    try:
        ds = loader(data_dir, **load_kwargs)
    except FileNotFoundError as e:
        sys.stderr.write(f"ERROR: {e}\n")
        return 2

    text = ds.text
    expected = ds.derived_boundaries
    tol = ds.tolerance_lines
    strategies = args.strategies.split(",") if args.strategies else ["auto", "heuristic"]

    print(f"Benchmark (dataset={args.dataset}) on {ds.source_path}")
    print(f"  n_lines: {ds.n_lines}  "
          f"| derived boundaries: {len(expected)}  "
          f"| tolerance (lines): {tol}")
    print(f"  target type: {ds.target_metadata.get('target_type')}")
    print()

    per_strategy = {}
    totals = {s: {"f1": [], "precision": [], "recall": [], "time": 0.0}
              for s in strategies}
    for strat in strategies:
        res = _run_one(text, strat)
        if expected:
            p, r, f1 = _match_boundaries(res["boundaries"], expected, tol)
            res.update({"precision": p, "recall": r, "f1": f1})
            totals[strat]["precision"].append(p)
            totals[strat]["recall"].append(r)
            totals[strat]["f1"].append(f1)
        totals[strat]["time"] += res["runtime_seconds"]
        per_strategy[strat] = res
    _print_file_row(ds.source_path.name, bool(expected), per_strategy, strategies)

    summary = _aggregate(totals, strategies)
    _print_aggregate(summary)

    report = {
        "mode": "dataset",
        "dataset": args.dataset,
        "source_path": str(ds.source_path),
        "n_lines": ds.n_lines,
        "tolerance_lines": tol,
        "strategies": strategies,
        "target_metadata": ds.target_metadata,
        "per_file": [{
            "file": ds.source_path.name,
            "n_lines": ds.n_lines,
            "expected_boundaries": list(expected),
            "annotated": bool(expected),
            "results": per_strategy,
        }],
        "summary": summary,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _save_report(report, out_path, write_md=not args.no_summary_md)
    print(f"\nSaved: {out_path}")
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    if args.dataset:
        return _cmd_benchmark_dataset(args)
    if args.input_dir:
        return _cmd_benchmark_dir(args)
    sys.stderr.write(
        "ERROR: benchmark requires either --input-dir DIR or "
        "--dataset NAME --data-dir DIR.\n"
    )
    return 2


# ── arg building ─────────────────────────────────────────────────

def _add_basic_segment_args(p: argparse.ArgumentParser) -> None:
    basic = p.add_argument_group("basic")
    basic.add_argument("--input", "-i", default="-",
                       help="Input file path (default: stdin)")
    basic.add_argument("--output", "-o", default="-",
                       help="Output JSON path (default: stdout)")
    basic.add_argument("--mode", choices=("char", "line"), default="line",
                       help="Analysis unit. Default 'line'. Use 'char' for "
                            "bit-exact reference-pipeline parity on small "
                            "text inputs.")
    basic.add_argument("--strategy",
                       choices=("auto", "heuristic", "qalign"),
                       default="auto",
                       help="Segmentation strategy. Default 'auto' uses "
                            "tokens+JSD+rolling+min_freq=2 for line mode "
                            "(the best-performing configuration on log "
                            "benchmarks) and the heuristic path for char "
                            "mode. 'heuristic' forces the coverage-routed "
                            "policy. 'qalign' uses Q-alignment with "
                            "explicit --signal / --q-mode overrides.")


def _add_advanced_segment_args(p: argparse.ArgumentParser) -> None:
    adv = p.add_argument_group(
        "advanced / experimental",
        "Most users do not need these; defaults are tuned per strategy.",
    )
    adv.add_argument("--sample-rate", type=int, default=1, metavar="N",
                     help="Evaluate curve every N-th position.")
    # Peak tuning
    adv.add_argument("--window", type=int, default=None)
    adv.add_argument("--quantile", type=float, default=None)
    adv.add_argument("--min-distance", type=int, default=None)
    adv.add_argument("--min-prominence", type=float, default=None)
    adv.add_argument("--nms-radius", type=int, default=None)
    adv.add_argument("--consolidation-radius", type=int, default=None)
    adv.add_argument("--min-segment-windows", type=int, default=None)
    # Q-alignment controls — default None = let strategy decide
    adv.add_argument("--scoring",
                     choices=("heuristic", "q_jsd", "q_kl"), default=None)
    adv.add_argument("--signal",
                     choices=("line_length", "tokens"), default=None)
    adv.add_argument("--q-mode",
                     choices=("global", "rolling", "prefix"), default=None)
    adv.add_argument("--min-token-freq", type=int, default=None)
    adv.add_argument("--dist-bins", type=int, default=32)
    adv.add_argument("--threshold-mode",
                     choices=("quantile", "mean_std", "topk"),
                     default="quantile")
    adv.add_argument("--mean-std-k", type=float, default=2.0)
    adv.add_argument("--topk-n", type=int, default=10)
    # Plotting
    adv.add_argument("--plot", metavar="PNG",
                     help="Also render single-panel segmentation plot.")
    adv.add_argument("--plot-comparison", metavar="PNG",
                     help="Also render two-panel raw-vs-structured plot.")
    adv.add_argument("--annotate-segments", action="store_true",
                     help="Annotate each segment in the plot(s) with its "
                          "structural profile (stable / transition / "
                          "volatile). Implies --interpret.")
    adv.add_argument("--include-curve", action="store_true",
                     help="Include per-position curve in JSON output.")
    adv.add_argument("--interpret", action="store_true",
                     help="Add a structural interpretation layer to the "
                          "segments (profile in {stable, transition, "
                          "volatile}, confidence, short summary). "
                          "Purely structural - no semantic labels.")
    adv.add_argument("--no-refine", action="store_true",
                     help="Disable the local multiscale boundary "
                          "refinement step (default: enabled in line "
                          "mode). Boundaries stay at the coarse "
                          "line-level detector output.")
    adv.add_argument("--refine-radius-lines", type=int, default=64,
                     help="Local window size (lines, half-width) used "
                          "by the multiscale refinement step.")
    adv.add_argument("--no-edge-cleanup", action="store_true",
                     help="Disable the post-refinement edge-segment "
                          "cleanup (default: enabled; drops the first "
                          "or last boundary if the resulting edge "
                          "segment is trivially short).")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="mentis_log_cli",
        description=(
            "Structural segmentation for text and log files. "
            "Default 'auto' strategy uses token-based Q-alignment in "
            "line mode — fast, semantic, and the best-performing "
            "configuration on log benchmarks."
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # segment
    p = sub.add_parser(
        "segment",
        help="Segment a single file and emit a JSON result.",
    )
    _add_basic_segment_args(p)
    _add_advanced_segment_args(p)
    p.set_defaults(func=_cmd_segment)

    # plot
    pp = sub.add_parser(
        "plot",
        help="Render a PNG visualising the structural-change signal.",
    )
    pp.add_argument("--input", "-i", default="-")
    pp.add_argument("--output", "-o", required=True, help="Output PNG path")
    pp.add_argument("--comparison", action="store_true",
                    help="Use the two-panel raw-vs-structured layout")
    pp.add_argument("--mode", choices=("char", "line"), default="line")
    pp.add_argument("--strategy",
                    choices=("auto", "heuristic", "qalign"),
                    default="auto")
    pp.add_argument("--sample-rate", type=int, default=1, metavar="N")
    pp.set_defaults(func=_cmd_plot)

    # benchmark
    pb = sub.add_parser(
        "benchmark",
        help="Run segmentation over logs; emit metrics vs expected boundaries.",
    )
    pb_mode = pb.add_mutually_exclusive_group(required=True)
    pb_mode.add_argument(
        "--input-dir", default=None,
        help="Directory of *.log files. If a *.json sibling shares a "
             "basename, its 'expected_line_boundaries' list is used.",
    )
    pb_mode.add_argument(
        "--dataset", choices=sorted(_DATASET_REGISTRY.keys()),
        default=None,
        help="Public-dataset adapter. Requires --data-dir. "
             "Derives expected boundaries from label / severity "
             "transitions; see benchmarks/datasets/public/<name>/README.md.",
    )
    pb.add_argument("--data-dir", default=None,
                    help="Dataset directory (required with --dataset).")
    pb.add_argument("--output", "-o", default="benchmark_results.json")
    pb.add_argument("--strategies", default="auto,heuristic",
                    help="Comma-separated strategies to compare.")
    pb.add_argument("--tolerance", type=int, default=None,
                    help="Boundary-match tolerance in lines. "
                         "Default: adapter-specific in --dataset mode, "
                         "100 in --input-dir mode.")
    pb.add_argument("--min-run", type=int, default=None,
                    help="(--dataset mode) Override the adapter's default "
                         "min-phase-length for target derivation.")
    pb.add_argument("--merge-window", type=int, default=None,
                    help="(--dataset mode) Override the adapter's default "
                         "merge window for nearby derived transitions.")
    pb.add_argument("--max-lines", type=int, default=None,
                    help="(--dataset mode) Truncate the source log to N "
                         "lines before deriving targets. Useful for smoke.")
    pb.add_argument("--no-summary-md", action="store_true",
                    help="Skip writing the compact .md summary next to "
                         "the JSON.")
    pb.set_defaults(func=_cmd_benchmark)
    # Back-compat for --input-dir mode without a dataset-specific default
    if pb.get_default("tolerance") is None:
        pb.set_defaults(tolerance=None)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
