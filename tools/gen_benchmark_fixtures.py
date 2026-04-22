"""Generate the synthetic benchmark fixtures in benchmarks/synthetic/.

Three fixtures of increasing difficulty:

  incident_small    600 lines,  3 regimes   (clear separation)
  incident_medium  1500 lines,  5 regimes   (mixed normal / error phases)
  incident_noisy    900 lines,  4 regimes   (same vocab, higher variability)

All seeded so reruns are reproducible. Each fixture is paired with a
*.json annotation that the benchmark CLI consumes.
"""
from __future__ import annotations

import json
import random
from pathlib import Path


USERS = ["alice", "bob", "carol", "dave", "erin", "frank", "grace"]
SKUS = [f"sku_{n}" for n in (4421, 1193, 2388, 7712, 9908, 3301)]


def _normal(n, rng):
    return [
        f"INFO checkout request_id={i:x} user={rng.choice(USERS)} "
        f"product={rng.choice(SKUS)} status=200"
        for i in range(n)
    ]


def _errors(n, rng):
    out = []
    trace = f"{rng.randint(1000, 9999):x}"
    for i in range(n):
        out.append(
            f"ERROR database connection refused pool exhausted "
            f"retry={(i % 3) + 1} trace_id={trace}"
        )
        if i % 4 == 3:
            out.append(
                f"WARN circuit_breaker open route=/checkout "
                f"requests_dropped={rng.randint(10, 200)}"
            )
    return out


def _fatals(n, rng):
    return [
        f"FATAL out_of_memory heap={rng.randint(90, 99)}% "
        f"gc_pause_ms={rng.randint(5000, 20000)} node=node_{i % 12:02d}"
        for i in range(n)
    ]


def _autoscale(n, rng):
    cur = rng.randint(4, 10)
    out = []
    for _ in range(n):
        target = cur + rng.randint(1, 4)
        out.append(
            f"INFO autoscaler scale_up from={cur} to={target} "
            f"reason=cpu_saturation"
        )
        cur = target
    return out


def _evict(n, rng):
    return [
        f"INFO pod_evicted name=checkout_{''.join(rng.choices('abcdef0123456789', k=4))} "
        f"reason=memory_pressure"
        for _ in range(n)
    ]


def _build(
    path: Path,
    schedule: list[tuple[str, int]],
    *,
    seed: int,
    tolerance: int,
    notes: str,
) -> dict:
    rng = random.Random(seed)
    all_lines: list[str] = []
    expected: list[int] = []
    fn_map = {
        "normal": _normal, "errors": _errors, "fatals": _fatals,
        "autoscale": _autoscale, "evict": _evict,
    }
    first = True
    for name, n in schedule:
        start = len(all_lines)
        if not first:
            expected.append(start)
        first = False
        all_lines.extend(fn_map[name](n, rng))

    text = "\n".join(all_lines) + "\n"
    path.write_text(text, encoding="utf-8")
    meta = {
        "expected_line_boundaries": expected,
        "tolerance_lines": tolerance,
        "notes": notes,
    }
    path.with_suffix(".json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8",
    )
    return {"n_lines": len(all_lines), "expected": expected}


def main() -> int:
    root = Path(__file__).resolve().parent.parent / "benchmarks" / "synthetic"
    root.mkdir(parents=True, exist_ok=True)

    # Phases are sized for the coarse line-mode defaults (min_distance=500).
    # Real log-regime transitions are typically hundreds to thousands of
    # lines apart (phase shifts, not events), which is what these fixtures
    # emulate.
    specs = [
        (
            "incident_small",
            [("normal", 800), ("errors", 600), ("normal", 700)],
            1,
            150,
            "2 clear regime transitions in a ~2.1k-line log",
        ),
        (
            "incident_medium",
            [("normal", 1000), ("errors", 600), ("normal", 900),
             ("fatals", 500), ("evict", 400), ("normal", 800)],
            2,
            200,
            "5 regime transitions with mixed phase durations "
            "(~4.2k lines)",
        ),
        (
            "incident_noisy",
            [("normal", 700), ("errors", 500), ("normal", 600),
             ("fatals", 500), ("normal", 800)],
            3,
            200,
            "4 regime transitions with shorter / more similar phases "
            "(~3.1k lines, higher variability)",
        ),
    ]

    for name, sched, seed, tol, notes in specs:
        path = root / f"{name}.log"
        m = _build(path, sched, seed=seed, tolerance=tol, notes=notes)
        print(f"Wrote {path.name}: {m['n_lines']} lines, "
              f"{len(m['expected'])} expected boundaries at {m['expected']}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
