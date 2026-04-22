"""Generate a larger realistic log file for scalability testing.

Produces a synthetic incident-style log with N clearly-bounded regimes.
Default: ~1 MB, ~15k lines, 5 regime transitions.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path


USERS = [
    "alice", "bob", "carol", "dave", "erin", "frank", "grace",
    "hank", "ivy", "jake", "karen", "liam", "mia", "noah",
    "olivia", "peter", "quinn", "ryan", "sara", "tom",
]
SKUS = [f"sku_{n}" for n in (4421, 1193, 2388, 7712, 9908, 3301, 8820, 6611)]
NODES = [f"node_{i:02d}" for i in range(12)]


def _normal_lines(n: int, rng: random.Random) -> list[str]:
    out = []
    for i in range(n):
        u = rng.choice(USERS)
        s = rng.choice(SKUS)
        amt = f"{rng.uniform(10, 200):.2f}"
        out.append(
            f"INFO checkout request_id={i:x} user={u} product={s} "
            f"amount={amt} status=200"
        )
    return out


def _error_storm_lines(n: int, rng: random.Random) -> list[str]:
    out = []
    trace = f"{rng.randint(1000, 9999):x}"
    for i in range(n):
        attempt = (i % 3) + 1
        dropped = rng.randint(10, 300)
        out.append(
            f"ERROR database connection refused pool exhausted "
            f"retry_attempt={attempt} trace_id={trace}"
        )
        if i % 4 == 3:
            out.append(
                f"WARN circuit_breaker open route=/checkout "
                f"latency_ms=30000 requests_dropped={dropped}"
            )
    return out


def _fatal_oom_lines(n: int, rng: random.Random) -> list[str]:
    out = []
    node = rng.choice(NODES)
    for i in range(n):
        heap = rng.randint(90, 99)
        gc = rng.randint(5000, 25000)
        blocked = rng.randint(100, 500)
        out.append(
            f"FATAL out_of_memory heap={heap}% gc_pause_ms={gc} "
            f"threads_blocked={blocked} node={node}"
        )
    return out


def _eviction_lines(n: int, rng: random.Random) -> list[str]:
    out = []
    for i in range(n):
        name = f"checkout_{''.join(rng.choices('abcdef0123456789', k=4))}"
        node = rng.choice(NODES)
        out.append(
            f"INFO pod_evicted name={name} reason=memory_pressure "
            f"node={node} rescheduled=true"
        )
    return out


def _autoscale_lines(n: int, rng: random.Random) -> list[str]:
    out = []
    cur = rng.randint(4, 10)
    for _ in range(n):
        target = cur + rng.randint(1, 4)
        out.append(
            f"INFO autoscaler scale_up replicas_from={cur} "
            f"replicas_to={target} reason=cpu_saturation target=70%"
        )
        cur = target
    return out


REGIMES = [
    ("normal", _normal_lines, 2000),
    ("error_storm", _error_storm_lines, 400),
    ("normal", _normal_lines, 2000),
    ("fatal_oom", _fatal_oom_lines, 300),
    ("eviction", _eviction_lines, 200),
    ("autoscale", _autoscale_lines, 100),
    ("normal", _normal_lines, 2500),
    ("error_storm", _error_storm_lines, 500),
    ("normal", _normal_lines, 3000),
]


def build(out_path: Path, seed: int = 42) -> dict:
    rng = random.Random(seed)
    all_lines = []
    regime_starts = {}  # regime_name -> line index of first occurrence
    for name, fn, n in REGIMES:
        regime_starts[f"{name}@{len(all_lines)}"] = len(all_lines)
        all_lines.extend(fn(n, rng))
    text = "\n".join(all_lines) + "\n"
    out_path.write_text(text, encoding="utf-8")
    return {
        "path": str(out_path),
        "n_lines": len(all_lines),
        "n_chars": len(text),
        "regime_starts": regime_starts,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="large_log.log")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    meta = build(Path(args.output), seed=args.seed)
    print(f"Wrote {meta['n_chars']} chars / {meta['n_lines']} lines "
          f"to {meta['path']}")
    print("Expected regime starts (line idx):")
    for k, v in meta["regime_starts"].items():
        print(f"  {k:>30s}  line {v}")
