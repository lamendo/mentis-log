"""Generate tiny synthetic fixtures that *mimic* the Loghub BGL / HDFS
formats so the benchmark command runs out of the box.

These are deliberately small and stylised — they are smoke inputs, not
representative of the real datasets.
"""
from __future__ import annotations

import random
from pathlib import Path


def _bgl_fixture(seed: int = 7) -> str:
    rng = random.Random(seed)
    out = []

    def ts() -> str:
        # Plausible-looking fields; exact values don't matter for the
        # adapter — only the first token (label) does.
        return (
            f"{rng.randint(1_100_000_000, 1_200_000_000)} "
            f"2005.06.03 R02-M1-N0-C:J{rng.randint(0, 15):02d}-U11 "
            f"2005-06-03-15.42.{rng.randint(10, 59):02d}.{rng.randint(100_000, 999_999)} "
            f"R02-M1-N0-C:J12-U11 RAS KERNEL INFO"
        )

    def normal(n: int):
        for _ in range(n):
            out.append(f"- {ts()} heartbeat ok")

    def alert(label: str, n: int, msg: str):
        for _ in range(n):
            out.append(f"{label} {ts()} {msg}")

    # Phase durations match the real-world BGL defaults (min_run=500 etc.)
    # so the benchmark command produces meaningful output on the smoke
    # fixture with no parameter overrides.
    normal(1000)
    alert("KERNDTLB", 600, "data TLB error encountered")
    normal(1000)
    alert("APPREAD", 600, "application read error at page")
    normal(800)
    return "\n".join(out) + "\n"


def _hdfs_fixture(seed: int = 11) -> str:
    rng = random.Random(seed)
    out = []

    def header(sev: str) -> str:
        return (
            f"081109 {rng.randint(100000, 999999):06d} "
            f"{rng.randint(1, 999):d} {sev}"
        )

    def info(n: int):
        for _ in range(n):
            blk = rng.randint(10**15, 10**16)
            out.append(
                f"{header('INFO')} dfs.DataNode$PacketResponder: "
                f"Received block blk_{blk} of size {rng.randint(1000, 9000)}"
            )

    def warn(n: int):
        for _ in range(n):
            out.append(
                f"{header('WARN')} dfs.DataNode$DataXceiver: "
                f"writeBlock blk_{rng.randint(10**15, 10**16)} "
                f"received exception"
            )

    def error(n: int):
        for _ in range(n):
            out.append(
                f"{header('ERROR')} dfs.DataNode$DataXceiver: "
                f"{rng.choice(['IOException', 'BrokenPipe', 'TimeoutException'])}: "
                f"Connection reset by peer"
            )

    # 3 phases: INFO → WARN burst → ERROR burst → INFO
    info(600)
    warn(400)
    error(400)
    info(500)
    return "\n".join(out) + "\n"


def main() -> int:
    root = Path(__file__).resolve().parent.parent / "benchmarks" / "datasets" / "public"
    bgl_path = root / "bgl" / "BGL_smoke.log"
    hdfs_path = root / "hdfs" / "HDFS_smoke.log"
    bgl_path.parent.mkdir(parents=True, exist_ok=True)
    hdfs_path.parent.mkdir(parents=True, exist_ok=True)
    bgl_path.write_text(_bgl_fixture(), encoding="utf-8")
    hdfs_path.write_text(_hdfs_fixture(), encoding="utf-8")
    for p in (bgl_path, hdfs_path):
        n = p.read_text(encoding="utf-8").count("\n")
        print(f"Wrote {p.relative_to(p.parent.parent.parent.parent)}: {n} lines")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
