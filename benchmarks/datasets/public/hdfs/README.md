# HDFS — Hadoop Distributed File System log

## Suitability caveat (read this first)

HDFS from Loghub labels **blocks**, not **lines**. That is a block
classification benchmark, not a phase-segmentation benchmark.

This adapter is included for structural symmetry with the BGL
workflow. It derives boundaries from **severity shifts**
(`INFO` / `WARN` / `ERROR` / `FATAL` per line). Severity is a
*weak* proxy for regime on HDFS. Treat HDFS results here as
illustrative, not as a meaningful detection benchmark.

Prefer BGL when evaluating pipeline quality.

## How to obtain the real dataset

- https://github.com/logpai/loghub
- direct link: https://zenodo.org/record/3227177

Download `HDFS.tar.gz`, extract so this directory contains `HDFS.log`:

```
benchmarks/datasets/public/hdfs/
├── README.md
├── HDFS_smoke.log     (committed; tiny synthetic fixture)
└── HDFS.log           (you place this; not committed)
```

Expected size: ~11 M lines, ~1.5 GB.

## Run the benchmark

```bash
# Smoke fixture (always works)
python mentis_log_cli.py benchmark \
    --dataset hdfs \
    --data-dir benchmarks/datasets/public/hdfs \
    --output benchmarks/results/hdfs_smoke_default_vs_heuristic.json

# Real HDFS.log
python mentis_log_cli.py benchmark \
    --dataset hdfs \
    --data-dir benchmarks/datasets/public/hdfs \
    --output benchmarks/results/hdfs_default_vs_heuristic.json
```

## Target derivation

The adapter applies `derive_boundaries_from_labels` to the severity
stream with defaults `min_run = 200`, `merge_window = 50`. Metadata
declared as `target_type = "derived_from_severity_transitions"` with
an explicit disclaimer.
