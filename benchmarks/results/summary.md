# Benchmark reference results

This directory ships the current reference numbers for the pipeline.
Every JSON here was produced by the committed code on the committed
fixtures (or on smoke fixtures that imitate the Loghub format for
public datasets). Reproduce them with the commands below.

## What ships in-repo

| Artifact | Source | Reproduce |
| --- | --- | --- |
| `synthetic_default_vs_heuristic.json` + `.md` | `benchmarks/synthetic/*.log` (committed) | `benchmark --input-dir benchmarks/synthetic --output benchmarks/results/synthetic_default_vs_heuristic.json` |
| `bgl_smoke_default_vs_heuristic.json` + `.md` | `benchmarks/datasets/public/bgl/BGL_smoke.log` (committed, ~4k lines, BGL-format stylised) | `benchmark --dataset bgl --data-dir benchmarks/datasets/public/bgl --output benchmarks/results/bgl_smoke_default_vs_heuristic.json` |
| `hdfs_smoke_default_vs_heuristic.json` + `.md` | `benchmarks/datasets/public/hdfs/HDFS_smoke.log` (committed, ~2k lines) | `benchmark --dataset hdfs --data-dir benchmarks/datasets/public/hdfs --output benchmarks/results/hdfs_smoke_default_vs_heuristic.json` |

## What requires local data download

| Artifact | Source | Reproduce |
| --- | --- | --- |
| `bgl_default_vs_heuristic.json` | the real `BGL.log` (~4.7 M lines, from Loghub) placed at `benchmarks/datasets/public/bgl/BGL.log` | `benchmark --dataset bgl --data-dir benchmarks/datasets/public/bgl --output benchmarks/results/bgl_default_vs_heuristic.json` |
| `hdfs_default_vs_heuristic.json` | the real `HDFS.log` (~11 M lines) placed at `benchmarks/datasets/public/hdfs/HDFS.log` | `benchmark --dataset hdfs --data-dir benchmarks/datasets/public/hdfs --output benchmarks/results/hdfs_default_vs_heuristic.json` |

Loghub download instructions are in each dataset's README:

- `benchmarks/datasets/public/bgl/README.md`
- `benchmarks/datasets/public/hdfs/README.md`

Large log files are **not** committed to this repository.

## Current reference numbers (from committed fixtures)

Run on 2026-04-22 on the code committed in this repository.

### Synthetic (`benchmarks/synthetic/`)

Three generated log fixtures with hand-defined expected boundaries
(committed in `*.json` alongside the `*.log`). Not a real benchmark —
a determinism / smoke test for the segmentation stack.

| strategy | total runtime | mean P | mean R | mean F1 |
| --- | ---: | ---: | ---: | ---: |
| auto | 0.55 s | 1.000 | 0.783 | **0.869** |
| heuristic | 0.47 s | 1.000 | 0.783 | **0.869** |

The two strategies are tied on these clean fixtures.

### BGL smoke (`benchmarks/datasets/public/bgl/BGL_smoke.log`)

4 000-line stylised BGL-format fixture (`heartbeat ok` / `KERNDTLB` /
`APPREAD` phases). Target derivation via
`derive_boundaries_from_labels(min_run=500, merge_window=100)`.

| strategy | runtime | n_boundaries | P | R | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| auto | 0.17 s | 0 | 1.000 | 0.000 | **0.000** |
| heuristic | 0.56 s | 2 | 1.000 | 0.500 | **0.667** |

On this smoke fixture heuristic beats auto. Why this matters and
what to expect on real BGL: see the caveats section below.

### HDFS smoke (`benchmarks/datasets/public/hdfs/HDFS_smoke.log`)

~1 900-line HDFS-format fixture with three severity phases
(INFO → WARN → ERROR → INFO). Targets derived from severity
transitions.

| strategy | runtime | n_boundaries | P | R | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| auto | 0.10 s | 1 | 1.000 | 0.333 | **0.500** |
| heuristic | 0.20 s | 2 | 1.000 | 0.667 | **0.800** |

## Caveats (read before reading any number above)

- **Derived targets ≠ manual segmentation ground truth.** Expected
  boundaries here are mechanically derived from anomaly-label or
  severity transitions. A different derivation would give different
  numbers.
- **Smoke fixtures are not the real datasets.** Stylised, small,
  deterministic. They validate the benchmark harness, not
  publication-scale performance.
- **`auto` underperforms on the BGL / HDFS smoke fixtures.** These
  fixtures have low token diversity within each phase (almost
  identical lines repeated). The default `auto` strategy uses
  token-based JSD with `min_token_freq=2`; on these fixtures the
  surviving vocabulary collapses and the divergence curve stays flat.
  On the committed `large_log.log` (11 k lines, more varied content)
  `auto` beats heuristic 5 / 8 vs 4 / 8. Real BGL.log is expected to
  behave closer to the latter — but the only way to verify is to run
  it locally.
- **HDFS is a weak fit** for regime segmentation; the Loghub HDFS
  benchmark labels blocks, not lines. The adapter uses severity shifts
  as a placeholder signal. See `benchmarks/datasets/public/hdfs/README.md`.

## Reproducing everything in one shot

```bash
# Synthetic (committed fixtures)
python mentis_log_cli.py benchmark \
    --input-dir benchmarks/synthetic \
    --output benchmarks/results/synthetic_default_vs_heuristic.json

# BGL smoke (committed stylised fixture)
python mentis_log_cli.py benchmark \
    --dataset bgl \
    --data-dir benchmarks/datasets/public/bgl \
    --output benchmarks/results/bgl_smoke_default_vs_heuristic.json

# HDFS smoke (committed stylised fixture)
python mentis_log_cli.py benchmark \
    --dataset hdfs \
    --data-dir benchmarks/datasets/public/hdfs \
    --output benchmarks/results/hdfs_smoke_default_vs_heuristic.json
```

For public real-scale reproduction, download the Loghub archives and
rerun against the real `BGL.log` / `HDFS.log`. Commands are identical;
output paths should drop the `_smoke` suffix.
