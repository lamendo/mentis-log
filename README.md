
# Mentis Log Segmentation

[![CI](https://github.com/lamendo/mentis-log/actions/workflows/ci.yml/badge.svg)](https://github.com/lamendo/mentis-log/actions/workflows/ci.yml)
[![Python: 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Detect structural change boundaries in large log streams.

## What it does

Given a log file, the tool finds the line positions where the per-line
*structure* of the log changes most strongly. These positions are
called **separators**: points of maximal structural separation between
one behavioural phase and the next.

It does **not**:

- classify log events
- label incidents, causes, or severities
- decide what is "normal" vs "anomalous"

It finds **where** structure changes, not **why**.

<img width="2056" height="1140" alt="plot" src="https://github.com/user-attachments/assets/a400294d-3ee2-4ee4-8951-fd0b32762bb8" />

## Quickstart

Install from source:

```bash
git clone https://github.com/lamendo/mentis-log.git
cd mentis-log
pip install -e ".[plot]"                  # installs the CLI entry point
```

Three commands:

```bash
# 1. Segment a log
mentis-log segment --input app.log --output result.json

# 2. Visualise the structural-change signal
mentis-log plot \
    --input app.log --output plot.png --comparison

# 3. Run the shipped benchmark on the in-repo synthetic fixtures
mentis-log benchmark \
    --input-dir benchmarks/synthetic \
    --output benchmarks/results/synthetic_default_vs_heuristic.json
```

The default segment invocation needs no flag tuning. Line mode +
token-based rolling JSD + local multiscale refinement + edge cleanup,
all on by default.

## Output overview

A segmentation run returns JSON with, at minimum:

- `boundaries` — line indices where structure changes (separators)
- `segments` — contiguous line spans between boundaries
- `raw_boundaries` — coarse detector output before local refinement
- `boundary_details` — per-boundary audit:
  `{raw, refined, separator, onset, status}`
- `boundary_refinement` — refinement metadata, including
  `public_boundary_semantics: "separator"`
- `edge_cleanup` — metadata and any boundaries dropped from edges

With `--interpret`, each segment additionally carries a structural
profile: `stable`, `transition`, or `volatile`. These labels describe
the *shape of the change signal inside the segment* only. No semantic
meaning is inferred.

## Benchmarks

The repository ships a benchmark harness and reference result artifacts:

- synthetic fixtures committed under `benchmarks/synthetic/`
- adapters for public Loghub datasets (BGL, HDFS) under
  `benchmarks/adapters/`
- committed reference JSON + markdown under `benchmarks/results/`

Reproduce everything locally:

```bash
# Public datasets (user-downloaded, not committed)
mentis-log benchmark \
    --dataset bgl \
    --data-dir benchmarks/datasets/public/bgl \
    --output benchmarks/results/bgl_default_vs_heuristic.json
```

Benchmark targets for public datasets are **mechanically derived**
from label / severity transitions. They are not manually annotated
segmentation ground truth. See `benchmarks/README.md` for the full
harness description and `benchmarks/results/summary.md` for the
current reference numbers and caveats.

## Limitations

- **Boundaries are separators, not semantic labels.** The tool
  identifies where the per-line structure of the log shifts. Whether
  a shift corresponds to a real incident, a deploy, or a routine
  schedule change is out of scope.
- **Behaviour depends on log characteristics.** Highly repetitive
  logs or logs with few distinguishable regimes produce few or no
  boundaries by design. Very noisy logs can produce boundaries that
  do not align with human intuition.
- **Benchmark targets are derived.** The F1 numbers under
  `benchmarks/results/` describe the match rate against automatically
  derived transition targets. A different derivation rule would give
  different numbers.

See [`USAGE.md`](USAGE.md) for the full operator manual.
