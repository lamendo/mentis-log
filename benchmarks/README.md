# Benchmarks

Two benchmark layers ship with this repository:

1. **Synthetic** — generated log fixtures with hand-defined expected
   boundaries, fully committed in-repo. The `benchmark` CLI runs out
   of the box against these.

2. **Public real-world** — adapters for Loghub BGL / HDFS logs. Log
   files are **not** committed (large, separately licensed); small
   format-compatible smoke fixtures are committed so the workflow
   runs immediately.

```
benchmarks/
├── README.md                                 (this file)
├── synthetic/                                # committed fixtures + JSON annotations
│   ├── incident_small.log  / .json
│   ├── incident_medium.log / .json
│   └── incident_noisy.log  / .json
├── real/                                     # placeholder for user-provided logs
│   └── README.md
├── datasets/
│   └── public/
│       ├── bgl/                              # committed: smoke + README
│       │   ├── README.md
│       │   └── BGL_smoke.log
│       └── hdfs/                             # committed: smoke + README
│           ├── README.md
│           └── HDFS_smoke.log
├── adapters/                                 # dataset-specific parsers
│   ├── __init__.py
│   ├── evaluation.py                         # P/R/F1, tolerance matching,
│   │                                         # target derivation, .md writer
│   ├── bgl.py
│   └── hdfs.py
└── results/                                  # committed reference artifacts
    ├── summary.md
    ├── synthetic_default_vs_heuristic.json  / .md
    ├── bgl_smoke_default_vs_heuristic.json  / .md
    └── hdfs_smoke_default_vs_heuristic.json / .md
```

---

## 1. Synthetic benchmark (included)

Three generated log fixtures committed with hand-written expected
boundaries:

```bash
python mentis_log_cli.py benchmark \
  --input-dir benchmarks/synthetic \
  --output benchmarks/results/synthetic_default_vs_heuristic.json
```

Compares `strategy="auto"` vs `strategy="heuristic"` on every `*.log`
file under the directory; pairs each with its sibling `*.json`
annotation for precision / recall / F1.

Annotation format per file (`<name>.json`):

```json
{
  "expected_line_boundaries": [800, 1550],
  "tolerance_lines": 150,
  "notes": "2 regime transitions in a 2250-line log"
}
```

If the sibling JSON is missing, runtime and boundary counts are
reported but F1 is skipped.

---

## 2. Public real-world benchmark workflow

Target datasets are provided via isolated adapters under
`benchmarks/adapters/`. Two are implemented:

- **BGL** (primary) — Loghub Blue Gene/L log. Per-line anomaly labels
  make phase-level segmentation meaningful.
- **HDFS** (weak fit) — Loghub HDFS log. No per-line anomaly labels;
  the adapter derives pseudo-phases from severity shifts. Use with
  caution — see `benchmarks/datasets/public/hdfs/README.md`.

### Target derivation (important — read this)

Expected boundaries for the public datasets are **not** manual
segmentation ground truth. They are **mechanically derived** from the
label / severity stream of the source log using
`derive_boundaries_from_labels` in `benchmarks/adapters/evaluation.py`:

- A *phase* is a maximal run of consecutive identical labels.
- Runs shorter than `min_run` (BGL default 500, HDFS default 200) are
  absorbed into the previous phase.
- Consecutive same-label runs after absorption are coalesced.
- Remaining transitions are the derived expected boundaries.
- Nearby transitions within `merge_window` collapse.

The JSON artifact records this under `target_metadata` with a
`target_type` field (`derived_from_label_transitions` for BGL,
`derived_from_severity_transitions` for HDFS) plus a disclaimer.

### Running

```bash
# BGL (primary benchmark) — smoke fixture is already committed
python mentis_log_cli.py benchmark \
  --dataset bgl \
  --data-dir benchmarks/datasets/public/bgl \
  --output benchmarks/results/bgl_smoke_default_vs_heuristic.json

# HDFS (weak fit)
python mentis_log_cli.py benchmark \
  --dataset hdfs \
  --data-dir benchmarks/datasets/public/hdfs \
  --output benchmarks/results/hdfs_smoke_default_vs_heuristic.json
```

The smoke fixtures are ~2 – 4 k stylised lines in the exact source
format. They exercise the full harness without a download.

### Reproducing against the real public datasets

Download the archives from the Loghub repository
(https://github.com/logpai/loghub) and place `BGL.log` / `HDFS.log` in
the corresponding `datasets/public/<name>/` directory. The adapters
use the canonical filename in preference to the smoke file, so the
same command just works:

```bash
# After placing real BGL.log
python mentis_log_cli.py benchmark \
  --dataset bgl \
  --data-dir benchmarks/datasets/public/bgl \
  --output benchmarks/results/bgl_default_vs_heuristic.json
```

Dataset-specific knobs for target derivation are exposed:

- `--min-run N`        override min-phase-length
- `--merge-window N`   override nearby-transition merge window
- `--tolerance N`      override match tolerance for P/R/F1
- `--max-lines N`      truncate the source log (useful when prototyping)

---

## 3. Committed reference results

Every `*.json` under `benchmarks/results/` was produced by the code in
this commit on the committed fixtures. Reproduction commands and a
compact overview live in
[`results/summary.md`](results/summary.md).

Current committed artifacts:

| artifact | source |
| --- | --- |
| `synthetic_default_vs_heuristic.json` + `.md` | `benchmarks/synthetic/*.log` |
| `bgl_smoke_default_vs_heuristic.json` + `.md` | `benchmarks/datasets/public/bgl/BGL_smoke.log` |
| `hdfs_smoke_default_vs_heuristic.json` + `.md` | `benchmarks/datasets/public/hdfs/HDFS_smoke.log` |

Artifacts for the real public datasets (`bgl_default_vs_heuristic.json`
etc.) are **not** committed — they depend on user-downloaded data.
Running the commands above against real `BGL.log` / `HDFS.log`
regenerates them in place.

---

## 4. Honest wording

This repository ships with a benchmark harness and reference results.
Synthetic fixtures and public-dataset smoke fixtures are included
in-repo; public real-world log benchmarks are reproducible locally
using automatically derived transition targets. Target derivation is
documented per adapter. **No artifact under `benchmarks/` is a
manually annotated segmentation benchmark.**

---

## Fixture regeneration

- `tools/gen_benchmark_fixtures.py` — regenerates `benchmarks/synthetic/`
- `tools/gen_public_smoke_fixtures.py` — regenerates the `BGL_smoke.log`
  and `HDFS_smoke.log` fixtures

Both scripts are deterministic (seeded).
