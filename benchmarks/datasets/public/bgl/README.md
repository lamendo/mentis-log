# BGL — Blue Gene/L system log

## What this directory is for

This is the expected location for the public BGL log file. The
repository **does not** bundle the full dataset; it is 700+ MB and
under a separate licence.

A small synthetic fixture (`BGL_smoke.log`) is committed so the
benchmark command runs immediately. Replace (or co-locate) with the
real dataset to reproduce published-scale numbers.

## How to obtain the real dataset

The dataset ships with the Loghub repository:

- https://github.com/logpai/loghub
- direct link: https://zenodo.org/record/3227177

Download `BGL.tar.gz` (or the split variant), extract so this directory
contains `BGL.log`:

```
benchmarks/datasets/public/bgl/
├── README.md          (this file)
├── BGL_smoke.log      (committed; ~800-line synthetic fixture)
└── BGL.log            (you place this; not committed)
```

Expected size: ~4.7 M lines, ~700 MB uncompressed.

## Run the benchmark

```bash
# Run on the committed smoke fixture (always works)
python mentis_log_cli.py benchmark \
    --dataset bgl \
    --data-dir benchmarks/datasets/public/bgl \
    --output benchmarks/results/bgl_smoke_default_vs_heuristic.json

# After downloading real BGL.log (canonical filename preferred)
python mentis_log_cli.py benchmark \
    --dataset bgl \
    --data-dir benchmarks/datasets/public/bgl \
    --output benchmarks/results/bgl_default_vs_heuristic.json \
    --strategies auto,heuristic
```

If both `BGL.log` and `BGL_smoke.log` are present, the adapter uses
`BGL.log` (canonical name wins).

## Target derivation (honest)

The BGL adapter parses each line's first whitespace-separated token
as its label:

- `-` → normal
- any other token → alert (e.g. `KERNDTLB`, `APPREAD`, `KERNRTSP`)

We binarise to `{normal, alert}` and derive phase transitions using
`derive_boundaries_from_labels` with defaults:

- `min_run = 500` — a phase must span at least 500 consecutive
  same-state lines to be counted (brief single-line alerts absorbed)
- `merge_window = 100` — adjacent transitions within 100 lines collapse

These are heuristic targets derived mechanically from the label stream.
**Not** a manual segmentation ground truth. The result artifact
includes this metadata under `target_metadata`.

## Licence

Loghub datasets are distributed under their own licence; see the
Loghub repository. Nothing under this directory beyond `README.md` and
`BGL_smoke.log` is committed to this repository.
