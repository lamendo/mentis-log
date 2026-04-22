# mentis_log_cli — Operator Manual

Practical reference for running the tool.

## 1. Installation

```bash
pip install -r requirements.txt                 # core: numpy
pip install -r requirements-plot.txt            # optional: matplotlib
python -m pytest tests/ -q                      # optional: verify install
```

Python 3.9+ recommended. No other runtime dependencies.

All commands below are run from the extract root, e.g.:

```bash
cd extracts/mentis_log_extract
python mentis_log_cli.py <subcommand> ...
```

## 2. CLI overview

Three subcommands:

| Subcommand | Purpose |
|---|---|
| `segment` | Segment one file; emit JSON result |
| `plot` | Render a PNG of the structural-change signal |
| `benchmark` | Run on a directory of logs or a public dataset |

Every subcommand has its own `--help` split into a **basic** and
**advanced / experimental** argument group. Most users only need
the basic group.

## 3. `segment`

### Default usage

```bash
python mentis_log_cli.py segment --input app.log --output result.json
```

Defaults (line mode):

- **strategy** = `auto` (token-based rolling JSD)
- **signal** = token vocabulary per line window
- **min_token_freq** = 2 (prune hapax legomena / unique IDs)
- **refinement** = local multiscale (enabled)
- **edge cleanup** = enabled

### Important flags

| Flag | Purpose |
|---|---|
| `--mode {line,char}` | Analysis unit. Default `line`. `char` is bit-exact vs the in-repo reference pipeline for small inputs; orders of magnitude slower. |
| `--strategy {auto,heuristic,qalign}` | Default `auto`. `heuristic` uses the coverage-routed divergence policy. |
| `--sample-rate N` | Evaluate the curve at every N-th position. |
| `--interpret` | Add a structural interpretation layer (see below). |
| `--no-refine` | Disable local multiscale boundary refinement. |
| `--no-edge-cleanup` | Keep trivially short edge segments. |
| `--include-curve` | Include the per-position curve in the JSON output. |

Peak-tuning flags (`--window`, `--min-distance`, `--min-prominence`,
`--nms-radius`, `--consolidation-radius`, `--min-segment-windows`)
live in the advanced group and are almost never needed.

### `raw_boundaries` vs `boundaries`

- `raw_boundaries` — the coarse detector output *before* local
  refinement. Preserved in the JSON as an audit trail.
- `boundaries` — the public boundary list, after refinement and edge
  cleanup. Values are **separator** positions.
- `boundary_details[i]` — per-boundary audit record.

### Interpretation output (`--interpret`)

Each segment receives:

- `profile` — one of `stable`, `transition`, `volatile`
- `profile_confidence` — in [0, 1]; a relative fit score, not a
  probability
- `summary` — short structural sentence
- numeric stats: `mean_score`, `std_score`, `max_score`, `p90_score`
- `boundary_strength_left` / `boundary_strength_right` — curve value
  at the segment's left / right boundary (null at the file edges)

The three labels describe only the **shape of the change curve
inside the segment**. They make no claim about what the segment
means.

## 4. `plot`

```bash
python mentis_log_cli.py plot \
    --input app.log --output plot.png --comparison
```

Two layouts:

- **default** — single-panel plot of the structural-change signal
  with boundary markers.
- **`--comparison`** — two-panel layout:
  top: raw per-line view (line length per line),
  bottom: the structural-change signal after processing.

### How to read a plot

- The **signal curve** is the per-position divergence score.
- **Vertical dashed lines** mark detected public boundaries.
- **Alternating background bands** show segments.
- The **top-right coloured badge** records the inner route
  selected by the coverage policy.
- The **stats block** shows route / coverage / boundary count.

`--annotate-segments` overlays each segment band with its
structural profile (implies `--interpret`).

## 5. `benchmark`

Two modes, mutually exclusive.

### Synthetic (in-repo)

```bash
python mentis_log_cli.py benchmark \
    --input-dir benchmarks/synthetic \
    --output benchmarks/results/synthetic_default_vs_heuristic.json
```

Recursively finds `*.log` files. If a `*.json` file shares a
basename with a log, its `expected_line_boundaries` list is used to
compute precision / recall / F1 per file.

### Public-dataset workflow

```bash
# Smoke fixture included in repo
python mentis_log_cli.py benchmark \
    --dataset bgl \
    --data-dir benchmarks/datasets/public/bgl \
    --output benchmarks/results/bgl_smoke_default_vs_heuristic.json

# With the real dataset placed in the same directory
#   benchmarks/datasets/public/bgl/BGL.log
# the canonical filename is preferred:
python mentis_log_cli.py benchmark \
    --dataset bgl \
    --data-dir benchmarks/datasets/public/bgl \
    --output benchmarks/results/bgl_default_vs_heuristic.json
```

Supported datasets: `bgl` (primary), `hdfs` (weak fit — see
`benchmarks/datasets/public/hdfs/README.md`).

### Strategy comparison

```bash
--strategies auto,heuristic,qalign
```

Default compares `auto` vs `heuristic`.

### Optional knobs

| Flag | Purpose |
|---|---|
| `--tolerance N` | Boundary-match tolerance in lines. |
| `--min-run N` | Override adapter min-phase-length for target derivation. |
| `--merge-window N` | Override adapter transition-merge window. |
| `--max-lines N` | Truncate the source log before deriving targets. |
| `--no-summary-md` | Skip the sibling `.md` summary. |

See `benchmarks/README.md` for a full description of the harness
and target-derivation logic.

## 6. Output JSON reference

Top-level keys in line mode:

| Key | Present when | Meaning |
|---|---|---|
| `mode` | always | `"line"` or `"char"` |
| `strategy`, `scoring` | always | What ran |
| `n_lines` / `n_chars` | always | Input size |
| `route` | always | Inner coverage-branch selection |
| `n_boundaries`, `boundaries` | always | Public boundary list |
| `n_segments`, `segments` | always | Segment list |
| `threshold_mode`, `effective_quantile` | always | Peak-select audit |
| `curve_stats` | when signal non-empty | mean / std / min / max |
| `raw_boundaries` | refinement ran | Coarse detector output |
| `boundary_details` | refinement ran | `{raw, refined, separator, onset, status}` per boundary |
| `boundary_refinement` | refinement ran | `{enabled, method, radius_lines, scales, fine_signal, public_boundary_semantics, onset_alpha, onset_persistence}` |
| `edge_cleanup` | always in line mode | `{enabled, threshold_lines, dropped_boundaries}` |
| `interpretation` | `--interpret` set | `{enabled, label_schema, disclaimer}` |
| `signal_type`, `divergence`, `q_mode` | `scoring != heuristic` | Which Q-align branch ran |
| `vocab_size`, `avg_tokens_per_line`, `sparsity`, `min_token_freq` | `signal_type == tokens` | Token-path diagnostics |
| `dist_bins` | `signal_type == line_length` | Histogram bin count |

In char mode, `raw_boundaries` / `boundary_refinement` /
`edge_cleanup` are not emitted — the refinement and cleanup stages
are line-mode-only.

## 7. Boundary semantics

Four related quantities, all explicitly named in the output:

| Field | Meaning |
|---|---|
| **raw** (in `boundary_details`) | Position from the coarse line-level detector, before any local recomputation. |
| **separator** | The position of maximal structural separation inside a small window around `raw`. Computed by re-running a multiscale character-class JSD locally and taking its argmax. |
| **onset** | The earliest position `≤ separator` where the same local signal enters a sustained elevated regime. Controlled by `onset_alpha` (relative threshold) and `onset_persistence` (minimum run length). |
| **refined** | Alias for `separator`. Kept so existing callers that read `boundary_details[i]["refined"]` continue to work. |

The public `boundaries` list uses **separator semantics**. That
contract is declared in
`boundary_refinement.public_boundary_semantics == "separator"`.

`onset` is reported alongside for users who need the earliest
lead-in point rather than the clearest cut. On sharp transitions
the two coincide; on gradual ones they differ.

## 8. Troubleshooting

### No boundaries detected

- Defaults for line mode (`DEFAULTS_LINE` in `runtime.py`) are tuned
  for real multi-thousand-line logs. On small inputs (a few hundred
  lines) the distance / prominence filters suppress every candidate.
  Override `--min-distance`, `--nms-radius`,
  `--consolidation-radius`, and `--min-segment-windows` if you need
  finer resolution.
- `--mode char` on a text over ~50 KB runs for minutes. Switch to
  line mode.

### Very repetitive logs

With `signal_type == tokens` and `min_token_freq == 2`, per-line
unique IDs are pruned. If the remaining vocabulary is very small
(e.g. a log whose lines only differ in a timestamp and a request
id) the change signal flattens and few boundaries are detected.
This is intentional — the tool does not invent change where there
is none.

### Very noisy logs (almost every line unique)

Token-based Q-alignment with `min_token_freq == 2` deliberately
prunes hapax legomena. If essentially every word is unique, nearly
the whole vocabulary gets pruned. In that case,
`--strategy heuristic` falls back on coarse line-length / format
shifts and can still produce boundaries.

### Very large logs (millions of lines)

The token-based path uses a sparse streaming implementation with
peak memory `O(nnz + V)` rather than `O(n_lines × V)`. Expect a few
tens of seconds per million lines.

If you need more throughput, pair with `--sample-rate 5`. The cost
of boundary refinement scales with the number of coarse boundaries
times the refinement window size, not with total log length.

### Plot too busy

- `--annotate-segments` adds a short profile label over each
  segment band (implies `--interpret`).
- Raising `--min-distance` and `--min-segment-windows` suppresses
  tightly-packed boundary clusters.
- Passing `--strategy heuristic` can produce a sparser curve on
  input where tokens vary a lot.

## 9. Limitations

- **Structural, not semantic.** The tool measures where the per-line
  structure shifts. It does not know what the shift means.
- **Benchmark targets are derived.** For public datasets, expected
  transitions are mechanically extracted from label / severity
  streams, not manually annotated. They are a proxy measurement.
- **Defaults are for real logs.** On small hand-crafted fixtures the
  coarse filters often suppress every candidate. Expose the
  peak-tuning flags when testing on small inputs.
- **Char mode is for research / parity.** Bit-exact vs the in-repo
  reference pipeline on small inputs, but orders of magnitude
  slower; do not use it on real logs.
- **Onset is conservative.** When no sustained-elevation run exists
  near the separator, onset falls back to separator. The two
  coincide on sharp transitions by design.
