# Benchmark report — hdfs

- Target type: `derived_from_severity_transitions` (source: Per-line severity token (INFO/WARN/ERROR/FATAL). Phases are runs of identical severity with min_run = 200; transitions merged within 50 lines.)
- Strategies compared: `auto, heuristic`
- Tolerance (lines): `100`

## Aggregate metrics

| strategy | runtime (s) | annotated files | mean P | mean R | mean F1 |
|---|---:|---:|---:|---:|---:|
| auto | 0.10 | 1 | 1.000 | 0.333 | 0.500 |
| heuristic | 0.24 | 1 | 1.000 | 0.667 | 0.800 |

## Per-file results

### `HDFS_smoke.log`
- n_lines: `1900`  | expected: `3`  | annotated: `True`
  - **auto**: 0.10s, bnds=1, P=1.000 R=0.333 F1=0.500
  - **heuristic**: 0.24s, bnds=2, P=1.000 R=0.667 F1=0.800

---

Target derivation is documented in the corresponding adapter. This is not a manually-annotated segmentation benchmark; expected boundaries are derived from label/severity transitions in the source log.