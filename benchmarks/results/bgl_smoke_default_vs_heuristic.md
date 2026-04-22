# Benchmark report — bgl

- Target type: `derived_from_label_transitions` (source: BGL per-line label: '-' = normal, any other token = alert. Phases are runs of identical binary states with min_run = 500; transitions are merged within 100 lines.)
- Strategies compared: `auto, heuristic`
- Tolerance (lines): `200`

## Aggregate metrics

| strategy | runtime (s) | annotated files | mean P | mean R | mean F1 |
|---|---:|---:|---:|---:|---:|
| auto | 0.18 | 1 | 1.000 | 0.000 | 0.000 |
| heuristic | 0.64 | 1 | 1.000 | 0.500 | 0.667 |

## Per-file results

### `BGL_smoke.log`
- n_lines: `4000`  | expected: `4`  | annotated: `True`
  - **auto**: 0.18s, bnds=0, P=1.000 R=0.000 F1=0.000
  - **heuristic**: 0.64s, bnds=2, P=1.000 R=0.500 F1=0.667

---

Target derivation is documented in the corresponding adapter. This is not a manually-annotated segmentation benchmark; expected boundaries are derived from label/severity transitions in the source log.