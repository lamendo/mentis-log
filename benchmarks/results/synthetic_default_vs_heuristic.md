# Benchmark report — unknown

- Input: `benchmarks\synthetic`
- Strategies compared: `auto, heuristic`
- Tolerance (lines): `100`

## Aggregate metrics

| strategy | runtime (s) | annotated files | mean P | mean R | mean F1 |
|---|---:|---:|---:|---:|---:|
| auto | 0.51 | 3 | 1.000 | 0.783 | 0.869 |
| heuristic | 0.59 | 3 | 1.000 | 0.783 | 0.869 |

## Per-file results

### `incident_medium.log`
- n_lines: `4350`  | expected: `5`  | annotated: `True`
  - **auto**: 0.23s, bnds=3, P=1.000 R=0.600 F1=0.750
  - **heuristic**: 0.29s, bnds=3, P=1.000 R=0.600 F1=0.750

### `incident_noisy.log`
- n_lines: `3225`  | expected: `4`  | annotated: `True`
  - **auto**: 0.17s, bnds=3, P=1.000 R=0.750 F1=0.857
  - **heuristic**: 0.19s, bnds=3, P=1.000 R=0.750 F1=0.857

### `incident_small.log`
- n_lines: `2250`  | expected: `2`  | annotated: `True`
  - **auto**: 0.12s, bnds=2, P=1.000 R=1.000 F1=1.000
  - **heuristic**: 0.11s, bnds=2, P=1.000 R=1.000 F1=1.000

---

Target derivation is documented in the corresponding adapter. This is not a manually-annotated segmentation benchmark; expected boundaries are derived from label/severity transitions in the source log.