# Changelog

All notable changes to this project will be documented here. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.0] — 2026-04-22

Initial public release.

### Added

- **Line-mode segmentation** — default analysis unit is a log line.
- **Token-based rolling JSD** as the default scoring strategy (`auto`).
  Uses a sparse streaming implementation with `O(nnz + V)` memory;
  handles million-line logs that would otherwise OOM on a dense
  `(N, V)` matrix.
- **Heuristic strategy** (`--strategy heuristic`) — coverage-routed
  divergence policy, three-branch selector (HGB / GKM / LXM).
- **Q-alignment strategy** (`--strategy qalign`) with configurable
  signal (`tokens` or `line_length`), `q_mode`
  (`global` / `rolling` / `prefix`), divergence (`q_jsd` / `q_kl`).
- **Local multiscale boundary refinement** — character-class JSD at
  scales `(4, 16, 64)`, vectorised via one-hot + cumsum. Produces
  both a **separator** (argmax) and an **onset** (earliest sustained
  elevation). Public boundaries use separator semantics; metadata
  declares `public_boundary_semantics: "separator"`.
- **Edge-segment cleanup** — drops the first or last boundary when
  the resulting edge segment is shorter than
  `max(32, int(0.001 × n_lines))` lines.
- **Optional structural interpretation** (`--interpret`) — per-segment
  profile label (`stable` / `transition` / `volatile`) with a
  confidence score and short structural summary. Labels are
  strictly structural; no semantic claims.
- **Char mode** (`--mode char`) kept for bit-exact parity against
  the in-repo reference pipeline on small text inputs.
- **CLI subcommands**: `segment`, `plot`, `benchmark`.
- **Benchmark harness** — synthetic fixtures committed in-repo,
  Loghub BGL / HDFS adapter stubs with download READMEs and
  committed smoke fixtures. Reference result artifacts under
  `benchmarks/results/`.
- **Memory guard** — `lines_to_matrix` raises `MemoryError` when an
  `(N, V)` dense allocation would exceed 2 GiB; steers users to
  the sparse streaming path.
- **Two-panel comparison plot** — raw per-line view vs structural
  change signal. Optional segment-profile annotations.
- **Documentation**: `README.md` (landing), `USAGE.md` (operator
  manual), `benchmarks/README.md` (harness), dataset-specific
  READMEs, `benchmarks/results/summary.md`.
- **Test suite** — 210 tests, deterministic, runs in ~20 s.

### Known limitations

- Boundaries are **separators**, not semantic labels. The tool does
  not classify incidents, anomalies, or root causes.
- Benchmark targets for public datasets are **mechanically derived**
  from label / severity transitions, not manually annotated.
- `char` mode runtime scales poorly on inputs above ~50 KB. Use
  `line` mode for real logs.
- The HDFS adapter is included for symmetry but is a **weak fit**
  for regime segmentation (block labels, not line labels).
