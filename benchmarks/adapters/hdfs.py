"""HDFS log adapter (**weak-fit**).

HDFS logs from Loghub ship with *block-level* anomaly labels, not
per-line labels. This does NOT align well with regime segmentation:
the task is block classification, not phase detection.

This adapter therefore uses a **pragmatic placeholder**:
  - parse each line for its severity token (``INFO`` / ``WARN`` /
    ``ERROR`` / ``FATAL``)
  - treat severity-level shifts as candidate phase transitions
  - document clearly that this is NOT a direct anomaly benchmark

BGL is the primary public benchmark target. HDFS is included so the
directory structure is complete and the command-line UX is symmetric.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .evaluation import derive_boundaries_from_labels


DATASET_NAME = "hdfs"
DEFAULT_MIN_RUN = 200
DEFAULT_MERGE_WINDOW = 50
DEFAULT_TOLERANCE = 100

# HDFS line format (example):
#   081109 203518 143 INFO dfs.DataNode$PacketResponder: ...
# The severity token is the 4th whitespace-separated field.
_SEVERITY_RE = re.compile(r"\b(INFO|WARN|WARNING|ERROR|FATAL|DEBUG)\b")


@dataclass
class HDFSDataset:
    name: str
    source_path: Path
    lines: List[str]
    labels: List[str]                 # severity per line
    derived_boundaries: List[int]
    tolerance_lines: int
    target_metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_lines(self) -> int:
        return len(self.lines)

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


def _severity(line: str) -> str:
    m = _SEVERITY_RE.search(line)
    return m.group(1) if m else "OTHER"


def _find_log_file(data_dir: Path) -> Optional[Path]:
    canonical = data_dir / "HDFS.log"
    if canonical.is_file():
        return canonical
    logs = sorted(data_dir.glob("*.log"))
    return logs[0] if logs else None


def load(
    data_dir: Path,
    *,
    max_lines: Optional[int] = None,
    min_run: int = DEFAULT_MIN_RUN,
    merge_window: int = DEFAULT_MERGE_WINDOW,
    tolerance: int = DEFAULT_TOLERANCE,
) -> HDFSDataset:
    """Load HDFS log from ``data_dir`` and derive severity-shift targets.

    Note: severity shifts are a WEAK proxy for regime boundaries on
    HDFS. This adapter is primarily for structural symmetry with the
    BGL workflow; prefer BGL for meaningful results.
    """
    data_dir = Path(data_dir)
    log_path = _find_log_file(data_dir)
    if log_path is None:
        raise FileNotFoundError(
            f"No *.log file found under {data_dir}. "
            f"Download HDFS.log from the Loghub repository and place it "
            f"there — see benchmarks/datasets/public/hdfs/README.md."
        )

    lines: List[str] = []
    labels: List[str] = []
    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for idx, raw in enumerate(f):
            if max_lines is not None and idx >= max_lines:
                break
            raw = raw.rstrip("\r\n")
            if not raw:
                continue
            lines.append(raw)
            labels.append(_severity(raw))

    boundaries = derive_boundaries_from_labels(
        labels, min_run=min_run, merge_window=merge_window,
    )

    return HDFSDataset(
        name=DATASET_NAME,
        source_path=log_path,
        lines=lines,
        labels=labels,
        derived_boundaries=boundaries,
        tolerance_lines=tolerance,
        target_metadata={
            "target_type": "derived_from_severity_transitions",
            "source": (
                "Per-line severity token (INFO/WARN/ERROR/FATAL). "
                f"Phases are runs of identical severity with min_run "
                f"= {min_run}; transitions merged within {merge_window} "
                f"lines."
            ),
            "min_run": int(min_run),
            "merge_window": int(merge_window),
            "disclaimer": (
                "HDFS is a weak fit for regime segmentation. The Loghub "
                "HDFS benchmark labels blocks, not lines. Severity shifts "
                "are used here only as a structural placeholder. Prefer "
                "the BGL benchmark."
            ),
        },
    )
