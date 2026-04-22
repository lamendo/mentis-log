"""BGL (Blue Gene/L) log adapter.

Format (Loghub / LogPAI):
    <label> <timestamp> <date> <node> <date-time> <node> <rrs> <core> \
    <severity> <component> <message>

The first whitespace-separated token is the label:
    "-"              → normal / non-alert
    <other string>   → alert type (e.g. KERNDTLB, APPREAD, KERNRTSP)

We binarise to ``{normal, alert}`` and derive expected boundaries
from transitions between stable runs of the binarised state via
``derive_boundaries_from_labels``.

**Target type:** ``derived_from_label_transitions``.
**Not** a manual segmentation ground truth. Short single-line alerts
embedded in otherwise-normal traffic are intentionally absorbed
(``min_run`` default 500 lines).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .evaluation import derive_boundaries_from_labels


DATASET_NAME = "bgl"
DEFAULT_MIN_RUN = 500
DEFAULT_MERGE_WINDOW = 100
DEFAULT_TOLERANCE = 200


@dataclass
class BGLDataset:
    name: str
    source_path: Path
    lines: List[str]                  # message portion (label stripped)
    labels: List[str]                 # one label per line
    binary_states: List[int]          # 0 = normal, 1 = alert
    derived_boundaries: List[int]
    tolerance_lines: int
    target_metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_lines(self) -> int:
        return len(self.lines)

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


def _parse_line(raw: str) -> tuple[str, str]:
    """Split a BGL line into (label, message). ``raw`` has no trailing newline."""
    stripped = raw.lstrip()
    if not stripped:
        return "-", ""
    # First whitespace-separated token is the label.
    first_space = stripped.find(" ")
    if first_space < 0:
        return stripped, ""
    label = stripped[:first_space]
    message = stripped[first_space + 1:]
    return label, message


def _find_log_file(data_dir: Path) -> Optional[Path]:
    """Prefer canonical ``BGL.log``; fall back to any other ``*.log``."""
    canonical = data_dir / "BGL.log"
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
) -> BGLDataset:
    """Load BGL log from ``data_dir`` and derive boundaries.

    Raises ``FileNotFoundError`` if no ``*.log`` file is present.
    """
    data_dir = Path(data_dir)
    log_path = _find_log_file(data_dir)
    if log_path is None:
        raise FileNotFoundError(
            f"No *.log file found under {data_dir}. "
            f"Download BGL.log from the Loghub repository and place it "
            f"there — see benchmarks/datasets/public/bgl/README.md."
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
            lab, msg = _parse_line(raw)
            labels.append(lab)
            lines.append(msg)

    binary = [0 if lab == "-" else 1 for lab in labels]
    boundaries = derive_boundaries_from_labels(
        [str(b) for b in binary],
        min_run=min_run,
        merge_window=merge_window,
    )

    return BGLDataset(
        name=DATASET_NAME,
        source_path=log_path,
        lines=lines,
        labels=labels,
        binary_states=binary,
        derived_boundaries=boundaries,
        tolerance_lines=tolerance,
        target_metadata={
            "target_type": "derived_from_label_transitions",
            "source": (
                "BGL per-line label: '-' = normal, any other token = alert. "
                "Phases are runs of identical binary states with min_run "
                f"= {min_run}; transitions are merged within "
                f"{merge_window} lines."
            ),
            "min_run": int(min_run),
            "merge_window": int(merge_window),
            "alert_fraction": (
                float(sum(binary)) / max(1, len(binary))
            ),
            "disclaimer": (
                "Not a manual segmentation ground truth. "
                "These boundaries are an honest but mechanical derivation "
                "from the anomaly-label stream."
            ),
        },
    )
