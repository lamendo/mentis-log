"""Dataset adapters for public-log benchmarks.

Each adapter exposes:
  - DATASET_NAME: str
  - load(data_dir, **kwargs) -> Dataset object with:
        .lines              list[str]
        .labels             list[str]
        .derived_boundaries list[int]
        .tolerance_lines    int
        .target_metadata    dict
        .text               str   (property: "\\n".join(lines))
"""
from __future__ import annotations

from typing import Callable, Dict

from . import bgl, hdfs

REGISTRY: Dict[str, Callable] = {
    bgl.DATASET_NAME: bgl.load,
    hdfs.DATASET_NAME: hdfs.load,
}

__all__ = ["bgl", "hdfs", "REGISTRY"]
