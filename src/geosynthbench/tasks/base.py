# src/geosynthbench/tasks/base.py
from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from geosynthbench.pipeline.types import RenderArtifacts


class BaseTask(Protocol):
    """
    Minimal task contract.
    """

    code: str  # e.g., "E1", "A1"
    name: str  # human-friendly
    is_temporal: bool  # single-image vs t0->t1

    def generate_t0(self, cfg: Any, rng: np.random.Generator) -> Any: ...

    def apply_change(
        self, world_t0: Any, cfg: Any, rng: np.random.Generator
    ) -> tuple[Any, np.ndarray | None, dict[str, Any] | None]:
        """
        Returns (world_t1, change_mask, change_log).
        change_mask may be None if task doesn't need it.
        change_log should be JSON-serializable.
        """
        ...

    def build_record(
        self,
        *,
        sample_idx: int,
        cfg: Any,
        world_t0: Any,
        world_t1: Any | None,
        render: RenderArtifacts,
        change_log: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """
        Returns JSON-serializable dict for JSONL.
        Must include prompt + answer + task code.
        """
        ...

    def evaluate(self, prediction: Any, record: dict[str, Any]) -> dict[str, float]:
        """
        Optional: used during benchmark evaluation (not generation).
        """
        ...
