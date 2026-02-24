# src/geosynthbench/tasks/base.py
from __future__ import annotations

from typing import Any, Protocol, TypeAlias, TypeVar

import numpy as np
import numpy.typing as npt
from attr import dataclass

from geosynthbench.gen.config import WorldGenConfig
from geosynthbench.pipeline.types import RenderArtifacts
from geosynthbench.world.world_state import WorldState


class TaskConfig(Protocol):
    # keep minimal; maybe just an id, difficulty, etc.
    pass


JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]
JSONObj: TypeAlias = dict[str, JSONValue]


@dataclass(frozen=True)
class ChangeResult:
    world_t1: WorldState
    change_mask: npt.NDArray[np.bool_] | None
    change_log: JSONObj | None


C = TypeVar("C", bound="TaskConfig", contravariant=True)


class BaseTask(Protocol[C]):
    """
    Minimal task contract.
    """

    code: str  # e.g., "E1", "A1"
    name: str  # human-friendly
    is_temporal: bool  # single-image vs t0->t1

    def generate_t0(self, world_cfg: WorldGenConfig, rng: np.random.Generator) -> WorldState: ...

    def apply_change(
        self, world_t0: WorldState, task_cfg: C, rng: np.random.Generator
    ) -> ChangeResult:
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
        task_cfg: C,
        world_t0: WorldState,
        world_t1: WorldState | None,
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
