# src/geosynthbench/pipeline/types.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RenderArtifacts:
    sample_dir: str

    t0_rgb: str
    t0_mask: str | None
    t0_elev: str | None

    t1_rgb: str | None = None
    t1_mask: str | None = None
    t1_elev: str | None = None

    change_mask: str | None = None  # uint8 HxW, 0..K (or binary)
    # use default factory for extra to avoid mutable default arg
    extra: dict[str, Any] = field(
        default_factory=lambda: {}
    )  # for any additional info your renderer wants to return
