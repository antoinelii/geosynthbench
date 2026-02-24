from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from geosynthbench.pipeline.types import RenderArtifacts
from geosynthbench.tasks.base import TaskConfig
from geosynthbench.tasks.e1_elevation import _px_loc


@dataclass(frozen=True)
class S1Config(TaskConfig):
    world_cfg: Any
    min_delta: float = 0.1  # slope difference threshold


class S1SlopeCompareTask:
    code = "S1"
    name = "Slope comparison between settlements"
    is_temporal = False

    def generate_t0(self, cfg: S1Config, rng: np.random.Generator):
        from geosynthbench.tasks.utils import generate_t0_sample

        # If your generate_t0_sample uses cfg.seed internally, set it here.
        # Otherwise pass rng down (if supported). Minimal approach: clone config with seed.
        world_cfg = cfg.world_cfg
        # if it's a dataclass/frozen, just mutate if allowed
        # if hasattr(world_cfg, "seed"):
        #    setattr(world_cfg, "seed", int(rng.integers(0, 2**31 - 1)))
        return generate_t0_sample(world_cfg)

    def build_record(
        self,
        *,
        sample_idx: int,
        cfg: S1Config,
        world_t0,
        render: RenderArtifacts,
        rng: np.random.Generator,
    ) -> dict[str, Any]:
        settlements = list(world_t0.settlements)
        if len(settlements) < 2:
            raise ValueError("Need ≥2 settlements")

        a, b = settlements[0], settlements[1]

        sa = world_t0.terrain.sample_slope_point(a.center.x, a.center.y)
        sb = world_t0.terrain.sample_slope_point(b.center.x, b.center.y)

        if abs(sa - sb) < cfg.min_delta:
            raise ValueError("Degenerate slope sample")

        answer = "A" if sa > sb else "B"

        a_px = _px_loc(a.center, world_t0)
        b_px = _px_loc(b.center, world_t0)

        prompt = (
            f"[{self.code}] Two settlements are given by pixel coordinates:\n"
            f"A at {a_px}\n"
            f"B at {b_px}\n\n"
            f"Question: Which settlement is on steeper terrain? Answer with A or B."
        )

        return {
            "sample_id": f"{sample_idx:05d}",
            "task_code": self.code,
            "task_name": self.name,
            "modality": "single",
            "inputs": {
                "image": render.t0_rgb,
                "mask": render.t0_mask,
            },
            "prompt": prompt,
            "answer": answer,
            "oracle": {
                "A": {"settlement_id": str(a.id), "px": list(a_px), "slope": float(sa)},
                "B": {"settlement_id": str(b.id), "px": list(b_px), "slope": float(sb)},
            },
        }
