from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from geosynthbench.pipeline.types import RenderArtifacts
from geosynthbench.tasks.base import TaskConfig
from geosynthbench.tasks.utils import px_loc


@dataclass(frozen=True)
class D1Config(TaskConfig):
    world_cfg: Any
    min_delta_m: float = 10.0
    strategy: str = "first_two"


class D1DistanceToWaterTask:
    code = "D1"
    name = "Distance to water comparison"
    is_temporal = False

    def generate_t0(self, cfg: D1Config):
        from geosynthbench.tasks.utils import generate_t0_sample

        # If your generate_t0_sample uses cfg.seed internally, set it here.
        # Otherwise pass rng down (if supported). Minimal approach: clone config with seed.
        world_cfg = cfg.world_cfg
        # if it's a dataclass/frozen, just mutate if allowed
        # if hasattr(world_cfg, "seed"):
        #    setattr(world_cfg, "seed", int(rng.integers(0, 2**31 - 1)))

        return generate_t0_sample(world_cfg)

    def _dist_to_water(self, settlement, world) -> float:
        if not world.water:
            return float("inf")
        return min(settlement.center.distance(w.polygon) for w in world.water)

    def build_record(
        self,
        *,
        sample_idx: int,
        cfg: D1Config,
        world_t0,
        render: RenderArtifacts,
        rng: np.random.Generator,
    ) -> dict[str, Any]:
        settlements = list(world_t0.settlements)
        if len(settlements) < 2:
            raise ValueError("Need ≥2 settlements")

        if cfg.strategy == "random_two":
            a, b = rng.choice(settlements, size=2, replace=False)
        else:
            a, b = settlements[0], settlements[1]

        da = self._dist_to_water(a, world_t0)
        db = self._dist_to_water(b, world_t0)

        if abs(da - db) < cfg.min_delta_m:
            raise ValueError("Degenerate distance sample")

        answer = "A" if da < db else "B"

        a_px = px_loc(a.center, world_t0)
        b_px = px_loc(b.center, world_t0)

        prompt = (
            f"[{self.code}] Two settlements are given by pixel coordinates:\n"
            f"A at {a_px}\n"
            f"B at {b_px}\n\n"
            f"Question: Which settlement is closer to water? Answer with A or B."
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
                "A": {"settlement_id": str(a.id), "px": list(a_px), "dist_to_water_m": float(da)},
                "B": {"settlement_id": str(b.id), "px": list(b_px), "dist_to_water_m": float(db)},
            },
        }
