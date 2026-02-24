from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from shapely.geometry import Point

from geosynthbench.gen.config import WorldGenConfig
from geosynthbench.pipeline.writer import RenderArtifacts
from geosynthbench.tasks.base import TaskConfig
from geosynthbench.tasks.utils import px_loc
from geosynthbench.world.world_state import WorldState


@dataclass(frozen=True)
class E1Config(TaskConfig):
    # uses your WorldGenConfig instance directly
    world_cfg: WorldGenConfig
    min_delta_m: float = 5.0
    settlement_strategy: str = "first_two"  # "random_two"


def _altitude_m(point: Point, world: WorldState) -> float:
    if world.terrain is None:
        raise ValueError("World terrain is not defined")
    return world.terrain.sample_point(point.x, point.y)


class E1ElevationCompareTask:
    code = "E1"
    name = "Elevation comparison of two settlements"
    is_temporal = False

    def generate_t0(self, task_cfg: E1Config):
        # Your existing generator wrapper

        from geosynthbench.tasks.utils import generate_t0_sample

        # IMPORTANT: make generation reproducible per sample:
        # If your generate_t0_sample uses cfg.seed internally, set it here.
        # Otherwise pass rng down (if supported). Minimal approach: clone config with seed.
        world_cfg = task_cfg.world_cfg
        # if it's a dataclass/frozen, just mutate if allowed
        # if hasattr(world_cfg, "seed"):
        #    setattr(world_cfg, "seed", int(rng.integers(0, 2**31 - 1)))

        return generate_t0_sample(world_cfg)

    def build_record(
        self,
        *,
        sample_idx: int,
        task_cfg: E1Config,
        world_t0: WorldState,
        render: RenderArtifacts,
        rng: np.random.Generator,
    ) -> dict[str, Any]:
        settlements = list(world_t0.settlements)
        if len(settlements) < 2:
            raise ValueError("E1 requires at least 2 settlements.")

        if task_cfg.settlement_strategy == "random_two":
            idx_a, idx_b = rng.choice(len(settlements), size=2, replace=False)
            a, b = settlements[int(idx_a)], settlements[int(idx_b)]
        else:
            a, b = settlements[0], settlements[1]

        pa = a.center
        pb = b.center

        ea = _altitude_m(pa, world_t0)
        eb = _altitude_m(pb, world_t0)

        # validate non-degenerate
        if abs(ea - eb) < task_cfg.min_delta_m:
            raise ValueError("Degenerate sample: elevations too close.")

        answer = "A" if ea > eb else "B"

        a_px = px_loc(pa, world_t0)
        b_px = px_loc(pb, world_t0)

        prompt = (
            f"[{self.code}] Two settlements are given by pixel coordinates:\n"
            f"A at {a_px}, B at {b_px}.\n"
            f"Which settlement is at higher elevation? Answer with A or B."
        )

        return {
            "sample_id": f"{sample_idx:05d}",
            "task_code": self.code,
            "task_name": self.name,
            "modality": "single",
            "inputs": {
                "image": render.t0_rgb,
                "mask": render.t0_mask,
                "elevation": render.t0_elev,
            },
            "prompt": prompt,
            "answer": answer,
            "oracle": {
                "settlement_A_id": str(a.id),
                "settlement_B_id": str(b.id),
                "A_px": list(a_px),
                "B_px": list(b_px),
                "elev_A_m": float(ea),
                "elev_B_m": float(eb),
                "delta_m": float(ea - eb),
            },
        }
