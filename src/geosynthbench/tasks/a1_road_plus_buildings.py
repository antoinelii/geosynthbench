# src/geosynthbench/tasks/a1_road_plus_buildings.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from geosynthbench.pipeline.types import RenderArtifacts
from geosynthbench.tasks.base import BaseTask


@dataclass(frozen=True)
class A1Config:
    # how many buildings to add near new road
    n_buildings_min: int = 2
    n_buildings_max: int = 6
    road_width_m: float = 8.0


class A1RoadPlusBuildingsTask(BaseTask):
    code = "A1"
    name = "Add road + new buildings, then categorize + count"
    is_temporal = True

    def generate_t0(self, cfg: Any, rng: np.random.Generator) -> Any:
        # TODO: wire your generate_t0 here
        raise NotImplementedError("Wire your generate_t0() here.")

    def apply_change(
        self, world_t0: Any, cfg: A1Config, rng: np.random.Generator
    ) -> tuple[Any, np.ndarray | None, dict[str, Any] | None]:
        """
        TODO: implement your custom operator:
          - add one road segment (and integrate into roads)
          - add K buildings near that road (K drawn from cfg range)
          - return world_t1, change_mask (uint8), change_log
        For demo, change_mask can be binary: 1 where changed.
        """
        raise NotImplementedError("Implement apply_change() for A1.")

    def build_record(
        self,
        *,
        sample_idx: int,
        cfg: A1Config,
        world_t0: Any,
        world_t1: Any | None,
        render: RenderArtifacts,
        change_log: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if world_t1 is None:
            raise ValueError("A1 requires t1.")

        # Expect your apply_change to provide at least:
        # change_log = {"ops":[{"type":"ROAD_ADDED",...},{"type":"BUILDING_ADDED","count":k,...}]}
        ops = (change_log or {}).get("ops", [])
        building_added = 0
        types: list[str] = []
        for op in ops:
            t = str(op.get("type", ""))
            if t:
                types.append(t)
            if t == "BUILDING_ADDED":
                building_added += int(op.get("count", 0))

        prompt = (
            f"[{self.code}] Compare the two images (t0 then t1). "
            f"Identify what changed and count how many buildings were added due to the new road. "
            f"Return a JSON object with keys: types (list of strings), buildings_added (int)."
        )

        answer = {
            "types": sorted(set(types)),
            "buildings_added": int(building_added),
        }

        return {
            "sample_id": f"{sample_idx:05d}",
            "task_code": self.code,
            "task_name": self.name,
            "modality": "temporal_pair",
            "inputs": {
                "t0_image": render.t0_rgb,
                "t1_image": render.t1_rgb,
                "change_mask": render.change_mask,  # for training/eval if you want
            },
            "prompt": prompt,
            "answer": answer,
            "oracle": {
                "ops": ops,
            },
        }

    def evaluate(self, prediction: Any, record: dict[str, Any]) -> dict[str, float]:
        """
        Minimal: exact match on buildings_added, and type F1 if you want later.
        If you also evaluate segmentation, you can compute IoU using saved change masks.
        """
        # Keep it minimal for now
        out: dict[str, float] = {}

        try:
            pred_b = int(prediction.get("buildings_added"))  # type: ignore[attr-defined]
            gt_b = int(record["answer"]["buildings_added"])
            out["buildings_added_mae"] = float(abs(pred_b - gt_b))
            out["buildings_added_exact"] = float(pred_b == gt_b)
        except Exception:
            out["buildings_added_mae"] = 1e9
            out["buildings_added_exact"] = 0.0

        return out
