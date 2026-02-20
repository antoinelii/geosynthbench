# src/geosynthbench/pipeline/writer.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from geosynthbench.gen.config import WorldGenConfig
from geosynthbench.pipeline.types import RenderArtifacts
from geosynthbench.world.world_state import WorldState


@dataclass(frozen=True)
class DatasetWriter:
    root_dir: Path
    jsonl_path: Path

    @classmethod
    def create(cls, root_dir: str | Path) -> DatasetWriter:
        root = Path(root_dir)
        root.mkdir(parents=True, exist_ok=True)
        return cls(root_dir=root, jsonl_path=root / "dataset.jsonl")

    def sample_dir(self, sample_idx: int) -> Path:
        d = self.root_dir / f"sample_{sample_idx:05d}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def append_jsonl(self, record: dict[str, Any]) -> None:
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _save_u8_mask(self, arr_u8: np.ndarray, path: Path) -> None:
        Image.fromarray(arr_u8, mode="L").save(path)

    def _save_rgb_u8(self, arr_u8: np.ndarray, path: Path) -> None:
        Image.fromarray(arr_u8, mode="RGB").save(path)

    def render_and_save_t0(
        self,
        *,
        sample_idx: int,
        world_t0: Any,
        rng: np.random.Generator,
        save_mask: bool = True,
        save_elev: bool = True,
    ) -> RenderArtifacts:
        """
        Uses your renderer + (optional) your mask builder.
        """
        sd = self.sample_dir(sample_idx)

        # ---- RGB (your existing renderer)
        # render_world_textured returns a PIL Image in your code.
        from geosynthbench.render.renderer import render_world_textured

        rgb_img = render_world_textured(world_t0, rng=rng)
        t0_rgb_path = sd / "t0_rgb.png"
        rgb_img.save(t0_rgb_path)

        # ---- Semantic mask (prefer your existing mask builder if available)
        t0_mask_path: Path | None = None
        if save_mask:
            sem_u8 = None

            # Preferred: if you have a mask builder implemented (recommended)
            try:
                from geosynthbench.render.textures.masks import (
                    build_masks_from_world,  # type: ignore
                )

                res = build_masks_from_world(
                    world_t0, settlement_mode="hull_then_circle", include_settlement_mask=True
                )
                masks = res.masks
                H, W = world_t0.tr.height_px, world_t0.tr.width_px
                sem = np.zeros((H, W), dtype=np.uint8)

                # NOTE: adapt ids if your enum differs
                from geosynthbench.render.semantic import SemanticClass

                sem[masks.get("water", np.zeros((H, W), bool))] = SemanticClass.WATER.value
                sem[masks.get("veg", np.zeros((H, W), bool))] = SemanticClass.VEGETATION.value
                sem[masks.get("settlement", np.zeros((H, W), bool))] = (
                    SemanticClass.SETTLEMENT.value
                )
                sem[masks.get("roads", np.zeros((H, W), bool))] = SemanticClass.ROAD.value

                for b in res.building_items:
                    sem[b["mask"]] = SemanticClass.BUILDING.value

                sem_u8 = sem
            except Exception:
                # Fallback: no semantic mask available (still fine for E1 demo)
                sem_u8 = None

            if sem_u8 is not None:
                t0_mask_path = sd / "t0_mask.png"
                Image.fromarray(sem_u8, mode="L").save(t0_mask_path)

        # ---- Elevation raster (optional, helpful for oracle/debug)
        t0_elev_path: Path | None = None
        if save_elev and getattr(world_t0, "terrain", None) is not None:
            t0_elev_path = sd / "t0_elev.npy"
            np.save(t0_elev_path, world_t0.terrain.elevation_m.astype(np.float32))

        return RenderArtifacts(
            sample_dir=str(sd),
            t0_rgb=str(t0_rgb_path),
            t0_mask=str(t0_mask_path) if t0_mask_path else None,
            t0_elev=str(t0_elev_path) if t0_elev_path else None,
            t1_rgb=None,
            t1_mask=None,
            t1_elev=None,
            change_mask=None,
            extra={},
        )

    def render_and_save(
        self,
        *,
        sample_idx: int,
        world_t0: WorldState,
        world_t1: WorldState | None,
        change_mask: np.ndarray | None,
        cfg: WorldGenConfig,
        rng: np.random.Generator,
    ) -> RenderArtifacts:
        """
        IMPORTANT: Replace the internals with your real renderer calls.
        Contract:
          - write t0_rgb.png (required)
          - write t0_mask.png (recommended)
          - write t0_elev.npy (optional)
          - for temporal tasks: t1_rgb.png, t1_mask.png, and change_mask.png (if provided)
        """
        # sd = self.sample_dir(sample_idx)

        # ---- TODO: integrate your renderer here
        # Expect something like:
        #   rgb_u8, sem_u8 = render_world(world_t0, rng=...)
        # For now: raise to force you to wire it.
        raise NotImplementedError(
            "Wire your renderer here: produce rgb_u8 (H,W,3) and sem_u8 (H,W) for t0 and optionally t1."
        )

        # Example saving (once you have arrays):
        # t0_rgb_path = sd / "t0_rgb.png"
        # self._save_rgb_u8(rgb0, t0_rgb_path)
        # t0_mask_path = sd / "t0_mask.png"
        # self._save_u8_mask(mask0, t0_mask_path)

        # If you want to store elevation:
        # t0_elev_path = sd / "t0_elev.npy"
        # np.save(t0_elev_path, world_t0.terrain.elevation_m.astype(np.float32))

        # Temporal:
        # if world_t1 is not None: save t1_rgb/t1_mask/t1_elev
        # if change_mask is not None: save change_mask.png

        # return RenderArtifacts(...)
