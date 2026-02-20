# Create a dataset structure
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from geosynthbench.io.deserialize import world_from_dict
from geosynthbench.io.jsonl_utils import JsonlWritePaths, append_world_t0_jsonl, read_jsonl_record
from geosynthbench.world.world_state import WorldState


class WorldItem:
    def __init__(
        self,
        sample_id: str,
        jsonl_path: Path,
        world_state: Optional[WorldState] = None,
        rgb_path: Optional[Path] = None,
        mask_path: Optional[Path] = None,
        height_path: Optional[Path] = None,
        slope_path: Optional[Path] = None,
    ):
        self.sample_id = sample_id
        self.jsonl_path = jsonl_path
        self.terrain_path = self.jsonl_path.parent / "terrain" / f"{sample_id}_elevation.npy"
        self.rgb_path = (
            self.jsonl_path.parent / "rgb" / f"{sample_id}_rgb.png"
            if rgb_path is not None
            else None
        )
        self.mask_path = (
            self.jsonl_path.parent / "mask" / f"{sample_id}_mask.png"
            if mask_path is not None
            else None
        )
        self.height_path = (
            self.jsonl_path.parent / "height" / f"{sample_id}_height.png"
            if height_path is not None
            else None
        )
        self.slope_path = (
            self.jsonl_path.parent / "slope" / f"{sample_id}_slope.png"
            if slope_path is not None
            else None
        )
        self.world_state = world_state
        self.has_rgb = self.rgb_path is not None and self.rgb_path.exists()
        self.has_mask = self.mask_path is not None and self.mask_path.exists()
        self.has_height = self.height_path is not None and self.height_path.exists()
        self.has_slope = self.slope_path is not None and self.slope_path.exists()

    def __str__(self) -> str:
        return self.sample_id

    def load_world_state(self) -> WorldState:
        idx = int(self.sample_id)
        record = read_jsonl_record(self.jsonl_path, idx)
        world_state = world_from_dict(record["t0"], base_dir=self.jsonl_path.parent)
        return world_state

    def load_terrain(self) -> np.ndarray[tuple[int, int], np.float32]:
        terrain = np.load(self.terrain_path).astype(np.dtype(np.float32))
        return terrain

    def load_rgb(self) -> Optional[np.ndarray[tuple[int, int, int], np.uint8]]:
        if self.has_rgb:
            assert self.rgb_path is not None
            rgb = np.load(self.rgb_path).astype(np.dtype(np.uint8))
            return rgb
        return None

    def load_mask(self) -> Optional[np.ndarray[tuple[int, int, int], np.uint8]]:
        if self.has_mask:
            assert self.mask_path is not None
            mask = np.load(self.mask_path).astype(np.dtype(np.uint8))
            return mask
        return None

    def load_height(self) -> Optional[np.ndarray[tuple[float], np.float32]]:
        if self.has_height:
            assert self.height_path is not None
            height = np.load(self.height_path).astype(np.dtype(np.float32))
            return height
        return None

    def load_slope(self) -> Optional[np.ndarray[tuple[float], np.float32]]:
        if self.has_slope:
            assert self.slope_path is not None
            slope = np.load(self.slope_path).astype(np.dtype(np.float32))
            return slope
        return None


class WorldsDataset:
    def __init__(
        self,
        base_dir: Path | str,
        jsonl_name: str = "t0.jsonl",
    ):
        self.base_dir = Path(base_dir)
        self.jsonl_path = self.base_dir / jsonl_name
        self.terrain_dir = self.base_dir / "terrain"
        self.items = []
        if self.jsonl_path.exists():
            with self.jsonl_path.open("r", encoding="utf-8") as f:
                for line in f:
                    record = json.loads(line)
                    sample_id = record["sample_id"]
                    item = WorldItem(
                        sample_id=sample_id,
                        jsonl_path=self.jsonl_path,
                    )
                    self.items.append(item)

    def create_folder_structure(self):
        if not self.base_dir.exists():
            self.base_dir.mkdir(parents=True, exist_ok=True)
            self.terrain_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> WorldItem:
        return self.items[idx]

    def load_world_state(self, idx: int) -> WorldState:
        world_state = self.items[idx].load_world_state()
        return world_state

    def load_terrain(self, idx: int) -> np.ndarray[tuple[float, float], np.float32]:
        terrain = self.items[idx].load_terrain()
        return terrain

    def load_rgb(self, idx: int) -> Optional[np.ndarray[tuple[int, int, int], np.uint8]]:
        item = self.items[idx]
        if item.rgb_path is not None and item.rgb_path.exists():
            rgb = np.load(item.rgb_path).astype(np.dtype(np.uint8))
            return rgb
        return None

    def add_record(self, world_state: WorldState):
        sample_id = f"{len(self.items):05d}"
        paths = JsonlWritePaths(self.jsonl_path, self.terrain_dir)
        # add to jsonl and save terrain
        append_world_t0_jsonl(
            paths=paths, sample_id=sample_id, world=world_state, save_terrain=True
        )
        self.items.append(WorldItem(sample_id=sample_id, jsonl_path=self.jsonl_path))
        print(
            f"Added record {sample_id} to dataset with {len(self.items)} total items."
            f" jsonl: {self.jsonl_path}, terrain: {paths.terrain_dir / f'{sample_id}_elevation.npy'}"
        )
