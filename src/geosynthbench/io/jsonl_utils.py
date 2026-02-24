from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from geosynthbench.io.serialize import world_to_dict
from geosynthbench.world.world_state import WorldState


@dataclass(frozen=True)
class JsonlWritePaths:
    jsonl_path: Path
    terrain_dir: Path | None = None


def save_terrain_sidecar(world: WorldState, terrain_path: Path) -> None:
    """
    Saves elevation array to .npy (fast, simple).
    """
    terrain_path.parent.mkdir(parents=True, exist_ok=True)
    if world.terrain is None:
        raise ValueError("WorldState has no terrain to save.")
    np.save(terrain_path, world.terrain.elevation_m.astype(np.float32), allow_pickle=False)


def append_world_t0_jsonl(
    *,
    paths: JsonlWritePaths,
    sample_id: str,
    world: WorldState,
    extra: dict[str, Any] | None = None,
    save_terrain: bool = True,
) -> None:
    """
    Appends one JSON object per line.

    Stores:
      - sample_id
      - t0 world state (geometries as WKT)
      - optional terrain sidecar path
      - extra metadata (config, counts, etc.)
    """
    terrain_ref: str | None = None

    if save_terrain and world.terrain is not None:
        if paths.terrain_dir is None:
            raise ValueError("terrain_dir must be set if save_terrain=True and world has terrain.")
        paths.terrain_dir.mkdir(parents=True, exist_ok=True)
        terrain_path = paths.terrain_dir / f"{sample_id}_elevation.npy"
        save_terrain_sidecar(world, terrain_path)
        # store relative path when possible
        try:
            terrain_ref = str(terrain_path.relative_to(paths.jsonl_path.parent))
        except Exception:
            terrain_ref = str(terrain_path)

    record: dict[str, Any] = {
        "sample_id": sample_id,
        "t0": world_to_dict(world, elevation_path=terrain_ref),
    }
    if extra:
        record["meta"] = extra

    paths.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with paths.jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl_record(path: Path, idx: int) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == idx:
                return json.loads(line)
    raise IndexError(f"jsonl index {idx} out of range: {path}")
