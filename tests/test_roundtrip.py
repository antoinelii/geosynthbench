from __future__ import annotations

import json
from pathlib import Path

from geosynthbench.gen.config import WorldGenConfig
from geosynthbench.gen.pipeline import generate_world
from geosynthbench.io.deserialize import world_from_jsonl
from geosynthbench.io.jsonl_utils import JsonlWritePaths, append_world_t0_jsonl
from geosynthbench.io.roundtrip_check import assert_same_world, world_fingerprint_debug
from geosynthbench.world.raster import RasterTransform


def test_jsonl_roundtrip(tmp_path: Path) -> None:
    tr = RasterTransform(extent=(0.0, 0.0, 2560.0, 2560.0), width_px=256, height_px=256)
    cfg = WorldGenConfig(seed=123)

    w0 = generate_world(tr, cfg, max_retries=50)

    paths = JsonlWritePaths(
        jsonl_path=tmp_path / "t0.jsonl",
        terrain_dir=tmp_path / "terrain",
    )

    append_world_t0_jsonl(
        paths=paths,
        sample_id="sample_00000",
        world=w0,
        save_terrain=True,
    )

    with paths.jsonl_path.open("r", encoding="utf-8") as f:
        record = json.loads(f.readline())

    w1 = world_from_jsonl(record["t0"], base_dir=paths.jsonl_path.parent)
    assert len(w1.settlements) == len(w0.settlements)
    assert len(w1.buildings) == len(w0.buildings)
    assert world_fingerprint_debug(w0, w1)
    assert_same_world(w0, w1)
