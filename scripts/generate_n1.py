from __future__ import annotations

import numpy as np

from geosynthbench.gen.config import WorldGenConfig
from geosynthbench.pipeline.writer import DatasetWriter
from geosynthbench.tasks.n1_isolation import (
    N1Config,
    N1IsolationTask,
    compute_isolation_scores,
    isolation_is_clear,
)
from geosynthbench.utils.logging import get_logger, setup_logging


def make_n1_world_cfg(seed: int) -> WorldGenConfig:
    return WorldGenConfig(
        seed=seed,
        # terrain: gentler => fewer slope rejections
        terrain_amplitude_m=25.0,
        terrain_n_hills=(1, 3),
        terrain_hill_sigma_m=(450.0, 1100.0),
        terrain_noise_scale_m=450.0,
        terrain_noise_strength_m=3.5,
        # keep light
        n_water=(0, 1),
        n_veg=(2, 5),
        # settlements: still 4–6, but allow them to fit
        n_settlements=(4, 6),
        settlement_radius_m=(180.0, 320.0),
        min_dist_settlements_m=520.0,  # ↓ was 800
        max_slope_settlement=0.30,  # slightly relaxed
        # roads: MST for clear leaves, relax slope a bit
        roads_mode="mst",
        extra_edges=0,
        road_width_m=8.0,
        max_slope_road=0.18,  # ↑ was 0.12
        # buildings: make them cheap (or effectively optional)
        buildings_per_settlement=(4, 10),  # ↓ was 8–20
        building_size_m=(12.0, 24.0),
        min_dist_buildings_m=5.0,
        max_slope_building=0.25,
        max_building_attempts=800,  # ↓ was 1500
        require_connected_roads=True,
        forbid_roads_in_water=True,
        prefer_settlement_near_water=False,
        prefer_water_distance_m=(60.0, 350.0),
    )


def main() -> None:
    setup_logging()
    log = get_logger()

    out_dir = "data/N1"
    writer = DatasetWriter.create(out_dir)

    task = N1IsolationTask()

    n_samples = 5
    base_seed = 30_000

    successful_samples = 0
    for i in range(n_samples):
        rng = np.random.default_rng(base_seed + i)

        cfg0 = make_n1_world_cfg(seed=0)  # seed will be mutated inside task.generate_t0
        task_cfg = N1Config(
            world_cfg=cfg0,
            clarity_ratio=1.10,  # ↓ from 1.15
            clarity_delta_m=200.0,  # ↓ from 300
            strategy="by_id",
        )

        for attempt in range(120):
            try:
                world_t0 = task.generate_t0(task_cfg, rng)
            except RuntimeError as e:
                log.info(f"[N1] world generation failed, config might be too restrictive: {e}")
                continue
            try:
                # Hard constraints
                n_sett = len(world_t0.settlements)
                if n_sett < 4 or n_sett > 6:
                    raise ValueError(f"need 4-6 settlements, got {n_sett}")

                # Compute isolation + clarity filter
                scores = compute_isolation_scores(world_t0)

                if not isolation_is_clear(
                    scores, ratio=task_cfg.clarity_ratio, delta_m=task_cfg.clarity_delta_m
                ):
                    raise ValueError("isolation not clear enough")

                # Render + save
                render = writer.render_and_save_t0(sample_idx=i, world_t0=world_t0, rng=rng)

                # Record
                record = task.build_record(
                    sample_idx=i,
                    cfg=task_cfg,
                    world_t0=world_t0,
                    render=render,
                    rng=rng,
                )
                writer.append_jsonl(record)

                log.success(f"[N1] sample {i:05d} OK (attempt={attempt}) | best={record['answer']}")
                successful_samples += 1
                break

            except ValueError as e:
                if attempt in (0, 10, 30, 60, 90, 119):
                    log.info(f"[N1] sample {i:05d} retry attempt={attempt}: {e}")
                if attempt == 119:
                    log.warning(f"[N1] sample {i:05d} FAILED after retries: {e}")
                continue

    log.info(f"[N1] wrote JSONL -> {writer.jsonl_path}")
    log.success(f"[N1] {successful_samples} / {n_samples} successful samples")


if __name__ == "__main__":
    main()
