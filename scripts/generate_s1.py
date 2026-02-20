# scripts/generate_s1.py
from __future__ import annotations

import numpy as np

from geosynthbench.gen.config import WorldGenConfig
from geosynthbench.pipeline.writer import DatasetWriter
from geosynthbench.tasks.s1_slope_compare import S1Config, S1SlopeCompareTask
from geosynthbench.utils.logging import get_logger, setup_logging


def make_s1_world_cfg(seed: int) -> WorldGenConfig:
    # Terrain tuned for slope diversity:
    # - many hills
    # - smaller sigma => steeper slopes
    # - stronger noise
    return WorldGenConfig(
        seed=seed,
        terrain_amplitude_m=350.0,
        terrain_n_hills=(5, 9),
        terrain_hill_sigma_m=(140.0, 320.0),
        terrain_noise_scale_m=180.0,
        terrain_noise_strength_m=12.0,
        n_water=(1, 2),
        n_veg=(3, 7),
        n_settlements=2,
    )


def main() -> None:
    setup_logging()
    log = get_logger()

    out_dir = "data/S1"
    writer = DatasetWriter.create(out_dir)

    task = S1SlopeCompareTask()

    n_samples = 5
    base_seed = 20_000

    for i in range(n_samples):
        rng = np.random.default_rng(base_seed + i)
        cfg0 = make_s1_world_cfg(seed=0)  # seed overridden via rng in task.generate_t0

        task_cfg = S1Config(
            world_cfg=cfg0,
            min_delta=0.02,  # increase if too many degenerate samples
        )

        for attempt in range(80):
            try:
                world_t0 = task.generate_t0(task_cfg, rng)

                if len(world_t0.settlements) < 2:
                    raise ValueError("Need ≥2 settlements")

                # Optional extra guard: ensure global slope distribution is not flat
                s = world_t0.terrain.slope()
                if float(np.percentile(s, 95) - np.percentile(s, 50)) < 0.02:
                    raise ValueError("Terrain slope diversity too low")

                render = writer.render_and_save_t0(sample_idx=i, world_t0=world_t0, rng=rng)

                record = task.build_record(
                    sample_idx=i,
                    cfg=task_cfg,
                    world_t0=world_t0,
                    render=render,
                    rng=rng,
                )
                writer.append_jsonl(record)
                log.success(f"[S1] sample {i:05d} OK (attempt={attempt})")
                break

            except ValueError as e:
                if attempt == 79:
                    log.warning(f"[S1] sample {i:05d} FAILED after retries: {e}")
                continue

    log.info(f"[S1] wrote JSONL -> {writer.jsonl_path}")


if __name__ == "__main__":
    main()
