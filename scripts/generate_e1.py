from __future__ import annotations

import time

import numpy as np

from geosynthbench.gen.config import WorldGenConfig
from geosynthbench.paths import DATA_DIR
from geosynthbench.pipeline.writer import DatasetWriter
from geosynthbench.tasks.e1_elevation import E1Config, E1ElevationCompareTask
from geosynthbench.utils.logging import get_logger, setup_logging


def main() -> None:
    setup_logging()
    log = get_logger()
    log.info("Starting dataset generation...")
    writer = DatasetWriter.create(DATA_DIR / "E1")
    t0 = time.time()

    # Your config (kept from your snippet)
    t0_cfg = WorldGenConfig(
        seed=0,
        terrain_amplitude_m=300.0,
        terrain_n_hills=(3, 5),
        terrain_hill_sigma_m=(250.0, 500.0),
        terrain_noise_scale_m=300.0,
        terrain_noise_strength_m=6.0,
        n_water=(1, 3),
        n_veg=(3, 6),
        n_settlements=(2, 5),  # at least 2 settlements to have landmarks
    )

    task = E1ElevationCompareTask()
    task_cfg = E1Config(world_cfg=t0_cfg, min_delta_m=5.0, settlement_strategy="first_two")

    n_samples = 5
    base_seed = 0

    for i in range(n_samples):
        rng = np.random.default_rng(base_seed + i)

        # keep trying until non-degenerate (cheap retry loop)
        for attempt in range(50):
            try:
                world_t0 = task.generate_t0(task_cfg, rng)
                render = writer.render_and_save_t0(sample_idx=i, world_t0=world_t0, rng=rng)
                record = task.build_record(
                    sample_idx=i, cfg=task_cfg, world_t0=world_t0, render=render, rng=rng
                )
                writer.append_jsonl(record)
                log.info(
                    f"Done {i+1}/{n_samples} ({(i+1)/n_samples:.0%}) | elapsed={time.time()-t0:.1f}s"
                )
                break
            except ValueError:
                # degenerate sample -> retry with same rng (it will advance internally)
                continue
        if attempt == 49:
            log.warning(f"Attempt {attempt+1}/50 failed for sample {i:05d}. Skipping this sample.")

    log.info(f"[DONE] dataset.jsonl -> {writer.jsonl_path}")


if __name__ == "__main__":
    main()
