# scripts/generate_d1.py
from __future__ import annotations

import numpy as np

from geosynthbench.gen.config import WorldGenConfig
from geosynthbench.pipeline.writer import DatasetWriter
from geosynthbench.tasks.d1_distance_to_water import D1Config, D1DistanceToWaterTask
from geosynthbench.utils.logging import get_logger, setup_logging


def make_d1_world_cfg(seed: int) -> WorldGenConfig:
    # Key constraints: >=2 lakes, >=2 settlements
    return WorldGenConfig(
        seed=seed,
        # terrain (moderate)
        terrain_amplitude_m=220.0,
        terrain_n_hills=(2, 4),
        terrain_hill_sigma_m=(280.0, 650.0),
        terrain_noise_scale_m=350.0,
        terrain_noise_strength_m=5.0,
        # enforce multiple lakes
        n_water=(2, 4),
        # vegetation ok
        n_veg=(3, 7),
        # need two settlements for A/B
        n_settlements=2,
        # roads optional
    )


def main() -> None:
    setup_logging()
    log = get_logger()

    out_dir = "data/D1"
    writer = DatasetWriter.create(out_dir)

    task = D1DistanceToWaterTask()

    n_samples = 5
    base_seed = 10_000

    for i in range(n_samples):
        rng = np.random.default_rng(base_seed + i)
        cfg0 = make_d1_world_cfg(seed=0)  # seed gets overridden in task.generate_t0 via rng

        task_cfg = D1Config(
            world_cfg=cfg0,
            min_delta_m=20.0,  # avoid ambiguous comparisons
            strategy="first_two",
        )

        for attempt in range(60):
            try:
                world_t0 = task.generate_t0(task_cfg, rng)

                # Hard requirements
                if len(world_t0.settlements) < 2:
                    raise ValueError("Need ≥2 settlements")
                if len(getattr(world_t0, "water", [])) < 2:
                    raise ValueError("Need ≥2 lakes")

                render = writer.render_and_save_t0(sample_idx=i, world_t0=world_t0, rng=rng)

                record = task.build_record(
                    sample_idx=i,
                    cfg=task_cfg,
                    world_t0=world_t0,
                    render=render,
                    rng=rng,
                )
                writer.append_jsonl(record)
                log.success(f"[D1] sample {i:05d} OK (attempt={attempt})")
                break

            except ValueError as e:
                if attempt == 59:
                    log.warning(f"[D1] sample {i:05d} FAILED after retries: {e}")
                continue

    log.info(f"[D1] wrote JSONL -> {writer.jsonl_path}")


if __name__ == "__main__":
    main()
