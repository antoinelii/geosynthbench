# src/geosynthbench/pipeline/run_task.py
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from geosynthbench.gen.config import WorldGenConfig
from geosynthbench.pipeline.writer import DatasetWriter
from geosynthbench.tasks.d1_distance_to_water import D1Config, D1DistanceToWaterTask
from geosynthbench.tasks.e1_elevation import E1Config, E1ElevationCompareTask
from geosynthbench.tasks.n1_isolation import (
    N1Config,
    N1IsolationTask,
    compute_isolation_scores,
    isolation_is_clear,
)
from geosynthbench.tasks.s1_slope_compare import S1Config, S1SlopeCompareTask
from geosynthbench.utils.logging import get_logger


def _parse_sample_idx(sample_id: str, default: int = 0) -> int:
    m = re.search(r"(\d+)$", sample_id)
    return int(m.group(1)) if m else default


def _normalize_record_for_viewer(
    record: dict[str, Any],
    *,
    task_code: str,
    sample_id: str,
) -> dict[str, Any]:
    """
    Make sure scripts/view_dataset.py can always display the record.
    It only fills missing keys; it does not rewrite your schema.
    """
    record.setdefault("sample_id", sample_id)
    record.setdefault("task_code", task_code)

    inputs = record.get("inputs")
    if not isinstance(inputs, dict):
        inputs = {}
        record["inputs"] = inputs

    if "modality" not in record:
        if any(k in inputs for k in ("t0_image", "t1_image", "change_mask")):
            record["modality"] = "pair"
        else:
            record["modality"] = "single"

    record.setdefault("prompt", "")
    record.setdefault("answer", None)
    record.setdefault("oracle", None)
    return record


# -------------------------
# Task-specific world cfgs
# -------------------------


def make_d1_world_cfg(seed: int) -> WorldGenConfig:
    # Mirrors scripts/generate_d1.py :contentReference[oaicite:4]{index=4}
    return WorldGenConfig(
        seed=seed,
        terrain_amplitude_m=220.0,
        terrain_n_hills=(2, 4),
        terrain_hill_sigma_m=(280.0, 650.0),
        terrain_noise_scale_m=350.0,
        terrain_noise_strength_m=5.0,
        n_water=(2, 4),
        n_veg=(3, 7),
        n_settlements=2,
    )


def make_s1_world_cfg(seed: int) -> WorldGenConfig:
    # Mirrors scripts/generate_s1.py :contentReference[oaicite:5]{index=5}
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


def make_n1_world_cfg(seed: int) -> WorldGenConfig:
    # Mirrors scripts/generate_n1.py :contentReference[oaicite:6]{index=6}
    return WorldGenConfig(
        seed=seed,
        terrain_amplitude_m=25.0,
        terrain_n_hills=(1, 3),
        terrain_hill_sigma_m=(450.0, 1100.0),
        terrain_noise_scale_m=450.0,
        terrain_noise_strength_m=3.5,
        n_water=(0, 1),
        n_veg=(2, 5),
        n_settlements=(4, 6),
        settlement_radius_m=(180.0, 320.0),
        min_dist_settlements_m=520.0,
        max_slope_settlement=0.30,
        roads_mode="mst",
        extra_edges=0,
        road_width_m=8.0,
        max_slope_road=0.18,
        buildings_per_settlement=(4, 10),
        building_size_m=(12.0, 24.0),
        min_dist_buildings_m=5.0,
        max_slope_building=0.25,
        max_building_attempts=800,
        require_connected_roads=True,
        forbid_roads_in_water=True,
        prefer_settlement_near_water=False,
        prefer_water_distance_m=(60.0, 350.0),
    )


# -------------------------
# Single-record builders
# -------------------------


def build_e1_record(
    *,
    sample_id: str,
    out_dir: str | Path,
    seed: int,
    max_attempts: int = 50,
) -> dict[str, Any]:
    """
    Single-record version of scripts/generate_e1.py :contentReference[oaicite:7]{index=7}
    """
    log = get_logger()
    out_path = Path(out_dir)
    writer = DatasetWriter.create(out_path)

    sample_idx = _parse_sample_idx(sample_id)
    rng = np.random.default_rng(seed)

    # Same config as your E1 script :contentReference[oaicite:8]{index=8}
    t0_cfg = WorldGenConfig(
        seed=0,
        terrain_amplitude_m=300.0,
        terrain_n_hills=(3, 5),
        terrain_hill_sigma_m=(250.0, 500.0),
        terrain_noise_scale_m=300.0,
        terrain_noise_strength_m=6.0,
        n_water=(1, 3),
        n_veg=(3, 6),
        n_settlements=(2, 5),
    )

    task = E1ElevationCompareTask()
    task_cfg = E1Config(world_cfg=t0_cfg, min_delta_m=5.0, settlement_strategy="first_two")

    last_err: Exception | None = None
    for attempt in range(max_attempts):
        try:
            world_t0 = task.generate_t0(task_cfg, rng)
            render = writer.render_and_save_t0(sample_idx=sample_idx, world_t0=world_t0, rng=rng)
            record = task.build_record(
                sample_idx=sample_idx, cfg=task_cfg, world_t0=world_t0, render=render, rng=rng
            )
            record = _normalize_record_for_viewer(record, task_code="e1", sample_id=sample_id)
            log.success(f"[E1] built {sample_id} OK (attempt={attempt})")
            return record
        except ValueError as e:
            last_err = e
            continue

    raise RuntimeError(
        f"[E1] FAILED to build {sample_id} after {max_attempts} attempts: {last_err}"
    )


def build_d1_record(
    *,
    sample_id: str,
    out_dir: str | Path,
    seed: int,
    max_attempts: int = 60,
) -> dict[str, Any]:
    """
    Single-record version of scripts/generate_d1.py :contentReference[oaicite:9]{index=9}
    """
    log = get_logger()
    out_path = Path(out_dir)
    writer = DatasetWriter.create(out_path)

    sample_idx = _parse_sample_idx(sample_id)
    rng = np.random.default_rng(seed)

    task = D1DistanceToWaterTask()
    cfg0 = make_d1_world_cfg(
        seed=0
    )  # seed overridden via rng in task.generate_t0 :contentReference[oaicite:10]{index=10}
    task_cfg = D1Config(world_cfg=cfg0, min_delta_m=20.0, strategy="first_two")

    last_err: Exception | None = None
    for attempt in range(max_attempts):
        try:
            world_t0 = task.generate_t0(task_cfg, rng)

            # Hard requirements (same as script) :contentReference[oaicite:11]{index=11}
            if len(world_t0.settlements) < 2:
                raise ValueError("Need ≥2 settlements")
            if len(getattr(world_t0, "water", [])) < 2:
                raise ValueError("Need ≥2 lakes")

            render = writer.render_and_save_t0(sample_idx=sample_idx, world_t0=world_t0, rng=rng)
            record = task.build_record(
                sample_idx=sample_idx,
                cfg=task_cfg,
                world_t0=world_t0,
                render=render,
                rng=rng,
            )
            record = _normalize_record_for_viewer(record, task_code="d1", sample_id=sample_id)
            log.success(f"[D1] built {sample_id} OK (attempt={attempt})")
            return record

        except ValueError as e:
            last_err = e
            continue

    raise RuntimeError(
        f"[D1] FAILED to build {sample_id} after {max_attempts} attempts: {last_err}"
    )


def build_s1_record(
    *,
    sample_id: str,
    out_dir: str | Path,
    seed: int,
    max_attempts: int = 80,
) -> dict[str, Any]:
    """
    Single-record version of scripts/generate_s1.py :contentReference[oaicite:12]{index=12}
    """
    log = get_logger()
    out_path = Path(out_dir)
    writer = DatasetWriter.create(out_path)

    sample_idx = _parse_sample_idx(sample_id)
    rng = np.random.default_rng(seed)

    task = S1SlopeCompareTask()
    cfg0 = make_s1_world_cfg(
        seed=0
    )  # seed overridden via rng in task.generate_t0 :contentReference[oaicite:13]{index=13}
    task_cfg = S1Config(world_cfg=cfg0, min_delta=0.02)

    last_err: Exception | None = None
    for attempt in range(max_attempts):
        try:
            world_t0 = task.generate_t0(task_cfg, rng)

            if len(world_t0.settlements) < 2:
                raise ValueError("Need ≥2 settlements")

            # slope diversity guard (same as script) :contentReference[oaicite:14]{index=14}
            s = world_t0.terrain.slope()
            if float(np.percentile(s, 95) - np.percentile(s, 50)) < 0.02:
                raise ValueError("Terrain slope diversity too low")

            render = writer.render_and_save_t0(sample_idx=sample_idx, world_t0=world_t0, rng=rng)
            record = task.build_record(
                sample_idx=sample_idx,
                cfg=task_cfg,
                world_t0=world_t0,
                render=render,
                rng=rng,
            )
            record = _normalize_record_for_viewer(record, task_code="s1", sample_id=sample_id)
            log.success(f"[S1] built {sample_id} OK (attempt={attempt})")
            return record

        except ValueError as e:
            last_err = e
            continue

    raise RuntimeError(
        f"[S1] FAILED to build {sample_id} after {max_attempts} attempts: {last_err}"
    )


def build_n1_record(
    *,
    sample_id: str,
    out_dir: str | Path,
    seed: int,
    max_attempts: int = 120,
) -> dict[str, Any]:
    """
    Single-record version of scripts/generate_n1.py :contentReference[oaicite:15]{index=15}
    """
    log = get_logger()
    out_path = Path(out_dir)
    writer = DatasetWriter.create(out_path)

    sample_idx = _parse_sample_idx(sample_id)
    rng = np.random.default_rng(seed)

    task = N1IsolationTask()
    cfg0 = make_n1_world_cfg(
        seed=0
    )  # seed mutated inside task.generate_t0 :contentReference[oaicite:16]{index=16}
    task_cfg = N1Config(world_cfg=cfg0, clarity_ratio=1.10, clarity_delta_m=200.0, strategy="by_id")

    last_err: Exception | None = None
    for attempt in range(max_attempts):
        try:
            try:
                world_t0 = task.generate_t0(task_cfg, rng)
            except RuntimeError as e:
                last_err = e
                continue

            # Hard constraints + clarity (same as script) :contentReference[oaicite:17]{index=17}
            n_sett = len(world_t0.settlements)
            if n_sett < 4 or n_sett > 6:
                raise ValueError(f"need 4-6 settlements, got {n_sett}")

            scores = compute_isolation_scores(world_t0)
            if not isolation_is_clear(
                scores, ratio=task_cfg.clarity_ratio, delta_m=task_cfg.clarity_delta_m
            ):
                raise ValueError("isolation not clear enough")

            render = writer.render_and_save_t0(sample_idx=sample_idx, world_t0=world_t0, rng=rng)
            record = task.build_record(
                sample_idx=sample_idx,
                cfg=task_cfg,
                world_t0=world_t0,
                render=render,
                rng=rng,
            )
            record = _normalize_record_for_viewer(record, task_code="n1", sample_id=sample_id)
            log.success(
                f"[N1] built {sample_id} OK (attempt={attempt}) | best={record.get('answer')}"
            )
            return record

        except ValueError as e:
            last_err = e
            continue

    raise RuntimeError(
        f"[N1] FAILED to build {sample_id} after {max_attempts} attempts: {last_err}"
    )
