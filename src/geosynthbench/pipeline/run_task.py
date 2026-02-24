# src/geosynthbench/pipeline/run_task.py
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from geosynthbench.gen.config import WorldGenConfig
from geosynthbench.pipeline.types import RenderArtifacts
from geosynthbench.pipeline.writer import DatasetWriter
from geosynthbench.tasks.a1_road_plus_building import A1Config, A1RoadPlusBuildingTask
from geosynthbench.tasks.d1_distance_to_water import D1Config, D1DistanceToWaterTask
from geosynthbench.tasks.e1_elevation import E1Config, E1ElevationCompareTask
from geosynthbench.tasks.exceptions import TaskGenerationFailed
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


def make_e1_world_cfg(seed: int) -> WorldGenConfig:
    return WorldGenConfig(
        seed=seed,
        terrain_amplitude_m=300.0,
        terrain_n_hills=(3, 5),
        terrain_hill_sigma_m=(250.0, 500.0),
        terrain_noise_scale_m=300.0,
        terrain_noise_strength_m=6.0,
        n_water=(1, 3),
        n_veg=(3, 6),
        n_settlements=(2, 5),
    )


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


def make_a1_world_cfg(seed: int) -> WorldGenConfig:
    # Favourable temporal config: enough settlements + connected roads + some existing buildings,
    # but not too dense (so counting new ones is doable).
    return WorldGenConfig(
        seed=seed,
        terrain_amplitude_m=40.0,
        terrain_n_hills=(1, 3),
        terrain_hill_sigma_m=(450.0, 1100.0),
        terrain_noise_scale_m=520.0,
        terrain_noise_strength_m=2.5,
        n_water=(0, 1),
        n_veg=(2, 5),
        n_settlements=(3, 5),
        settlement_radius_m=(180.0, 320.0),
        min_dist_settlements_m=520.0,
        max_slope_settlement=0.30,
        roads_mode="mst",
        extra_edges=0,
        road_width_m=8.0,
        max_slope_road=0.18,
        buildings_per_settlement=(2, 5),
        building_size_m=(12.0, 22.0),
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
    world_seed: int,
    render_seed: int,
    task_seed: int,
    max_attempts: int = 50,
) -> dict[str, Any]:
    """
    Single-record version of scripts/generate_e1.py :contentReference[oaicite:7]{index=7}
    """
    log = get_logger()
    out_path = Path(out_dir)
    writer = DatasetWriter.create(out_path)

    sample_idx = _parse_sample_idx(sample_id)

    #### CONFIGURATION ####
    # world
    cfg0 = make_e1_world_cfg(seed=world_seed)
    # render
    render_rng = np.random.default_rng(render_seed)
    # task
    task = E1ElevationCompareTask()
    task_cfg = E1Config(world_cfg=cfg0, min_delta_m=5.0, settlement_strategy="first_two")
    task_rng = np.random.default_rng(task_seed)

    last_err: Exception | None = None
    for attempt in range(max_attempts):
        try:
            world_t0 = task.generate_t0(task_cfg)
            render = writer.render_and_save_t0(
                sample_idx=sample_idx, world_t0=world_t0, rng=render_rng
            )
            record = task.build_record(
                sample_idx=sample_idx,
                task_cfg=task_cfg,
                world_t0=world_t0,
                render=render,
                rng=task_rng,
            )
            record = _normalize_record_for_viewer(record, task_code="e1", sample_id=sample_id)
            log.success(f"[E1] built {sample_id} OK (attempt={attempt})")  # type: ignore
            return record
        except ValueError as e:
            last_err = e
            continue

    raise TaskGenerationFailed(
        f"[E1] FAILED to build {sample_id} after {max_attempts} attempts: {last_err}"
    )


def build_d1_record(
    *,
    sample_id: str,
    out_dir: str | Path,
    world_seed: int,
    render_seed: int,
    task_seed: int,
    max_attempts: int = 60,
) -> dict[str, Any]:
    """
    Single-record version of scripts/generate_d1.py :contentReference[oaicite:9]{index=9}
    """
    log = get_logger()
    out_path = Path(out_dir)
    writer = DatasetWriter.create(out_path)

    sample_idx = _parse_sample_idx(sample_id)

    cfg0 = make_d1_world_cfg(seed=world_seed)
    render_rng = np.random.default_rng(render_seed)
    task_rng = np.random.default_rng(task_seed)

    task = D1DistanceToWaterTask()
    cfg0 = make_d1_world_cfg(
        seed=0
    )  # seed overridden via rng in task.generate_t0 :contentReference[oaicite:10]{index=10}
    task_cfg = D1Config(world_cfg=cfg0, min_delta_m=20.0, strategy="first_two")

    last_err: Exception | None = None
    for attempt in range(max_attempts):
        try:
            world_t0 = task.generate_t0(task_cfg)

            # Hard requirements (same as script) :contentReference[oaicite:11]{index=11}
            if len(world_t0.settlements) < 2:
                raise ValueError("Need ≥2 settlements")
            if len(getattr(world_t0, "water", [])) < 2:
                raise ValueError("Need ≥2 lakes")

            render = writer.render_and_save_t0(
                sample_idx=sample_idx, world_t0=world_t0, rng=render_rng
            )
            record = task.build_record(
                sample_idx=sample_idx,
                task_cfg=task_cfg,
                world_t0=world_t0,
                render=render,
                rng=task_rng,
            )
            record = _normalize_record_for_viewer(record, task_code="d1", sample_id=sample_id)
            log.success(f"[D1] built {sample_id} OK (attempt={attempt})")  # type: ignore
            return record

        except ValueError as e:
            last_err = e
            continue

    raise TaskGenerationFailed(
        f"[D1] FAILED to build {sample_id} after {max_attempts} attempts: {last_err}"
    )


def build_s1_record(
    *,
    sample_id: str,
    out_dir: str | Path,
    world_seed: int,
    render_seed: int,
    task_seed: int,
    max_attempts: int = 80,
) -> dict[str, Any]:
    """
    Single-record version of scripts/generate_s1.py :contentReference[oaicite:12]{index=12}
    """
    log = get_logger()
    out_path = Path(out_dir)
    writer = DatasetWriter.create(out_path)

    sample_idx = _parse_sample_idx(sample_id)

    task = S1SlopeCompareTask()
    cfg0 = make_s1_world_cfg(seed=world_seed)
    task_cfg = S1Config(world_cfg=cfg0, min_delta=0.02)
    render_rng = np.random.default_rng(render_seed)
    task_rng = np.random.default_rng(task_seed)

    last_err: Exception | None = None
    for attempt in range(max_attempts):
        try:
            world_t0 = task.generate_t0(task_cfg)

            if len(world_t0.settlements) < 2:
                raise ValueError("Need ≥2 settlements")

            # slope diversity guard (same as script) :contentReference[oaicite:14]{index=14}
            if world_t0.terrain is None:
                raise ValueError("Terrain is required for slope-based task")
            s = world_t0.terrain.slope()
            if float(np.percentile(s, 95) - np.percentile(s, 50)) < 0.02:
                raise ValueError("Terrain slope diversity too low")

            render = writer.render_and_save_t0(
                sample_idx=sample_idx, world_t0=world_t0, rng=render_rng
            )
            record = task.build_record(
                sample_idx=sample_idx,
                task_cfg=task_cfg,
                world_t0=world_t0,
                render=render,
                rng=task_rng,
            )
            record = _normalize_record_for_viewer(record, task_code="s1", sample_id=sample_id)
            log.success(f"[S1] built {sample_id} OK (attempt={attempt})")  # type: ignore
            return record

        except ValueError as e:
            last_err = e
            continue

    raise TaskGenerationFailed(
        f"[S1] FAILED to build {sample_id} after {max_attempts} attempts: {last_err}"
    )


def build_n1_record(
    *,
    sample_id: str,
    out_dir: str | Path,
    world_seed: int,
    render_seed: int,
    task_seed: int,
    max_attempts: int = 120,
) -> dict[str, Any]:
    """
    Single-record version of scripts/generate_n1.py :contentReference[oaicite:15]{index=15}
    """
    log = get_logger()
    out_path = Path(out_dir)
    writer = DatasetWriter.create(out_path)

    sample_idx = _parse_sample_idx(sample_id)

    task = N1IsolationTask()
    cfg0 = make_n1_world_cfg(
        seed=world_seed
    )  # seed mutated inside task.generate_t0 :contentReference[oaicite:16]{index=16}
    task_cfg = N1Config(world_cfg=cfg0, clarity_ratio=1.10, clarity_delta_m=200.0, strategy="by_id")
    render_rng = np.random.default_rng(render_seed)
    task_rng = np.random.default_rng(task_seed)

    last_err: Exception | None = None
    for attempt in range(max_attempts):
        try:
            try:
                world_t0 = task.generate_t0(task_cfg)
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

            render = writer.render_and_save_t0(
                sample_idx=sample_idx, world_t0=world_t0, rng=render_rng
            )
            record = task.build_record(
                sample_idx=sample_idx,
                task_cfg=task_cfg,
                world_t0=world_t0,
                render=render,
                rng=task_rng,
            )
            record = _normalize_record_for_viewer(record, task_code="n1", sample_id=sample_id)
            log.success(  # type: ignore
                f"[N1] built {sample_id} OK (attempt={attempt}) | best={record.get('answer')}"
            )
            return record

        except ValueError as e:
            last_err = e
            continue
    raise TaskGenerationFailed(
        f"[N1] FAILED to build {sample_id} after {max_attempts} attempts: {last_err}"
    )


def build_a1_record(
    *,
    sample_id: str,
    out_dir: str | Path,
    world_seed: int,
    render_seed: int,
    task_seed: int,
    max_attempts: int = 120,
) -> dict[str, Any]:
    log = get_logger()
    out_path = Path(out_dir)
    writer = DatasetWriter.create(out_path)

    sample_idx = _parse_sample_idx(sample_id)

    task = A1RoadPlusBuildingTask()
    cfg0 = make_a1_world_cfg(seed=world_seed)
    task_cfg = A1Config(world_cfg=cfg0)
    render_rng = np.random.default_rng(render_seed)
    task_rng = np.random.default_rng(task_seed)

    last_err: Exception | None = None
    for attempt in range(max_attempts):
        try:
            world_t0 = task.generate_t0(task_cfg)

            # Build t1 inside task, then render pair
            tmp = task.build_record(
                sample_idx=sample_idx,
                task_cfg=task_cfg,
                world_t0=world_t0,
                render=RenderArtifacts(  # will be overwritten below (we just need a placeholder type)
                    sample_dir="",
                    t0_rgb="",
                    t0_mask=None,
                    t0_elev=None,
                    t1_rgb=None,
                    t1_mask=None,
                    change_mask=None,
                ),
                rng=task_rng,
            )
            world_t1 = tmp.get("_debug_world_t1")
            if world_t1 is None:
                raise ValueError("A1 internal error: missing world_t1")

            # Render pair robustly: use a dedicated method if present, else fallback
            render = None
            if hasattr(writer, "render_and_save_pair"):
                render = writer.render_and_save_pair(
                    sample_idx=sample_idx, world_t0=world_t0, world_t1=world_t1, rng=render_rng
                )
            else:
                # minimal fallback (depends on your writer implementation)
                r0 = writer.render_and_save_t0(
                    sample_idx=sample_idx, world_t0=world_t0, rng=render_rng
                )
                if hasattr(writer, "render_and_save_t1"):
                    # reset render_rng to ensure t1 rendering is done in the
                    # same way than t0, important to control differences
                    render_rng = np.random.default_rng(render_seed)
                    r1 = writer.render_and_save_t1(
                        sample_idx=sample_idx, world_t1=world_t1, rng=render_rng
                    )
                    # merge the two objects (duck-typing)
                    for k, v in r1.__dict__.items():
                        setattr(r0, k, v)
                    render = r0
                else:
                    raise ValueError(
                        "DatasetWriter missing render_and_save_pair/render_and_save_t1"
                    )

            # Rebuild final record with real render artifacts
            record = task.build_record(
                sample_idx=sample_idx,
                task_cfg=task_cfg,
                world_t0=world_t0,
                render=render,
                rng=task_rng,
            )
            record.pop("_debug_world_t1", None)

            record = _normalize_record_for_viewer(record, task_code="a1", sample_id=sample_id)
            log.success(  # type: ignore
                f"[A1] built {sample_id} OK (attempt={attempt}) | answer={record.get('answer')}"
            )
            return record

        except ValueError as e:
            last_err = e
            continue

    raise TaskGenerationFailed(
        f"[A1] FAILED to build {sample_id} after {max_attempts} attempts: {last_err}"
    )
