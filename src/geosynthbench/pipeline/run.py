# src/geosynthbench/pipeline/run.py
from __future__ import annotations

from pathlib import Path
from typing import Any

from geosynthbench.pipeline.run_task import (
    build_a1_record,
    build_d1_record,
    build_e1_record,
    build_n1_record,
    build_s1_record,
)


def build_one_record(
    *,
    task_code: str,
    sample_id: str,
    out_dir: str | Path,
    world_seed: int,
    render_seed: int,
    task_seed: int,
) -> dict[str, Any]:
    t = task_code.lower()
    if t == "e1":
        return build_e1_record(
            sample_id=sample_id,
            out_dir=out_dir,
            world_seed=world_seed,
            render_seed=render_seed,
            task_seed=task_seed,
            max_attempts=10,
        )
    if t == "d1":
        return build_d1_record(
            sample_id=sample_id,
            out_dir=out_dir,
            world_seed=world_seed,
            render_seed=render_seed,
            task_seed=task_seed,
            max_attempts=10,
        )
    if t == "s1":
        return build_s1_record(
            sample_id=sample_id,
            out_dir=out_dir,
            world_seed=world_seed,
            render_seed=render_seed,
            task_seed=task_seed,
            max_attempts=10,
        )
    if t == "n1":
        return build_n1_record(
            sample_id=sample_id,
            out_dir=out_dir,
            world_seed=world_seed,
            render_seed=render_seed,
            task_seed=task_seed,
            max_attempts=10,
        )
    if t == "a1":
        return build_a1_record(
            sample_id=sample_id,
            out_dir=out_dir,
            world_seed=world_seed,
            render_seed=render_seed,
            task_seed=task_seed,
            max_attempts=10,
        )
    raise ValueError(f"Unknown task_code={task_code!r}")
