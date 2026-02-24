# src/geosynthbench/pipeline/core.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from geosynthbench.pipeline.writer import DatasetWriter
from geosynthbench.tasks.base import BaseTask, ChangeResult


@dataclass(frozen=True)
class PipelineConfig:
    seed: int
    n_samples: int


class DatasetPipeline:
    def __init__(self, task: BaseTask):
        self.task = task

    def run(self, cfg: PipelineConfig, task_cfg: Any, writer: DatasetWriter) -> None:
        for i in range(cfg.n_samples):
            rng = np.random.default_rng(cfg.seed + i)

            # 1) t0
            world_t0 = self.task.generate_t0(task_cfg, rng)

            # 2) optional t1
            world_t1 = None
            change_mask = None
            change_log = None
            if self.task.is_temporal:
                change_result: ChangeResult = self.task.apply_change(world_t0, task_cfg, rng)
                world_t1 = change_result.world_t1
                change_mask = change_result.change_mask
                change_log = change_result.change_log

            # 3) render + save files
            render = writer.render_and_save(
                sample_idx=i,
                world_t0=world_t0,
                world_t1=world_t1,
                change_mask=change_mask,
                cfg=task_cfg,
                rng=rng,
            )

            # 4) record
            record = self.task.build_record(
                sample_idx=i,
                task_cfg=task_cfg,
                world_t0=world_t0,
                world_t1=world_t1,
                render=render,
                change_log=change_log,
            )

            # 5) append jsonl
            writer.append_jsonl(record)
