from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from geosynthbench.gen.exceptions import WorldGenerationFailed
from geosynthbench.pipeline.random import derive_seeds
from geosynthbench.pipeline.run import build_one_record
from geosynthbench.tasks import TASK_IDS
from geosynthbench.tasks.exceptions import TaskGenerationFailed
from geosynthbench.utils.logging import get_logger, setup_logging


@dataclass(frozen=True)
class DemoConfig:
    out_root: Path  # root folder containing data/<task>/dataset.jsonl
    base_seed: int
    per_task: tuple[int, ...]
    task_ids: tuple[str, ...]
    overwrite: bool


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    setup_logging()
    log = get_logger()

    ap = argparse.ArgumentParser(
        description="Generate reproducible demo datasets (per-task folders)."
    )
    ap.add_argument(
        "--out",
        type=str,
        default="data_demo",
        help="Output root directory (default: data_demo). Will write <out>/<task>/dataset.jsonl",
    )
    ap.add_argument(
        "--tasks",
        type=str,
        default=",".join(TASK_IDS),
        help="Comma-separated list of task ids (default: e1,d1,s1,n1,a1)",
    )
    # add an argument for number of samples per task if list make it match listof task ids
    # if int make it same for all task ids
    ap.add_argument(
        "--per-task",
        type=str,
        default="3",
        help="Samples per task id (int or comma-separated list of ints) to match task ids (default: 3)\
                        . If list, must match number of task ids. If int, same for all task ids.",
    )
    ap.add_argument("--seed", type=int, default=0, help="Base seed for reproducibility")
    ap.add_argument(
        "--overwrite",
        type=bool,
        default=False,
        help="If True, overwrite <out>/<task>/dataset.jsonl. Otherwise appends.",
    )
    args = ap.parse_args()
    task_ids: tuple[str, ...] = tuple(t.strip() for t in args.tasks.split(",") if t.strip())
    n_tasks = len(task_ids)
    per_tasks = [n.strip() for n in args.per_task.split(",") if n.strip()]
    if len(per_tasks) == 1:
        per_tasks = per_tasks * n_tasks
    assert len(per_tasks) == n_tasks
    per_tasks = tuple([int(n) for n in per_tasks])

    total = 0
    cfg = DemoConfig(
        out_root=Path(args.out),
        base_seed=int(args.seed),
        per_task=per_tasks,
        task_ids=task_ids,
        overwrite=bool(args.overwrite),
    )
    # Set up a base RNG to derive all other seeds to ensure reproducibility and avoid correlations
    # between samples (e.g. if using sample_id as seed directly, it may correlate with task_id and
    # lead to similar samples for same k across tasks)

    cfg.out_root.mkdir(parents=True, exist_ok=True)
    for i, task_id in enumerate(cfg.task_ids):
        total = 0
        task_dir = cfg.out_root / task_id.lower()
        task_dir.mkdir(parents=True, exist_ok=True)

        dataset_path = task_dir / "dataset.jsonl"
        records: list[dict[str, Any]] = []

        # If not overwriting and dataset exists, we continue sample indices after current length
        start_k = 0
        if dataset_path.exists() and not cfg.overwrite:
            # Count lines for stable continuation
            records = [json.loads(line) for line in dataset_path.open("r", encoding="utf-8")]
            start_k = len(records)

        for k in range(start_k, start_k + cfg.per_task[i]):
            world_seed, render_seed, task_seed = derive_seeds(cfg.base_seed, task_id, k)

            sample_id = f"{task_id.lower()}_{k:05d}"
            try:
                rec = build_one_record(
                    task_code=task_id,
                    sample_id=sample_id,
                    out_dir=task_dir,  # IMPORTANT: per-task directory (viewer expects relative paths to exist)
                    world_seed=world_seed,
                    render_seed=render_seed,
                    task_seed=task_seed,
                )
                # Add minimal provenance
                rec.setdefault("world_seed", world_seed)
                rec.setdefault("render_seed", render_seed)
                rec.setdefault("task_seed", task_seed)
                rec.setdefault("task_code", task_id.lower())
                rec.setdefault("sample_id", sample_id)

                records.append(rec)
                # save after each sample to ensure progress is not lost on failure and for easier inspection during generation
                write_jsonl(dataset_path, records)
                total += 1

            except WorldGenerationFailed as e:
                log.warning(f"[SKIP] sample {sample_id} failed to build_one_record: {e}")
                continue

            except TaskGenerationFailed as e:
                log.warning(f"[SKIP] sample {sample_id} failed to build_one_record: {e}")
                continue

        log.success(f"✅ {task_id}: wrote {len(records)} records -> {dataset_path}")

    log.success(f"✅ Done. Total new records: {total}")
    log.info("Tip: open the viewer with: uv run streamlit run scripts/view_dataset.py")


if __name__ == "__main__":
    main()
