from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from geosynthbench.pipeline.run import build_one_record
from geosynthbench.utils.logging import get_logger, setup_logging

TASK_IDS_DEFAULT = ("e1", "d1", "s1", "n1", "a1")


def _stable_seed(base_seed: int, task_id: str, k: int) -> int:
    """
    Deterministic seed derivation so anyone reproduces the same demo.
    Avoid python's built-in hash() (salted per process).
    """
    acc = base_seed & 0xFFFFFFFF
    for ch in task_id.encode("utf-8"):
        acc = (acc * 16777619) ^ ch
        acc &= 0xFFFFFFFF
    acc = (acc * 16777619) ^ (k & 0xFFFFFFFF)
    acc &= 0xFFFFFFFF
    return int(acc)


@dataclass(frozen=True)
class DemoConfig:
    out_root: Path  # root folder containing data/<task>/dataset.jsonl
    base_seed: int
    per_task: int
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
        default=",".join(TASK_IDS_DEFAULT),
        help="Comma-separated list of task ids (default: e1,d1,s1,n1,a1)",
    )
    ap.add_argument("--per-task", type=int, default=3, help="Samples per task id")
    ap.add_argument("--seed", type=int, default=12345, help="Base seed for reproducibility")
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, overwrite <out>/<task>/dataset.jsonl. Otherwise appends.",
    )
    args = ap.parse_args()

    cfg = DemoConfig(
        out_root=Path(args.out),
        base_seed=int(args.seed),
        per_task=int(args.per_task),
        task_ids=tuple(t.strip() for t in args.tasks.split(",") if t.strip()),
        overwrite=bool(args.overwrite),
    )

    cfg.out_root.mkdir(parents=True, exist_ok=True)

    total = 0

    for task_id in cfg.task_ids:
        task_dir = cfg.out_root / task_id.lower()
        task_dir.mkdir(parents=True, exist_ok=True)

        dataset_path = task_dir / "dataset.jsonl"
        records: list[dict[str, Any]] = []

        # If not overwriting and dataset exists, we continue sample indices after current length
        start_k = 0
        if dataset_path.exists() and not cfg.overwrite:
            # Count lines for stable continuation
            start_k = sum(1 for _ in dataset_path.open("r", encoding="utf-8"))

        for k in range(start_k, start_k + cfg.per_task):
            seed = _stable_seed(cfg.base_seed, task_id, k)
            sample_id = f"{task_id.lower()}_{k:03d}"

            rec = build_one_record(
                task_code=task_id,
                sample_id=sample_id,
                out_dir=task_dir,  # IMPORTANT: per-task directory (viewer expects relative paths to exist)
                seed=seed,
            )

            # Add minimal provenance
            rec.setdefault("seed", seed)
            rec.setdefault("task_code", task_id.lower())
            rec.setdefault("sample_id", sample_id)

            records.append(rec)
            total += 1

        # Write dataset.jsonl
        if cfg.overwrite:
            write_jsonl(dataset_path, records)
        else:
            # append
            dataset_path.parent.mkdir(parents=True, exist_ok=True)
            with dataset_path.open("a", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

        log.success(f"✅ {task_id}: wrote {len(records)} records -> {dataset_path}")

    log.success(f"✅ Done. Total new records: {total}")
    log.info("Tip: open the viewer with: uv run streamlit run scripts/view_dataset.py")


if __name__ == "__main__":
    main()
