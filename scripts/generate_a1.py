from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from geosynthbench.pipeline.run_task import build_a1_record
from geosynthbench.utils.logging import get_logger, setup_logging


def main() -> None:
    setup_logging()
    log = get_logger()

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="data/A1", help="output directory")
    ap.add_argument("--n", type=int, default=5, help="number of samples")
    ap.add_argument("--seed", type=int, default=0, help="master seed")
    ap.add_argument("--max-attempts", type=int, default=120, help="max attempts per sample")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "dataset.jsonl"

    rng = np.random.default_rng(args.seed)

    n_ok = 0
    with jsonl_path.open("w", encoding="utf-8") as f:
        for i in range(args.n):
            sample_id = f"a1_{i:05d}"
            seed_i = int(rng.integers(0, 2**31 - 1))
            record: dict[str, Any] = build_a1_record(
                sample_id=sample_id,
                out_dir=out_dir,
                seed=seed_i,
                max_attempts=args.max_attempts,
            )
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_ok += 1
            if (i + 1) % 10 == 0:
                log.info(f"[A1] progress {i+1}/{args.n} (written={n_ok})")

    log.success(f"[A1] DONE. Wrote {n_ok} records to {jsonl_path}")


if __name__ == "__main__":
    main()
