# scripts/inspect_jsonl.py
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from geosynthbench.utils.logging import get_logger, setup_logging


def _read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _get(d: dict[str, Any], keys: list[str], default=None):
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _fmt(x: Any) -> str:
    if x is None:
        return "-"
    if isinstance(x, (int, float, str)):
        s = str(x)
        return s if len(s) <= 80 else s[:77] + "..."
    s = json.dumps(x, ensure_ascii=False)
    return s if len(s) <= 80 else s[:77] + "..."


def _print_table(rows: list[list[str]], headers: list[str]) -> None:
    # simple fixed-width table without dependencies
    cols = list(zip(*([headers] + rows)))
    widths = [min(60, max(len(c) for c in col)) for col in cols]

    def line(parts):
        return " | ".join(p[:w].ljust(w) for p, w in zip(parts, widths))

    print(line(headers))
    print("-+-".join("-" * w for w in widths))
    for r in rows:
        print(line(r))


def main() -> None:
    setup_logging()
    log = get_logger()

    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=30)
    args = ap.parse_args()

    recs = _read_jsonl(args.jsonl, limit=args.limit)
    if not recs:
        log.warning("No records found.")
        return

    # summarize
    task_codes = Counter(
        [str(r.get("task_code", r.get("task", {}).get("type", "UNK"))) for r in recs]
    )
    modalities = Counter([str(r.get("modality", "UNK")) for r in recs])

    log.info(f"Loaded {len(recs)} record(s) from {args.jsonl}")
    log.info(f"Task codes (top): {task_codes.most_common(10)}")
    log.info(f"Modalities: {modalities}")

    # display rows
    rows: list[list[str]] = []
    for r in recs:
        sid = str(r.get("sample_id", r.get("world_id", "UNK")))
        code = str(r.get("task_code", r.get("task", {}).get("type", "UNK")))
        mod = str(r.get("modality", "UNK"))
        prompt = _get(r, ["prompt"], _get(r, ["task", "question"], ""))
        ans = _get(r, ["answer"], _get(r, ["task", "answer"], ""))
        img = _get(r, ["inputs", "image"], _get(r, ["rgb_path"], _get(r, ["paths", "t0_rgb"], "")))
        rows.append([sid, code, mod, _fmt(img), _fmt(prompt), _fmt(ans)])

    _print_table(rows, headers=["sample_id", "task", "modality", "image", "prompt", "answer"])


if __name__ == "__main__":
    main()
