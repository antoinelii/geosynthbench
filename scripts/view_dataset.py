# scripts/view_dataset.py
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
from PIL import Image

from geosynthbench.paths import ROOT_PROJECT_DIR
from geosynthbench.render.semantic import semantic_mask_to_rgb
from geosynthbench.utils.logging import get_logger, setup_logging


def _scan_datasets(data_root: Path = Path("data_demo")) -> dict[str, Path]:
    """
    Finds <data_root>/<NAME>/dataset.jsonl
    Returns mapping: dataset_name -> jsonl_path
    """
    root = Path(data_root)
    out: dict[str, Path] = {}
    if not root.exists():
        return out
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        p = d / "dataset.jsonl"
        if p.exists():
            out[d.name] = p
    return out


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def get_absolute_path(p: Path) -> Path:
    """
    If p is absolute, return as Path.
    If p is relative, interpret as relative to ROOT_PROJECT_DIR and return absolute Path.
    """
    path = Path(p)
    if path.is_absolute():
        return path
    else:
        return ROOT_PROJECT_DIR / path


def load_img(path: str | None) -> Image.Image | None:
    if not path:
        return None
    p = get_absolute_path(Path(path))
    if not p.exists():
        return None
    return Image.open(str(p))


def load_mask_as_rgb(path: str | None) -> Image.Image | None:
    if not path:
        return None
    p = get_absolute_path(Path(path))
    if not p.exists():
        return None
    m = np.array(Image.open(str(p)))
    if m.ndim == 3:
        return Image.fromarray(m)
    return Image.fromarray(semantic_mask_to_rgb(m))


def _safe_text(x: Any, max_len: int = 120) -> str:
    if x is None:
        return ""
    if isinstance(x, (dict, list)):
        s = json.dumps(x, ensure_ascii=False)
    else:
        s = str(x)
    return s if len(s) <= max_len else s[: max_len - 3] + "..."


def _answer_block(ans: Any) -> None:
    if isinstance(ans, (dict, list)):
        st.json(ans)
    else:
        st.code("" if ans is None else str(ans), language="text")


def _dataset_summary(recs: list[dict[str, Any]], max_preview: int = 30) -> None:
    # Counts
    task_codes = Counter([str(r.get("task_code", "UNK")) for r in recs])
    modalities = Counter([str(r.get("modality", "UNK")) for r in recs])

    c1, c2, c3 = st.columns(3)
    c1.metric("Records", len(recs))
    c2.metric("Task codes", len(task_codes))
    c3.metric("Modalities", len(modalities))

    st.markdown("#### Task code distribution")
    st.write(dict(task_codes.most_common()))

    st.markdown("#### Modality distribution")
    st.write(dict(modalities.most_common()))

    # Preview table (safe, compact)
    rows = []
    for r in recs[:max_preview]:
        sid = str(r.get("sample_id", "UNK"))
        code = str(r.get("task_code", "UNK"))
        mod = str(r.get("modality", "UNK"))

        inputs = r.get("inputs", {}) or {}
        img = inputs.get("image") or inputs.get("t0_image") or ""

        rows.append(
            {
                "sample_id": sid,
                "task_code": code,
                "modality": mod,
                "image": _safe_text(img, 80),
                "answer": _safe_text(r.get("answer"), 60),
            }
        )

    st.markdown("#### Preview (first records)")
    st.dataframe(rows, width=True)


def main() -> None:
    root_data_dir_str = st.sidebar.text_input("Root data dir", value="data_demo")
    root_data_dir = Path(root_data_dir_str)

    setup_logging()
    log = get_logger()

    st.set_page_config(page_title="GeoSynthBench Dataset Viewer", layout="wide")
    st.title("GeoSynthBench Dataset Viewer")

    # --- dataset selection
    datasets = _scan_datasets(root_data_dir)
    dataset_names = ["(manual path)"] + list(datasets.keys())

    selected_dataset = st.sidebar.selectbox("Dataset", options=dataset_names, index=0)

    default_path = "data_demo/e1/dataset.jsonl"
    if selected_dataset != "(manual path)":
        default_path = str(datasets[selected_dataset])

    jsonl_path_str = st.sidebar.text_input("dataset.jsonl path", value=default_path)
    jsonl_path = Path(jsonl_path_str)

    if not jsonl_path.exists():
        st.warning("JSONL path not found.")
        st.info(
            "Tip: Put datasets under data_demo/<TASK>/dataset.jsonl (e.g., data_demo/e1/dataset.jsonl)."
        )
        return

    # --- load records
    recs = read_jsonl(jsonl_path)
    if not recs:
        st.warning("No records in JSONL.")
        return

    # --- Filters
    task_codes = sorted({str(r.get("task_code", "UNK")) for r in recs})
    modalities = sorted({str(r.get("modality", "UNK")) for r in recs})

    selected_task = st.sidebar.selectbox("task_code", options=["(all)"] + task_codes)
    selected_mod = st.sidebar.selectbox("modality", options=["(all)"] + modalities)

    def keep(rec: dict[str, Any]) -> bool:
        if selected_task != "(all)" and str(rec.get("task_code", "UNK")) != selected_task:
            return False
        if selected_mod != "(all)" and str(rec.get("modality", "UNK")) != selected_mod:
            return False
        return True

    filtered = [r for r in recs if keep(r)]
    st.sidebar.write(f"{len(filtered)} / {len(recs)} records")

    if not filtered:
        st.warning("No records match the filters.")
        return

    # --- navigation (Prev/Next + index)
    if "idx" not in st.session_state:
        st.session_state.idx = 0

    cprev, cnext = st.sidebar.columns(2)
    if cprev.button("◀ Prev"):
        st.session_state.idx = max(0, st.session_state.idx - 1)
    if cnext.button("Next ▶"):
        st.session_state.idx = min(len(filtered) - 1, st.session_state.idx + 1)

    idx = st.sidebar.number_input(
        "record index",
        min_value=0,
        max_value=max(0, len(filtered) - 1),
        value=int(st.session_state.idx),
        step=1,
    )
    st.session_state.idx = int(idx)

    rec = filtered[int(idx)]

    # Header
    st.subheader(
        f"Sample {rec.get('sample_id', 'UNK')} — {rec.get('task_code', 'UNK')} ({rec.get('modality','UNK')})"
    )

    inputs = rec.get("inputs", {}) or {}
    modality = str(rec.get("modality", "single"))

    # Images
    st.markdown("### Visuals")
    if modality == "single":
        img = load_img(inputs.get("image"))
        mask_rgb = load_mask_as_rgb(inputs.get("mask"))

        c1, c2 = st.columns(2)
        with c1:
            st.write("RGB")
            if img is None:
                st.warning("image not found")
            else:
                st.image(img, width="content")
        with c2:
            st.write("Mask (colored)")
            if mask_rgb is None:
                st.info("mask not available")
            else:
                st.image(mask_rgb, width="content")

    else:
        t0 = load_img(inputs.get("t0_image"))
        t1 = load_img(inputs.get("t1_image"))

        c1, c2 = st.columns(2)
        with c1:
            st.write("t0")
            if t0 is None:
                st.warning("t0 not found")
            else:
                st.image(t0, width="content")
        with c2:
            st.write("t1")
            if t1 is None:
                st.warning("t1 not found")
            else:
                st.image(t1, width="content")

    # Text fields
    colA, colB = st.columns([2, 1])
    with colA:
        st.markdown("### Prompt")
        st.code(rec.get("prompt", ""), language="text")

        st.markdown("### Answer")
        ans = rec.get("answer", None)
        log.info(f"Answer (raw): {ans}")
        _answer_block(ans)

    with colB:
        st.markdown("### Oracle")
        orc = rec.get("oracle", None)
        if isinstance(orc, (dict, list)):
            st.json(orc)
        else:
            st.code("" if orc is None else str(orc), language="text")

        st.markdown("### Inputs")
        st.json(rec.get("inputs", {}))

    # --- optional dataset summary pane
    with st.expander("Dataset summary", expanded=True):
        _dataset_summary(recs, max_preview=40)

    log.info(f"Viewed idx={idx} sample_id={rec.get('sample_id')} task={rec.get('task_code')}")


if __name__ == "__main__":
    main()
