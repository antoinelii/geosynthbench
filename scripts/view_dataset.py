# scripts/view_dataset.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
from PIL import Image

from geosynthbench.render.semantic import semantic_mask_to_rgb
from geosynthbench.utils.logging import get_logger, setup_logging


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def load_img(path: str | None):
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return Image.open(p)


def load_mask_as_rgb(path: str | None):
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    m = np.array(Image.open(p))
    if m.ndim == 3:
        # already rgb-like
        return Image.fromarray(m)
    return Image.fromarray(semantic_mask_to_rgb(m))


def main() -> None:
    setup_logging()
    log = get_logger()

    st.set_page_config(page_title="GeoSynthBench Dataset Viewer", layout="wide")
    st.title("GeoSynthBench Dataset Viewer")

    jsonl_path_str = st.sidebar.text_input("dataset.jsonl path", value="data/E1/dataset.jsonl")
    jsonl_path = Path(jsonl_path_str)

    if not jsonl_path.exists():
        st.warning("JSONL path not found.")
        return

    recs = read_jsonl(jsonl_path)
    if not recs:
        st.warning("No records in JSONL.")
        return

    # Filters
    task_codes = sorted({str(r.get("task_code", "UNK")) for r in recs})
    modalities = sorted({str(r.get("modality", "UNK")) for r in recs})

    selected_task = st.sidebar.selectbox("task_code", options=["(all)"] + task_codes)
    selected_mod = st.sidebar.selectbox("modality", options=["(all)"] + modalities)

    def keep(r: dict[str, Any]) -> bool:
        if selected_task != "(all)" and str(r.get("task_code", "UNK")) != selected_task:
            return False
        if selected_mod != "(all)" and str(r.get("modality", "UNK")) != selected_mod:
            return False
        return True

    filtered = [r for r in recs if keep(r)]
    st.sidebar.write(f"{len(filtered)} / {len(recs)} records")

    idx = st.sidebar.slider(
        "record index", min_value=0, max_value=max(0, len(filtered) - 1), value=0, step=1
    )
    r = filtered[idx]

    # Header
    st.subheader(
        f"Sample {r.get('sample_id', 'UNK')} — {r.get('task_code', 'UNK')} ({r.get('modality','UNK')})"
    )

    # Text fields
    colA, colB = st.columns([2, 1])
    with colA:
        st.markdown("### Prompt")
        st.code(r.get("prompt", ""), language="text")

        st.markdown("### Answer")
        log.info(f"Answer: {r.get('answer', {})}")
        st.json(r.get("answer", {}))

    with colB:
        st.markdown("### Oracle")
        st.json(r.get("oracle", {}))

        st.markdown("### Inputs")
        st.json(r.get("inputs", {}))

    inputs = r.get("inputs", {}) or {}
    modality = str(r.get("modality", "single"))

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
                st.image(img, width=True)
        with c2:
            st.write("Mask (colored)")
            if mask_rgb is None:
                st.info("mask not available")
            else:
                st.image(mask_rgb, width=True)

    else:
        t0 = load_img(inputs.get("t0_image"))
        t1 = load_img(inputs.get("t1_image"))
        change = load_mask_as_rgb(
            inputs.get("change_mask")
        )  # supports colored if semantic ids; otherwise shows binary as gray

        c1, c2 = st.columns(2)
        with c1:
            st.write("t0")
            if t0 is None:
                st.warning("t0 not found")
            else:
                st.image(t0, width=True)
        with c2:
            st.write("t1")
            if t1 is None:
                st.warning("t1 not found")
            else:
                st.image(t1, width=True)

        st.write("Change mask")
        if change is None:
            st.info("change_mask not available")
        else:
            st.image(change, width=True)

    # Quick debug log
    log.info(f"Viewed record idx={idx} sample_id={r.get('sample_id')} task={r.get('task_code')}")


if __name__ == "__main__":
    main()
