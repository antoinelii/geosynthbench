# src/geosynthbench/render/textures/compose.py
from __future__ import annotations

import numpy as np


def alpha_over(dst_rgb: np.ndarray, src_rgba: np.ndarray) -> np.ndarray:
    """
    dst_rgb: float32 [H,W,3] in 0..1
    src_rgba: float32 [H,W,4] in 0..1
    """
    a = np.clip(src_rgba[..., 3:4], 0.0, 1.0)
    dst_rgb = dst_rgb * (1.0 - a) + src_rgba[..., :3] * a
    return np.clip(dst_rgb, 0.0, 1.0).astype(np.float32)


def render_full_rgb(
    background_rgb: np.ndarray,
    layers_rgba: list[np.ndarray],
) -> np.ndarray:
    rgb = background_rgb.astype(np.float32)
    for layer in layers_rgba:
        rgb = alpha_over(rgb, layer)
    return rgb
