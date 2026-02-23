# src/geosynthbench/render/textures/compose.py
from __future__ import annotations

import numpy as np
import numpy.typing as npt


def alpha_over(
    dst_rgb: npt.NDArray[np.float32], src_rgba: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    """
    dst_rgb: float32 [H,W,3] in 0..1
    src_rgba: float32 [H,W,4] in 0..1
    """
    a = np.clip(src_rgba[..., 3:4], 0.0, 1.0).astype(np.float32, copy=False)
    out = dst_rgb * (1.0 - a) + src_rgba[..., :3] * a
    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)


def render_full_rgb(
    background_rgb: npt.NDArray[np.float32],
    layers_rgba: list[npt.NDArray[np.float32]],
) -> npt.NDArray[np.float32]:
    rgb = background_rgb
    for layer in layers_rgba:
        rgb = alpha_over(rgb, layer)
    return rgb
