# src/geosynthbench/render/textures/post.py
from __future__ import annotations

import numpy as np


def _box_blur_2d(x: np.ndarray, k: int) -> np.ndarray:
    """
    Fast-ish separable box blur for float32 images.
    k must be odd.
    """
    if k <= 1:
        return x
    k = int(k)
    if k % 2 == 0:
        k += 1
    pad = k // 2

    # blur y then x
    ypad = np.pad(x, ((pad, pad), (0, 0), (0, 0)), mode="reflect")
    c = np.cumsum(ypad, axis=0)
    y = (c[k:, ...] - c[:-k, ...]) / k

    xpad = np.pad(y, ((0, 0), (pad, pad), (0, 0)), mode="reflect")
    c2 = np.cumsum(xpad, axis=1)
    out = (c2[:, k:, :] - c2[:, :-k, :]) / k
    return out.astype(np.float32)


def postprocess(
    rgb: np.ndarray,
    exposure: float,
    gamma: float,
    saturation: float,
    grain: float,
    blur_k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    rgb float32 [0,1] -> uint8
    """
    x = np.clip(rgb * float(exposure), 0.0, 1.0).astype(np.float32)

    # mild blur (anti-alias & anti-mask)
    if blur_k > 1:
        x = _box_blur_2d(x, blur_k)

    # saturation adjust in HSV-lite space
    if saturation != 1.0:
        light = (0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2])[..., None]
        x = np.clip(light + (x - light) * float(saturation), 0.0, 1.0)

    # grain
    if grain > 0:
        g = (rng.normal(0.0, 1.0, size=x.shape).astype(np.float32)) * float(grain)
        x = np.clip(x + g, 0.0, 1.0)

    # gamma
    x = np.clip(x, 0.0, 1.0) ** (1.0 / max(1e-6, float(gamma)))

    return (x * 255.0 + 0.5).astype(np.uint8)
