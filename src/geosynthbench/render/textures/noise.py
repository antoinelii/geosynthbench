# src/geosynthbench/render/textures/noise.py
from __future__ import annotations

import numpy as np
import numpy.typing as npt


def _lerp(
    a: npt.NDArray[np.float32], b: npt.NDArray[np.float32], t: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    """Linear interpolation between a and b with weight t in [0,1]."""
    return (a + (b - a) * t).astype(np.float32, copy=False)


def value_noise_2d(
    h: int, w: int, scale: float, rng: np.random.Generator
) -> npt.NDArray[np.float32]:
    """
    Coarse grid random -> bilinear upsample. Returns [0,1].
    scale ~ desired feature size in pixels (bigger = smoother).
    """
    scale = max(1.0, float(scale))
    gh = int(np.ceil(h / scale)) + 2  # grid height, add 2 for padding to avoid out-of-bounds
    gw = int(np.ceil(w / scale)) + 2
    grid = rng.random((gh, gw), dtype=np.float32)

    # Coordinates in grid space
    y = np.linspace(0, gh - 2, h, dtype=np.float32)
    x = np.linspace(0, gw - 2, w, dtype=np.float32)
    y_int = np.floor(y).astype(np.int32)
    x_int = np.floor(x).astype(np.int32)
    y_frac = (y - y_int).reshape(-1, 1)
    x_frac = (x - x_int).reshape(1, -1)

    g00 = grid[y_int[:, None], x_int[None, :]]
    g10 = grid[y_int[:, None] + 1, x_int[None, :]]
    g01 = grid[y_int[:, None], x_int[None, :] + 1]
    g11 = grid[y_int[:, None] + 1, x_int[None, :] + 1]

    a = _lerp(g00, g01, x_frac)
    b = _lerp(g10, g11, x_frac)
    out = _lerp(a, b, y_frac)
    return out.clip(0.0, 1.0)


def fbm_2d(
    h: int,
    w: int,
    base_scale_px: float,
    octaves: int,
    lacunarity: float,
    gain: float,
    rng: np.random.Generator,
) -> npt.NDArray[np.float32]:
    """
    Fractal Brownian motion from value noise. Returns [0,1].
    """
    amp = 1.0
    freq_scale = float(base_scale_px)
    total = np.zeros((h, w), dtype=np.float32)
    norm = 0.0
    for _ in range(int(octaves)):
        total += amp * value_noise_2d(h, w, freq_scale, rng)
        norm += amp
        amp *= float(gain)
        freq_scale /= float(lacunarity)
        freq_scale = max(1.0, freq_scale)
    out = (total / max(1e-6, norm)).astype(np.float32)
    return out.clip(0.0, 1.0)
