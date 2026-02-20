# src/geosynthbench/render/textures/noise.py
from __future__ import annotations

import numpy as np


def _lerp(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray:
    return a + (b - a) * t


def value_noise_2d(h: int, w: int, scale: float, rng: np.random.Generator) -> np.ndarray:
    """
    Coarse grid random -> bilinear upsample. Returns [0,1].
    scale ~ desired feature size in pixels (bigger = smoother).
    """
    scale = max(1.0, float(scale))
    gh = int(np.ceil(h / scale)) + 2
    gw = int(np.ceil(w / scale)) + 2
    grid = rng.random((gh, gw), dtype=np.float32)

    # Coordinates in grid space
    y = np.linspace(0, gh - 2, h, dtype=np.float32)
    x = np.linspace(0, gw - 2, w, dtype=np.float32)
    yi = np.floor(y).astype(np.int32)
    xi = np.floor(x).astype(np.int32)
    yf = (y - yi).reshape(-1, 1)
    xf = (x - xi).reshape(1, -1)

    g00 = grid[yi[:, None], xi[None, :]]
    g10 = grid[yi[:, None] + 1, xi[None, :]]
    g01 = grid[yi[:, None], xi[None, :] + 1]
    g11 = grid[yi[:, None] + 1, xi[None, :] + 1]

    a = _lerp(g00, g01, xf)
    b = _lerp(g10, g11, xf)
    out = _lerp(a, b, yf)
    return out.clip(0.0, 1.0)


def fbm_2d(
    h: int,
    w: int,
    base_scale_px: float,
    octaves: int,
    lacunarity: float,
    gain: float,
    rng: np.random.Generator,
) -> np.ndarray:
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
