# src/geosynthbench/render/textures/mask_ops.py
from __future__ import annotations

import numpy as np


def _dilate_bool(mask: np.ndarray, radius_px: int) -> np.ndarray:
    """
    Cheap binary dilation using repeated 3x3 max filter.
    radius_px ~ number of iterations.
    """
    m = mask.astype(bool, copy=False)
    r = int(radius_px)
    if r <= 0 or not np.any(m):
        return m.copy()

    out = m.copy()
    for _ in range(r):
        p = np.pad(out, ((1, 1), (1, 1)), mode="edge")
        out = (
            p[0:-2, 0:-2]
            | p[0:-2, 1:-1]
            | p[0:-2, 2:]
            | p[1:-1, 0:-2]
            | p[1:-1, 1:-1]
            | p[1:-1, 2:]
            | p[2:, 0:-2]
            | p[2:, 1:-1]
            | p[2:, 2:]
        )
    return out


def shoreline_band(water_mask: np.ndarray, width_px: int) -> np.ndarray:
    """
    Returns a ring mask around water: dilate(water,width) - water.
    """
    w = water_mask.astype(bool, copy=False)
    if width_px <= 0 or not np.any(w):
        return np.zeros_like(w, dtype=bool)
    dil = _dilate_bool(w, int(width_px))
    return dil & ~w


def thin_mask_by_density(
    mask: np.ndarray,
    density: float,
    rng: np.random.Generator,
    *,
    clumpy: bool = True,
    clump_scale_px: float = 32.0,
) -> np.ndarray:
    """
    Randomly keeps ~density fraction of True pixels inside mask.
    If clumpy=True, produces patchy thinning rather than i.i.d speckle.
    """
    m = mask.astype(bool, copy=False)
    d = float(np.clip(density, 0.0, 1.0))
    if d <= 0.0 or not np.any(m):
        return np.zeros_like(m, dtype=bool)
    if d >= 1.0:
        return m.copy()

    h, w = m.shape

    if not clumpy:
        keep = rng.random((h, w), dtype=np.float32) < d
        return m & keep

    # clumpy: create a smooth-ish random field then threshold it
    # (no extra deps; build coarse grid & bilinear upsample)
    scale = max(2.0, float(clump_scale_px))
    gh = int(np.ceil(h / scale)) + 2
    gw = int(np.ceil(w / scale)) + 2
    grid = rng.random((gh, gw), dtype=np.float32)

    yy = np.linspace(0, gh - 2, h, dtype=np.float32)
    xx = np.linspace(0, gw - 2, w, dtype=np.float32)
    yi = np.floor(yy).astype(np.int32)
    xi = np.floor(xx).astype(np.int32)
    yf = (yy - yi).reshape(-1, 1)
    xf = (xx - xi).reshape(1, -1)

    g00 = grid[yi[:, None], xi[None, :]]
    g10 = grid[yi[:, None] + 1, xi[None, :]]
    g01 = grid[yi[:, None], xi[None, :] + 1]
    g11 = grid[yi[:, None] + 1, xi[None, :] + 1]

    a = g00 + (g01 - g00) * xf
    b = g10 + (g11 - g10) * xf
    field = a + (b - a) * yf  # ~smooth 0..1

    # choose threshold so that fraction ~ density inside mask
    vals = field[m]
    # guard
    if vals.size == 0:
        return np.zeros_like(m, dtype=bool)
    thr = np.quantile(vals, 1.0 - d)
    keep = field >= thr
    return m & keep
