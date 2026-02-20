# src/geosynthbench/render/textures/shadows.py
from __future__ import annotations

import numpy as np


def _shift_mask(mask: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """
    Shift boolean mask by (dx,dy) in pixels. Outside = False.
    dx>0 shifts right. dy>0 shifts down.
    """
    h, w = mask.shape
    out = np.zeros_like(mask, dtype=bool)

    x0_src = max(0, -dx)
    x1_src = min(w, w - dx)  # exclusive
    y0_src = max(0, -dy)
    y1_src = min(h, h - dy)

    x0_dst = max(0, dx)
    x1_dst = x0_dst + (x1_src - x0_src)
    y0_dst = max(0, dy)
    y1_dst = y0_dst + (y1_src - y0_src)

    if x1_src <= x0_src or y1_src <= y0_src:
        return out

    out[y0_dst:y1_dst, x0_dst:x1_dst] = mask[y0_src:y1_src, x0_src:x1_src]
    return out


def add_building_shadows(
    rgb: np.ndarray,
    building_mask: np.ndarray,
    *,
    sun_azimuth_deg: float,
    strength: float,
    length_px: int,
) -> np.ndarray:
    """
    Darkens pixels 'behind' buildings w.r.t sun direction.
    sun_azimuth_deg: 0=north, 90=east
    """
    if strength <= 0 or length_px <= 0 or not np.any(building_mask):
        return rgb

    # Shadow goes opposite of light direction.
    az = np.deg2rad(float(sun_azimuth_deg))
    # image coords: +x right (east), +y down (south)
    lx = np.sin(az)
    ly = np.cos(az)
    sx = -lx
    sy = -ly

    # pixel shift
    dx = int(np.round(sx * float(length_px)))
    dy = int(np.round(sy * float(length_px)))

    shadow = _shift_mask(building_mask.astype(bool), dx, dy)
    shadow &= ~building_mask.astype(bool)

    # feather shadow by doing a couple of small shifts and summing
    acc = shadow.astype(np.float32)
    for t in (0.33, 0.66):
        ddx = int(np.round(dx * t))
        ddy = int(np.round(dy * t))
        acc += _shift_mask(building_mask.astype(bool), ddx, ddy).astype(np.float32)

    acc = np.clip(acc / (1.0 + 2.0), 0.0, 1.0)[..., None]
    k = float(np.clip(strength, 0.0, 1.0)) * 0.35  # cap darkness

    out = rgb.astype(np.float32)
    out *= 1.0 - k * acc
    return np.clip(out, 0.0, 1.0).astype(np.float32)
