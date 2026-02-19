from __future__ import annotations

import numpy as np

from geosynthbench.world.raster import HeightField, RasterTransform


def _value_noise_2d(rng: np.random.Generator, h: int, w: int, scale_px: float) -> np.ndarray:
    """
    Very simple smooth-ish noise: sample coarse grid, bilinear upsample.
    Keeps deps minimal (no external noise lib).
    """
    scale_px = max(scale_px, 2.0)
    gh = max(int(np.ceil(h / scale_px)), 2)
    gw = max(int(np.ceil(w / scale_px)), 2)
    coarse = rng.normal(0.0, 1.0, size=(gh, gw)).astype(np.float32)

    # bilinear upsample
    ys = np.linspace(0, gh - 1, h, dtype=np.float32)
    xs = np.linspace(0, gw - 1, w, dtype=np.float32)
    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    y1 = np.clip(y0 + 1, 0, gh - 1)
    x1 = np.clip(x0 + 1, 0, gw - 1)
    wy = (ys - y0).reshape(-1, 1)
    wx = (xs - x0).reshape(1, -1)

    a = coarse[y0][:, x0]
    b = coarse[y0][:, x1]
    c = coarse[y1][:, x0]
    d = coarse[y1][:, x1]
    out = (1 - wy) * ((1 - wx) * a + wx * b) + wy * ((1 - wx) * c + wx * d)
    return out.astype(np.float32)


def generate_terrain(tr: RasterTransform, rng: np.random.Generator, amplitude_m: float, n_hills: int,
                     hill_sigma_m_range: tuple[float, float], noise_scale_m: float, noise_strength_m: float) -> HeightField:
    H, W = tr.height_px, tr.width_px
    elev = np.zeros((H, W), dtype=np.float32)

    # hills as gaussian bumps (game terrain classic)
    for _ in range(n_hills):
        cx = rng.uniform(tr.xmin, tr.xmax)
        cy = rng.uniform(tr.ymin, tr.ymax)
        sigma = rng.uniform(hill_sigma_m_range[0], hill_sigma_m_range[1])
        height = rng.uniform(0.4, 1.0) * amplitude_m

        # build in pixel space for speed
        u0, v0 = tr.world_to_px(cx, cy)
        uu = np.arange(W, dtype=np.float32)
        vv = np.arange(H, dtype=np.float32)
        # convert sigma meters to sigma pixels roughly
        sig_u = sigma / tr.dx
        sig_v = sigma / tr.dy
        du2 = (uu - float(u0)) ** 2 / (2.0 * sig_u * sig_u + 1e-6)
        dv2 = (vv - float(v0)) ** 2 / (2.0 * sig_v * sig_v + 1e-6)
        bump = np.exp(-(dv2.reshape(-1, 1) + du2.reshape(1, -1))).astype(np.float32)
        elev += height * bump

    # add smooth noise
    scale_px = noise_scale_m / max(tr.dx, tr.dy)
    noise = _value_noise_2d(rng, H, W, scale_px=scale_px)
    elev += noise * float(noise_strength_m)

    # normalize a bit (optional)
    elev -= float(elev.min())
    return HeightField(tr=tr, elevation_m=elev)
