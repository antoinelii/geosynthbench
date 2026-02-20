# src/geosynthbench/render/textures/terrain.py
from __future__ import annotations

import numpy as np

from .noise import fbm_2d
from .params import SceneParams


def hillshade(elev: np.ndarray, azimuth_deg: float, altitude_deg: float) -> np.ndarray:
    """
    Classic hillshade from elevation using gradient-based normals.
    elev: float32 [H,W]
    returns: float32 [H,W] in [0,1]
    """
    z = elev.astype(np.float32)
    dzdy, dzdx = np.gradient(z)

    # Normal vector components
    nx = -dzdx
    ny = -dzdy
    nz = np.ones_like(z, dtype=np.float32)
    n_norm = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-6
    nx /= n_norm
    ny /= n_norm
    nz /= n_norm

    az = np.deg2rad(azimuth_deg)
    alt = np.deg2rad(altitude_deg)

    # Light vector: azimuth clockwise from north; map to image coords (x=east, y=south)
    lx = np.sin(az) * np.cos(alt)
    ly = np.cos(az) * np.cos(alt)
    lz = np.sin(alt)

    luminance = nx * lx + ny * ly + nz * lz
    luminance = np.clip(luminance, 0.0, 1.0)
    return luminance.astype(np.float32)


def _biome_base_rgb(scene: SceneParams) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (lowland_rgb, highland_rgb) in 0..1, shape (3,)
    """
    b = scene.biome
    if b == "temperate":
        return (np.array([0.40, 0.45, 0.35], np.float32), np.array([0.55, 0.55, 0.50], np.float32))
    if b == "arid":
        return (np.array([0.55, 0.50, 0.38], np.float32), np.array([0.62, 0.60, 0.52], np.float32))
    if b == "tropical":
        return (np.array([0.32, 0.44, 0.30], np.float32), np.array([0.50, 0.55, 0.45], np.float32))
    # boreal
    return (np.array([0.38, 0.42, 0.40], np.float32), np.array([0.55, 0.57, 0.60], np.float32))


def background_palette_from_elevation(
    elev: np.ndarray,
    scene: SceneParams,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Creates base terrain RGB in float32 [0,1] using elevation percentile + fbm albedo + hillshade.
    """
    h, w = elev.shape
    z = elev.astype(np.float32)

    # Normalize elevation robustly
    lo = np.percentile(z, 2).astype(np.float32)
    hi = np.percentile(z, 98).astype(np.float32)
    zn = np.clip((z - lo) / max(1e-6, (hi - lo)), 0.0, 1.0)

    low_rgb, high_rgb = _biome_base_rgb(scene)
    base = low_rgb[None, None, :] * (1.0 - zn[..., None]) + high_rgb[None, None, :] * zn[..., None]

    # Albedo variation (prevents “flat paint”)
    albedo = fbm_2d(h, w, base_scale_px=48, octaves=4, lacunarity=2.0, gain=0.55, rng=rng)
    albedo = (0.90 + 0.20 * (albedo - 0.5)).astype(np.float32)  # ~0.8..1.0
    base *= albedo[..., None]

    # Shading (geometry cue; keep subtle)
    hs = hillshade(z, scene.sun_azimuth_deg, scene.sun_altitude_deg)
    shade = 0.75 + 0.25 * (hs * scene.sun_intensity)
    base *= shade[..., None]

    # Haze wash
    if scene.haze > 0:
        base = base * (1.0 - scene.haze) + scene.haze * np.array([0.55, 0.60, 0.65], np.float32)

    return np.clip(base, 0.0, 1.0).astype(np.float32)
