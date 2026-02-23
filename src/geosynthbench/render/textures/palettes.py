# src/geosynthbench/render/textures/palettes.py
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .mask_ops import shoreline_band, thin_mask_by_density
from .noise import fbm_2d
from .params import (
    BuildingParams,
    RoadParams,
    SceneParams,
    SettlementParams,
    VegetationParams,
    WaterParams,
)


def _rgba_layer(h: int, w: int) -> npt.NDArray[np.float32]:
    return np.zeros((h, w, 4), dtype=np.float32)


def water_palette(
    water_mask: npt.NDArray[np.bool_],
    scene: SceneParams,
    params: WaterParams,
    rng: np.random.Generator,
) -> npt.NDArray[np.float32]:
    h, w = water_mask.shape
    out = _rgba_layer(h, w)
    if not np.any(water_mask):
        return out

    # Base water color by biome
    if scene.biome in ("temperate", "boreal"):
        base = np.array([0.10, 0.20, 0.28], np.float32)
    elif scene.biome == "tropical":
        base = np.array([0.08, 0.28, 0.30], np.float32)
    else:
        base = np.array([0.12, 0.22, 0.24], np.float32)

    # Turbidity adds green/brown + noise
    turb = params.turbidity
    turb_noise = fbm_2d(h, w, 64, 3, 2.0, 0.6, rng)
    turb_map = (turb_noise - 0.5) * turb * 0.25
    color = np.broadcast_to(base, (h, w, 3)).copy()
    color[:, :, 0] += turb_map
    color[:, :, 1] += 0.8 * turb_map

    # Specular-ish glint aligned with sun (fake): brighten a low-freq band
    if params.specular > 0:
        gl = fbm_2d(h, w, 96, 2, 2.0, 0.5, rng)
        gl = np.clip((gl - 0.65) * 2.5, 0, 1) * params.specular * 0.15
        color += gl[..., None]

    out[..., :3] = color
    out[..., 3] = params.alpha * water_mask.astype(np.float32)
    # Add shoreline highlight band if desired
    if params.shoreline_width_px > 0:
        shore = shoreline_band(water_mask, params.shoreline_width_px)
        if np.any(shore):
            # light edge, slightly warm
            shore_rgb = np.array([0.62, 0.62, 0.58], np.float32)
            # overlay into out directly (since water_palette returns one RGBA)
            # we’ll brighten edge a bit but keep alpha modest
            out[shore, 0:3] = shore_rgb
            out[shore, 3] = np.maximum(out[shore, 3], (params.alpha * 0.35))
    return np.clip(out, 0.0, 1.0)


def vegetation_palette(
    veg_mask: npt.NDArray[np.bool_],
    scene: SceneParams,
    params: VegetationParams,
    rng: np.random.Generator,
) -> npt.NDArray[np.float32]:
    h, w = veg_mask.shape
    out = _rgba_layer(h, w)
    # Patchy density mask: thins out the input mask in a spatially coherent way, for more visual interest.
    veg_eff = thin_mask_by_density(
        veg_mask,
        density=params.density,
        rng=rng,
        clumpy=True,
        clump_scale_px=max(12.0, params.texture_scale_px * 2.0),
    )
    if not np.any(veg_eff):
        return out

    # Base veg color by biome
    if scene.biome == "temperate":
        base = np.array([0.16, 0.32, 0.16], np.float32)
    elif scene.biome == "tropical":
        base = np.array([0.10, 0.38, 0.16], np.float32)
    elif scene.biome == "arid":
        base = np.array([0.24, 0.34, 0.18], np.float32)
    else:  # boreal
        base = np.array([0.12, 0.26, 0.16], np.float32)

    tex = fbm_2d(h, w, params.texture_scale_px, 4, 2.0, 0.55, rng)
    tex = tex - 0.5

    # Patchiness creates clumps, density controls coverage intensity inside mask
    patches = fbm_2d(h, w, 96, 3, 2.0, 0.6, rng)
    patches = np.clip((patches - (1.0 - params.patchiness)) * 2.0, 0.0, 1.0)

    color = base[None, None, :] + np.stack([0.04 * tex, 0.06 * tex, 0.03 * tex], axis=-1)
    alpha = params.alpha * params.density * (0.65 + 0.35 * patches)
    out[..., :3] = color
    out[..., 3] = alpha * veg_eff.astype(np.float32)
    return np.clip(out, 0.0, 1.0)


def roads_palette(
    roads_mask: npt.NDArray[np.bool_],
    scene: SceneParams,
    params: RoadParams,
    rng: np.random.Generator,
) -> npt.NDArray[np.float32]:
    h, w = roads_mask.shape
    out = _rgba_layer(h, w)
    if not np.any(roads_mask):
        return out

    # Asphalt-ish
    base = np.array([0.22, 0.22, 0.23], np.float32)
    if scene.biome == "arid":
        base = np.array([0.25, 0.24, 0.22], np.float32)

    tex = fbm_2d(h, w, params.texture_scale_px, 4, 2.0, 0.6, rng)
    cracks = np.clip((tex - 0.55) * 3.0, 0.0, 1.0) * params.wear

    color = base[None, None, :] + (cracks[..., None] * np.array([0.05, 0.05, 0.05], np.float32))
    out[..., :3] = color
    out[..., 3] = params.alpha * roads_mask.astype(np.float32)
    return np.clip(out, 0.0, 1.0)


def settlement_palette(
    settlement_mask: npt.NDArray[np.bool_],
    scene: SceneParams,
    params: SettlementParams,
    rng: np.random.Generator,
) -> npt.NDArray[np.float32]:
    h, w = settlement_mask.shape
    out = _rgba_layer(h, w)
    if not np.any(settlement_mask):
        return out

    # Impervious surface mix
    concrete = np.array([0.55, 0.54, 0.52], np.float32)
    soil = (
        np.array([0.48, 0.46, 0.40], np.float32)
        if scene.biome != "arid"
        else np.array([0.56, 0.50, 0.40], np.float32)
    )

    mix = np.clip(params.impervious, 0.0, 1.0)
    base = soil * (1.0 - mix) + concrete * mix

    grime = fbm_2d(h, w, 64, 3, 2.0, 0.6, rng)
    grime = (grime - 0.5) * params.grime * 0.12

    color = base[None, None, :] + grime[..., None]
    out[..., :3] = color
    out[..., 3] = params.alpha * settlement_mask.astype(np.float32)
    return np.clip(out, 0.0, 1.0)


def building_palette(
    building_mask: npt.NDArray[np.bool_],
    settlement_id: str,
    scene: SceneParams,
    params: BuildingParams,
    rng: np.random.Generator,
) -> npt.NDArray[np.float32]:
    h, w = building_mask.shape
    out = _rgba_layer(h, w)
    if not np.any(building_mask):
        return out

    # Choose roof family by biome, then per-building variation
    if scene.biome in ("temperate", "boreal"):
        base = np.array([0.62, 0.30, 0.26], np.float32)  # red-ish roof
    elif scene.biome == "arid":
        base = np.array([0.62, 0.52, 0.40], np.float32)  # sandy roof
    else:
        base = np.array([0.30, 0.42, 0.50], np.float32)  # blue-ish / metal-ish

    # per-building jitter: use RNG already seeded per building outside, or add noise field
    jitter = (rng.random(3, dtype=np.float32) - 0.5) * params.roof_variation * 0.15
    color = np.clip(base + jitter, 0.0, 1.0)

    # subtle roof texture
    tex = fbm_2d(h, w, 14, 3, 2.0, 0.6, rng)
    tex = (tex - 0.5) * 0.06
    rgb = color[None, None, :] + tex[..., None]

    out[..., :3] = rgb
    out[..., 3] = params.alpha * building_mask.astype(np.float32)
    return np.clip(out, 0.0, 1.0)
