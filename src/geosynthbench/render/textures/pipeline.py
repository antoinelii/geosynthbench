from __future__ import annotations

import numpy as np
import numpy.typing as npt

from geosynthbench.render.textures import (
    BuildingParams,
    RoadParams,
    SceneParams,
    SettlementParams,
    VegetationParams,
    WaterParams,
    background_palette_from_elevation,
    building_palette,
    postprocess,
    render_full_rgb,
    roads_palette,
    settlement_palette,
    vegetation_palette,
    water_palette,
)
from geosynthbench.world.entities import BuildingMaskItem
from geosynthbench.world.types import LayerKind


def render_scene_rgb(
    elev: npt.NDArray[np.float32],
    scene: SceneParams,
    masks: dict[LayerKind, npt.NDArray[np.bool_]],
    buildings: list[BuildingMaskItem],
    rng: np.random.Generator,
) -> np.ndarray:
    """
    masks contains bool arrays: "water", "veg", "roads", "settlement" etc.
    buildings: list of BuildingMaskItem objects with attributes "mask", "settlement_id", and "id"
    returns uint8 RGB [H,W,3]
    """
    bg = background_palette_from_elevation(elev, scene, rng)

    # entity params (could be sampled per scene)
    wp = WaterParams(alpha=0.85, turbidity=0.35, specular=0.25, shoreline_width_px=3)
    vp = VegetationParams(alpha=0.85, density=0.75, patchiness=0.65, texture_scale_px=16)
    rp = RoadParams(alpha=0.95, wear=0.45, lane_hint=0.15, texture_scale_px=10)
    sp = SettlementParams(alpha=0.65, impervious=0.75, grime=0.35)
    bp = BuildingParams(alpha=0.98, roof_variation=0.8, shadow_strength=0.2)

    layers: list[npt.NDArray[np.float32]] = []
    layers.append(water_palette(masks.get("water", np.zeros_like(elev, bool)), scene, wp, rng))
    layers.append(
        vegetation_palette(masks.get("vegetation", np.zeros_like(elev, bool)), scene, vp, rng)
    )
    layers.append(
        settlement_palette(masks.get("settlement", np.zeros_like(elev, bool)), scene, sp, rng)
    )
    layers.append(roads_palette(masks.get("road", np.zeros_like(elev, bool)), scene, rp, rng))

    for b in buildings:
        layers.append(building_palette(b.mask, b.settlement_id, scene, bp, rng))

    rgb = render_full_rgb(bg, layers)

    # postprocess: anti-mask look
    out = postprocess(
        rgb,
        exposure=scene.exposure,
        gamma=scene.gamma,
        saturation=scene.saturation,
        grain=0.015,
        blur_k=3,
        rng=rng,
    )
    return out
