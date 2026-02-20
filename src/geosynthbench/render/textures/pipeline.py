from __future__ import annotations

import numpy as np

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


def render_scene_rgb(
    elev: np.ndarray,
    scene: SceneParams,
    masks: dict[str, np.ndarray],
    buildings: list[dict],
    rng: np.random.Generator,
) -> np.ndarray:
    """
    masks contains bool arrays: "water", "veg", "roads", "settlement" etc.
    buildings: list of { "mask": bool[H,W], "settlement_id": str, "id": str }
    returns uint8 RGB [H,W,3]
    """
    bg = background_palette_from_elevation(elev, scene, rng)

    # entity params (could be sampled per scene)
    wp = WaterParams(alpha=0.85, turbidity=0.35, specular=0.25, shoreline_width_px=3)
    vp = VegetationParams(alpha=0.85, density=0.75, patchiness=0.65, texture_scale_px=16)
    rp = RoadParams(alpha=0.95, wear=0.45, lane_hint=0.15, texture_scale_px=10)
    sp = SettlementParams(alpha=0.65, impervious=0.75, grime=0.35)
    bp = BuildingParams(alpha=0.98, roof_variation=0.8, shadow_strength=0.2)

    layers = []
    layers.append(water_palette(masks.get("water", np.zeros_like(elev, bool)), scene, wp, rng))
    layers.append(vegetation_palette(masks.get("veg", np.zeros_like(elev, bool)), scene, vp, rng))
    layers.append(
        settlement_palette(masks.get("settlement", np.zeros_like(elev, bool)), scene, sp, rng)
    )
    layers.append(roads_palette(masks.get("roads", np.zeros_like(elev, bool)), scene, rp, rng))

    # buildings: seed each building for stable intra-scene variety if you want
    for b in buildings:
        b_rng = np.random.default_rng(rng.integers(0, 2**63 - 1, dtype=np.int64))
        layers.append(building_palette(b["mask"], b["settlement_id"], scene, bp, b_rng))

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
