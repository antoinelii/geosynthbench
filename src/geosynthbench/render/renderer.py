from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import Polygon

from geosynthbench.render.textures.compose import render_full_rgb
from geosynthbench.render.textures.masks import (
    MaskLayers,
    build_masks_from_world,
    poly_to_px_rings,
)
from geosynthbench.render.textures.palettes import (
    building_palette,
    roads_palette,
    settlement_palette,
    vegetation_palette,
    water_palette,
)
from geosynthbench.render.textures.params import (
    BuildingParams,
    RoadParams,
    SceneParams,
    SettlementParams,
    VegetationParams,
    WaterParams,
)
from geosynthbench.render.textures.post import postprocess
from geosynthbench.render.textures.shadows import add_building_shadows
from geosynthbench.render.textures.terrain import background_palette_from_elevation
from geosynthbench.world.world_state import WorldState


@dataclass(frozen=True)
class RenderResult:
    rgb: Image.Image
    mask: Image.Image
    height: Image.Image | None = None
    slope: Image.Image | None = None


def make_empty_mask(W: int, H: int) -> Image.Image:
    return Image.new("L", (W, H), color=0)


def pil_mask_to_bool(im: Image.Image) -> np.ndarray:
    return np.array(im, dtype=np.uint8) > 0


def draw_poly_mask(
    draw: ImageDraw.ImageDraw, world: WorldState, poly: Polygon, value: int = 255
) -> None:
    if poly.is_empty:
        return
    ext, holes = poly_to_px_rings(world, poly)
    draw.polygon(ext, fill=value)
    for hole in holes:
        draw.polygon(hole, fill=0)


def rng_from_id(global_rng: np.random.Generator, stable_id: str) -> np.random.Generator:
    # mix global randomness with stable id (so different scenes differ, but buildings are unique)
    h = hashlib.blake2b(stable_id.encode("utf-8"), digest_size=8).digest()
    sid = int.from_bytes(h, "little", signed=False)
    salt = int(global_rng.integers(0, 2**63 - 1, dtype=np.int64))
    seed = (sid ^ salt) & ((1 << 63) - 1)
    return np.random.default_rng(seed)
    # Will be called like this:
    # b_rng = rng_from_id(rng, b.id)  # or f"{world_id}:{b.id}"
    # layers.append(building_palette(b_mask, b.settlement_id, scene, bp, b_rng))


def render_world_textured(world: WorldState, rng: np.random.Generator) -> Image.Image:
    W, H = world.tr.width_px, world.tr.height_px

    elev = (
        world.terrain.elevation_m.astype(np.float32)
        if world.terrain is not None
        else np.zeros((H, W), np.float32)
    )

    scene = SceneParams(
        biome="temperate",
        sun_azimuth_deg=float(rng.uniform(0, 360)),
        sun_altitude_deg=float(rng.uniform(20, 60)),
        sun_intensity=float(rng.uniform(0.8, 1.2)),
        exposure=float(rng.uniform(0.9, 1.1)),
        gamma=float(rng.uniform(0.95, 1.15)),
        haze=float(rng.uniform(0.0, 0.08)),
        saturation=float(rng.uniform(0.9, 1.15)),
    )

    # 1) Build masks (bool arrays) using PIL drawing (recommended for now)
    mask_build_res = build_masks_from_world(
        world
    )  # you implement using your polygon->px ring helper
    masks = mask_build_res.masks
    building_items = mask_build_res.building_items
    # settlement_polys = mask_build_res.settlement_polys  # for debug / visualization if

    # 2) Background
    bg = background_palette_from_elevation(elev, scene, rng)

    # 3) Params
    wp = WaterParams(
        alpha=0.88,
        turbidity=float(rng.uniform(0.1, 0.6)),
        specular=float(rng.uniform(0.0, 0.35)),
        shoreline_width_px=3,
    )
    vp = VegetationParams(
        alpha=0.85,
        density=float(
            rng.uniform(0.95, 1.0)
        ),  # need more veg for masks to be visible in test render
        patchiness=float(rng.uniform(0.4, 0.9)),
        texture_scale_px=float(rng.uniform(10, 22)),
    )
    rp = RoadParams(
        alpha=0.96,
        wear=float(rng.uniform(0.2, 0.7)),
        lane_hint=float(rng.uniform(0.0, 0.25)),
        texture_scale_px=float(rng.uniform(8, 14)),
    )
    sp = SettlementParams(
        alpha=0.65, impervious=float(rng.uniform(0.5, 0.95)), grime=float(rng.uniform(0.1, 0.6))
    )
    bp = BuildingParams(
        alpha=0.98,
        roof_variation=float(rng.uniform(0.4, 1.0)),
        shadow_strength=float(rng.uniform(0.1, 0.35)),
    )

    # 4) Layers
    layers = [
        water_palette(masks.get("water", np.zeros((H, W), bool)), scene, wp, rng),
        vegetation_palette(masks.get("veg", np.zeros((H, W), bool)), scene, vp, rng),
        settlement_palette(masks.get("settlement", np.zeros((H, W), bool)), scene, sp, rng),
        roads_palette(masks.get("roads", np.zeros((H, W), bool)), scene, rp, rng),
    ]

    for b in building_items:
        b_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1, dtype=np.int64)))
        layers.append(building_palette(b["mask"], b["settlement_id"], scene, bp, b_rng))

    rgb = render_full_rgb(bg, layers)

    # 5) Shadows from union of all buildings (cheap + effective)
    all_b = np.zeros((H, W), dtype=bool)
    for b in building_items:
        all_b |= b["mask"]
    rgb = add_building_shadows(
        rgb,
        all_b,
        sun_azimuth_deg=scene.sun_azimuth_deg,
        strength=bp.shadow_strength,
        length_px=5,
    )

    # 6) Postprocess
    out_u8 = postprocess(
        rgb,
        exposure=scene.exposure,
        gamma=scene.gamma,
        saturation=scene.saturation,
        grain=0.012,
        blur_k=3,
        rng=rng,
    )
    return Image.fromarray(out_u8, mode="RGB")


def render_world_textured_with_mask(
    world: WorldState, rng: np.random.Generator
) -> tuple[Image.Image, MaskLayers]:
    W, H = world.tr.width_px, world.tr.height_px

    elev = (
        world.terrain.elevation_m.astype(np.float32)
        if world.terrain is not None
        else np.zeros((H, W), np.float32)
    )

    scene = SceneParams(
        biome="temperate",
        sun_azimuth_deg=float(rng.uniform(0, 360)),
        sun_altitude_deg=float(rng.uniform(20, 60)),
        sun_intensity=float(rng.uniform(0.8, 1.2)),
        exposure=float(rng.uniform(0.9, 1.1)),
        gamma=float(rng.uniform(0.95, 1.15)),
        haze=float(rng.uniform(0.0, 0.08)),
        saturation=float(rng.uniform(0.9, 1.15)),
    )

    # 1) Build masks (bool arrays) using PIL drawing (recommended for now)
    mask_build_res = build_masks_from_world(
        world
    )  # you implement using your polygon->px ring helper
    masks = mask_build_res.masks
    building_items = mask_build_res.building_items
    # settlement_polys = mask_build_res.settlement_polys  # for debug / visualization if

    # 2) Background
    bg = background_palette_from_elevation(elev, scene, rng)

    # 3) Params
    wp = WaterParams(
        alpha=0.88,
        turbidity=float(rng.uniform(0.1, 0.6)),
        specular=float(rng.uniform(0.0, 0.35)),
        shoreline_width_px=3,
    )
    vp = VegetationParams(
        alpha=0.85,
        density=float(rng.uniform(0.95, 1.0)),  # need to be 1 veg for masks to be interesting
        patchiness=float(rng.uniform(0.4, 0.9)),
        texture_scale_px=float(rng.uniform(10, 22)),
    )
    rp = RoadParams(
        alpha=0.96,
        wear=float(rng.uniform(0.2, 0.7)),
        lane_hint=float(rng.uniform(0.0, 0.25)),
        texture_scale_px=float(rng.uniform(8, 14)),
    )
    sp = SettlementParams(
        alpha=0.65, impervious=float(rng.uniform(0.5, 0.95)), grime=float(rng.uniform(0.1, 0.6))
    )
    bp = BuildingParams(
        alpha=0.98,
        roof_variation=float(rng.uniform(0.4, 1.0)),
        shadow_strength=float(rng.uniform(0.1, 0.35)),
    )

    # 4) Layers
    layers = [
        water_palette(masks.get("water", np.zeros((H, W), bool)), scene, wp, rng),
        vegetation_palette(masks.get("veg", np.zeros((H, W), bool)), scene, vp, rng),
        settlement_palette(masks.get("settlement", np.zeros((H, W), bool)), scene, sp, rng),
        roads_palette(masks.get("roads", np.zeros((H, W), bool)), scene, rp, rng),
    ]

    for b in building_items:
        b_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1, dtype=np.int64)))
        layers.append(building_palette(b["mask"], b["settlement_id"], scene, bp, b_rng))

    rgb = render_full_rgb(bg, layers)

    # 5) Shadows from union of all buildings (cheap + effective)
    all_b = np.zeros((H, W), dtype=bool)
    for b in building_items:
        all_b |= b["mask"]
    rgb = add_building_shadows(
        rgb,
        all_b,
        sun_azimuth_deg=scene.sun_azimuth_deg,
        strength=bp.shadow_strength,
        length_px=5,
    )

    # 6) Postprocess
    out_u8 = postprocess(
        rgb,
        exposure=scene.exposure,
        gamma=scene.gamma,
        saturation=scene.saturation,
        grain=0.012,
        blur_k=3,
        rng=rng,
    )
    return (
        Image.fromarray(out_u8, mode="RGB"),
        mask_build_res,
    )


def mask_layers_to_mask_image(mask_build_res: MaskLayers) -> Image.Image:
    # convert the dict of bool masks to a single PIL image with different values for each class
    # (for visualization / debug purposes, not used in rendering)
    masks = mask_build_res.masks
    W, H = masks["water"].shape[1], masks["water"].shape[0]
    out = np.zeros((H, W), dtype=np.uint8)
    class_to_val = {
        "background": 0,
        "water": 1,
        "veg": 2,
        "settlement": 3,
        "roads": 4,
        "building": 5,
    }
    for cls, mask in masks.items():
        val = class_to_val.get(cls, 255)
        out[mask] = val
    # also add buildings (overwrite any other class, since they are on top)
    for b in mask_build_res.building_items:
        out[b["mask"]] = class_to_val["building"]
    return Image.fromarray(out, mode="L")
