# src/geosynthbench/render/textures/__init__.py

from .compose import alpha_over, render_full_rgb
from .noise import fbm_2d, value_noise_2d
from .palettes import (
    building_palette,
    roads_palette,
    settlement_palette,
    vegetation_palette,
    water_palette,
)
from .params import (
    BuildingParams,
    RoadParams,
    SceneParams,
    SettlementParams,
    VegetationParams,
    WaterParams,
)
from .post import postprocess
from .shadows import add_building_shadows
from .terrain import background_palette_from_elevation, hillshade

__all__ = [
    "alpha_over",
    "background_palette_from_elevation",
    "fbm_2d",
    "hillshade",
    "render_full_rgb",
    "value_noise_2d",
    "water_palette",
    "roads_palette",
    "settlement_palette",
    "vegetation_palette",
    "building_palette",
    "add_building_shadows",
    "postprocess",
    "SceneParams",
    "WaterParams",
    "VegetationParams",
    "RoadParams",
    "SettlementParams",
    "BuildingParams",
]
