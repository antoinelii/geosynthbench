# src/geosynthbench/render/textures/params.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Biome = Literal["temperate", "arid", "tropical", "boreal"]


@dataclass(frozen=True)
class SceneParams:
    biome: Biome
    sun_azimuth_deg: float  # 0..360 (0=north, 90=east)
    sun_altitude_deg: float  # 5..80
    sun_intensity: float  # 0.6..1.4 (multiplies hillshade)
    exposure: float  # 0.85..1.15
    gamma: float  # 0.9..1.2
    haze: float  # 0..0.12 small atmospheric wash
    saturation: float  # 0.85..1.2


@dataclass(frozen=True)
class WaterParams:
    alpha: float  # 0..1 blend over background
    turbidity: float  # 0..1
    specular: float  # 0..1 (fake sun glint)
    shoreline_width_px: int  # 0..8


@dataclass(frozen=True)
class VegetationParams:
    alpha: float
    density: float  # 0..1 user-controlled
    patchiness: float  # 0..1
    texture_scale_px: float  # e.g. 8..32


@dataclass(frozen=True)
class RoadParams:
    alpha: float
    wear: float  # 0..1 cracks / noise
    lane_hint: float  # 0..1 faint center highlight
    texture_scale_px: float  # 6..24


@dataclass(frozen=True)
class SettlementParams:
    alpha: float
    impervious: float  # 0..1 how “concrete” vs soil
    grime: float  # 0..1


@dataclass(frozen=True)
class BuildingParams:
    alpha: float
    roof_variation: float  # 0..1 per-building color jitter
    shadow_strength: float  # 0..1 fake small shadow
