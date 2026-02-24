from __future__ import annotations

from typing import NotRequired, TypedDict


class RasterDict(TypedDict):
    extent: list[float]  # [minx, miny, maxx, maxy]
    width_px: int
    height_px: int
    dx: float
    dy: float
    gsd_m: NotRequired[float]  # optional if you expose it


class TerrainStatsDict(TypedDict):
    min: float
    max: float
    mean: float


class TerrainDict(TypedDict):
    type: str  # e.g. "heightfield"
    elevation_path: str | None  # sidecar reference
    stats: TerrainStatsDict


class WaterDict(TypedDict):
    id: str
    polygon_wkt: str


class VegetationDict(TypedDict):
    id: str
    polygon_wkt: str
    density: float


class SettlementDict(TypedDict):
    id: str
    center_wkt: str
    radius_m: float


class RoadSegDict(TypedDict):
    id: str
    a_id: str
    b_id: str
    centerline_wkt: str
    width_m: float
    length_m: float


class RoadEdgeDict(TypedDict):
    a_id: str
    b_id: str
    road_id: str
    length_m: float


class RoadsDict(TypedDict):
    segments: list[RoadSegDict]
    edges: list[RoadEdgeDict]


class BuildingDict(TypedDict):
    id: str
    settlement_id: str
    footprint_wkt: str
    near_road_id: str | None
    area_m2: float


class WorldDict(TypedDict):
    version: str
    raster: RasterDict
    terrain: TerrainDict | None
    water: list[WaterDict]
    vegetation: list[VegetationDict]
    settlements: list[SettlementDict]
    roads: RoadsDict
    buildings: list[BuildingDict]
