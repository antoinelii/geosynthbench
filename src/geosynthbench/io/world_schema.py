from __future__ import annotations

from typing import Annotated, Literal, NotRequired, TypedDict

from pydantic import BaseModel, ConfigDict, Field


class RasterDict(TypedDict):
    extent: list[float]  # [minx, miny, maxx, maxy]
    width_px: int
    height_px: int
    dx: float
    dy: float
    gsd_m: NotRequired[float]  # optional if you expose it


Extent4 = Annotated[list[float], Field(min_length=4, max_length=4)]


class RasterModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # enforce length 4
    extent: Extent4
    width_px: int
    height_px: int
    dx: float
    dy: float


class TerrainStatsDict(TypedDict):
    min: float
    max: float
    mean: float


class TerrainStatsModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
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


class TerrainModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["heightfield"]
    elevation_path: str | None
    stats: TerrainStatsModel


class WaterModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    polygon_wkt: str


class VegetationModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    polygon_wkt: str
    density: float


class SettlementModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    center_wkt: str
    radius_m: float


class RoadSegModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    a_id: str
    b_id: str
    centerline_wkt: str
    width_m: float
    length_m: float


class RoadEdgeModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    a_id: str
    b_id: str
    road_id: str
    length_m: float


class RoadsModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    segments: list[RoadSegModel]
    edges: list[RoadEdgeModel]


class BuildingModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    settlement_id: str
    footprint_wkt: str
    near_road_id: str | None
    area_m2: float


class WorldModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: str
    raster: RasterModel
    terrain: TerrainModel | None
    water: list[WaterModel] = Field(default_factory=lambda: [])
    vegetation: list[VegetationModel] = Field(default_factory=lambda: [])
    settlements: list[SettlementModel] = Field(default_factory=lambda: [])
    roads: RoadsModel
    buildings: list[BuildingModel] = Field(default_factory=lambda: [])
