from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import TypedDict

import networkx as nx
import numpy as np
import numpy.typing as npt
from shapely.geometry import LineString, Point, Polygon

from geosynthbench.world.types import (
    BuildingId,
    RoadId,
    SettlementId,
    VegId,
    WaterId,
)


# Create entitities enum
class EntityType(IntEnum):
    BG = 0
    WATER = 1
    VEG = 2
    ROAD = 3
    SETTLEMENT = 4
    BUILDING = 5


@dataclass(frozen=True)
class WaterBody:
    id: WaterId
    polygon: Polygon


@dataclass(frozen=True)
class VegetationPatch:
    id: VegId
    polygon: Polygon
    density: float = 1.0


@dataclass(frozen=True)
class Settlement:
    id: SettlementId
    center: Point
    radius_m: float


@dataclass(frozen=True)
class RoadSegment:
    id: RoadId
    a_id: SettlementId
    b_id: SettlementId
    centerline: LineString
    width_m: float = 8.0


@dataclass
class RoadNetwork:
    segments: list[RoadSegment] = field(default_factory=lambda: [])
    graph: nx.Graph[SettlementId] = field(default_factory=lambda: nx.Graph(), init=False)

    def rebuild_graph(self) -> None:
        g: nx.Graph[SettlementId] = nx.Graph()
        for seg in self.segments:
            g.add_node(seg.a_id)
            g.add_node(seg.b_id)
            # store edge attributes
            g.add_edge(seg.a_id, seg.b_id, road_id=seg.id, length=float(seg.centerline.length))
        self.graph = g


@dataclass(frozen=True)
class Building:
    id: BuildingId
    settlement_id: SettlementId
    footprint: Polygon
    near_road_id: RoadId | None = None


@dataclass(frozen=True)
class BuildingMaskItem(TypedDict):
    id: BuildingId
    settlement_id: SettlementId
    mask: npt.NDArray[np.bool_]  # bool[H,W]
