from __future__ import annotations

from dataclasses import dataclass, field

from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from geosynthbench.world.entities import (
    Building,
    RoadNetwork,
    Settlement,
    VegetationPatch,
    WaterBody,
)
from geosynthbench.world.raster import HeightField, RasterTransform
from geosynthbench.world.types import SettlementId


@dataclass
class WorldState:
    tr: RasterTransform
    terrain: HeightField | None = None

    water: list[WaterBody] = field(default_factory=lambda: [])
    vegetation: list[VegetationPatch] = field(default_factory=lambda: [])
    settlements: list[Settlement] = field(default_factory=lambda: [])
    roads: RoadNetwork = field(default_factory=lambda: RoadNetwork())
    buildings: list[Building] = field(default_factory=lambda: [])

    _extent_poly: Polygon | None = None

    def extent_polygon(self) -> Polygon:
        if self._extent_poly is None:
            self._extent_poly = self.tr.extent_polygon()
        return self._extent_poly

    def water_union(self) -> BaseGeometry:
        if not self.water:
            return Polygon()
        return unary_union([w.polygon for w in self.water])

    def vegetation_union(self) -> BaseGeometry:
        if not self.vegetation:
            return Polygon()
        return unary_union([v.polygon for v in self.vegetation])

    def buildings_by_settlement(self) -> dict[SettlementId, list[Building]]:
        out: dict[SettlementId, list[Building]] = {}
        for b in self.buildings:
            out.setdefault(b.settlement_id, []).append(b)
        return out
