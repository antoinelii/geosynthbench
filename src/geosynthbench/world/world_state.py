from __future__ import annotations

from dataclasses import dataclass, field

from shapely.geometry import MultiPolygon, Polygon
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

    def settlement_by_id(self, id: SettlementId) -> Settlement:
        for s in self.settlements:
            if s.id == id:
                return s
        raise ValueError(f"Settlement with id {id} not found")

    def get_settlement_polygon(
        self, settlement_id: SettlementId, settlement_mode: str = "hull_then_circle"
    ) -> Polygon:  # type: ignore
        """settlement_mode:
        - "hull_only": settlement polygon = convex hull of union of buildings (if none -> empty)
        - "circle_only": settlement polygon = circle from settlement.center + settlement.radius_m
        - "hull_then_circle": hull if buildings exist else circle
        """
        s = self.settlement_by_id(settlement_id)
        by_sett: dict[SettlementId, list[Polygon]] = {}
        for b in self.buildings:
            s_id = getattr(b, "settlement_id")
            by_sett.setdefault(s_id, []).append(b.footprint)

        for s in getattr(self, "settlements", []):
            s_id = getattr(s, "id")
            polys = by_sett.get(s_id, [])

            poly: Polygon
            if settlement_mode == "circle_only":
                geom = s.center.buffer(float(s.radius_m), resolution=64)
                poly = _build_convex_hull(geom)
            elif settlement_mode == "hull_only":
                if polys:
                    u = unary_union(polys)
                    poly = _build_convex_hull(u)
                else:
                    poly = Polygon()
            else:  # "hull_then_circle"
                if polys:
                    u = unary_union(polys)
                    poly = _build_convex_hull(u)
                else:
                    geom = s.center.buffer(float(s.radius_m), resolution=64)
                    poly = _build_convex_hull(geom)
            return poly


def _build_convex_hull(geom: BaseGeometry) -> Polygon:
    """Best-effort conversion of the shapely retrieved convex hull to a Polygon for strict typing.
    Given the input geometries are building footprints, the convex hull should normally
    be a Polygon, but we handle edge cases."""
    if geom.is_empty:
        return Polygon()

    # Convex hull can be Polygon, LineString, Point, etc
    # Here normally a Polygon object
    hull = geom.convex_hull
    if hull.is_empty:
        return Polygon()

    if isinstance(hull, Polygon):
        return hull
    if isinstance(hull, MultiPolygon):
        # convex_hull usually isn't MultiPolygon, but being defensive is fine
        # pick largest piece if it ever happens
        return max(hull.geoms, key=lambda p: p.area)

    # If hull is a LineString/Point (e.g., 1-2 buildings), return empty polygon
    return Polygon()
