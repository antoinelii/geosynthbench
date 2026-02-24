from __future__ import annotations

import numpy as np
import numpy.typing as npt

from geosynthbench.io.world_schema import (
    BuildingDict,
    RasterDict,
    RoadEdgeDict,
    RoadsDict,
    RoadSegDict,
    SettlementDict,
    TerrainDict,
    TerrainStatsDict,
    VegetationDict,
    WaterDict,
)
from geosynthbench.world.world_state import WorldState

WORLD_SCHEMA_VERSION = "0.1"


def raster_to_dict(world: WorldState) -> RasterDict:
    tr = world.tr
    out: RasterDict = {
        "extent": [
            float(tr.extent[0]),
            float(tr.extent[1]),
            float(tr.extent[2]),
            float(tr.extent[3]),
        ],
        "width_px": int(tr.width_px),
        "height_px": int(tr.height_px),
        "dx": float(tr.dx),
        "dy": float(tr.dy),
    }
    # If you expose gsd_m on RasterTransform, uncomment:
    # out["gsd_m"] = float(tr.gsd_m)
    return out


def _terrain_stats(elev_m: npt.NDArray[np.floating]) -> TerrainStatsDict:
    # Ensure pure Python floats for JSON
    return {
        "min": float(np.min(elev_m)),
        "max": float(np.max(elev_m)),
        "mean": float(np.mean(elev_m)),
    }


def terrain_to_dict(world: WorldState, *, elevation_path: str | None) -> TerrainDict | None:
    if world.terrain is None:
        return None
    return {
        "type": "heightfield",
        "elevation_path": elevation_path,
        "stats": _terrain_stats(world.terrain.elevation_m),
    }


def water_to_dict(world: WorldState) -> list[WaterDict]:
    out: list[WaterDict] = []
    for w in world.water:
        out.append(
            {
                "id": str(w.id),
                "polygon_wkt": w.polygon.wkt,
            }
        )
    return out


def vegetation_to_dict(world: WorldState) -> list[VegetationDict]:
    out: list[VegetationDict] = []
    for v in world.vegetation:
        out.append(
            {
                "id": str(v.id),
                "polygon_wkt": v.polygon.wkt,
                "density": float(v.density),
            }
        )
    return out


def settlements_to_dict(world: WorldState) -> list[SettlementDict]:
    out: list[SettlementDict] = []
    for s in world.settlements:
        out.append(
            {
                "id": str(s.id),
                "center_wkt": s.center.wkt,
                "radius_m": float(s.radius_m),
            }
        )
    return out


def road_segments_to_dict(world: WorldState) -> list[RoadSegDict]:
    out: list[RoadSegDict] = []
    for seg in world.roads.segments:
        out.append(
            {
                "id": str(seg.id),
                "a_id": str(seg.a_id),
                "b_id": str(seg.b_id),
                "centerline_wkt": seg.centerline.wkt,
                "width_m": float(seg.width_m),
                "length_m": float(seg.centerline.length),
            }
        )
    return out


def road_edges_to_dict(world: WorldState) -> list[RoadEdgeDict]:
    """
    Derived road edges from the graph.

    Note: networkx edge data typing is often dynamic. We keep this contained here.
    """
    out: list[RoadEdgeDict] = []

    g = world.roads.graph  # networkx.Graph or your wrapper
    # If you later wrap graph, expose edges_with_data() and use that instead.
    for u, v, data in g.edges(data=True):
        road_id = data.get("road_id", "")
        length = data.get("length", 0.0)
        out.append(
            {
                "a_id": str(u),
                "b_id": str(v),
                "road_id": str(road_id),
                "length_m": float(length),  # float(...) handles int/np scalar too
            }
        )
    return out


def roads_to_dict(world: WorldState) -> RoadsDict:
    return {
        "segments": road_segments_to_dict(world),
        "edges": road_edges_to_dict(world),
    }


def buildings_to_dict(world: WorldState) -> list[BuildingDict]:
    out: list[BuildingDict] = []
    for b in world.buildings:
        out.append(
            {
                "id": str(b.id),
                "settlement_id": str(b.settlement_id),
                "footprint_wkt": b.footprint.wkt,
                "near_road_id": None if b.near_road_id is None else str(b.near_road_id),
                "area_m2": float(b.footprint.area),
            }
        )
    return out
