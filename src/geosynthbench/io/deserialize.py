from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from shapely import wkt
from shapely.geometry import LineString, Point, Polygon

from geosynthbench.world.entities import (
    Building,
    RoadNetwork,
    RoadSegment,
    Settlement,
    VegetationPatch,
    WaterBody,
)
from geosynthbench.world.raster import HeightField, RasterTransform
from geosynthbench.world.types import BuildingId, RoadId, SettlementId, VegId, WaterId
from geosynthbench.world.world_state import WorldState


def _as_path(base_dir: Path, maybe_rel: str | None) -> Path | None:
    if maybe_rel is None:
        return None
    p = Path(maybe_rel)
    if not p.is_absolute():
        return base_dir / p
    return p


def world_from_dict(t0: dict[str, Any], *, base_dir: Path) -> WorldState:
    """
    Inverse of world_to_dict() (WKT -> shapely, terrain sidecar -> numpy).
    """
    raster = t0["raster"]
    tr = RasterTransform(
        extent=tuple(raster["extent"]),
        width_px=int(raster["width_px"]),
        height_px=int(raster["height_px"]),
    )

    world = WorldState(tr=tr)

    # Terrain (optional) — loads from elevation_path sidecar if present
    terrain = t0.get("terrain")
    if terrain and terrain.get("type") == "heightfield":
        elev_path = _as_path(base_dir, terrain.get("elevation_path"))
        if elev_path is not None and elev_path.exists():
            elev = np.load(elev_path).astype(np.float32, copy=False)
            world.terrain = HeightField(tr=tr, elevation_m=elev)
        else:
            world.terrain = None

    # Water
    water_list = []
    for wrec in t0.get("water", []):
        poly = wkt.loads(wrec["polygon_wkt"])
        water_list.append(WaterBody(id=WaterId(str(wrec["id"])), polygon=poly))
    world.water = water_list

    # Vegetation
    veg_list = []
    for vrec in t0.get("vegetation", []):
        poly = wkt.loads(vrec["polygon_wkt"])
        veg_list.append(
            VegetationPatch(
                id=VegId(str(vrec["id"])),
                polygon=poly,
                density=float(vrec.get("density", 1.0)),
            )
        )
    world.vegetation = veg_list

    # Settlements
    sett_list = []
    for srec in t0.get("settlements", []):
        center = wkt.loads(srec["center_wkt"])
        if not isinstance(center, Point):
            raise TypeError(f"Settlement center WKT is not a Point: {srec['center_wkt']}")
        sett_list.append(
            Settlement(
                id=SettlementId(str(srec["id"])),
                center=center,
                radius_m=float(srec["radius_m"]),
            )
        )
    world.settlements = sett_list

    # Roads
    segs = []
    roads = t0.get("roads", {})
    for rrec in roads.get("segments", []):
        line = wkt.loads(rrec["centerline_wkt"])
        if not isinstance(line, LineString):
            raise TypeError(f"Road centerline WKT is not a LineString: {rrec['centerline_wkt']}")
        segs.append(
            RoadSegment(
                id=RoadId(str(rrec["id"])),
                a_id=SettlementId(str(rrec["a_id"])),
                b_id=SettlementId(str(rrec["b_id"])),
                centerline=line,
                width_m=float(rrec.get("width_m", 8.0)),
            )
        )

    rn = RoadNetwork(segments=segs)
    rn.rebuild_graph()  # rebuild from segments (edges in JSONL are redundant)
    world.roads = rn

    # Buildings
    bld_list = []
    for brec in t0.get("buildings", []):
        fp = wkt.loads(brec["footprint_wkt"])
        if not isinstance(fp, Polygon):
            raise TypeError(f"Building footprint WKT is not a Polygon: {brec['footprint_wkt']}")
        near = brec.get("near_road_id")
        bld_list.append(
            Building(
                id=BuildingId(str(brec["id"])),
                settlement_id=SettlementId(str(brec["settlement_id"])),
                footprint=fp,
                near_road_id=None if near is None else RoadId(str(near)),
            )
        )
    world.buildings = bld_list

    return world
