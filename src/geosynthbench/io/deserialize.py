from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
from shapely import wkt
from shapely.geometry import LineString, Point, Polygon

from geosynthbench.io.world_schema import WorldModel
from geosynthbench.world.entities import (
    Building,
    RoadSegment,
    Settlement,
    VegetationPatch,
    WaterBody,
)
from geosynthbench.world.raster import HeightField, RasterTransform
from geosynthbench.world.types import BuildingId, RoadId, SettlementId, VegId, WaterId
from geosynthbench.world.world_state import WorldState


def world_from_jsonl(raw_jsonl: dict[str, Any], base_dir: Path) -> WorldState:
    # validate and coerce types with pydantic
    m = WorldModel.model_validate(raw_jsonl)  # validated, coerced types

    xmin, ymin, xmax, ymax = m.raster.extent
    tr = RasterTransform(
        extent=(xmin, ymin, xmax, ymax),
        width_px=m.raster.width_px,
        height_px=m.raster.height_px,
        # dx/dy if your RasterTransform stores them
    )
    world = WorldState(tr=tr)

    # Terrain (optional) — loads from elevation_path sidecar if present
    if m.terrain and m.terrain.type == "heightfield":
        elev_path = _as_path(base_dir, m.terrain.elevation_path)
        if elev_path is not None and elev_path.exists():
            elev = np.load(elev_path).astype(np.float32, copy=False)
            world.terrain = HeightField(tr=tr, elevation_m=elev)
        else:
            world.terrain = None

    # Geometry back from WKT
    for w_ in m.water:
        poly = cast(Polygon, wkt.loads(w_.polygon_wkt))
        # create Water entity, append to world.water
        world.water.append(WaterBody(id=WaterId(str(w_.id)), polygon=poly))

    for v_ in m.vegetation:
        poly = cast(Polygon, wkt.loads(v_.polygon_wkt))
        world.vegetation.append(
            VegetationPatch(
                id=VegId(str(v_.id)),
                polygon=poly,
                density=float(v_.density),
            )
        )

    for s_ in m.settlements:
        center = cast(Point, wkt.loads(s_.center_wkt))
        world.settlements.append(
            Settlement(
                id=SettlementId(str(s_.id)),
                center=center,
                radius_m=float(s_.radius_m),
            )
        )

    segs: list[RoadSegment] = []
    for r_ in m.roads.segments:
        line = cast(LineString, wkt.loads(r_.centerline_wkt))
        segs.append(
            RoadSegment(
                id=RoadId(str(r_.id)),
                a_id=SettlementId(str(r_.a_id)),
                b_id=SettlementId(str(r_.b_id)),
                centerline=line,
                width_m=float(r_.width_m),
            )
        )
    world.roads.segments = segs
    world.roads.rebuild_graph()  # rebuild from segments (edges in JSONL are redundant)

    for b_ in m.buildings:
        fp = cast(Polygon, wkt.loads(b_.footprint_wkt))
        world.buildings.append(
            Building(
                id=BuildingId(str(b_.id)),
                settlement_id=SettlementId(str(b_.settlement_id)),
                footprint=fp,
                near_road_id=RoadId(str(b_.near_road_id)) if b_.near_road_id is not None else None,
            )
        )

    _ = world.extent_polygon()  # ensure extent is cached for later geometry checks
    return world


def _as_path(base_dir: Path, maybe_rel: str | None) -> Path | None:
    if maybe_rel is None:
        return None
    p = Path(maybe_rel)
    if not p.is_absolute():
        return base_dir / p
    return p
