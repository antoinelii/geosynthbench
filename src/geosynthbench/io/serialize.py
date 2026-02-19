from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np
from shapely.geometry.base import BaseGeometry

from geosynthbench.world.world_state import WorldState


def geom_to_wkt(g: BaseGeometry | None) -> str | None:
    if g is None:
        return None
    return g.wkt


def _jsonify(obj: Any) -> Any:
    """
    Convert common non-JSON objects into JSON-friendly types.
    - Shapely geometry -> WKT
    - numpy scalars -> python scalars
    - dataclasses -> dict recursively
    """
    if obj is None:
        return None

    if isinstance(obj, BaseGeometry):
        return obj.wkt

    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)

    if is_dataclass(obj):
        d = asdict(obj)
        return {k: _jsonify(v) for k, v in d.items()}

    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_jsonify(x) for x in obj]

    return obj


def world_to_dict(world: WorldState, *, include_terrain_ref: str | None = None) -> dict[str, Any]:
    """
    Serialize WorldState to a JSON-friendly dict.
    Geometry is encoded as WKT strings.
    Terrain is referenced by file path (sidecar), not embedded.
    """
    out: dict[str, Any] = {
        "version": "0.1",
        "raster": {
            "extent": list(world.tr.extent),
            "width_px": world.tr.width_px,
            "height_px": world.tr.height_px,
            "dx": world.tr.dx,
            "dy": world.tr.dy,
        },
        "terrain": None,
        "water": [],
        "vegetation": [],
        "settlements": [],
        "roads": {
            "segments": [],
            "edges": [],
        },
        "buildings": [],
    }

    if world.terrain is not None:
        out["terrain"] = {
            "type": "heightfield",
            "elevation_path": include_terrain_ref,  # may be None
            "stats": {
                "min": float(np.min(world.terrain.elevation_m)),
                "max": float(np.max(world.terrain.elevation_m)),
                "mean": float(np.mean(world.terrain.elevation_m)),
            },
        }

    for w in world.water:
        out["water"].append({"id": str(w.id), "polygon_wkt": w.polygon.wkt})

    for v in world.vegetation:
        out["vegetation"].append(
            {"id": str(v.id), "polygon_wkt": v.polygon.wkt, "density": float(v.density)}
        )

    for s in world.settlements:
        out["settlements"].append(
            {
                "id": str(s.id),
                "center_wkt": s.center.wkt,
                "radius_m": float(s.radius_m),
            }
        )

    # road segments
    for seg in world.roads.segments:
        out["roads"]["segments"].append(
            {
                "id": str(seg.id),
                "a_id": str(seg.a_id),
                "b_id": str(seg.b_id),
                "centerline_wkt": seg.centerline.wkt,
                "width_m": float(seg.width_m),
                "length_m": float(seg.centerline.length),
            }
        )

    # road edges derived from graph
    g = world.roads.graph
    for u, v, data in g.edges(data=True):
        out["roads"]["edges"].append(
            {
                "a_id": str(u),
                "b_id": str(v),
                "road_id": str(data.get("road_id", "")),
                "length_m": float(data.get("length", 0.0)),
            }
        )

    for b in world.buildings:
        out["buildings"].append(
            {
                "id": str(b.id),
                "settlement_id": str(b.settlement_id),
                "footprint_wkt": b.footprint.wkt,
                "near_road_id": None if b.near_road_id is None else str(b.near_road_id),
                "area_m2": float(b.footprint.area),
            }
        )

    return _jsonify(out)
