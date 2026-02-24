from __future__ import annotations

from geosynthbench.io.world_schema import WorldDict
from geosynthbench.io.world_subserialize import (
    WORLD_SCHEMA_VERSION,
    buildings_to_dict,
    raster_to_dict,
    roads_to_dict,
    settlements_to_dict,
    terrain_to_dict,
    vegetation_to_dict,
    water_to_dict,
)
from geosynthbench.world.world_state import WorldState

JSONScalar = str | int | float | bool | None
JSONValue = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]


def world_to_dict(world: WorldState, *, elevation_path: str | None = None) -> WorldDict:
    """
    Serialize WorldState to a JSON-friendly dict with a stable schema.
    Geometry is encoded as WKT strings.
    Terrain is referenced via 'elevation_path' (sidecar), not embedded.
    """
    return {
        "version": WORLD_SCHEMA_VERSION,
        "raster": raster_to_dict(world),
        "terrain": terrain_to_dict(world, elevation_path=elevation_path),
        "water": water_to_dict(world),
        "vegetation": vegetation_to_dict(world),
        "settlements": settlements_to_dict(world),
        "roads": roads_to_dict(world),
        "buildings": buildings_to_dict(world),
    }
