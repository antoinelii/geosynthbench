### WATER checks
from shapely.geometry.polygon import Polygon

from geosynthbench.world.rules import r1_check_in_extent
from geosynthbench.world.types import RuleType, Violation
from geosynthbench.world.world_state import WorldState


def r2_check_waters_in_extent(world: WorldState, extent: Polygon) -> list[Violation]:
    V: list[Violation] = []
    for w in world.water:
        v = r1_check_in_extent(w.polygon, extent, str(w.id), "R2_WATER_IN_EXTENT")
        if isinstance(v, Violation):
            V.append(v)
    return V


def r3_check_no_water_overlap(world: WorldState) -> Violation | None:
    for i in range(len(world.water)):
        for j in range(i + 1, len(world.water)):
            if world.water[i].polygon.intersects(world.water[j].polygon):
                inter = world.water[i].polygon.intersection(world.water[j].polygon)
                if not inter.is_empty and inter.area > 1e-6:
                    return Violation(
                        code="R3_WATER_NO_OVERLAP",
                        rule_type=RuleType.HARD,
                        severity="ERROR",
                        message="Water bodies overlap (consider union/merge).",
                        entity_id=f"{world.water[i].id},{world.water[j].id}",
                    )
