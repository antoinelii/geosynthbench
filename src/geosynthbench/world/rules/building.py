### WATER checks
from shapely.geometry.base import BaseGeometry
from shapely.geometry.polygon import Polygon

from geosynthbench.world.rules import r1_check_in_extent
from geosynthbench.world.types import RuleType, Violation
from geosynthbench.world.world_state import WorldState

### Buildings


def r1_check_buildings_in_extent(world: WorldState, extent: Polygon) -> list[Violation]:
    V: list[Violation] = []
    for b in world.buildings:
        v = r1_check_in_extent(b.footprint, extent, str(b.id), "R1_BUILDING_IN_EXTENT")
        if isinstance(v, Violation):
            V.append(v)
    return V


def r15_check_buildings_not_in_water(world: WorldState, water_u: BaseGeometry) -> list[Violation]:
    V: list[Violation] = []
    if not water_u.is_empty:
        for b in world.buildings:
            if b.footprint.intersects(water_u):
                V.append(
                    Violation(
                        code="R15_BUILDING_NOT_IN_WATER",
                        rule_type=RuleType.HARD,
                        severity="ERROR",
                        message="Building intersects water.",
                        entity_id=str(b.id),
                    )
                )
    return V


def r15_check_building_valid(world: WorldState) -> list[Violation]:
    V: list[Violation] = []
    for b in world.buildings:
        if b.footprint.is_empty:
            V.append(
                Violation(
                    code="R15_BUILDING_VALID",
                    rule_type=RuleType.HARD,
                    severity="ERROR",
                    message="Building footprint is empty.",
                    entity_id=str(b.id),
                )
            )
        elif not b.footprint.is_valid:
            V.append(
                Violation(
                    code="R15_BUILDING_VALID",
                    rule_type=RuleType.HARD,
                    severity="ERROR",
                    message="Building footprint is invalid.",
                    entity_id=str(b.id),
                )
            )
        elif not world.extent_polygon().contains(b.footprint):
            V.append(
                Violation(
                    code="R15_BUILDING_VALID",
                    rule_type=RuleType.HARD,
                    severity="ERROR",
                    message="Building footprint outside extent.",
                    entity_id=str(b.id),
                )
            )
    return V


def r16_check_building_slope(
    world: WorldState, max_slope_building: float = 0.20
) -> list[Violation]:
    V: list[Violation] = []
    if world.terrain is not None:
        for b in world.buildings:
            stats = world.terrain.poly_stats(b.footprint, max_points=512)
            if stats["slope_max"] > max_slope_building:
                V.append(
                    Violation(
                        code="R16_BUILDING_SLOPE_MAX",
                        rule_type=RuleType.HARD,
                        severity="ERROR",
                        message=f"Building slope too high: {stats['slope_max']:.3f} > {max_slope_building:.3f}",
                        entity_id=str(b.id),
                    )
                )
    return V


def r17_check_building_collision(world: WorldState) -> list[Violation]:
    V: list[Violation] = []
    for i in range(len(world.buildings)):
        bi = world.buildings[i].footprint
        for j in range(i + 1, len(world.buildings)):
            bj = world.buildings[j].footprint
            if bi.intersects(bj):
                inter = bi.intersection(bj)
                if not inter.is_empty and inter.area > 1e-6:
                    V.append(
                        Violation(
                            code="R17_BUILDING_NO_OVERLAP",
                            rule_type=RuleType.HARD,
                            severity="ERROR",
                            message="Buildings overlap.",
                            entity_id=f"{world.buildings[i].id},{world.buildings[j].id}",
                        )
                    )
    return V
