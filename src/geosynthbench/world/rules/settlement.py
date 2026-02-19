### WATER checks
from shapely.geometry.base import BaseGeometry

from geosynthbench.world.types import RuleType, Violation
from geosynthbench.world.world_state import WorldState


### Settlements checks
def r7_check_settlement_not_in_water(world: WorldState, water_u: BaseGeometry) -> list[Violation]:
    V: list[Violation] = []
    for s in world.settlements:
        if not water_u.is_empty and water_u.contains(s.center):
            V.append(
                Violation(
                    code="R7_SETTLEMENT_NOT_IN_WATER",
                    rule_type=RuleType.HARD,
                    severity="ERROR",
                    message="Settlement center in water.",
                    entity_id=str(s.id),
                )
            )
    return V


def r8_check_settlement_min_dist(world: WorldState, min_dist_m: float = 500.0) -> list[Violation]:
    V: list[Violation] = []
    for i in range(len(world.settlements)):
        for j in range(i + 1, len(world.settlements)):
            d = world.settlements[i].center.distance(world.settlements[j].center)
            if d < min_dist_m:
                V.append(
                    Violation(
                        code="R8_SETTLEMENT_MIN_DIST",
                        rule_type=RuleType.HARD,
                        severity="ERROR",
                        message=f"Settlements too close: {d:.1f}m < {min_dist_m:.1f}m",
                        entity_id=f"{world.settlements[i].id},{world.settlements[j].id}",
                    )
                )
    return V


def r9_check_settlement_slope(
    world: WorldState, max_slope_settlement: float = 0.25
) -> list[Violation]:
    V: list[Violation] = []
    if world.terrain is not None:
        for s in world.settlements:
            sl = world.terrain.sample_slope_point(s.center.x, s.center.y)
            if sl > max_slope_settlement:
                V.append(
                    Violation(
                        code="R9_SETTLEMENT_SLOPE_MAX",
                        rule_type=RuleType.HARD,
                        severity="ERROR",
                        message=f"Settlement slope too high: {sl:.3f} > {max_slope_settlement:.3f}",
                        entity_id=str(s.id),
                    )
                )
    return V
