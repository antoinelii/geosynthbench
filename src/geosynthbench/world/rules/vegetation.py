### WATER checks
from shapely.geometry.base import BaseGeometry
from shapely.geometry.polygon import Polygon
from shapely.ops import unary_union

from geosynthbench.world.rules import r1_check_in_extent
from geosynthbench.world.types import RuleType, Violation
from geosynthbench.world.world_state import WorldState


### Vegetation checks
def r4_check_vegs_in_extent(world: WorldState, extent: Polygon) -> list[Violation]:
    V: list[Violation] = []
    for v in world.vegetation:
        v = r1_check_in_extent(v.polygon, extent, str(v.id), "R4_VEG_IN_EXTENT")
        if isinstance(v, Violation):
            V.append(v)
    return V


def r5_check_veg_not_over_water(veg_u: BaseGeometry, water_u: BaseGeometry) -> list[Violation]:
    V: list[Violation] = []
    if not veg_u.is_empty and not water_u.is_empty:
        if veg_u.intersects(water_u):
            inter = veg_u.intersection(water_u)
            if not inter.is_empty and inter.area > 1e-6:
                V.append(
                    Violation(
                        code="R5_VEG_NOT_OVER_WATER",
                        rule_type=RuleType.HARD,
                        severity="ERROR",
                        message="Vegetation overlaps water.",
                        entity_id=None,
                    )
                )
    return V


def s6_check_veg_overlaps_settlement(world: WorldState, veg_u: BaseGeometry) -> list[Violation]:
    V: list[Violation] = []
    if veg_u.is_empty or not world.settlements:
        return V
    buffers = [s.center.buffer(s.radius_m * 1.2) for s in world.settlements]
    settlements_inf = unary_union(buffers)
    if not veg_u.is_empty and veg_u.intersects(settlements_inf):
        area = float(veg_u.intersection(settlements_inf).area)
        if area > 1.0:
            V.append(
                Violation(
                    code="R6_VEG_AVOID_URBAN",
                    rule_type=RuleType.SOFT,
                    severity="WARN",
                    message=f"Vegetation overlaps settlement influence area={area:.1f}.",
                    entity_id=None,
                )
            )
    return V
