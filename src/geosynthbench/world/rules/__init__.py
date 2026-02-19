from shapely.geometry.base import BaseGeometry
from shapely.geometry.polygon import Polygon

from geosynthbench.world.types import RuleType, Violation


def r1_check_in_extent(
    geom: BaseGeometry, extent: Polygon, eid: str, code: str
) -> Violation | None:
    if geom.is_empty:
        return Violation(
            code=code,
            rule_type=RuleType.HARD,
            severity="ERROR",
            message="Geometry is empty",
            entity_id=eid,
        )
    if not geom.is_valid:
        return Violation(
            code=code,
            rule_type=RuleType.HARD,
            severity="ERROR",
            message="Geometry is invalid",
            entity_id=eid,
        )
    if not extent.contains(geom):
        return Violation(
            code=code,
            rule_type=RuleType.HARD,
            severity="ERROR",
            message="Geometry outside extent",
            entity_id=eid,
        )
