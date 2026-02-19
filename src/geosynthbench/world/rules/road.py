from shapely.geometry.base import BaseGeometry
from shapely.geometry.polygon import Polygon

from geosynthbench.world.rules import r1_check_in_extent
from geosynthbench.world.types import RuleType, Violation
from geosynthbench.world.world_state import WorldState


### ROADS checks
def r1_check_roads_in_extent(world: WorldState, extent: Polygon) -> list[Violation]:
    V: list[Violation] = []
    for seg in world.roads.segments:
        seg = r1_check_in_extent(seg.centerline, extent, str(seg.id), "R1_ROAD_IN_EXTENT")
        if isinstance(seg, Violation):
            V.append(seg)
    return V


def r12_check_road_not_in_water(
    world: WorldState,
    water_u: BaseGeometry,
    forbid_roads_in_water: bool = True,
) -> list[Violation]:
    V: list[Violation] = []
    if forbid_roads_in_water and not water_u.is_empty:
        for seg in world.roads.segments:
            # --- R12 HARD: roads not in water (buffered)
            road_poly = seg.centerline.buffer(
                seg.width_m / 2.0, cap_style="round", join_style="mitre"
            )
            if road_poly.intersects(water_u):
                inter = road_poly.intersection(water_u)
                if not inter.is_empty and inter.area > 1e-6:
                    V.append(
                        Violation(
                            code="R12_ROAD_NOT_IN_WATER",
                            rule_type=RuleType.HARD,
                            severity="ERROR",
                            message="Road intersects water.",
                            entity_id=str(seg.id),
                        )
                    )
    return V


def r13_check_road_slope(world: WorldState, max_slope_road: float = 0.35) -> list[Violation]:
    V: list[Violation] = []
    if world.terrain is not None:
        for seg in world.roads.segments:
            # --- R13 HARD: slope max along roads (sample points)
            # sample a few points along the line
            L = float(seg.centerline.length)
            n = max(10, min(40, int(L / 80.0)))
            for k in range(n + 1):
                t = k / float(n)
                p = seg.centerline.interpolate(t, normalized=True)
                sl = world.terrain.sample_slope_point(p.x, p.y)
                if sl > max_slope_road:
                    V.append(
                        Violation(
                            code="R13_ROAD_SLOPE_MAX",
                            rule_type=RuleType.HARD,
                            severity="ERROR",
                            message=f"Road slope too high at sample: {sl:.3f} > {max_slope_road:.3f}",
                            entity_id=str(seg.id),
                        )
                    )
                    break
    return V


def r11_check_road_connected(
    world: WorldState, require_connected_roads: bool = True
) -> list[Violation]:
    V: list[Violation] = []
    if require_connected_roads and len(world.settlements) >= 2:
        # build graph is expected; still handle empty
        g = world.roads.graph
        if g.number_of_nodes() > 0:
            # all settlement ids should be in one component
            comps = list(__import__("networkx").connected_components(g))
            if comps:
                largest = max(comps, key=len)
                missing = [str(s.id) for s in world.settlements if s.id not in largest]
                if missing:
                    V.append(
                        Violation(
                            code="R11_ROADS_CONNECTED",
                            rule_type=RuleType.HARD,
                            severity="ERROR",
                            message=f"Some settlements disconnected from road network: {missing}",
                            entity_id=None,
                        )
                    )
            else:
                V.append(
                    Violation(
                        code="R11_ROADS_CONNECTED",
                        rule_type=RuleType.HARD,
                        severity="ERROR",
                        message="Road graph has no connected components.",
                        entity_id=None,
                    )
                )
        else:
            V.append(
                Violation(
                    code="R11_ROADS_CONNECTED",
                    rule_type=RuleType.HARD,
                    severity="ERROR",
                    message="Road graph empty.",
                    entity_id=None,
                )
            )
    return V
