from __future__ import annotations

from shapely.geometry import Point
from shapely.ops import unary_union

from geosynthbench.world.types import RuleType, Violation
from geosynthbench.world.world_state import WorldState


def validate_world(world: WorldState, *, require_connected_roads: bool = True, forbid_roads_in_water: bool = True,
                   max_slope_settlement: float = 0.25, max_slope_road: float = 0.35, max_slope_building: float = 0.20,
                   min_dist_settlements_m: float = 500.0, min_dist_buildings_m: float = 6.0) -> list[Violation]:
    """
    Implements 15 elemental rules. HARD ones must be 0 for acceptance.
    """
    V: list[Violation] = []
    extent = world.extent_polygon()
    water_u = world.water_union()

    # --- R1 HARD: everything in extent (check per entity)
    def _check_in_extent(geom, eid: str, code: str) -> None:
        if geom.is_empty:
            V.append(Violation(code=code, rule_type=RuleType.HARD, severity="ERROR",
                               message="Geometry is empty", entity_id=eid))
            return
        if not geom.is_valid:
            V.append(Violation(code=code, rule_type=RuleType.HARD, severity="ERROR",
                               message="Geometry is invalid", entity_id=eid))
            return
        if not extent.contains(geom):
            V.append(Violation(code=code, rule_type=RuleType.HARD, severity="ERROR",
                               message="Geometry outside extent", entity_id=eid))

    # --- R2 HARD: valid geometries (merged into check above by is_valid + non-empty)

    # Water checks
    for w in world.water:
        _check_in_extent(w.polygon, str(w.id), "R1_WATER_IN_EXTENT")

    # --- R3 HARD: water components should not overlap (if overlaps exist => violation)
    for i in range(len(world.water)):
        for j in range(i + 1, len(world.water)):
            if world.water[i].polygon.intersects(world.water[j].polygon):
                inter = world.water[i].polygon.intersection(world.water[j].polygon)
                if not inter.is_empty and inter.area > 1e-6:
                    V.append(Violation(
                        code="R3_WATER_NO_OVERLAP",
                        rule_type=RuleType.HARD,
                        severity="ERROR",
                        message="Water bodies overlap (consider union/merge).",
                        entity_id=f"{world.water[i].id},{world.water[j].id}",
                    ))

    # Vegetation checks
    for v in world.vegetation:
        _check_in_extent(v.polygon, str(v.id), "R1_VEG_IN_EXTENT")

    # --- R5 HARD: vegetation must not cover water
    if world.vegetation and not water_u.is_empty:
        veg_u = world.vegetation_union()
        if veg_u.intersects(water_u):
            inter = veg_u.intersection(water_u)
            if not inter.is_empty and inter.area > 1e-6:
                V.append(Violation(
                    code="R5_VEG_NOT_OVER_WATER",
                    rule_type=RuleType.HARD,
                    severity="ERROR",
                    message="Vegetation overlaps water.",
                    entity_id=None,
                ))

    # Settlements checks
    for s in world.settlements:
        # --- R7 HARD: settlement center not in water
        if not water_u.is_empty and water_u.contains(s.center):
            V.append(Violation(
                code="R7_SETTLEMENT_NOT_IN_WATER",
                rule_type=RuleType.HARD,
                severity="ERROR",
                message="Settlement center in water.",
                entity_id=str(s.id),
            ))

        # --- R9 HARD: settlement slope acceptable (if terrain)
        if world.terrain is not None:
            sl = world.terrain.sample_slope_point(s.center.x, s.center.y)
            if sl > max_slope_settlement:
                V.append(Violation(
                    code="R9_SETTLEMENT_SLOPE_MAX",
                    rule_type=RuleType.HARD,
                    severity="ERROR",
                    message=f"Settlement slope too high: {sl:.3f} > {max_slope_settlement:.3f}",
                    entity_id=str(s.id),
                ))

    # --- R8 HARD: settlement min distance
    for i in range(len(world.settlements)):
        for j in range(i + 1, len(world.settlements)):
            d = world.settlements[i].center.distance(world.settlements[j].center)
            if d < min_dist_settlements_m:
                V.append(Violation(
                    code="R8_SETTLEMENT_MIN_DIST",
                    rule_type=RuleType.HARD,
                    severity="ERROR",
                    message=f"Settlements too close: {d:.1f}m < {min_dist_settlements_m:.1f}m",
                    entity_id=f"{world.settlements[i].id},{world.settlements[j].id}",
                ))

    # Roads checks
    for seg in world.roads.segments:
        _check_in_extent(seg.centerline, str(seg.id), "R1_ROAD_IN_EXTENT")

        # --- R12 HARD: roads not in water (buffered)
        if forbid_roads_in_water and not water_u.is_empty:
            road_poly = seg.centerline.buffer(seg.width_m / 2.0, cap_style=2, join_style=2)
            if road_poly.intersects(water_u):
                inter = road_poly.intersection(water_u)
                if not inter.is_empty and inter.area > 1e-6:
                    V.append(Violation(
                        code="R12_ROAD_NOT_IN_WATER",
                        rule_type=RuleType.HARD,
                        severity="ERROR",
                        message="Road intersects water.",
                        entity_id=str(seg.id),
                    ))

        # --- R13 HARD: slope max along roads (sample points)
        if world.terrain is not None:
            # sample a few points along the line
            L = float(seg.centerline.length)
            n = max(10, min(40, int(L / 80.0)))
            for k in range(n + 1):
                t = k / float(n)
                p = seg.centerline.interpolate(t, normalized=True)
                sl = world.terrain.sample_slope_point(p.x, p.y)
                if sl > max_slope_road:
                    V.append(Violation(
                        code="R13_ROAD_SLOPE_MAX",
                        rule_type=RuleType.HARD,
                        severity="ERROR",
                        message=f"Road slope too high at sample: {sl:.3f} > {max_slope_road:.3f}",
                        entity_id=str(seg.id),
                    ))
                    break

    # --- R11 HARD: road network connectivity
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
                    V.append(Violation(
                        code="R11_ROADS_CONNECTED",
                        rule_type=RuleType.HARD,
                        severity="ERROR",
                        message=f"Some settlements disconnected from road network: {missing}",
                        entity_id=None,
                    ))
            else:
                V.append(Violation(
                    code="R11_ROADS_CONNECTED",
                    rule_type=RuleType.HARD,
                    severity="ERROR",
                    message="Road graph has no connected components.",
                    entity_id=None,
                ))
        else:
            V.append(Violation(
                code="R11_ROADS_CONNECTED",
                rule_type=RuleType.HARD,
                severity="ERROR",
                message="Road graph empty.",
                entity_id=None,
            ))

    # Buildings checks
    for b in world.buildings:
        _check_in_extent(b.footprint, str(b.id), "R1_BUILDING_IN_EXTENT")

        # --- R15 HARD: buildings not in water
        if not water_u.is_empty and b.footprint.intersects(water_u):
            V.append(Violation(
                code="R15_BUILDING_NOT_IN_WATER",
                rule_type=RuleType.HARD,
                severity="ERROR",
                message="Building intersects water.",
                entity_id=str(b.id),
            ))

        # --- R15 HARD: building slope max
        if world.terrain is not None:
            stats = world.terrain.poly_stats(b.footprint, max_points=512)
            if stats["slope_max"] > max_slope_building:
                V.append(Violation(
                    code="R15_BUILDING_SLOPE_MAX",
                    rule_type=RuleType.HARD,
                    severity="ERROR",
                    message=f"Building slope too high: {stats['slope_max']:.3f} > {max_slope_building:.3f}",
                    entity_id=str(b.id),
                ))

    # --- R15 HARD: building collision / min distance
    for i in range(len(world.buildings)):
        bi = world.buildings[i].footprint
        for j in range(i + 1, len(world.buildings)):
            bj = world.buildings[j].footprint
            if bi.intersects(bj):
                inter = bi.intersection(bj)
                if not inter.is_empty and inter.area > 1e-6:
                    V.append(Violation(
                        code="R15_BUILDING_NO_OVERLAP",
                        rule_type=RuleType.HARD,
                        severity="ERROR",
                        message="Buildings overlap.",
                        entity_id=f"{world.buildings[i].id},{world.buildings[j].id}",
                    ))
            if bi.distance(bj) < min_dist_buildings_m:
                V.append(Violation(
                    code="R15_BUILDING_MIN_DIST",
                    rule_type=RuleType.HARD,
                    severity="ERROR",
                    message=f"Buildings too close (< {min_dist_buildings_m}m).",
                    entity_id=f"{world.buildings[i].id},{world.buildings[j].id}",
                ))

    # --- SOFT rules (reported as WARN, not blocking)
    # R6 SOFT: veg overlaps settlement influence
    if world.vegetation and world.settlements:
        from shapely.geometry import Point as _Point
        buffers = [s.center.buffer(s.radius_m * 1.2) for s in world.settlements]
        inf = unary_union(buffers)
        veg_u = world.vegetation_union()
        if not veg_u.is_empty and veg_u.intersects(inf):
            area = float(veg_u.intersection(inf).area)
            if area > 1.0:
                V.append(Violation(
                    code="R6_VEG_AVOID_URBAN",
                    rule_type=RuleType.SOFT,
                    severity="WARN",
                    message=f"Vegetation overlaps settlement influence area={area:.1f}.",
                    entity_id=None,
                ))

    # R10 SOFT: settlements near water band (if enabled by caller in pipeline scoring)
    # R14 SOFT: road total length reasonable (not enforced here)

    return V


def hard_violations(violations: list[Violation]) -> list[Violation]:
    return [v for v in violations if v.rule_type == RuleType.HARD and v.severity == "ERROR"]
