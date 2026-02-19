from __future__ import annotations

from geosynthbench.world.rules.building import (
    r1_check_buildings_in_extent,
    r15_check_buildings_not_in_water,
    r16_check_building_slope,
    r17_check_building_collision,
)
from geosynthbench.world.rules.road import (
    r1_check_roads_in_extent,
    r11_check_road_connected,
    r12_check_road_not_in_water,
    r13_check_road_slope,
)
from geosynthbench.world.rules.settlement import (
    r7_check_settlement_not_in_water,
    r8_check_settlement_min_dist,
    r9_check_settlement_slope,
)
from geosynthbench.world.rules.vegetation import (
    r4_check_vegs_in_extent,
    r5_check_veg_not_over_water,
    s6_check_veg_overlaps_settlement,
)
from geosynthbench.world.rules.water import r2_check_waters_in_extent, r3_check_no_water_overlap
from geosynthbench.world.types import RuleType, Violation
from geosynthbench.world.world_state import WorldState


def validate_world(
    world: WorldState,
    *,
    require_connected_roads: bool = True,
    forbid_roads_in_water: bool = True,
    max_slope_settlement: float = 0.25,
    max_slope_road: float = 0.35,
    max_slope_building: float = 0.20,
    min_dist_settlements_m: float = 500.0,
    min_dist_buildings_m: float = 6.0,
) -> list[Violation]:
    """
    Implements 15 elemental rules. HARD ones must be 0 for acceptance.
    """
    V: list[Violation] = []
    extent = world.extent_polygon()
    water_u = world.water_union()

    # --- R2 HARD: valid geometries (merged into check above by is_valid + non-empty)

    # 1st Water checks (Water precedence over rest)
    V.extend(r2_check_waters_in_extent(world, extent))
    # --- R3 HARD: water components should not overlap (if overlaps exist => violation)
    v = r3_check_no_water_overlap(world)
    if v is not None:
        V.append(v)

    # Vegetation checks
    veg_u = world.vegetation_union()
    # --- R4 HARD: vegetation must be in extent
    V.extend(r4_check_vegs_in_extent(world, extent))
    # --- R5 HARD: vegetation must not cover water
    V.extend(r5_check_veg_not_over_water(veg_u, water_u))

    # Settlements checks
    # --- R7 HARD: settlements must not be in water (check center point, not polygon, for simplicity)
    V.extend(r7_check_settlement_not_in_water(world, water_u))
    # --- R9 HARD: settlement slope acceptable (if terrain)
    V.extend(r9_check_settlement_slope(world, max_slope_settlement))
    # --- R8 HARD: settlement min distance
    V.extend(r8_check_settlement_min_dist(world, min_dist_settlements_m))

    # Roads checks
    # --- R1 HARD: roads must be in extent
    V.extend(r1_check_roads_in_extent(world, extent))
    # --- R12 HARD: roads not in water (buffered by width)
    V.extend(r12_check_road_not_in_water(world, water_u, forbid_roads_in_water))
    # --- R13 HARD: roads slope max
    V.extend(r13_check_road_slope(world, max_slope_road))

    # --- R11 HARD: road network connectivity
    V.extend(r11_check_road_connected(world, require_connected_roads))

    # Buildings checks
    V.extend(r1_check_buildings_in_extent(world, extent))
    V.extend(r15_check_buildings_not_in_water(world, water_u))
    V.extend(r16_check_building_slope(world, max_slope_building))
    V.extend(r17_check_building_collision(world))

    # --- SOFT rules (reported as WARN, not blocking)

    # R6 SOFT: veg overlaps settlement influence
    V.extend(s6_check_veg_overlaps_settlement(world, veg_u))

    # R10 SOFT: settlements near water band (if enabled by caller in pipeline scoring)
    # R14 SOFT: road total length reasonable (not enforced here)

    return V


def hard_violations(violations: list[Violation]) -> list[Violation]:
    return [v for v in violations if v.rule_type == RuleType.HARD and v.severity == "ERROR"]
