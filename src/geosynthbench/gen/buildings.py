from __future__ import annotations

import math

import numpy as np
from shapely.affinity import rotate
from shapely.geometry import Point, Polygon

from geosynthbench.world.entities import Building
from geosynthbench.world.types import BuildingId
from geosynthbench.world.world_state import WorldState


def _rect(center: Point, size: float) -> Polygon:
    # axis-aligned square, later rotated
    half = size / 2.0
    x, y = center.x, center.y
    return Polygon(
        [(x - half, y - half), (x + half, y - half), (x + half, y + half), (x - half, y + half)]
    )


def _ok_building(
    world: WorldState,
    footprint: Polygon,
    settlement_center: Point,
    settlement_radius: float,
    min_dist_buildings: float,
    max_slope_building: float,
) -> bool:
    if footprint.is_empty or not footprint.is_valid:
        return False

    # inside extent
    if not world.extent_polygon().contains(footprint):
        return False

    # within settlement influence (HARD-ish, keeps structure plausible)
    if footprint.centroid.distance(settlement_center) > 1.15 * settlement_radius:
        return False

    # not in water
    wu = world.water_union()
    if not wu.is_empty and footprint.intersects(wu):
        return False

    # slope constraint (if terrain)
    if world.terrain is not None:
        stats = world.terrain.poly_stats(footprint, max_points=512)
        if stats["slope_max"] > max_slope_building:
            return False

    # avoid overlap / too close
    for b in world.buildings:
        if footprint.intersects(b.footprint):
            return False
        if footprint.distance(b.footprint) < min_dist_buildings:
            return False

    return True


def generate_buildings(
    world: WorldState,
    rng: np.random.Generator,
    buildings_per_settlement: tuple[int, int],
    size_range: tuple[float, float],
    min_dist_buildings_m: float,
    max_slope_building: float,
    max_attempts: int,
) -> None:
    world.buildings = []
    bid_counter = 0

    for s in world.settlements:
        target = int(rng.integers(buildings_per_settlement[0], buildings_per_settlement[1] + 1))
        placed = 0
        attempts = 0

        while placed < target and attempts < max_attempts:
            attempts += 1

            # sample a ring-ish distribution around settlement center
            ang = float(rng.uniform(0.0, 2.0 * math.pi))
            r = float(rng.uniform(0.10 * s.radius_m, 1.05 * s.radius_m))
            cx = float(s.center.x + r * math.cos(ang))
            cy = float(s.center.y + r * math.sin(ang))
            size = float(rng.uniform(size_range[0], size_range[1]))
            fp = _rect(Point(cx, cy), size=size)
            fp = rotate(
                fp, angle=float(rng.uniform(0.0, 180.0)), origin=(cx, cy), use_radians=False
            )

            if not _ok_building(
                world,
                fp,
                settlement_center=s.center,
                settlement_radius=s.radius_m,
                min_dist_buildings=min_dist_buildings_m,
                max_slope_building=max_slope_building,
            ):
                continue

            world.buildings.append(
                Building(
                    id=BuildingId(f"b{bid_counter}"),
                    settlement_id=s.id,
                    footprint=fp,
                    near_road_id=None,
                )
            )
            bid_counter += 1
            placed += 1
