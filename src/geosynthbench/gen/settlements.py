from __future__ import annotations

import numpy as np
from shapely.geometry import Point

from geosynthbench.world.entities import Settlement
from geosynthbench.world.world_state import WorldState
from geosynthbench.world.types import SettlementId


def _ok_center(world: WorldState, x: float, y: float, min_dist: float, max_slope: float) -> bool:
    p = Point(x, y)
    # not in water
    if not world.water_union().is_empty and world.water_union().contains(p):
        return False
    # slope check (if terrain)
    if world.terrain is not None:
        if world.terrain.sample_slope_point(x, y) > max_slope:
            return False
    # distance to existing
    for s in world.settlements:
        if p.distance(s.center) < min_dist:
            return False
    return True


def generate_settlements(world: WorldState, rng: np.random.Generator, n_settlements: int,
                         radius_range: tuple[float, float], min_dist_settlements_m: float,
                         max_slope_settlement: float) -> None:
    out: list[Settlement] = []
    world.settlements = []

    attempts = 0
    max_attempts = max(2000, 500 * n_settlements)

    while len(out) < n_settlements and attempts < max_attempts:
        attempts += 1
        x = float(rng.uniform(world.tr.xmin, world.tr.xmax))
        y = float(rng.uniform(world.tr.ymin, world.tr.ymax))
        if not _ok_center(world, x, y, min_dist=min_dist_settlements_m, max_slope=max_slope_settlement):
            continue
        radius = float(rng.uniform(radius_range[0], radius_range[1]))
        sid = SettlementId(f"s{len(out)}")
        out.append(Settlement(id=sid, center=Point(x, y), radius_m=radius))

    world.settlements = out
