from __future__ import annotations

import numpy as np
from shapely.geometry import MultiPolygon, Polygon

from geosynthbench.world.entities import WaterBody
from geosynthbench.world.types import WaterId
from geosynthbench.world.world_state import WorldState


def _blob(
    rng: np.random.Generator, center: tuple[float, float], base_r: float, n_pts: int = 16
) -> Polygon:
    cx, cy = center
    angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)

    radii = base_r * (0.75 + 0.50 * rng.random(n_pts))
    xs = cx + radii * np.cos(angles)
    ys = cy + radii * np.sin(angles)
    poly = Polygon(np.column_stack([xs, ys]))
    # smooth a bit with buffer in/out (cheap)
    poly = poly.buffer(base_r * 0.15).buffer(-base_r * 0.10)
    return poly


def generate_water(world: WorldState, rng: np.random.Generator, n_water: int) -> None:
    extent = world.extent_polygon()
    out: list[WaterBody] = []

    for i in range(n_water):
        # place somewhat randomly
        cx = rng.uniform(world.tr.xmin, world.tr.xmax)
        cy = rng.uniform(world.tr.ymin, world.tr.ymax)
        base_r = rng.uniform(220.0, 650.0)

        poly = _blob(rng, (cx, cy), base_r=base_r, n_pts=int(rng.integers(16, 32)))
        # clip to extent
        poly = poly.intersection(extent)

        if poly.is_empty or (hasattr(poly, "area") and poly.area < 1.0):
            continue

        # convert to list of polys to handle multi-poly result from intersection
        if isinstance(poly, Polygon):
            polys = [poly]
        elif isinstance(poly, MultiPolygon):
            polys = list(poly.geoms)
        else:  # shouldn't happen with current code, but just in case
            continue

        for p in polys:
            out.append(WaterBody(id=WaterId(f"w{i}"), polygon=p))

    world.water = out
