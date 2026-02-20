from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon

from geosynthbench.world.entities import VegetationPatch
from geosynthbench.world.types import VegId
from geosynthbench.world.world_state import WorldState


def _blob(rng: np.random.Generator, cx: float, cy: float, base_r: float, n_pts: int) -> Polygon:
    import numpy as _np
    from shapely.geometry import Polygon as _Polygon

    angles = _np.linspace(0, 2 * _np.pi, n_pts, endpoint=False)
    radii = base_r * (0.70 + 0.60 * rng.random(n_pts))
    xs = cx + radii * _np.cos(angles)
    ys = cy + radii * _np.sin(angles)
    poly = _Polygon(_np.column_stack([xs, ys]))
    poly = poly.buffer(base_r * 0.12).buffer(-base_r * 0.08)
    return poly


def generate_vegetation(world: WorldState, rng: np.random.Generator, n_veg: int) -> None:
    extent = world.extent_polygon()
    water_u = world.water_union()
    out: list[VegetationPatch] = []

    for i in range(n_veg):
        cx = rng.uniform(world.tr.xmin, world.tr.xmax)
        cy = rng.uniform(world.tr.ymin, world.tr.ymax)
        base_r = rng.uniform(350.0, 1100.0)
        poly = _blob(rng, cx, cy, base_r=base_r, n_pts=int(rng.integers(64, 128)))
        poly = poly.intersection(extent)

        # HARD RULE: vegetation must not cover water -> clip/difference
        if not water_u.is_empty:
            poly = poly.difference(water_u)

        if poly.is_empty or poly.area < 10.0:
            continue

        # convert to list of polys if we get a multi-poly result from difference
        if poly.geom_type == "Polygon":
            polys = [poly]
        else:
            polys = list(poly.geoms)

        for p in polys:
            out.append(
                VegetationPatch(id=VegId(f"v{i}"), polygon=p, density=float(rng.uniform(0.6, 1.0)))
            )

    world.vegetation = out
