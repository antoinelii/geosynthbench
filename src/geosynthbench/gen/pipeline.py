from __future__ import annotations

import numpy as np

from geosynthbench.gen.buildings import generate_buildings
from geosynthbench.gen.config import WorldGenConfig
from geosynthbench.gen.roads import generate_roads
from geosynthbench.gen.settlements import generate_settlements
from geosynthbench.gen.terrain import generate_terrain
from geosynthbench.gen.vegetation import generate_vegetation
from geosynthbench.gen.water import generate_water
from geosynthbench.world.validate import hard_violations, validate_world
from geosynthbench.world.world_state import WorldState
from geosynthbench.world.raster import RasterTransform


def generate_world(tr: RasterTransform, cfg: WorldGenConfig, *, max_retries: int = 30) -> WorldState:
    """
    Full world generation with hard-rule enforcement:
    retry whole world if hard violations remain.
    """
    for attempt in range(max_retries):
        rng = np.random.default_rng(cfg.seed + attempt * 10_000)

        world = WorldState(tr=tr)

        # 1) terrain
        n_hills = cfg.pick_int(rng, cfg.terrain_n_hills)
        world.terrain = generate_terrain(
            tr=tr,
            rng=rng,
            amplitude_m=cfg.terrain_amplitude_m,
            n_hills=n_hills,
            hill_sigma_m_range=cfg.terrain_hill_sigma_m,
            noise_scale_m=cfg.terrain_noise_scale_m,
            noise_strength_m=cfg.terrain_noise_strength_m,
        )

        # 2) water
        n_water = cfg.pick_int(rng, cfg.n_water)
        generate_water(world, rng, n_water=n_water)

        # 3) vegetation (clips water internally)
        n_veg = cfg.pick_int(rng, cfg.n_veg)
        generate_vegetation(world, rng, n_veg=n_veg)

        # 4) settlements
        n_sett = cfg.pick_int(rng, cfg.n_settlements)
        generate_settlements(
            world,
            rng,
            n_settlements=n_sett,
            radius_range=cfg.settlement_radius_m,
            min_dist_settlements_m=cfg.min_dist_settlements_m,
            max_slope_settlement=cfg.max_slope_settlement,
        )

        # 5) roads
        generate_roads(world, rng, mode=cfg.roads_mode, extra_edges=cfg.extra_edges, width_m=cfg.road_width_m)

        # 6) buildings
        generate_buildings(
            world,
            rng,
            buildings_per_settlement=cfg.buildings_per_settlement,
            size_range=cfg.building_size_m,
            min_dist_buildings_m=cfg.min_dist_buildings_m,
            max_slope_building=cfg.max_slope_building,
            max_attempts=cfg.max_building_attempts,
        )

        # validate
        viol = validate_world(
            world,
            require_connected_roads=cfg.require_connected_roads,
            forbid_roads_in_water=cfg.forbid_roads_in_water,
            max_slope_settlement=cfg.max_slope_settlement,
            max_slope_road=cfg.max_slope_road,
            max_slope_building=cfg.max_slope_building,
            min_dist_settlements_m=cfg.min_dist_settlements_m,
            min_dist_buildings_m=cfg.min_dist_buildings_m,
        )
        if not hard_violations(viol):
            return world

    raise RuntimeError(f"Failed to generate a valid world after {max_retries} retries.")
