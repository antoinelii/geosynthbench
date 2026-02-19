from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

IntRange = tuple[int, int]
FloatRange = tuple[float, float]


def _pick_int(rng : np.random.Generator, v: int | IntRange) -> int:
    if isinstance(v, int):
        return v
    a, b = v
    return int(rng.integers(a, b + 1))


def _pick_float(rng : np.random.Generator, v: float | FloatRange) -> float:
    if isinstance(v, float):
        return v
    a, b = v
    return float(rng.uniform(a, b))


@dataclass(frozen=True)
class WorldGenConfig:
    seed: int = 0

    # terrain
    terrain_amplitude_m: float = 30.0
    terrain_n_hills: IntRange = (2, 6)
    terrain_hill_sigma_m: FloatRange = (250.0, 900.0)
    terrain_noise_scale_m: float = 300.0
    terrain_noise_strength_m: float = 6.0

    # water / vegetation counts
    n_water: int | IntRange = (0, 2)
    n_veg: int | IntRange = (1, 3)

    # settlements
    n_settlements: int | IntRange = (2, 5)
    settlement_radius_m: FloatRange = (180.0, 400.0)
    min_dist_settlements_m: float = 500.0
    max_slope_settlement: float = 0.25  # ~ 25%

    # roads
    roads_mode: Literal["mst", "mst+extras", "ring", "dense"] = "mst+extras"
    extra_edges: int = 1
    road_width_m: float = 8.0
    max_slope_road: float = 0.35

    # buildings
    buildings_per_settlement: IntRange = (10, 30)
    building_size_m: FloatRange = (12.0, 30.0)  # square-ish side
    min_dist_buildings_m: float = 6.0
    max_slope_building: float = 0.20
    max_building_attempts: int = 2000

    # hard rule toggles
    require_connected_roads: bool = True
    forbid_roads_in_water: bool = True

    # soft preferences
    prefer_settlement_near_water: bool = True
    prefer_water_distance_m: FloatRange = (60.0, 350.0)  # target band

    # internal helpers exposed (optional)
    def pick_int(self, rng: np.random.Generator, v: int | IntRange) -> int:
        return _pick_int(rng, v)

    def pick_float(self, rng: np.random.Generator, v: float | FloatRange) -> float:
        return _pick_float(rng, v)
