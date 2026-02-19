from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Palette:
    """
    Semantic ids (mask) + simple RGB colors for debug rendering.
    mask: uint8 image where each pixel is a class id.
    """

    # class ids
    BG: int = 0
    WATER: int = 1
    VEGETATION: int = 2
    ROAD: int = 3
    BUILDING: int = 4
    SETTLEMENT: int = 5  # (optional debug)

    # RGB (debug)
    bg_rgb: tuple[int, int, int] = (200, 200, 200)
    water_rgb: tuple[int, int, int] = (60, 120, 200)
    veg_rgb: tuple[int, int, int] = (60, 140, 70)
    road_rgb: tuple[int, int, int] = (120, 120, 120)
    building_rgb: tuple[int, int, int] = (190, 170, 140)
    settlement_rgb: tuple[int, int, int] = (220, 80, 80)
