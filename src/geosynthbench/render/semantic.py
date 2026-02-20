from __future__ import annotations

from enum import IntEnum
from typing import Dict, Tuple

import numpy as np


class SemanticClass(IntEnum):
    BG = 0
    WATER = 1
    VEGETATION = 2
    ROAD = 3
    SETTLEMENT = 4
    BUILDING = 5

    @property
    def rgb(self) -> Tuple[int, int, int]:
        return _COLORS[self]


_COLORS: Dict[SemanticClass, Tuple[int, int, int]] = {
    SemanticClass.BG: (0, 0, 0),
    SemanticClass.WATER: (0, 90, 180),
    SemanticClass.VEGETATION: (30, 140, 30),
    SemanticClass.ROAD: (120, 120, 120),
    SemanticClass.SETTLEMENT: (200, 180, 140),
    SemanticClass.BUILDING: (180, 60, 60),
}


def semantic_mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for c in SemanticClass:
        out[mask == c.value] = c.rgb
    return out
