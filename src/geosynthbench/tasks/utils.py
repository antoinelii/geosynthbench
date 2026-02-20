import numpy as np
from PIL.Image import Image
from shapely.geometry import Point

from geosynthbench.gen.config import WorldGenConfig
from geosynthbench.gen.pipeline import generate_world
from geosynthbench.render.renderer import (
    mask_layers_to_mask_image,
    render_world_textured_with_mask,
)
from geosynthbench.world.raster import RasterTransform
from geosynthbench.world.world_state import WorldState

tr_ref_config = {
    "extent": (0.0, 0.0, float(512 * 5), float(512 * 5)),
    "width_px": 512,
    "height_px": 512,
}

tr_ref = RasterTransform(
    extent=tr_ref_config["extent"],
    width_px=tr_ref_config["width_px"],
    height_px=tr_ref_config["height_px"],
)


def generate_t0_sample(cfg: WorldGenConfig) -> WorldState:
    world0 = generate_world(tr=tr_ref, cfg=cfg)
    return world0


def render_t0_t1_samples(
    w0: WorldState, w1: WorldState
) -> tuple[tuple[Image, Image], tuple[Image, Image]]:
    random_seed = np.random.default_rng().integers(0, 2**63 - 1, dtype=np.int64)
    common_rng = np.random.default_rng(random_seed)
    img0, mask_res0 = render_world_textured_with_mask(w0, rng=common_rng)
    img1, mask_res1 = render_world_textured_with_mask(w1, rng=common_rng)
    sem_mask0 = mask_layers_to_mask_image(mask_res0)
    sem_mask1 = mask_layers_to_mask_image(mask_res1)
    return (img0, sem_mask0), (img1, sem_mask1)


def _px_loc(point: Point, world: WorldState) -> tuple[int, int]:
    u, v = world.tr.world_to_px(point.x, point.y)
    return int(u), int(v)


def _labels(n: int) -> list[str]:
    # Supports up to 26 settlements; your config is 4–6 so it's fine
    return [chr(ord("A") + i) for i in range(n)]
