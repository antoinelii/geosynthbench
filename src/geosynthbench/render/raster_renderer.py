from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import LineString, Polygon

from geosynthbench.render.palette import Palette
from geosynthbench.world.world_state import WorldState


@dataclass(frozen=True)
class RenderResult:
    rgb: Image.Image
    mask: Image.Image
    height: Image.Image | None = None
    slope: Image.Image | None = None


def _poly_to_px_rings(
    world: WorldState, poly: Polygon
) -> tuple[list[tuple[float, float]], list[list[tuple[float, float]]]]:
    """
    Convert shapely Polygon to pixel ring coordinates for PIL.
    Returns (exterior, holes[]).
    """
    tr = world.tr

    ext = [(tr.world_to_px(x, y)[0], tr.world_to_px(x, y)[1]) for (x, y) in poly.exterior.coords]
    holes: list[list[tuple[float, float]]] = []
    for ring in poly.interiors:
        holes.append([(tr.world_to_px(x, y)[0], tr.world_to_px(x, y)[1]) for (x, y) in ring.coords])
    return ext, holes


def _line_to_px(world: WorldState, line: LineString) -> list[tuple[float, float]]:
    tr = world.tr
    return [(tr.world_to_px(x, y)[0], tr.world_to_px(x, y)[1]) for (x, y) in line.coords]


def _draw_polygon(draw: ImageDraw.ImageDraw, poly: Polygon, fill, outline=None) -> None:
    if poly.is_empty:
        return
    ext = poly.exterior.coords.xy
    draw.polygon(ext, fill=fill, outline=outline)
    # "Punch" holes by filling with 0 alpha / background is not trivial in RGB,
    # but for our layers we rarely have holes. If you do, render holes separately in mask logic.
    # For mask, holes are best handled by draw.polygon(hole, fill=bg).
    # We'll handle holes explicitly in mask drawing below if needed.


def render_world_debug(
    world: WorldState, palette: Palette | None = None, *, draw_settlement_circles: bool = False
) -> RenderResult:
    """
    Renders:
      - RGB debug image (PIL)
      - semantic mask (uint8 class ids)
      - optional height/slope grayscale debug
    """
    pal = palette or Palette()
    W, H = world.tr.width_px, world.tr.height_px

    rgb = Image.new("RGB", (W, H), pal.bg_rgb)
    mask = Image.new("L", (W, H), color=pal.BG)  # L=8-bit

    draw_rgb = ImageDraw.Draw(rgb, "RGBA")
    draw_m = ImageDraw.Draw(mask)

    # --- optional terrain visual (not in rgb, but via separate outputs)
    height_img: Image.Image | None = None
    slope_img: Image.Image | None = None
    if world.terrain is not None:
        elev = world.terrain.elevation_m
        s = world.terrain.slope()

        def _to_gray(a: np.ndarray) -> Image.Image:
            aa = a.astype(np.float32)
            lo, hi = float(np.percentile(aa, 2)), float(np.percentile(aa, 98))
            if hi <= lo + 1e-6:
                hi = lo + 1.0
            g = np.clip((aa - lo) / (hi - lo), 0.0, 1.0)
            g8 = (g * 255.0).astype(np.uint8)
            return Image.fromarray(g8, mode="L")

        height_img = _to_gray(elev)
        slope_img = _to_gray(s)

    # Helper to fill polygon on both rgb + mask (with holes)
    def paint_poly(
        poly: Polygon, rgb_fill: tuple[int, int, int], mask_id: int, alpha: int = 255
    ) -> None:
        if poly.is_empty:
            return
        ext, holes = _poly_to_px_rings(world, poly)
        draw_rgb.polygon(ext, fill=(rgb_fill[0], rgb_fill[1], rgb_fill[2], alpha))
        draw_m.polygon(ext, fill=mask_id)

        # holes -> revert to background in mask + rgb
        for hole in holes:
            draw_rgb.polygon(hole, fill=(pal.bg_rgb[0], pal.bg_rgb[1], pal.bg_rgb[2], 255))
            draw_m.polygon(hole, fill=pal.BG)

    # LAYER ORDER (game-like):
    # water -> vegetation -> roads -> buildings (roads above veg often looks better; adjust as you want)

    # Water
    for w in world.water:
        paint_poly(w.polygon, pal.water_rgb, pal.WATER, alpha=255)

    # Vegetation
    for v in world.vegetation:
        # slight transparency on rgb so terrain can be added later if you blend
        paint_poly(v.polygon, pal.veg_rgb, pal.VEGETATION, alpha=230)

    # Roads (draw as thick poly from buffered centerline)
    for seg in world.roads.segments:
        road_poly = seg.centerline.buffer(seg.width_m / 2.0, cap_style="round", join_style="mitre")
        paint_poly(road_poly, pal.road_rgb, pal.ROAD, alpha=255)

    # Buildings
    for b in world.buildings:
        paint_poly(b.footprint, pal.building_rgb, pal.BUILDING, alpha=255)

    # Optional settlement circles (debug)
    if draw_settlement_circles:
        for s in world.settlements:
            c = s.center
            # approximate circle by shapely buffer
            circ = c.buffer(s.radius_m, resolution=64)
            ext, holes = _poly_to_px_rings(world, circ)
            draw_rgb.line(
                ext,
                fill=(pal.settlement_rgb[0], pal.settlement_rgb[1], pal.settlement_rgb[2], 255),
                width=2,
            )

    return RenderResult(rgb=rgb, mask=mask, height=height_img, slope=slope_img)
