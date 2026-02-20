# src/geosynthbench/render/textures/masks.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

from geosynthbench.world.world_state import WorldState


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


def _pil_to_bool(im: Image.Image) -> np.ndarray:
    # im is L mode, 0..255
    return np.asarray(im, dtype=np.uint8) > 0


def _new_L(W: int, H: int) -> Image.Image:
    return Image.new("L", (W, H), color=0)


def _draw_poly_L(
    draw: ImageDraw.ImageDraw,
    ext: list[tuple[float, float]],
    holes: list[list[tuple[float, float]]],
) -> None:
    # fill exterior then punch holes
    if ext:
        draw.polygon(ext, fill=255)
    for hole in holes:
        if hole:
            draw.polygon(hole, fill=0)


def _draw_polygon_mask(draw: ImageDraw.ImageDraw, world: Any, poly: Polygon) -> None:
    """
    Draw shapely Polygon (with holes) into an L mask.
    Uses your existing _poly_to_px_rings(world, poly) helper.
    """
    if poly.is_empty:
        return
    ext, holes = _poly_to_px_rings(world, poly)  # noqa: F821 (provided by your renderer module)
    _draw_poly_L(draw, ext, holes)


def _draw_linestring_buffer_mask(
    draw: ImageDraw.ImageDraw, world: Any, line: LineString, width_m: float
) -> None:
    if line.is_empty:
        return
    poly = line.buffer(width_m / 2.0, cap_style=2, join_style=2)
    _draw_polygon_mask(draw, world, poly)


@dataclass(frozen=True)
class MaskBuildResult:
    masks: dict[str, np.ndarray]  # bool[H,W]
    building_items: list[dict[str, Any]]  # {id, settlement_id, mask}
    settlement_polys: dict[str, Polygon]  # for debug


def build_masks_from_world(
    world: Any,
    *,
    settlement_mode: str = "hull_then_circle",  # "hull_only" | "circle_only" | "hull_then_circle"
    include_settlement_mask: bool = True,
) -> MaskBuildResult:
    """
    Builds boolean masks from your WorldState (water/veg/roads/settlement) + per-building masks.

    settlement_mode:
      - "hull_only": settlement polygon = convex hull of union of buildings (if none -> empty)
      - "circle_only": settlement polygon = circle from settlement.center + settlement.radius_m
      - "hull_then_circle": hull if buildings exist else circle

    Returns:
      MaskBuildResult(masks, building_items, settlement_polys)
    """
    W, H = world.tr.width_px, world.tr.height_px

    # --- Base masks
    water_im = _new_L(W, H)
    veg_im = _new_L(W, H)
    roads_im = _new_L(W, H)
    settlement_im = _new_L(W, H) if include_settlement_mask else None

    dw = ImageDraw.Draw(water_im)
    dv = ImageDraw.Draw(veg_im)
    dr = ImageDraw.Draw(roads_im)
    ds = ImageDraw.Draw(settlement_im) if settlement_im is not None else None

    # Water polygons
    for w in getattr(world, "water", []):
        _draw_polygon_mask(dw, world, w.polygon)

    # Vegetation polygons
    for v in getattr(world, "vegetation", []):
        _draw_polygon_mask(dv, world, v.polygon)

    # Roads: buffer each segment
    roads = getattr(world, "roads", None)
    if roads is not None:
        for seg in getattr(roads, "segments", []):
            _draw_linestring_buffer_mask(dr, world, seg.centerline, float(seg.width_m))

    # Buildings: per-building masks
    building_items: list[dict[str, Any]] = []
    buildings = getattr(world, "buildings", [])
    for b in buildings:
        bim = _new_L(W, H)
        db = ImageDraw.Draw(bim)
        _draw_polygon_mask(db, world, b.footprint)
        building_items.append(
            {
                "id": str(getattr(b, "id")),
                "settlement_id": str(getattr(b, "settlement_id")),
                "mask": _pil_to_bool(bim),
            }
        )

    # Settlement polygons (hull per settlement, else circle)
    settlement_polys: dict[str, Polygon] = {}
    if include_settlement_mask:
        # index buildings by settlement_id
        by_set: dict[str, list[Polygon]] = {}
        for b in buildings:
            sid = str(getattr(b, "settlement_id"))
            by_set.setdefault(sid, []).append(b.footprint)

        for s in getattr(world, "settlements", []):
            sid = str(getattr(s, "id"))
            polys = by_set.get(sid, [])

            poly: Polygon
            if settlement_mode == "circle_only":
                poly = s.center.buffer(float(s.radius_m), resolution=64)
            elif settlement_mode == "hull_only":
                if polys:
                    u = unary_union(polys)
                    poly = u.convex_hull
                else:
                    poly = Polygon()
            else:  # "hull_then_circle"
                if polys:
                    u = unary_union(polys)
                    poly = u.convex_hull
                else:
                    poly = s.center.buffer(float(s.radius_m), resolution=64)

            settlement_polys[sid] = poly
            if ds is not None and not poly.is_empty:
                _draw_polygon_mask(ds, world, poly)

    masks: dict[str, np.ndarray] = {
        "water": _pil_to_bool(water_im),
        "veg": _pil_to_bool(veg_im),
        "roads": _pil_to_bool(roads_im),
    }
    if include_settlement_mask and settlement_im is not None:
        masks["settlement"] = _pil_to_bool(settlement_im)

    return MaskBuildResult(
        masks=masks, building_items=building_items, settlement_polys=settlement_polys
    )
