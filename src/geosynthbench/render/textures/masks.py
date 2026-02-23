# src/geosynthbench/render/textures/masks.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw
from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from geosynthbench.world.entities import BuildingMaskItem
from geosynthbench.world.types import LayerKind, SettlementId
from geosynthbench.world.world_state import WorldState


def poly_to_px_rings(
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


def _pil_to_bool(im: Image.Image) -> npt.NDArray[np.bool_]:
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


def _draw_polygon_mask(draw: ImageDraw.ImageDraw, world: WorldState, poly: Polygon) -> None:
    """
    Draw shapely Polygon (with holes) into an L mask.
    Uses your existing poly_to_px_rings(world, poly) helper.
    """
    if poly.is_empty:
        return
    ext, holes = poly_to_px_rings(world, poly)
    _draw_poly_L(draw, ext, holes)


def _draw_linestring_buffer_mask(
    draw: ImageDraw.ImageDraw, world: WorldState, line: LineString, width_m: float
) -> None:
    if line.is_empty:
        return
    poly = line.buffer(width_m / 2.0, cap_style="round", join_style="mitre")
    _draw_polygon_mask(draw, world, poly)


@dataclass(frozen=True)
class MaskLayers:
    masks: dict[LayerKind, npt.NDArray[np.bool_]]  # bool[H,W]
    building_items: list[BuildingMaskItem]  # {id, settlement_id, mask}
    settlement_polys: dict[SettlementId, Polygon]  # for debug


def build_masks_from_world(
    world: WorldState,
    *,
    settlement_mode: str = "hull_then_circle",  # "hull_only" | "circle_only" | "hull_then_circle"
    include_settlement_mask: bool = True,
) -> MaskLayers:
    """
    Builds boolean masks from your WorldState (water/veg/roads/settlement) + per-building masks.

    settlement_mode:
      - "hull_only": settlement polygon = convex hull of union of buildings (if none -> empty)
      - "circle_only": settlement polygon = circle from settlement.center + settlement.radius_m
      - "hull_then_circle": hull if buildings exist else circle

    Returns:
      MaskLayers(masks, building_items, settlement_polys)
    """
    W, H = world.tr.width_px, world.tr.height_px

    # --- Base masks
    water_im = _new_L(W, H)
    veg_im = _new_L(W, H)
    roads_im = _new_L(W, H)
    settlement_im = _new_L(W, H) if include_settlement_mask else None

    w_drawer = ImageDraw.Draw(water_im)
    v_drawer = ImageDraw.Draw(veg_im)
    r_drawer = ImageDraw.Draw(roads_im)
    s_drawer = ImageDraw.Draw(settlement_im) if settlement_im is not None else None

    # Water polygons
    for w in getattr(world, "water", []):
        _draw_polygon_mask(w_drawer, world, w.polygon)

    # Vegetation polygons
    for v in getattr(world, "vegetation", []):
        _draw_polygon_mask(v_drawer, world, v.polygon)

    # Roads: buffer each segment
    roads = getattr(world, "roads", None)
    if roads is not None:
        for seg in getattr(roads, "segments", []):
            _draw_linestring_buffer_mask(r_drawer, world, seg.centerline, float(seg.width_m))

    # Buildings: per-building masks
    building_items: list[BuildingMaskItem] = []
    buildings = getattr(world, "buildings", [])
    for b in buildings:
        b_im = _new_L(W, H)
        b_drawer = ImageDraw.Draw(b_im)
        _draw_polygon_mask(b_drawer, world, b.footprint)
        building_items.append(
            BuildingMaskItem(
                id=getattr(b, "id"),
                settlement_id=getattr(b, "settlement_id"),
                mask=_pil_to_bool(b_im),
            )
        )

    # Settlement polygons (hull per settlement, else circle)
    settlement_polys: dict[SettlementId, Polygon] = {}
    if include_settlement_mask:
        # index buildings by settlement_id
        by_sett: dict[SettlementId, list[Polygon]] = {}
        for b in buildings:
            s_id = getattr(b, "settlement_id")
            by_sett.setdefault(s_id, []).append(b.footprint)

        for s in getattr(world, "settlements", []):
            s_id = getattr(s, "id")
            polys = by_sett.get(s_id, [])

            poly: Polygon
            if settlement_mode == "circle_only":
                poly = s.center.buffer(float(s.radius_m), resolution=64)
            elif settlement_mode == "hull_only":
                if polys:
                    u = unary_union(polys)
                    poly = _build_convex_hull(u)
                else:
                    poly = Polygon()
            else:  # "hull_then_circle"
                if polys:
                    u = unary_union(polys)
                    poly = _build_convex_hull(u)
                else:
                    poly = s.center.buffer(float(s.radius_m), resolution=64)

            settlement_polys[s_id] = poly
            if s_drawer is not None and not poly.is_empty:
                _draw_polygon_mask(s_drawer, world, poly)

    masks: dict[LayerKind, npt.NDArray[np.bool_]] = {
        "water": _pil_to_bool(water_im),
        "vegetation": _pil_to_bool(veg_im),
        "road": _pil_to_bool(roads_im),
    }
    if include_settlement_mask and settlement_im is not None:
        masks["settlement"] = _pil_to_bool(settlement_im)

    return MaskLayers(masks=masks, building_items=building_items, settlement_polys=settlement_polys)


def _build_convex_hull(geom: BaseGeometry) -> Polygon:
    """Best-effort conversion of the shapely retrieved convex hull to a Polygon for strict typing.
    Given the input geometries are building footprints, the convex hull should normally
    be a Polygon, but we handle edge cases."""
    if geom.is_empty:
        return Polygon()

    # Convex hull can be Polygon, LineString, Point, etc
    # Here normally a Polygon object
    hull = geom.convex_hull
    if hull.is_empty:
        return Polygon()

    if isinstance(hull, Polygon):
        return hull
    if isinstance(hull, MultiPolygon):
        # convex_hull usually isn't MultiPolygon, but being defensive is fine
        # pick largest piece if it ever happens
        return max(hull.geoms, key=lambda p: p.area)

    # If hull is a LineString/Point (e.g., 1-2 buildings), return empty polygon
    return Polygon()
