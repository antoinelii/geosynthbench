from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from shapely import wkt
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from geosynthbench.io.deserialize import world_from_dict
from geosynthbench.io.jsonl_utils import read_jsonl_record

# --- Import your texture pipeline entrypoint(s)
# Adjust these imports to match your code.
from geosynthbench.render.textures import (
    BuildingParams,
    RoadParams,
    SceneParams,
    SettlementParams,
    VegetationParams,
    WaterParams,
    background_palette_from_elevation,
    building_palette,
    postprocess,
    render_full_rgb,
    roads_palette,
    settlement_palette,
    vegetation_palette,
    water_palette,
)
from geosynthbench.render.textures.shadows import add_building_shadows
from geosynthbench.world.entities import EntityType
from geosynthbench.world.raster import RasterTransform


def parse_transform(rec: dict[str, Any]) -> RasterTransform:
    # Supports {"transform": {"extent": [...], "width_px":..., "height_px":...}}
    # or {"tr": {...}}
    tr = rec.get("transform") or rec.get("tr") or {}
    extent = tr.get("extent") or tr.get("bbox") or rec.get("extent")
    if extent is None:
        raise ValueError("No extent found (expected rec['transform']['extent']).")
    width_px = int(tr.get("width_px") or tr.get("W") or rec.get("width_px"))
    height_px = int(tr.get("height_px") or tr.get("H") or rec.get("height_px"))
    return RasterTransform(extent=tuple(map(float, extent)), width_px=width_px, height_px=height_px)


def _geom_from_any(d: dict[str, Any], *keys: str):
    for k in keys:
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return wkt.loads(v)
    return None


def parse_water_polys(rec: dict[str, Any]) -> list[Polygon]:
    out: list[Polygon] = []
    for obj in rec.get("water", []) or rec.get("water_bodies", []) or []:
        g = _geom_from_any(obj, "polygon_wkt", "wkt", "geom_wkt", "geometry_wkt")
        if isinstance(g, Polygon) and not g.is_empty:
            out.append(g)
    return out


def parse_veg_polys(rec: dict[str, Any]) -> list[Polygon]:
    out: list[Polygon] = []
    for obj in rec.get("vegetation", []) or rec.get("veg", []) or []:
        g = _geom_from_any(obj, "polygon_wkt", "wkt", "geom_wkt", "geometry_wkt")
        if isinstance(g, Polygon) and not g.is_empty:
            out.append(g)
    return out


def parse_roads(rec: dict[str, Any]) -> list[tuple[LineString, float]]:
    out: list[tuple[LineString, float]] = []
    roads = rec.get("roads") or {}
    segs = roads.get("segments") if isinstance(roads, dict) else rec.get("road_segments")
    if segs is None:
        segs = rec.get("segments") if "segments" in rec else []
    for seg in segs or []:
        g = _geom_from_any(seg, "centerline_wkt", "linestring_wkt", "wkt", "geom_wkt")
        if isinstance(g, LineString) and not g.is_empty:
            width_m = float(seg.get("width_m", 8.0))
            out.append((g, width_m))
    return out


def parse_settlements(rec: dict[str, Any]) -> list[dict[str, Any]]:
    # Expect: [{"id": "...", "center_wkt": "POINT (...)", "radius_m": 250.0}, ...]
    out: list[dict[str, Any]] = []
    for s in rec.get("settlements", []) or []:
        sid = str(s.get("id", ""))
        c = _geom_from_any(s, "center_wkt", "wkt", "geom_wkt")
        if isinstance(c, Point):
            out.append({"id": sid, "center": c, "radius_m": float(s.get("radius_m", 250.0))})
    return out


def parse_buildings(rec: dict[str, Any]) -> list[dict[str, Any]]:
    # Expect: [{"id":"b0","settlement_id":"s0","footprint_wkt":"POLYGON(...)"}]
    out: list[dict[str, Any]] = []
    for b in rec.get("buildings", []) or []:
        bid = str(b.get("id", ""))
        sid = str(b.get("settlement_id", b.get("settlement", "")))
        g = _geom_from_any(b, "footprint_wkt", "polygon_wkt", "wkt", "geom_wkt")
        if isinstance(g, Polygon) and not g.is_empty:
            out.append({"id": bid, "settlement_id": sid, "footprint": g})
    return out


def poly_to_px_rings(
    tr: RasterTransform, poly: Polygon
) -> tuple[list[tuple[float, float]], list[list[tuple[float, float]]]]:
    ext = [(tr.world_to_px(x, y)[0], tr.world_to_px(x, y)[1]) for (x, y) in poly.exterior.coords]
    holes: list[list[tuple[float, float]]] = []
    for ring in poly.interiors:
        holes.append([(tr.world_to_px(x, y)[0], tr.world_to_px(x, y)[1]) for (x, y) in ring.coords])
    return ext, holes


def draw_polygon_L(
    draw: ImageDraw.ImageDraw, tr: RasterTransform, poly: Polygon, value: int
) -> None:
    if poly.is_empty:
        return
    ext, holes = poly_to_px_rings(tr, poly)
    draw.polygon(ext, fill=value)
    for hole in holes:
        draw.polygon(hole, fill=0)


def build_masks(
    tr: RasterTransform,
    water_polys: list[Polygon],
    veg_polys: list[Polygon],
    road_lines: list[tuple[LineString, float]],
    settlements: list[dict[str, Any]],
    buildings: list[dict[str, Any]],
    *,
    settlement_mode: str = "hull_then_circle",
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], np.ndarray]:
    """
    Returns:
      masks: water/veg/roads/settlement bool[H,W]
      building_items: [{id, settlement_id, mask(bool)}]
      semantic_id: uint8[H,W] with BG/WATER/VEG/ROAD/SETTLEMENT/BUILDING
    """
    W, H = tr.width_px, tr.height_px

    # bool masks via PIL L images
    water_im = Image.new("L", (W, H), 0)
    veg_im = Image.new("L", (W, H), 0)
    roads_im = Image.new("L", (W, H), 0)
    settlement_im = Image.new("L", (W, H), 0)

    dw = ImageDraw.Draw(water_im)
    dv = ImageDraw.Draw(veg_im)
    dr = ImageDraw.Draw(roads_im)
    ds = ImageDraw.Draw(settlement_im)

    for p in water_polys:
        draw_polygon_L(dw, tr, p, 255)

    for p in veg_polys:
        draw_polygon_L(dv, tr, p, 255)

    for line, width_m in road_lines:
        road_poly = line.buffer(width_m / 2.0, cap_style=2, join_style=2)
        draw_polygon_L(dr, tr, road_poly, 255)

    # buildings: per-building masks
    building_items: list[dict[str, Any]] = []
    for b in buildings:
        bim = Image.new("L", (W, H), 0)
        db = ImageDraw.Draw(bim)
        draw_polygon_L(db, tr, b["footprint"], 255)
        building_items.append(
            {
                "id": b["id"],
                "settlement_id": b["settlement_id"],
                "mask": (np.array(bim, np.uint8) > 0),
            }
        )

    # settlement polygons
    by_sid: dict[str, list[Polygon]] = {}
    for b in buildings:
        by_sid.setdefault(b["settlement_id"], []).append(b["footprint"])

    for s in settlements:
        sid = s["id"]
        polys = by_sid.get(sid, [])
        if settlement_mode == "circle_only":
            poly = s["center"].buffer(float(s["radius_m"]), resolution=64)
        elif settlement_mode == "hull_only":
            poly = unary_union(polys).convex_hull if polys else Polygon()
        else:  # hull_then_circle
            poly = (
                unary_union(polys).convex_hull
                if polys
                else s["center"].buffer(float(s["radius_m"]), resolution=64)
            )
        if not poly.is_empty:
            draw_polygon_L(ds, tr, poly, 255)

    masks = {
        "water": (np.array(water_im, np.uint8) > 0),
        "veg": (np.array(veg_im, np.uint8) > 0),
        "roads": (np.array(roads_im, np.uint8) > 0),
        "settlement": (np.array(settlement_im, np.uint8) > 0),
    }

    # semantic id map with simple priority order:
    # BG < water < veg < settlement < roads < buildings (buildings highest)
    sem = np.zeros((H, W), dtype=np.uint8)
    sem[masks["water"]] = EntityType.WATER.value
    sem[masks["veg"]] = EntityType.VEG.value
    sem[masks["settlement"]] = EntityType.SETTLEMENT.value
    sem[masks["roads"]] = EntityType.ROAD.value
    for b in building_items:
        sem[b["mask"]] = EntityType.BUILDING.value

    return masks, building_items, sem


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=Path, required=True, help="Path to t0.jsonl")
    ap.add_argument("--elev", type=Path, required=True, help="Path to sample_elevation_map.npy")
    ap.add_argument("--idx", type=int, default=0, help="Which jsonl record to load")
    ap.add_argument("--out", type=Path, default=Path("out"), help="Output directory")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for textures")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rec = read_jsonl_record(args.jsonl, args.idx)
    world = world_from_dict(rec["t0"], base_dir=args.jsonl.parent)
    tr = world.tr

    elev = np.load(args.elev).astype(np.float32)
    if elev.shape != (tr.height_px, tr.width_px):
        raise ValueError(f"Elevation shape {elev.shape} != (H,W)=({tr.height_px},{tr.width_px})")

    # Parse world entities from JSON
    # water_polys = parse_water_polys(rec)
    # veg_polys = parse_veg_polys(rec)
    # road_lines = parse_roads(rec)
    # settlements = parse_settlements(rec)
    # buildings = parse_buildings(rec)
    water_polys = [w.polygon for w in world.water]
    veg_polys = [v.polygon for v in world.vegetation]
    road_lines = [(s.centerline, s.width_m) for s in world.roads.segments]
    settlements = [
        {"id": s.id, "center": s.center, "radius_m": s.radius_m} for s in world.settlements
    ]
    buildings = [
        {"id": b.id, "settlement_id": b.settlement_id, "footprint": b.footprint}
        for b in world.buildings
    ]

    # Build masks + semantic id map
    masks, building_items, sem = build_masks(
        tr,
        water_polys,
        veg_polys,
        road_lines,
        settlements,
        buildings,
        settlement_mode="hull_then_circle",
    )

    # Save semantic mask (uint8 ids)
    Image.fromarray(sem, mode="L").save(args.out / "mask_semantic.png")

    # --- Render textured RGB using your texture modules
    rng = np.random.default_rng(args.seed)

    scene = SceneParams(
        biome="temperate",
        sun_azimuth_deg=float(rng.uniform(0, 360)),
        sun_altitude_deg=float(rng.uniform(20, 60)),
        sun_intensity=float(rng.uniform(0.85, 1.2)),
        exposure=float(rng.uniform(0.92, 1.08)),
        gamma=float(rng.uniform(0.98, 1.12)),
        haze=float(rng.uniform(0.0, 0.06)),
        saturation=float(rng.uniform(0.92, 1.12)),
    )

    wp = WaterParams(
        alpha=0.88,
        turbidity=float(rng.uniform(0.1, 0.6)),
        specular=float(rng.uniform(0.0, 0.35)),
        shoreline_width_px=3,
    )
    vp = VegetationParams(
        alpha=0.85,
        density=float(rng.uniform(0.35, 0.95)),
        patchiness=float(rng.uniform(0.4, 0.9)),
        texture_scale_px=float(rng.uniform(10, 22)),
    )
    rp = RoadParams(
        alpha=0.96,
        wear=float(rng.uniform(0.2, 0.7)),
        lane_hint=float(rng.uniform(0.0, 0.25)),
        texture_scale_px=float(rng.uniform(8, 14)),
    )
    sp = SettlementParams(
        alpha=0.65, impervious=float(rng.uniform(0.5, 0.95)), grime=float(rng.uniform(0.1, 0.6))
    )
    bp = BuildingParams(
        alpha=0.98,
        roof_variation=float(rng.uniform(0.4, 1.0)),
        shadow_strength=float(rng.uniform(0.1, 0.35)),
    )

    bg = background_palette_from_elevation(elev, scene, rng)

    print(elev.shape)
    layers = [
        water_palette(masks["water"], scene, wp, rng),
        vegetation_palette(masks["veg"], scene, vp, rng),
        settlement_palette(masks["settlement"], scene, sp, rng),
        roads_palette(masks["roads"], scene, rp, rng),
    ]

    # Per-building individualized textures
    for b in building_items:
        b_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1, dtype=np.int64)))
        layers.append(building_palette(b["mask"], b["settlement_id"], scene, bp, b_rng))

    rgb = render_full_rgb(bg, layers)

    # shadows from union of buildings
    all_b = np.zeros((tr.height_px, tr.width_px), dtype=bool)
    for b in building_items:
        all_b |= b["mask"]
    rgb = add_building_shadows(
        rgb,
        all_b,
        sun_azimuth_deg=scene.sun_azimuth_deg,
        strength=bp.shadow_strength,
        length_px=5,
    )

    out_u8 = postprocess(
        rgb,
        exposure=scene.exposure,
        gamma=scene.gamma,
        saturation=scene.saturation,
        grain=0.012,
        blur_k=3,
        rng=rng,
    )
    Image.fromarray(out_u8, mode="RGB").save(args.out / "rgb_textured.png")

    print(f"[OK] wrote {args.out / 'mask_semantic.png'}")
    print(f"[OK] wrote {args.out / 'rgb_textured.png'}")


if __name__ == "__main__":
    main()
