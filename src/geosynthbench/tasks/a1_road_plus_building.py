from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from shapely.affinity import rotate
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from geosynthbench.pipeline.writer import RenderArtifacts
from geosynthbench.tasks.utils import generate_t0_sample, px_loc


@dataclass(frozen=True)
class A1Config:
    # uses your WorldGenConfig instance directly
    world_cfg: Any

    # --- operator knobs (kept intentionally simple/robust) ---
    connect_strategy: str = "nearest"  # "nearest" (only one supported for MVP)
    road_width_m: float = 8.0

    # building cluster around the NEW road / NEW settlement
    n_new_buildings: tuple[int, int] = (4, 9)  # keep small to be countable
    cluster_radius_m: float = 220.0
    road_buffer_m: float = 55.0  # buildings counted if within this of new road
    building_size_m: tuple[float, float] = (12.0, 22.0)
    min_dist_new_buildings_m: float = 6.0
    max_attempts_place_buildings: int = 400

    # acceptance guards
    min_existing_settlements: int = 2


def _within_extent(p: Point, extent: tuple[float, float, float, float], margin: float) -> bool:
    xmin, ymin, xmax, ymax = extent
    return (xmin + margin) <= p.x <= (xmax - margin) and (ymin + margin) <= p.y <= (ymax - margin)


def _sample_new_settlement_center(
    *,
    rng: np.random.Generator,
    extent: tuple[float, float, float, float],
    forbidden: Polygon,
    existing_centers: list[Point],
    min_dist_m: float,
    margin_m: float,
    max_tries: int = 500,
) -> Point:
    xmin, ymin, xmax, ymax = extent
    for _ in range(max_tries):
        p = Point(
            float(rng.uniform(xmin + margin_m, xmax - margin_m)),
            float(rng.uniform(ymin + margin_m, ymax - margin_m)),
        )
        if forbidden.contains(p):
            continue
        if any(p.distance(q) < min_dist_m for q in existing_centers):
            continue
        return p
    raise ValueError("Could not sample a valid new settlement center (too constrained).")


def _road_angle_deg(a: Point, b: Point) -> float:
    dx = b.x - a.x
    dy = b.y - a.y
    # angle for rotate(): degrees CCW, 0 means aligned with +x axis
    return float(np.degrees(np.arctan2(dy, dx)))


def _make_rect_footprint(center: Point, w: float, h: float, angle_deg: float) -> Polygon:
    # axis-aligned rectangle around origin then rotate + translate
    x0, y0 = center.x, center.y
    rect = Polygon(
        [
            (x0 - w / 2, y0 - h / 2),
            (x0 + w / 2, y0 - h / 2),
            (x0 + w / 2, y0 + h / 2),
            (x0 - w / 2, y0 + h / 2),
        ]
    )
    return rotate(rect, angle_deg, origin=(x0, y0), use_radians=False)


def _safe_union_water(world_t0) -> Polygon:
    waters = list(getattr(world_t0, "water", []))
    if not waters:
        return Polygon()
    geoms = [w.geometry for w in waters if getattr(w, "geometry", None) is not None]
    if not geoms:
        return Polygon()
    u = unary_union(geoms)
    # if union is MultiPolygon, returning as-is is fine (it still has contains/intersects)
    return u


class A1RoadPlusBuildingTask:
    """
    Temporal task:
      t0: baseline world
      t1: add 1 settlement + 1 connecting road + a small cluster of NEW buildings near the new road
    Question:
      Count how many NEW buildings were constructed in the NEW cluster (near the new road).
    """

    code = "A1"
    name = "Road + settlement addition, count new buildings in new cluster"
    is_temporal = True

    def generate_t0(self, cfg: A1Config, rng: np.random.Generator):
        # Keep identical style to other tasks (E1/N1)
        return generate_t0_sample(cfg.world_cfg)

    def _apply_operator_t0_to_t1(
        self,
        *,
        cfg: A1Config,
        world_t0,
        rng: np.random.Generator,
    ):
        import copy

        if len(list(world_t0.settlements)) < cfg.min_existing_settlements:
            raise ValueError("A1 requires at least 2 existing settlements.")

        world_t1 = copy.deepcopy(world_t0)

        # --- choose new settlement center (avoid water + keep distance) ---
        extent = tuple(world_t0.tr.extent)  # (xmin,ymin,xmax,ymax)
        water_union = _safe_union_water(world_t0)
        existing_centers = [s.center for s in world_t0.settlements]

        new_center = _sample_new_settlement_center(
            rng=rng,
            extent=extent,
            forbidden=water_union,
            existing_centers=existing_centers,
            min_dist_m=520.0,
            margin_m=120.0,
        )

        # new settlement id (simple deterministic string)
        new_settlement_id = f"s_new_{int(rng.integers(1_000_000, 9_999_999))}"
        # radius kept modest
        new_radius = float(rng.uniform(180.0, 300.0))

        # Settlement class exists in your world model; we construct by reusing the same type as existing
        SettlementCls = type(world_t0.settlements[0])
        new_settlement = SettlementCls(id=new_settlement_id, center=new_center, radius_m=new_radius)
        world_t1.settlements.append(new_settlement)

        # --- connect with 1 road to nearest existing settlement ---
        if cfg.connect_strategy != "nearest":
            raise ValueError("Only connect_strategy='nearest' supported in MVP.")

        nearest = min(world_t0.settlements, key=lambda s: new_center.distance(s.center))
        a_id = str(nearest.id)
        b_id = str(new_settlement_id)

        road_id = f"r_new_{int(rng.integers(1_000_000, 9_999_999))}"
        centerline = LineString(
            [(nearest.center.x, nearest.center.y), (new_center.x, new_center.y)]
        )

        RoadSegCls = type(world_t0.roads.segments[0]) if world_t0.roads.segments else None
        if RoadSegCls is None:
            # fallback: rely on duck-typing fields used elsewhere (id,a_id,b_id,centerline,width_m)
            raise ValueError("No existing road segment type found (roads.segments empty).")

        new_seg = RoadSegCls(
            id=road_id,
            a_id=a_id,
            b_id=b_id,
            centerline=centerline,
            width_m=float(cfg.road_width_m),
        )
        world_t1.roads.segments.append(new_seg)

        # rebuild road graph if supported
        if hasattr(world_t1.roads, "build_graph_from_segments"):
            world_t1.roads.build_graph_from_segments()

        # --- generate NEW buildings near new road / new settlement ---
        n_new = int(rng.integers(cfg.n_new_buildings[0], cfg.n_new_buildings[1] + 1))
        angle = _road_angle_deg(nearest.center, new_center)

        existing_buildings = list(world_t1.buildings)
        existing_union = (
            unary_union([b.footprint for b in existing_buildings]) if existing_buildings else None
        )

        new_building_ids: list[str] = []

        # sample in a disk around the new settlement, but require proximity to the new road
        for _ in range(cfg.max_attempts_place_buildings):
            if len(new_building_ids) >= n_new:
                break

            # polar sample around new settlement
            r = float(rng.uniform(20.0, cfg.cluster_radius_m))
            t = float(rng.uniform(0.0, 2 * np.pi))
            cx = float(new_center.x + r * np.cos(t))
            cy = float(new_center.y + r * np.sin(t))
            c = Point(cx, cy)

            if not _within_extent(c, extent, margin=40.0):
                continue
            if water_union and water_union.contains(c):
                continue
            if c.distance(centerline) > cfg.road_buffer_m:
                continue

            w = float(rng.uniform(cfg.building_size_m[0], cfg.building_size_m[1]))
            h = float(rng.uniform(cfg.building_size_m[0], cfg.building_size_m[1]))
            fp = _make_rect_footprint(c, w=w, h=h, angle_deg=angle)

            if water_union and water_union.intersects(fp):
                continue

            # avoid overlaps with existing buildings (cheap + robust)
            if existing_union is not None and existing_union.intersects(fp):
                continue

            # also avoid placing too close to other NEW buildings
            if any(
                fp.distance(b.footprint) < cfg.min_dist_new_buildings_m for b in world_t1.buildings
            ):
                continue

            b_id = f"b_new_{int(rng.integers(1_000_000, 9_999_999))}"
            BuildingCls = type(world_t1.buildings[0]) if world_t1.buildings else None
            if BuildingCls is None:
                raise ValueError("No existing Building type found (world.buildings empty).")

            new_b = BuildingCls(id=b_id, footprint=fp, settlement_id=new_settlement_id)
            world_t1.buildings.append(new_b)
            new_building_ids.append(b_id)

            # update union incrementally (fast enough for small n)
            existing_union = unary_union([existing_union, fp]) if existing_union is not None else fp

        if len(new_building_ids) < n_new:
            raise ValueError("Could not place enough new buildings (degenerate sample).")

        return world_t1, new_settlement_id, road_id, new_building_ids

    def build_record(
        self,
        *,
        sample_idx: int,
        cfg: A1Config,
        world_t0,
        render: RenderArtifacts,
        rng: np.random.Generator,
    ) -> dict[str, Any]:
        # Operator: build t1 + remember what is NEW
        world_t1, new_sid, new_rid, new_bids = self._apply_operator_t0_to_t1(
            cfg=cfg, world_t0=world_t0, rng=rng
        )

        # Render pair (run_task/scripts may have already rendered; keep record construction pure)
        # We expect caller to pass a RenderArtifacts that contains t0 + t1 paths
        # (see run_task update below).

        # Pixel anchors for reasoning prompt
        new_settlement = next(s for s in world_t1.settlements if str(s.id) == str(new_sid))
        new_px = px_loc(new_settlement.center, world_t1)

        # Find road segment we created
        new_seg = next(seg for seg in world_t1.roads.segments if str(seg.id) == str(new_rid))
        a_pt = Point(new_seg.centerline.coords[0])
        b_pt = Point(new_seg.centerline.coords[-1])
        a_px = px_loc(a_pt, world_t1)
        b_px = px_loc(b_pt, world_t1)

        answer_int = int(len(new_bids))

        # “Intelligent enough” prompt:
        # - forces temporal comparison (t0 vs t1)
        # - restricts region (near NEW settlement + near NEW road)
        # - avoids trivial global counting
        prompt = (
            f"[{self.code}] You are given two satellite-like images of the same area: t0 (before) and t1 (after).\n"
            f"Between t0 and t1, ONE new settlement was added and connected by ONE new road.\n\n"
            # f"Helpful anchors (pixel coordinates):\n"
            # f"- New settlement center ≈ {tuple(new_px)}\n"
            # f"- New road endpoints ≈ A{tuple(a_px)} to B{tuple(b_px)}\n\n"
            f"Task: Count how many NEW buildings were constructed in the NEW cluster.\n"
            f"Definition of 'NEW cluster' for this question:\n"
            f"  (1) the building appears in t1 but not in t0, AND\n"
            f"  (2) it is near the new road (visually adjacent to the A–B road), AND\n"
            f"  (3) it is in the vicinity of the new settlement (around the given center).\n\n"
            f"Answer with a single integer."
        )

        # Optional oracle px centers (useful for debugging/eval)
        new_buildings = [b for b in world_t1.buildings if str(b.id) in set(new_bids)]
        new_building_centers_px: list[list[int]] = []
        for b in new_buildings:
            c = b.footprint.centroid
            px = px_loc(Point(c.x, c.y), world_t1)
            new_building_centers_px.append([px[0], px[1]])

        return {
            "sample_id": f"{sample_idx:05d}",
            "task_code": self.code,
            "task_name": self.name,
            "modality": "pair",
            "inputs": {
                # expected by your viewer normalizer in run_task.py :contentReference[oaicite:1]{index=1}
                "t0_image": getattr(render, "t0_rgb", None),
                "t1_image": getattr(render, "t1_rgb", None),
                "t0_mask": getattr(render, "t0_mask", None),
                "t1_mask": getattr(render, "t1_mask", None),
                "change_mask": getattr(render, "change_mask", None),
            },
            "prompt": prompt,
            "answer": answer_int,
            "oracle": {
                "new_settlement_id": str(new_sid),
                "new_road_id": str(new_rid),
                "new_building_ids": [str(x) for x in new_bids],
                "new_settlement_center_px": [int(new_px[0]), int(new_px[1])],
                "new_road_endpoints_px": [
                    [int(a_px[0]), int(a_px[1])],
                    [int(b_px[0]), int(b_px[1])],
                ],
                "new_building_centers_px": new_building_centers_px,
                "answer_int": answer_int,
            },
            # keep world_t1 accessible to caller if needed (not written to jsonl by default)
            "_debug_world_t1": world_t1,
        }
