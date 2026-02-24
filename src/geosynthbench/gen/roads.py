from __future__ import annotations

import networkx as nx
import numpy as np
from shapely.geometry import LineString, Point

from geosynthbench.world.entities import RoadNetwork, RoadSegment
from geosynthbench.world.types import RoadId, SettlementId
from geosynthbench.world.world_state import WorldState


def _complete_graph_edges(world: WorldState) -> list[tuple[SettlementId, SettlementId, float]]:
    # settlements indexed by list order
    edges: list[tuple[SettlementId, SettlementId, float]] = []
    for i, a in enumerate(world.settlements):
        for j, b in enumerate(world.settlements):
            if j <= i:
                continue
            d = float(a.center.distance(b.center))
            # edges.append((i, j, d))
            edges.append((a.id, b.id, d))  # use SettlementId instead of index
    return edges


def _mst_edges(world: WorldState) -> list[tuple[SettlementId, SettlementId]]:
    g: nx.Graph = nx.Graph()  # later convert to SettlementId keys
    edges = _complete_graph_edges(world)
    for a_id, b_id, d in edges:
        g.add_edge(a_id, b_id, weight=d)

    mst: nx.Graph = nx.minimum_spanning_tree(g, weight="weight")  # type: ignore

    return [(u, v) for (u, v) in mst.edges()]


def _ring_edges(world: WorldState, n: int) -> list[tuple[SettlementId, SettlementId]]:
    if n < 2:
        return []
    return [(world.settlements[i].id, world.settlements[(i + 1) % n].id) for i in range(n)]


def _chaikin_smooth(coords: list[tuple[float, float]], n_iters: int) -> list[tuple[float, float]]:
    """Chaikin corner-cutting. Keeps endpoints. Returns a denser, smoother polyline.

    https://en.wikipedia.org/wiki/Ca%C3%AFkin%27s_corner-cutting_algorithm
    """
    if len(coords) < 2:
        return coords
    out = coords
    for _ in range(n_iters):
        nxt: list[tuple[float, float]] = [out[0]]
        for i in range(len(out) - 1):
            (x0, y0), (x1, y1) = out[i], out[i + 1]
            q = (0.75 * x0 + 0.25 * x1, 0.75 * y0 + 0.25 * y1)
            r = (0.25 * x0 + 0.75 * x1, 0.25 * y0 + 0.75 * y1)
            nxt.append(q)
            nxt.append(r)
        nxt.append(out[-1])
        out = nxt
    return out


def _curved_link(
    a: Point,
    b: Point,
    rng: np.random.Generator,
    jitter_frac: float = 0.2,
    smooth_iters: int = 2,
) -> LineString:
    """
    Create a curved polyline between a and b by adding 1-2 control points with perpendicular jitter,
    then smoothing.
    """
    ax, ay = a.x, a.y
    bx, by = b.x, b.y
    dx, dy = bx - ax, by - ay
    dist = float(np.hypot(dx, dy))
    if dist <= 1e-6:
        return LineString([a, b])

    # unit perpendicular
    px, py = (-dy / dist, dx / dist)

    # 1 or 2 control points
    k = int(rng.integers(1, 3))
    ts = np.linspace(0.0, 1.0, num=k + 2)[1:-1]
    coords: list[tuple[float, float]] = [(ax, ay)]
    for t in ts:
        base_x = ax + float(t) * dx
        base_y = ay + float(t) * dy
        # jitter amplitude scales with distance
        amp = jitter_frac * dist
        off = float(rng.uniform(-amp, amp))
        coords.append((base_x + off * px, base_y + off * py))
    coords.append((bx, by))

    smoothed = _chaikin_smooth(coords, n_iters=smooth_iters)
    return LineString(smoothed)


def generate_roads(
    world: WorldState, rng: np.random.Generator, mode: str, extra_edges: int, width_m: float
) -> None:
    n = len(world.settlements)
    segs: list[RoadSegment] = []

    if n <= 1:
        world.roads = RoadNetwork([])
        world.roads.rebuild_graph()
        return

    if mode in ("mst", "mst+extras", "dense"):
        base = _mst_edges(world)
    elif mode == "ring":
        base = _ring_edges(world, n)
    else:
        base = _mst_edges(world)

    chosen = set(tuple(sorted(e)) for e in base)

    # extras: add short edges not already in set
    if mode in ("mst+extras", "dense"):
        candidates: list[tuple[float, SettlementId, SettlementId]] = []
        for i in range(n):
            for j in range(i + 1, n):
                if (i, j) in chosen:
                    continue
                settlement_i = world.settlements[i]
                settlement_j = world.settlements[j]
                d = float(settlement_i.center.distance(settlement_j.center))
                candidates.append((d, settlement_i.id, settlement_j.id))
        candidates.sort(key=lambda t: t[0])

        k = extra_edges if mode == "mst+extras" else min(len(candidates), max(extra_edges, n))
        # pick among the shortest with some randomness
        top = candidates[: max(10, 3 * k)]
        rng.shuffle(top)
        for _, i_id, j_id in top[:k]:
            chosen.add((i_id, j_id))

    # build segments
    for idx, (i_id, j_id) in enumerate(sorted(chosen)):
        a = world.settlement_by_id(i_id)
        b = world.settlement_by_id(j_id)
        line = LineString([(a.center.x, a.center.y), (b.center.x, b.center.y)])
        line = _curved_link(a.center, b.center, rng=rng, jitter_frac=0.2, smooth_iters=2)
        # later add some road generation rules based on physics
        segs.append(
            RoadSegment(
                id=RoadId(f"r{idx}"),
                a_id=a.id,
                b_id=b.id,
                centerline=line,
                width_m=width_m,
            )
        )

    rn = RoadNetwork(segments=segs)
    rn.rebuild_graph()
    world.roads = rn
