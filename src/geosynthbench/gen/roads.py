from __future__ import annotations

import numpy as np
import networkx as nx
from shapely.geometry import LineString

from geosynthbench.world.entities import RoadNetwork, RoadSegment
from geosynthbench.world.world_state import WorldState
from geosynthbench.world.types import RoadId


def _complete_graph_edges(world: WorldState) -> list[tuple[int, int, float]]:
    # settlements indexed by list order
    edges: list[tuple[int, int, float]] = []
    for i, a in enumerate(world.settlements):
        for j, b in enumerate(world.settlements):
            if j <= i:
                continue
            d = float(a.center.distance(b.center))
            edges.append((i, j, d))
    return edges


def _mst_edges(world: WorldState) -> list[tuple[int, int]]:
    g = nx.Graph()
    edges = _complete_graph_edges(world)
    for i, j, d in edges:
        g.add_edge(i, j, weight=d)
    mst = nx.minimum_spanning_tree(g, weight="weight")
    return [(int(u), int(v)) for (u, v) in mst.edges()]


def _ring_edges(n: int) -> list[tuple[int, int]]:
    if n < 2:
        return []
    return [(i, (i + 1) % n) for i in range(n)]


def generate_roads(world: WorldState, rng: np.random.Generator, mode: str, extra_edges: int, width_m: float) -> None:
    n = len(world.settlements)
    segs: list[RoadSegment] = []

    if n <= 1:
        world.roads = RoadNetwork([])
        world.roads.rebuild_graph()
        return

    if mode in ("mst", "mst+extras", "dense"):
        base = _mst_edges(world)
    elif mode == "ring":
        base = _ring_edges(n)
    else:
        base = _mst_edges(world)

    chosen = set(tuple(sorted(e)) for e in base)

    # extras: add short edges not already in set
    if mode in ("mst+extras", "dense"):
        candidates = []
        for i in range(n):
            for j in range(i + 1, n):
                if (i, j) in chosen:
                    continue
                d = float(world.settlements[i].center.distance(world.settlements[j].center))
                candidates.append((d, i, j))
        candidates.sort(key=lambda t: t[0])

        k = extra_edges if mode == "mst+extras" else min(len(candidates), max(extra_edges, n))
        # pick among the shortest with some randomness
        top = candidates[: max(10, 3 * k)]
        rng.shuffle(top)
        for _, i, j in top[:k]:
            chosen.add((i, j))

    # build segments
    for idx, (i, j) in enumerate(sorted(chosen)):
        a = world.settlements[i]
        b = world.settlements[j]
        line = LineString([(a.center.x, a.center.y), (b.center.x, b.center.y)])
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
