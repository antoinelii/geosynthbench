from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np

from geosynthbench.pipeline.writer import RenderArtifacts


@dataclass(frozen=True)
class N1Config:
    world_cfg: Any
    clarity_ratio: float = 1.15
    clarity_delta_m: float = 300.0
    strategy: str = "by_id"  # tie-break: "by_id" for determinism


def compute_isolation_scores(world) -> dict[str, float]:
    """
    Isolation score per settlement id:
      iso(i) = mean_j shortest_path_length(i,j)
    Uses actual curved RoadSegment.centerline.length for edge weights.
    """
    G = nx.Graph()
    for s in world.settlements:
        G.add_node(s.id)

    for seg in world.roads.segments:
        w = float(seg.centerline.length)  # IMPORTANT: curved length
        G.add_edge(seg.a_id, seg.b_id, weight=w)

    # if graph somehow disconnected (shouldn't if require_connected_roads=True), handle safely:
    # compute per connected component; unreachable pairs get +inf => isolated will be in smaller component.
    lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight="weight"))

    ids = [s.id for s in world.settlements]
    scores: dict[str, float] = {}

    for i in ids:
        dsum = 0.0
        cnt = 0
        for j in ids:
            if i == j:
                continue
            dij = lengths.get(i, {}).get(j, float("inf"))
            dsum += float(dij)
            cnt += 1
        scores[str(i)] = dsum / max(cnt, 1)

    return scores


def pick_most_isolated(scores: dict[str, float], *, tie_break: str = "by_id") -> str:
    """
    Returns settlement id of maximum isolation. Tie-break deterministic by id.
    """
    if tie_break == "by_id":
        return max(scores.items(), key=lambda kv: (kv[1], kv[0]))[0]
    return max(scores.items(), key=lambda kv: kv[1])[0]


def isolation_is_clear(scores: dict[str, float], ratio: float, delta_m: float) -> bool:
    vals = sorted(scores.values(), reverse=True)
    if len(vals) < 2:
        return False
    return (vals[0] >= vals[1] * ratio) or ((vals[0] - vals[1]) >= delta_m)


class N1IsolationTask:
    code = "N1"
    name = "Most isolated settlement in road network"
    is_temporal = False

    def generate_t0(self, cfg: N1Config, rng: np.random.Generator):
        from geosynthbench.tasks.utils import generate_t0_sample

        world_cfg = cfg.world_cfg

        return generate_t0_sample(world_cfg)

    def build_record(
        self,
        *,
        sample_idx: int,
        cfg: N1Config,
        world_t0,
        render: RenderArtifacts,
        rng: np.random.Generator,
    ) -> dict[str, Any]:
        scores = compute_isolation_scores(world_t0)
        best = pick_most_isolated(scores, tie_break=cfg.strategy)

        prompt = (
            f"[{self.code}] A road network connects multiple settlements.\n"
            f"Which settlement is the MOST isolated, defined as the one with the largest "
            f"average shortest-path distance (by road) to all other settlements?\n"
            f"Answer with the settlement id (e.g., s0, s1, ...)."
        )

        return {
            "sample_id": f"{sample_idx:05d}",
            "task_code": self.code,
            "task_name": self.name,
            "modality": "single",
            "inputs": {
                "image": render.t0_rgb,
                "mask": render.t0_mask,
            },
            "prompt": prompt,
            "answer": best,
            "oracle": {
                "isolation_scores_m": scores,
                "best_id": best,
            },
        }
