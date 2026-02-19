from __future__ import annotations

import hashlib

import numpy as np
from shapely.geometry.base import BaseGeometry

from geosynthbench.world.world_state import WorldState


def _h_update(h: "hashlib._Hash", s: str) -> None:
    h.update(s.encode("utf-8"))
    h.update(b"\x1f")


def _geom_sig(g: BaseGeometry) -> bytes:
    """
    Use WKB for stable geometry bytes. This avoids minor WKT formatting differences.
    """
    return g.wkb


def world_fingerprint(world: WorldState) -> str:
    """
    Stable signature for round-trip equivalence checks.
    Includes:
      - raster transform params
      - terrain elevation bytes (if present)
      - entity geometry bytes + key numeric fields
      - road segment definitions (a_id/b_id/width)
      - building settlement links
    """
    h = hashlib.blake2b(digest_size=16)

    # Raster
    _h_update(h, "raster")
    _h_update(h, ",".join(map(str, world.tr.extent)))
    _h_update(h, f"{world.tr.width_px}x{world.tr.height_px}")

    # Terrain
    _h_update(h, "terrain")
    if world.terrain is None:
        _h_update(h, "none")
    else:
        elev = world.terrain.elevation_m.astype(np.float32, copy=False)
        _h_update(h, f"shape={elev.shape}")
        # include raw bytes (exact equality)
        h.update(elev.tobytes(order="C"))

    # Water
    _h_update(h, "water")
    for w in sorted(world.water, key=lambda x: str(x.id)):
        _h_update(h, str(w.id))
        h.update(_geom_sig(w.polygon))

    # Vegetation
    _h_update(h, "veg")
    for v in sorted(world.vegetation, key=lambda x: str(x.id)):
        _h_update(h, str(v.id))
        _h_update(h, f"density={v.density:.6f}")
        h.update(_geom_sig(v.polygon))

    # Settlements
    _h_update(h, "settlements")
    for s in sorted(world.settlements, key=lambda x: str(x.id)):
        _h_update(h, str(s.id))
        _h_update(h, f"radius={s.radius_m:.6f}")
        h.update(_geom_sig(s.center))

    # Roads (segments only — graph is derived)
    _h_update(h, "roads")
    for r in sorted(world.roads.segments, key=lambda x: str(x.id)):
        _h_update(h, str(r.id))
        _h_update(h, f"{r.a_id}->{r.b_id}")
        _h_update(h, f"width={r.width_m:.6f}")
        h.update(_geom_sig(r.centerline))

    # Buildings
    _h_update(h, "buildings")
    for b in sorted(world.buildings, key=lambda x: str(x.id)):
        _h_update(h, str(b.id))
        _h_update(h, f"settlement={b.settlement_id}")
        _h_update(h, f"near_road={'' if b.near_road_id is None else b.near_road_id}")
        h.update(_geom_sig(b.footprint))

    return h.hexdigest()


def assert_same_world(w0: WorldState, w1: WorldState) -> None:
    fp0 = world_fingerprint(w0)
    fp1 = world_fingerprint(w1)
    if fp0 != fp1:
        raise AssertionError(f"World mismatch:\n  fp0={fp0}\n  fp1={fp1}")
