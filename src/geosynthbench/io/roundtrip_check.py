from __future__ import annotations

import hashlib
from typing import Protocol

import numpy as np
from shapely.geometry.base import BaseGeometry

from geosynthbench.world.world_state import WorldState


class HashLike(Protocol):
    def update(self, data: bytes | bytearray | memoryview, /) -> None: ...


def _h_update(h: HashLike, s: str) -> None:
    h.update(s.encode("utf-8"))
    h.update(b"\x1f")


def _geom_sig(g: BaseGeometry) -> bytes:
    return g.wkb


def world_fingerprint(world: WorldState) -> str:
    h = hashlib.blake2b(digest_size=16)

    _h_update(h, "raster")
    _h_update(h, ",".join(map(str, world.tr.extent)))
    _h_update(h, f"{world.tr.width_px}x{world.tr.height_px}")

    _h_update(h, "terrain")
    if world.terrain is None:
        _h_update(h, "none")
    else:
        elev = world.terrain.elevation_m.astype(np.float32, copy=False)
        _h_update(h, f"shape={elev.shape}")
        h.update(elev.tobytes(order="C"))

    _h_update(h, "water")
    for w in sorted(world.water, key=lambda x: str(x.id)):
        _h_update(h, str(w.id))
        h.update(_geom_sig(w.polygon))

    _h_update(h, "veg")
    for v in sorted(world.vegetation, key=lambda x: str(x.id)):
        _h_update(h, str(v.id))
        _h_update(h, f"density={v.density:.6f}")
        h.update(_geom_sig(v.polygon))

    _h_update(h, "settlements")
    for s in sorted(world.settlements, key=lambda x: str(x.id)):
        _h_update(h, str(s.id))
        _h_update(h, f"radius={s.radius_m:.6f}")
        h.update(_geom_sig(s.center))

    _h_update(h, "roads")
    for r in sorted(world.roads.segments, key=lambda x: str(x.id)):
        _h_update(h, str(r.id))
        _h_update(h, f"{r.a_id}->{r.b_id}")
        _h_update(h, f"width={r.width_m:.6f}")
        h.update(_geom_sig(r.centerline))

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


def world_fingerprint_debug(w0: WorldState, w1: WorldState) -> bool:
    """
    Stable signature for round-trip equivalence checks.
    Includes:
      - raster transform params
      - terrain elevation bytes (if present)
      - entity geometry bytes + key numeric fields
      - road segment definitions (a_id/b_id/width)
      - building settlement links
    """
    h0 = hashlib.blake2b(digest_size=16)
    h1 = hashlib.blake2b(digest_size=16)

    # Raster
    _h_update(h0, "raster")
    _h_update(h1, "raster")
    assert h0.hexdigest() == h1.hexdigest()  # should be identical up to this point
    _h_update(h0, ",".join(map(str, w0.tr.extent)))
    _h_update(h1, ",".join(map(str, w1.tr.extent)))
    print(f"After extent: \n w0={w0.tr.extent}\n w1={w1.tr.extent}")
    assert h0.hexdigest() == h1.hexdigest()  # should be identical up to this point
    _h_update(h0, f"{w0.tr.width_px}x{w0.tr.height_px}")
    _h_update(h1, f"{w1.tr.width_px}x{w1.tr.height_px}")
    assert h0.hexdigest() == h1.hexdigest()

    # Terrain
    _h_update(h0, "terrain")
    _h_update(h1, "terrain")

    if w0.terrain is None:
        _h_update(h0, "none")
    else:
        elev = w0.terrain.elevation_m.astype(np.float32, copy=False)
        _h_update(h0, f"shape={elev.shape}")
        # include raw bytes (exact equality)
        h0.update(elev.tobytes(order="C"))

    if w1.terrain is None:
        _h_update(h1, "none")
    else:
        elev = w1.terrain.elevation_m.astype(np.float32, copy=False)
        _h_update(h1, f"shape={elev.shape}")
        # include raw bytes (exact equality)
        h1.update(elev.tobytes(order="C"))

    assert h0.hexdigest() == h1.hexdigest()  # should be identical up to this point

    # Water
    _h_update(h0, "water")
    for w in sorted(w0.water, key=lambda x: str(x.id)):
        _h_update(h0, str(w.id))
        h0.update(_geom_sig(w.polygon))

    _h_update(h1, "water")
    for w in sorted(w1.water, key=lambda x: str(x.id)):
        _h_update(h1, str(w.id))
        h1.update(_geom_sig(w.polygon))

    assert h0.hexdigest() == h1.hexdigest()  # should be identical up to this point

    # Vegetation
    _h_update(h0, "veg")
    for v in sorted(w0.vegetation, key=lambda x: str(x.id)):
        _h_update(h0, str(v.id))
        _h_update(h0, f"density={v.density:.6f}")
        h0.update(_geom_sig(v.polygon))

    _h_update(h1, "veg")
    for v in sorted(w1.vegetation, key=lambda x: str(x.id)):
        _h_update(h1, str(v.id))
        _h_update(h1, f"density={v.density:.6f}")
        h1.update(_geom_sig(v.polygon))

    assert h0.hexdigest() == h1.hexdigest()  # should be identical up to this point

    # Settlements
    _h_update(h0, "settlements")
    for s in sorted(w0.settlements, key=lambda x: str(x.id)):
        _h_update(h0, str(s.id))
        _h_update(h0, f"radius={s.radius_m:.6f}")
        h0.update(_geom_sig(s.center))

    _h_update(h1, "settlements")
    for s in sorted(w1.settlements, key=lambda x: str(x.id)):
        _h_update(h1, str(s.id))
        _h_update(h1, f"radius={s.radius_m:.6f}")
        h1.update(_geom_sig(s.center))

    assert h0.hexdigest() == h1.hexdigest()  # should be identical up to this point

    # Roads (segments only — graph is derived)
    _h_update(h0, "roads")
    for r in sorted(w0.roads.segments, key=lambda x: str(x.id)):
        _h_update(h0, str(r.id))
        _h_update(h0, f"{r.a_id}->{r.b_id}")
        _h_update(h0, f"width={r.width_m:.6f}")
        h0.update(_geom_sig(r.centerline))

    _h_update(h1, "roads")
    for r in sorted(w1.roads.segments, key=lambda x: str(x.id)):
        _h_update(h1, str(r.id))
        _h_update(h1, f"{r.a_id}->{r.b_id}")
        _h_update(h1, f"width={r.width_m:.6f}")
        h1.update(_geom_sig(r.centerline))

    assert h0.hexdigest() == h1.hexdigest()  # should be identical up to this point

    # Buildings
    _h_update(h0, "buildings")
    for b in sorted(w0.buildings, key=lambda x: str(x.id)):
        _h_update(h0, str(b.id))
        _h_update(h0, f"settlement={b.settlement_id}")
        _h_update(h0, f"near_road={'' if b.near_road_id is None else b.near_road_id}")
        h0.update(_geom_sig(b.footprint))

    _h_update(h1, "buildings")
    for b in sorted(w1.buildings, key=lambda x: str(x.id)):
        _h_update(h1, str(b.id))
        _h_update(h1, f"settlement={b.settlement_id}")
        _h_update(h1, f"near_road={'' if b.near_road_id is None else b.near_road_id}")
        h1.update(_geom_sig(b.footprint))

    assert h0.hexdigest() == h1.hexdigest()  # should be identical up to this point

    return h0.hexdigest() == h1.hexdigest()
