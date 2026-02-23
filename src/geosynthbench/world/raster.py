from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from shapely.geometry import Point, Polygon
from shapely.prepared import prep


@dataclass(frozen=True)
class RasterTransform:
    """
    Maps world coordinates (meters) to pixel coordinates.
    extent=(xmin,ymin,xmax,ymax), raster shape (H,W).
    """

    extent: tuple[float, float, float, float]
    width_px: int
    height_px: int

    @property
    def xmin(self) -> float:
        return self.extent[0]

    @property
    def ymin(self) -> float:
        return self.extent[1]

    @property
    def xmax(self) -> float:
        return self.extent[2]

    @property
    def ymax(self) -> float:
        return self.extent[3]

    @property
    def dx(self) -> float:
        return (self.xmax - self.xmin) / float(self.width_px)

    @property
    def dy(self) -> float:
        return (self.ymax - self.ymin) / float(self.height_px)

    def world_to_px(self, x: float, y: float) -> tuple[float, float]:
        # px space: (0..W-1, 0..H-1) with y downward
        u = (x - self.xmin) / self.dx
        v = (self.ymax - y) / self.dy
        return u, v

    # vectorized
    def world_to_px_vec(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        u = (x - self.xmin) / self.dx
        v = (self.ymax - y) / self.dy
        return u, v

    def px_to_world(self, u: float, v: float) -> tuple[float, float]:
        x = self.xmin + u * self.dx
        y = self.ymax - v * self.dy
        return x, y

    def px_to_world_vec(self, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = self.xmin + u * self.dx
        y = self.ymax - v * self.dy
        return x, y

    def extent_polygon(self) -> Polygon:
        return Polygon(
            [
                (self.xmin, self.ymin),
                (self.xmax, self.ymin),
                (self.xmax, self.ymax),
                (self.xmin, self.ymax),
            ]
        )


@dataclass
class HeightField:
    tr: RasterTransform
    elevation_m: np.ndarray  # shape (H,W), float32

    _slope_cache: np.ndarray | None = None  # shape (H,W)

    def slope(self) -> np.ndarray:
        """
        Returns slope magnitude in meters per meter (approx),
        computed from gradient in world units.
        """
        if self._slope_cache is not None:
            return self._slope_cache

        elev = self.elevation_m.astype(np.float32, copy=False)
        # gradient in pixels:
        gy_px, gx_px = np.gradient(elev)
        # convert to world units: dz/dx, dz/dy (meters per meter)
        gx = gx_px / self.tr.dx
        gy = gy_px / self.tr.dy
        slope = np.sqrt(gx * gx + gy * gy).astype(np.float32)
        self._slope_cache = slope
        return slope

    def sample_point(self, x: float, y: float) -> float:
        u, v = self.tr.world_to_px(x, y)
        ui = int(np.clip(round(u), 0, self.tr.width_px - 1))
        vi = int(np.clip(round(v), 0, self.tr.height_px - 1))
        return float(self.elevation_m[vi, ui])

    def sample_slope_point(self, x: float, y: float) -> float:
        s = self.slope()
        u, v = self.tr.world_to_px(x, y)
        ui = int(np.clip(round(u), 0, self.tr.width_px - 1))
        vi = int(np.clip(round(v), 0, self.tr.height_px - 1))
        return float(s[vi, ui])

    def poly_stats(self, poly: Polygon, max_points: int = 2048) -> dict[str, float]:
        """
        Approximate stats under polygon by sampling a grid of points in its bbox.
        """
        if poly.is_empty:
            return {
                "elev_mean": 0.0,
                "elev_max": 0.0,
                "slope_mean": 0.0,
                "slope_max": 0.0,
                "n": 0.0,
            }

        minx, miny, maxx, maxy = poly.bounds
        # choose sampling resolution based on bbox size
        # target about max_points samples
        area = max((maxx - minx) * (maxy - miny), 1e-6)
        step = np.sqrt(area / float(max_points))
        step = max(step, max(self.tr.dx, self.tr.dy))  # not smaller than 1 px

        xs = np.arange(minx, maxx + 1e-9, step)
        ys = np.arange(miny, maxy + 1e-9, step)
        if xs.size == 0 or ys.size == 0:
            cx, cy = poly.centroid.x, poly.centroid.y
            return {
                "elev_mean": self.sample_point(cx, cy),
                "elev_max": self.sample_point(cx, cy),
                "slope_mean": self.sample_slope_point(cx, cy),
                "slope_max": self.sample_slope_point(cx, cy),
                "n": 1.0,
            }

        P = prep(poly)
        elev_vals: list[float] = []
        slope_vals: list[float] = []
        for y in ys:
            for x in xs:
                if P.contains(Point(float(x), float(y))):
                    elev_vals.append(self.sample_point(float(x), float(y)))
                    slope_vals.append(self.sample_slope_point(float(x), float(y)))

        if not elev_vals:
            cx, cy = poly.centroid.x, poly.centroid.y
            return {
                "elev_mean": self.sample_point(cx, cy),
                "elev_max": self.sample_point(cx, cy),
                "slope_mean": self.sample_slope_point(cx, cy),
                "slope_max": self.sample_slope_point(cx, cy),
                "n": 1.0,
            }

        e = np.array(elev_vals, dtype=np.float32)
        s = np.array(slope_vals, dtype=np.float32)
        return {
            "elev_mean": float(e.mean()),
            "elev_max": float(e.max()),
            "slope_mean": float(s.mean()),
            "slope_max": float(s.max()),
            "n": float(e.size),
        }
