"""2-D route geometry helpers (coordinate-frame agnostic, no ROS dependency)."""

import math
from typing import List, Optional, Sequence, Tuple


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class RoutePolyline:
    """Polyline with arc-length parameterisation and point projection."""

    def __init__(self, points: Sequence[Tuple[float, float]]):
        pts: List[Tuple[float, float]] = []
        for x, y in points:
            if not pts or math.hypot(x - pts[-1][0], y - pts[-1][1]) > 1e-3:
                pts.append((float(x), float(y)))
        self.pts = pts
        self.s = [0.0]
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            self.s.append(self.s[-1] + math.hypot(x1 - x0, y1 - y0))

    def __len__(self):
        return len(self.pts)

    @property
    def length(self) -> float:
        return self.s[-1] if self.s else 0.0

    def project(self, x: float, y: float, s_min: Optional[float] = None,
                s_max: Optional[float] = None) -> Tuple[float, float, int]:
        """Closest point on the polyline: returns (arc length s, lateral distance, segment idx).

        Only segments overlapping ``[s_min, s_max]`` are considered (useful for
        routes that pass the same place twice).
        """
        if not self.pts:
            return 0.0, float('inf'), 0
        if len(self.pts) == 1:
            return 0.0, math.hypot(x - self.pts[0][0], y - self.pts[0][1]), 0
        best = None
        for i in range(len(self.pts) - 1):
            if s_max is not None and self.s[i] > s_max:
                break
            if s_min is not None and self.s[i + 1] < s_min:
                continue
            ax, ay = self.pts[i]
            bx, by = self.pts[i + 1]
            dx, dy = bx - ax, by - ay
            seg2 = dx * dx + dy * dy
            t = 0.0 if seg2 <= 0 else max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / seg2))
            px, py = ax + t * dx, ay + t * dy
            d = math.hypot(x - px, y - py)
            if best is None or d < best[1]:
                best = (self.s[i] + t * math.sqrt(seg2), d, i)
        if best is None:
            return 0.0, float('inf'), 0
        return best

    def point_at(self, s: float) -> Tuple[float, float]:
        if not self.pts:
            return 0.0, 0.0
        s = max(0.0, min(self.length, s))
        for i in range(len(self.pts) - 1):
            if self.s[i + 1] >= s:
                seg = self.s[i + 1] - self.s[i]
                t = 0.0 if seg <= 0 else (s - self.s[i]) / seg
                (ax, ay), (bx, by) = self.pts[i], self.pts[i + 1]
                return ax + t * (bx - ax), ay + t * (by - ay)
        return self.pts[-1]

    def heading_at(self, s: float) -> float:
        if len(self.pts) < 2:
            return 0.0
        s = max(0.0, min(self.length, s))
        for i in range(len(self.pts) - 1):
            if self.s[i + 1] >= s:
                (ax, ay), (bx, by) = self.pts[i], self.pts[i + 1]
                return math.atan2(by - ay, bx - ax)
        (ax, ay), (bx, by) = self.pts[-2], self.pts[-1]
        return math.atan2(by - ay, bx - ax)

    def max_curvature(self, s0: float, s1: float, step: float = 5.0) -> float:
        """Approximate maximum curvature (1/m) between arc lengths s0 and s1."""
        k = 0.0
        s = s0
        while s + step <= min(s1, self.length):
            dh = abs(wrap_angle(self.heading_at(s + step) - self.heading_at(s)))
            k = max(k, dh / step)
            s += step
        return k


def route_distance(route: RoutePolyline, ego_xy: Tuple[float, float],
                   target_xy: Tuple[float, float], corridor_m: float = 6.0,
                   ego_search_m: float = 60.0, behind_m: float = 30.0) -> Optional[float]:
    """Signed distance along ``route`` from the ego to the target.

    Returns ``None`` when the target is further than ``corridor_m`` from the
    route (i.e. not on the planned path). Negative values mean the target has
    just been passed.
    """
    if len(route) == 0:
        return None
    s_e, _, _ = route.project(ego_xy[0], ego_xy[1], s_max=ego_search_m)
    s_t, lat, _ = route.project(target_xy[0], target_xy[1], s_min=s_e - behind_m)
    if lat > corridor_m:
        return None
    if s_t <= 1e-6 and route.length > 0:
        # target projects onto the very start of the route: it may be behind us
        hx = route.heading_at(0.0)
        dx, dy = target_xy[0] - ego_xy[0], target_xy[1] - ego_xy[1]
        along = dx * math.cos(hx) + dy * math.sin(hx)
        if along < 0:
            return along
    return s_t - s_e
