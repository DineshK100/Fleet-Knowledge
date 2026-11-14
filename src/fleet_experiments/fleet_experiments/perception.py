"""Simulated (privileged) perception: nearest obstacle in the planned path.

The baseline driver needs to "see" obstacles to brake for them, and the
surprise trigger needs time-to-collision. We use CARLA ground truth, limited
by a per-object visibility range. This models late detection (cones that are
hard to see, a car hidden behind a curve). The pothole is deliberately not
perceivable: only the fleet memory can warn about it.

All computations here are in CARLA coordinates.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from memory_reader.geometry import RoutePolyline

from .carla_utils import actor_kind


@dataclass
class Obstacle:
    actor_id: int
    kind: str
    distance: float           # free distance along the path (bumper to bumper), m
    closing_speed: float      # m/s
    ttc: float                # s
    x: float                  # CARLA coordinates
    y: float
    stationary: bool


@dataclass
class PerceptionConfig:
    corridor_half_width_m: float = 1.4
    default_range_m: float = 45.0
    horizon_m: float = 70.0


class GroundTruthPerception:
    def __init__(self, world, ego, cfg: PerceptionConfig = None):
        self.world = world
        self.ego = ego
        self.cfg = cfg or PerceptionConfig()
        self.ego_half_len = ego.bounding_box.extent.x

    def detect(self, path_pts: List[Tuple[float, float]],
               extra: Dict[int, Tuple[object, float]]) -> Optional[Obstacle]:
        """``path_pts``: ego position followed by the upcoming route points (CARLA x, y).
        ``extra``: hazard actors {id: (actor, visibility_m)} that are not vehicles/walkers."""
        if len(path_pts) < 2:
            return None
        route = RoutePolyline(path_pts)
        ego_loc = self.ego.get_location()
        ego_v = self.ego.get_velocity()
        ego_speed = math.sqrt(ego_v.x ** 2 + ego_v.y ** 2)
        horizon = min(self.cfg.horizon_m, route.length)

        candidates = {}
        for a in self.world.get_actors().filter('vehicle.*'):
            if a.id != self.ego.id:
                candidates[a.id] = (a, self.cfg.default_range_m)
        for a in self.world.get_actors().filter('walker.pedestrian.*'):
            candidates[a.id] = (a, self.cfg.default_range_m)
        candidates.update(extra)         # hazard-specific visibility wins

        best = None
        for aid, (actor, vis) in candidates.items():
            try:
                loc = actor.get_location()
            except RuntimeError:
                continue
            if loc.distance(ego_loc) > horizon + 10.0:
                continue
            s, lat, _ = route.project(loc.x, loc.y)
            if s <= 0.5 or s > horizon:
                continue
            ext = actor.bounding_box.extent
            half_w = max(ext.x, ext.y) if actor_kind(actor) == 'walker' else min(ext.x, ext.y)
            if lat > self.cfg.corridor_half_width_m + half_w:
                continue
            dist = s - self.ego_half_len - max(ext.x, ext.y) * 0.5
            if dist > vis:
                continue
            dist = max(0.0, dist)
            av = actor.get_velocity()
            hdg = route.heading_at(s)
            along_v = av.x * math.cos(hdg) + av.y * math.sin(hdg)
            closing = ego_speed - along_v
            ttc = dist / closing if closing > 0.1 else 1e9
            if best is None or dist < best.distance:
                best = Obstacle(aid, actor_kind(actor), dist, closing, ttc, loc.x, loc.y,
                                math.sqrt(av.x ** 2 + av.y ** 2) < 0.3)
        return best
