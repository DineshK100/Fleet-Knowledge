"""Hazard scenarios placed along a CARLA route (Section 3.3 of the proposal).

Hazards are not hard-coded to map coordinates. Each one is placed
automatically at the first route position, after ``after_m`` metres, that
fits its type:

* ``road_damage``          - straight, non-junction road. Visually a debris
  decal. Physically, driving over it faster than ``safe_speed_kmh`` gives a
  violent jolt (a sudden speed loss), i.e. a hard brake / jerk spike.
* ``lane_closure``         - non-junction, multi-lane road. Traffic cones in a
  taper and barriers close the route lane. Hard to see (``visibility_m``).
* ``pedestrian_crossing``  - the approach to an intersection. A van is parked
  at the kerb and a pedestrian hidden in front of it steps out when the ego is
  ``trigger_distance_m`` away.
* ``stalled_vehicle``      - (stretch) a stopped car with hazard lights just
  after a curve.
"""

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .carla_utils import actor_kind, carla_to_ros_xy, carla_yaw_to_ros

HAZARD_TYPES = ('pedestrian_crossing', 'lane_closure', 'road_damage', 'stalled_vehicle')


@dataclass
class HazardSpec:
    name: str
    type: str
    after_m: float = 50.0
    enabled: bool = True
    visibility_m: float = 25.0
    active_from_car: int = 1
    active_until_car: Optional[int] = None     # None = forever (stale-hazard experiments)
    params: Dict = field(default_factory=dict)

    @staticmethod
    def from_dict(d: dict) -> 'HazardSpec':
        known = {'name', 'type', 'after_m', 'enabled', 'visibility_m', 'active_from_car',
                 'active_until_car'}
        spec = HazardSpec(name=d.get('name', d['type']), type=d['type'],
                          **{k: d[k] for k in known - {'name', 'type'} if k in d})
        spec.params = {k: v for k, v in d.items() if k not in known}
        return spec


# ----------------------------------------------------------------- geometry
def route_with_s(route) -> List[Tuple[float, object, object]]:
    """[(wp, road_option)] -> [(s, wp, road_option)] with cumulative arc length."""
    out, s, prev = [], 0.0, None
    for wp, opt in route:
        loc = wp.transform.location
        if prev is not None:
            s += loc.distance(prev)
        out.append((s, wp, opt))
        prev = loc
    return out


def _offset(carla, loc, vec, d, dz=0.0):
    return carla.Location(x=loc.x + vec.x * d, y=loc.y + vec.y * d, z=loc.z + dz)


def _along(wp, d):
    """Waypoint ``d`` metres along the lane (negative = backwards)."""
    if abs(d) < 0.05:
        return wp
    nxt = wp.next(d) if d > 0 else wp.previous(-d)
    return nxt[0] if nxt else wp


def same_dir_adjacent(carla, wp, side: str):
    adj = wp.get_left_lane() if side == 'left' else wp.get_right_lane()
    if adj is None or adj.lane_type != carla.LaneType.Driving or adj.lane_id * wp.lane_id <= 0:
        return None
    return adj


def _yaw_diff(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _window(route_s, i, before, after):
    """Route entries within [s_i - before, s_i + after] (route_s is sorted by s)."""
    s0 = route_s[i][0]
    lo = i
    while lo > 0 and route_s[lo - 1][0] >= s0 - before:
        lo -= 1
    hi = i
    while hi + 1 < len(route_s) and route_s[hi + 1][0] <= s0 + after:
        hi += 1
    return route_s[lo:hi + 1]


def _window_clear(route_s, i, before, after) -> bool:
    return all(not wp.is_junction for _, wp, _ in _window(route_s, i, before, after))


def _wp_at_s(route_s, s):
    best = min(route_s, key=lambda r: abs(r[0] - s))
    return best[1]


def find_spot(carla, route_s, htype: str, s_from: float, params: dict) -> Optional[int]:
    """Index into ``route_s`` of the first suitable spot for ``htype`` after ``s_from``."""
    total = route_s[-1][0] if route_s else 0.0
    for i, (s, wp, _opt) in enumerate(route_s):
        if s < s_from or s > total - 40.0:
            continue
        if wp.is_junction:
            continue
        if htype == 'road_damage':
            if _window_clear(route_s, i, 20, 15):
                return i
        elif htype == 'lane_closure':
            if not _window_clear(route_s, i, 40, 30):
                continue
            side = None
            for cand in ('left', 'right'):
                if all(same_dir_adjacent(carla, w, cand) is not None
                       for _, w, _ in _window(route_s, i, 35, 25)):
                    side = cand
                    break
            if side is not None:
                return i
        elif htype == 'pedestrian_crossing':
            first_j = next((r for r in _window(route_s, i, 0.0, 14.0)[1:] if r[1].is_junction), None)
            if first_j is None or first_j[0] - s < 5.0:
                continue
            if _window_clear(route_s, i, 30, 0):
                return i
        elif htype == 'stalled_vehicle':
            if not _window_clear(route_s, i, 10, 30):
                continue
            w_before = _wp_at_s(route_s, s - 35.0)
            w_mid = _wp_at_s(route_s, s - 5.0)
            if _yaw_diff(w_before.transform.rotation.yaw, w_mid.transform.rotation.yaw) < \
                    params.get('min_curve_deg', 30.0):
                continue
            if any(same_dir_adjacent(carla, wp, sd) for sd in ('left', 'right')):
                return i
    return None


def plan_placements(carla, route_s, specs: List[HazardSpec], min_gap_m: float = 70.0
                    ) -> Optional[List[Tuple[HazardSpec, int]]]:
    out, last_s = [], -1e9
    for spec in sorted([s for s in specs if s.enabled], key=lambda s: s.after_m):
        idx = find_spot(carla, route_s, spec.type, max(spec.after_m, last_s + min_gap_m), spec.params)
        if idx is None:
            return None
        out.append((spec, idx))
        last_s = route_s[idx][0]
    return out


# ------------------------------------------------------------------ hazards
class Hazard:
    def __init__(self, carla, world, spec: HazardSpec, route_s, idx: int, rng: random.Random):
        self.carla, self.world, self.spec, self.rng = carla, world, spec, rng
        self.s, self.wp, _ = route_s[idx]
        self.location = self.wp.transform.location
        self.actors: List = []
        self.perceivable: Dict[int, Tuple[object, float]] = {}   # actor id -> (actor, visibility)
        self.ignored_collision_ids = set()
        self.active = False
        self.lane = 'unknown'

    # common -----------------------------------------------------------
    @property
    def name(self):
        return self.spec.name

    @property
    def type(self):
        return self.spec.type

    def ros_xy(self):
        return carla_to_ros_xy(self.location.x, self.location.y)

    def ros_heading(self):
        return carla_yaw_to_ros(self.wp.transform.rotation.yaw)

    def is_active_for(self, car: int) -> bool:
        until = self.spec.active_until_car
        return self.spec.active_from_car <= car and (until is None or car <= until)

    def _bp(self, *names):
        lib = self.world.get_blueprint_library()
        for n in names:
            found = lib.filter(n)
            if len(found):
                return found[0] if '*' not in n else self.rng.choice(list(found))
        return None

    def _spawn(self, bp, loc, yaw, physics=False, visible_m=None):
        if bp is None:
            return None
        tf = self.carla.Transform(loc, self.carla.Rotation(yaw=yaw))
        actor = self.world.try_spawn_actor(bp, tf)
        if actor is None:
            return None
        if not physics:
            try:
                actor.set_simulate_physics(False)
            except RuntimeError:
                pass
        self.actors.append(actor)
        if visible_m is not None:
            self.perceivable[actor.id] = (actor, visible_m)
        return actor

    def spawn(self):
        raise NotImplementedError

    def destroy(self):
        for a in self.actors:
            try:
                if a.is_alive:
                    a.destroy()
            except RuntimeError:
                pass
        self.actors.clear()
        self.perceivable.clear()
        self.ignored_collision_ids.clear()

    def prepare_for_car(self, car: int):
        want = self.is_active_for(car)
        if want and not self.active:
            self.spawn()
        elif not want and self.active:
            self.destroy()
        self.active = want
        if want:
            self.reset_for_car(car)

    def reset_for_car(self, car: int):
        pass

    def tick(self, ego, dt: float):
        pass

    def oracle_text(self) -> str:
        return self.type.replace('_', ' ')

    def info(self) -> dict:
        x, y = self.ros_xy()
        return {'name': self.name, 'type': self.type, 'x': x, 'y': y, 'route_s': self.s,
                'lane': self.lane, 'road_id': self.wp.road_id, 'lane_id': self.wp.lane_id,
                'active_from_car': self.spec.active_from_car,
                'active_until_car': self.spec.active_until_car}


class RoadDamage(Hazard):
    """Hidden pothole: invisible to perception, punishes speed with a jolt."""

    def spawn(self):
        p = self.spec.params
        bp = self._bp(p.get('visual_prop', 'static.prop.dirtdebris01'), 'static.prop.dirtdebris02')
        a = self._spawn(bp, self.carla.Location(self.location.x, self.location.y,
                                                self.location.z + 0.02),
                        self.wp.transform.rotation.yaw)
        if a is not None:
            self.ignored_collision_ids.add(a.id)
        self._inside = False
        self.lane = 'center'

    def tick(self, ego, dt):
        p = self.spec.params
        radius = p.get('radius_m', 2.0)
        d = ego.get_location().distance(self.location)
        if d > radius + 6.0:
            self._inside = False
            return
        if self._inside or d > radius:
            return
        self._inside = True
        v = ego.get_velocity()
        speed = math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)
        if speed < 0.5:
            return
        safe = p.get('safe_speed_kmh', 20.0) / 3.6
        dv = p.get('bump_min_mps', 0.3) + p.get('jolt_gain', 0.35) * max(0.0, speed - safe)
        k = max(0.0, speed - dv) / speed
        ego.set_target_velocity(self.carla.Vector3D(v.x * k, v.y * k, v.z * k))

    def oracle_text(self):
        return 'Deep pothole / damaged road surface in the lane, hit hard at speed'


class LaneClosure(Hazard):
    def spawn(self):
        carla = self.carla
        p = self.spec.params
        free = 'left' if same_dir_adjacent(carla, self.wp, 'left') else 'right'
        self.free_side = free
        self.lane = 'right' if free == 'left' else 'left'
        w = self.wp.lane_width
        closed_edge = 0.42 * w if free == 'left' else -0.42 * w   # along the right vector
        open_edge = -closed_edge
        vis = self.spec.visibility_m
        cone = self._bp(p.get('cone', 'static.prop.trafficcone01'), 'static.prop.constructioncone')
        barrier = self._bp(p.get('barrier', 'static.prop.streetbarrier'),
                           'static.prop.warningconstruction', 'static.prop.trafficwarning')
        taper = p.get('taper_m', 12.0)
        n_taper = int(p.get('taper_cones', 5))
        for i in range(n_taper):
            frac = i / max(1, n_taper - 1)
            along = -taper * (1.0 - frac)
            lat = closed_edge + (open_edge - closed_edge) * frac
            wp = _along(self.wp, along)
            loc = _offset(carla, wp.transform.location, wp.transform.get_right_vector(), lat, 0.05)
            self._spawn(cone, loc, wp.transform.rotation.yaw, visible_m=vis)
        length = p.get('length_m', 18.0)
        along = 3.0
        while along <= length:
            wp = _along(self.wp, along)
            loc = _offset(carla, wp.transform.location, wp.transform.get_right_vector(),
                          open_edge, 0.05)
            self._spawn(cone, loc, wp.transform.rotation.yaw, visible_m=vis)
            along += 3.0
        for along in (2.0, 9.0):
            wp = _along(self.wp, along)
            self._spawn(barrier, _offset(carla, wp.transform.location,
                                         wp.transform.get_right_vector(), 0.0, 0.05),
                        wp.transform.rotation.yaw + 90.0, visible_m=vis)

    def oracle_text(self):
        return (f'Lane closed by traffic cones and barriers (road works); '
                f'use the {getattr(self, "free_side", "other")} lane')


class PedestrianCrossing(Hazard):
    def spawn(self):
        carla = self.carla
        p = self.spec.params
        r = self.wp
        while True:
            nxt = same_dir_adjacent(carla, r, 'right')
            if nxt is None:
                break
            r = nxt
        self.kerb_wp = r
        w = r.lane_width
        self.lane = 'right'
        # parked van at the kerb, just before the crossing point
        van_wp = _along(r, -p.get('van_before_m', 7.0))
        van_bp = self._bp(p.get('van', 'vehicle.volkswagen.t2'), 'vehicle.mercedes.sprinter',
                          'vehicle.carlamotors.carlacola', 'vehicle.ford.ambulance')
        self.van = None
        for lat in (w / 2 + 1.1, w / 2 + 0.8, w / 2 + 0.5):
            loc = _offset(carla, van_wp.transform.location,
                          van_wp.transform.get_right_vector(), lat, 0.4)
            self.van = self._spawn(van_bp, loc, van_wp.transform.rotation.yaw, physics=True,
                                   visible_m=self.spec.params.get('van_visibility_m', 50.0))
            if self.van is not None:
                self.van.apply_control(carla.VehicleControl(hand_brake=True, brake=1.0))
                break
        # pedestrian hidden in front of the van
        ped_wp = _along(r, -p.get('walker_before_m', 3.0))
        right = ped_wp.transform.get_right_vector()
        self.walker_start = _offset(carla, ped_wp.transform.location, right,
                                    w / 2 + p.get('walker_kerb_offset_m', 0.8), 1.0)
        self.walk_dir = carla.Vector3D(-right.x, -right.y, 0.0)
        self.walker_yaw = math.degrees(math.atan2(-right.y, -right.x))
        bp = self._bp(p.get('walker', 'walker.pedestrian.*'))
        if bp is not None and bp.has_attribute('is_invincible'):
            bp.set_attribute('is_invincible', 'false')
        self.walker = self._spawn(bp, self.walker_start, self.walker_yaw, physics=True,
                                  visible_m=self.spec.visibility_m)
        if self.walker is None:
            raise RuntimeError(f'could not spawn pedestrian for hazard {self.name}')
        # Where the walker crosses the ego's lane (the route lane, which may be one or more
        # lanes left of the kerb) and how far it has to walk to get there.
        cross_wp = _along(self.wp, -p.get('walker_before_m', 3.0))
        self.cross_loc = cross_wp.transform.location
        self.lateral_m = math.hypot(self.walker_start.x - self.cross_loc.x,
                                    self.walker_start.y - self.cross_loc.y)
        self.state = 'waiting'

    def reset_for_car(self, car):
        p = self.spec.params
        self.appear_distance = p.get('appear_distance_m', 12.0) + \
            self.rng.uniform(-1.0, 1.0) * p.get('appear_jitter_m', 2.0)
        self.walker.apply_control(self.carla.WalkerControl(self.walk_dir, 0.0))
        self.walker.set_transform(self.carla.Transform(self.walker_start,
                                                       self.carla.Rotation(yaw=self.walker_yaw)))
        self.state = 'waiting'
        self.walked = 0.0

    def tick(self, ego, dt):
        p = self.spec.params
        if self.state == 'waiting':
            # Step out so that the walker enters the ego's path when the ego is
            # ``appear_distance_m`` from the crossing point: start earlier by the ego's travel
            # during the walk to the lane. A fixed trigger distance let a car in an inner lane
            # pass before the walker arrived; a slower (memory-capped) car now meets the walker
            # at the same distance but with a larger TTC, which is the effect we measure.
            el = ego.get_location()
            cp = self.cross_loc
            f = ego.get_transform().get_forward_vector()
            ahead = (cp.x - el.x) * f.x + (cp.y - el.y) * f.y > 0
            v = ego.get_velocity()
            v_ego = math.sqrt(v.x ** 2 + v.y ** 2)
            v_walk = p.get('walker_speed_mps', 2.2)
            t_walk = max(0.0, self.lateral_m - p.get('path_half_width_m', 1.6)) / v_walk
            d = math.hypot(cp.x - el.x, cp.y - el.y)
            # appear at a fixed time-to-collision (but at least appear_distance_m), so a fast
            # car still has room to stop: a surprise, not an unavoidable crash
            appear = max(self.appear_distance, v_ego * p.get('appear_ttc_s', 1.4))
            if ahead and d <= appear + v_ego * t_walk:
                self.state = 'crossing'
                self.speed = v_walk
                self.walker.apply_control(self.carla.WalkerControl(self.walk_dir, self.speed))
        elif self.state == 'crossing':
            self.walked += self.speed * dt
            walk_m = p.get('walk_distance_m', max(2 * self.kerb_wp.lane_width + 3.0,
                                                  self.lateral_m + 4.0))
            if self.walked >= walk_m:
                self.walker.apply_control(self.carla.WalkerControl(self.walk_dir, 0.0))
                self.state = 'done'

    def oracle_text(self):
        return ('Pedestrian steps out from behind a parked van just before the intersection, '
                'right side')


class StalledVehicle(Hazard):
    def spawn(self):
        carla = self.carla
        p = self.spec.params
        bp = self._bp(p.get('vehicle', 'vehicle.tesla.model3'), 'vehicle.audi.a2', 'vehicle.*')
        loc = self.carla.Location(self.location.x, self.location.y, self.location.z + 0.3)
        self.vehicle = self._spawn(bp, loc, self.wp.transform.rotation.yaw, physics=True,
                                   visible_m=self.spec.visibility_m)
        if self.vehicle is None:
            raise RuntimeError(f'could not spawn stalled vehicle for hazard {self.name}')
        self.vehicle.apply_control(carla.VehicleControl(hand_brake=True, brake=1.0))
        try:
            ls = carla.VehicleLightState
            self.vehicle.set_light_state(ls(ls.LeftBlinker | ls.RightBlinker | ls.Position))
        except Exception:
            pass
        self.lane = 'right' if same_dir_adjacent(carla, self.wp, 'left') else 'left'

    def oracle_text(self):
        return 'Stalled car with hazard lights stopped in the lane right after the curve'


HAZARD_CLASSES = {
    'road_damage': RoadDamage,
    'lane_closure': LaneClosure,
    'pedestrian_crossing': PedestrianCrossing,
    'stalled_vehicle': StalledVehicle,
}


def build_hazards(carla, world, route_s, placements, rng) -> List[Hazard]:
    return [HAZARD_CLASSES[spec.type](carla, world, spec, route_s, idx, rng)
            for spec, idx in placements]


def free_positions(route_s, hazards: List[Hazard], n: int, rng: random.Random,
                   min_gap_m: float = 60.0) -> List[Tuple[float, float, float, float]]:
    """Route positions far from real hazards, for injected false memories.
    Returns [(ros_x, ros_y, ros_heading, route_s)]."""
    total = route_s[-1][0]
    taken = [h.s for h in hazards]
    out = []
    for _ in range(500):
        if len(out) >= n:
            break
        s = rng.uniform(40.0, max(41.0, total - 40.0))
        if any(abs(s - t) < min_gap_m for t in taken):
            continue
        wp = _wp_at_s(route_s, s)
        x, y = carla_to_ros_xy(wp.transform.location.x, wp.transform.location.y)
        out.append((x, y, carla_yaw_to_ros(wp.transform.rotation.yaw), s))
        taken.append(s)
    return out


def actor_type(actor) -> str:
    return actor_kind(actor)
