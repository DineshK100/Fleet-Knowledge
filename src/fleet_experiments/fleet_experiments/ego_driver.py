"""Ego vehicle driver: CARLA BehaviorAgent + Fleet-Memory policy hooks.

The ``BehaviorAgent`` is the memory-free baseline driver. On top of it this
class:

* enforces a cruise speed and a curvature speed limit;
* applies the **speed caps** sent by the memory reader with a comfortable
  braking profile ``v_allow(d) = sqrt(v_cap^2 + 2 a (d - buffer))``, so
  braking starts at the earliest-braking distance;
* executes **lane changes** (bypass) and **reroutes** requested by the policy;
* brakes for obstacles reported by the simulated perception (the agent
  ignores static props), and when stuck behind a static or stalled blocker it
  recovers by going around it. The first car of the fleet therefore still
  reaches its destination, only later and with hard braking.

All positions inside this class are CARLA coordinates.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

from memory_reader.geometry import RoutePolyline

from .hazards import _along, same_dir_adjacent


@dataclass
class DriverConfig:
    behavior: str = 'normal'
    cruise_speed_kmh: float = 45.0
    min_speed_kmh: float = 8.0
    a_comfort: float = 2.0
    cap_buffer_m: float = 5.0
    pass_margin_m: float = 8.0
    constraint_corridor_m: float = 8.0
    lat_accel_max: float = 3.0
    curve_lookahead_m: float = 80.0
    speed_pid_max_brake: float = 0.15
    emergency_max_decel: float = 8.0
    stop_gap_m: float = 3.0
    bypass_trigger_m: float = 90.0
    reroute_trigger_m: float = 160.0
    bypass_change_before_m: float = 30.0
    bypass_past_m: float = 50.0
    recover_after_s: float = 3.0
    recover_vehicle_after_s: float = 8.0
    ignore_traffic_lights: bool = False


class EgoDriver:
    def __init__(self, carla, world, cmap, grp, vehicle, route_plan, destination,
                 cfg: DriverConfig, logger=None):
        from agents.navigation.behavior_agent import BehaviorAgent
        from agents.navigation.local_planner import RoadOption
        self.carla, self.world, self.cmap, self.grp = carla, world, cmap, grp
        self.vehicle = vehicle
        self.destination = destination
        self.cfg = cfg
        self.log = logger
        self.RoadOption = RoadOption
        try:
            self.agent = BehaviorAgent(vehicle, behavior=cfg.behavior, map_inst=cmap, grp_inst=grp)
        except TypeError:
            self.agent = BehaviorAgent(vehicle, behavior=cfg.behavior)
        if cfg.ignore_traffic_lights and hasattr(self.agent, 'ignore_traffic_lights'):
            self.agent.ignore_traffic_lights(True)
        self._set_plan(route_plan)
        # The agent's speed PID brakes with its maximum (0.3) for any small overspeed, i.e.
        # bang-bang throttle/brake at 20 Hz. Limit it to cruise-control braking; obstacle
        # braking (below) and the agent's emergency stop set the brake directly.
        lp = self.agent.get_local_planner()
        ctl = getattr(lp, '_vehicle_controller', None)
        if ctl is not None and hasattr(ctl, 'max_brake'):
            lp._max_brake = ctl.max_brake = cfg.speed_pid_max_brake
        self.half_len = vehicle.bounding_box.extent.x
        self.constraints: List[Dict] = []
        self.handled = set()
        self.recovered = set()
        self.stuck_time = 0.0
        self.bypass_blocker = None
        self.maneuver = 'cruise'
        self.lane_changes = 0
        self.reroutes = 0

    # ---------------------------------------------------------------- plans
    def _set_plan(self, plan):
        try:
            self.agent.set_global_plan(plan, stop_waypoint_creation=True, clean_queue=True)
        except TypeError:
            self.agent.set_global_plan(plan)

    def remaining_plan(self):
        lp = self.agent.get_local_planner()
        q = getattr(lp, '_waypoints_queue', None)
        if q is None and hasattr(lp, 'get_plan'):
            q = lp.get_plan()
        return list(q or [])

    def path_points(self, max_m: float = 250.0, step_m: float = 0.0):
        """Ego position followed by the upcoming route points (CARLA x, y)."""
        loc = self.vehicle.get_location()
        pts = [(loc.x, loc.y)]
        dist = 0.0
        for wp, _ in self.remaining_plan():
            p = wp.transform.location
            d = math.hypot(p.x - pts[-1][0], p.y - pts[-1][1])
            if d < step_m:
                continue
            dist += d
            pts.append((p.x, p.y))
            if dist > max_m:
                break
        return pts

    def set_constraints(self, constraints: List[Dict]):
        """``constraints``: dicts with memory_id, x, y (CARLA), speed_cap_kmh, action,
        preferred_lane, hazard_type, description."""
        self.constraints = constraints

    # ------------------------------------------------------------ speed cap
    def _constraint_distance(self, route: RoutePolyline, c) -> Optional[float]:
        s, lat, _ = route.project(c['x'], c['y'])
        if s <= 0.01:
            ego = self.vehicle.get_transform()
            f = ego.get_forward_vector()
            dx, dy = c['x'] - ego.location.x, c['y'] - ego.location.y
            along = dx * f.x + dy * f.y
            if along < -self.cfg.pass_margin_m or abs(dx * -f.y + dy * f.x) > self.cfg.constraint_corridor_m:
                return None
            return along
        if lat > self.cfg.constraint_corridor_m:
            return None
        return s - self.half_len

    def _speed_cap(self, route: RoutePolyline):
        cap, src = float('inf'), None
        for c in self.constraints:
            d = self._constraint_distance(route, c)
            if d is None:
                continue
            c['_d'] = d
            v_cap = c['speed_cap_kmh'] / 3.6
            v_allow = math.sqrt(v_cap ** 2 + 2.0 * self.cfg.a_comfort *
                                max(0.0, d - self.cfg.cap_buffer_m)) * 3.6
            if v_allow < cap:
                cap, src = v_allow, c
        return cap, src

    def _curve_limit(self, route: RoutePolyline) -> float:
        """Speed limit for the curves ahead, approached at ``a_comfort``.

        A curve at distance d with comfortable speed v_c allows sqrt(v_c^2 + 2 a d) now.
        Without this ramp the target jumped (45 -> 21 km/h) as soon as a turn entered the
        window and the agent's PID braked at ~1 g before every turn, which the surprise
        trigger then reported as hard brakes."""
        if route.length < 10:
            return float('inf')
        v_min = 15.0 / 3.6
        best = float('inf')
        s = 0.0
        horizon = min(self.cfg.curve_lookahead_m, route.length - 5.0)
        while s <= horizon:
            k = route.max_curvature(s, s + 10.0, step=5.0)
            if k >= 1e-3:
                v_c = max(v_min, math.sqrt(self.cfg.lat_accel_max / k))
                best = min(best, math.sqrt(v_c ** 2 + 2.0 * self.cfg.a_comfort * s))
            s += 5.0
        return best * 3.6

    # ------------------------------------------------------------ maneuvers
    def _route_wp_near(self, loc, max_d=10.0):
        best, bd = None, max_d
        for wp, _ in self.remaining_plan()[:200]:
            d = wp.transform.location.distance(loc)
            if d < bd:
                best, bd = wp, d
        return best

    def plan_bypass(self, blocker, prefer: str = '') -> bool:
        carla = self.carla
        ego_loc = self.vehicle.get_location()
        wp_b = self._route_wp_near(blocker) or self.cmap.get_waypoint(blocker)
        if wp_b is None:
            return False
        dist = ego_loc.distance(blocker)
        sides = [prefer] + [s for s in ('left', 'right') if s != prefer] if prefer else ['left', 'right']
        for side in sides:
            back = min(self.cfg.bypass_change_before_m, max(0.0, dist - 8.0))
            if back > 1.0:
                change_wp = _along(wp_b, -back)
            else:
                change_wp = _along(self.cmap.get_waypoint(ego_loc), 2.0)
            adj = same_dir_adjacent(carla, change_wp, side)
            if adj is None:
                continue
            plan = []
            if ego_loc.distance(change_wp.transform.location) > 6.0:
                plan += self.grp.trace_route(ego_loc, change_wp.transform.location)
            opt = self.RoadOption.CHANGELANELEFT if side == 'left' else self.RoadOption.CHANGELANERIGHT
            w = _along(adj, 8.0)
            plan.append((w, opt))
            travelled, needed = 8.0, back + self.cfg.bypass_past_m
            while travelled < needed:
                nxt = w.next(3.0)
                if not nxt:
                    break
                w = nxt[0]
                travelled += 3.0
                plan.append((w, self.RoadOption.LANEFOLLOW))
            rest = self.grp.trace_route(w.transform.location, self.destination)
            plan += rest[1:]
            self._set_plan(plan)
            self.bypass_blocker = blocker
            if hasattr(self.agent, 'ignore_vehicles'):
                self.agent.ignore_vehicles(True)
            self.lane_changes += 1
            self._log(f'bypass {side} around ({blocker.x:.1f},{blocker.y:.1f})')
            return True
        return False

    def plan_reroute(self, blocker, radius: float = 20.0) -> bool:
        ego_loc = self.vehicle.get_location()
        graph = getattr(self.grp, '_graph', None)
        if graph is None:
            return False
        changed = []
        for _u, _v, data in graph.edges(data=True):
            pts = [data['entry_waypoint'].transform.location, data['exit_waypoint'].transform.location]
            pts += [w.transform.location for w in data.get('path', [])]
            if any(p.distance(blocker) < radius for p in pts):
                changed.append((data, data['length']))
                data['length'] = 10 ** 7
        try:
            plan = self.grp.trace_route(ego_loc, self.destination)
        except Exception:
            plan = None
        finally:
            for data, length in changed:
                data['length'] = length
        if not plan or plan[0][0].transform.location.distance(ego_loc) > 15.0:
            return False
        if any(wp.transform.location.distance(blocker) < radius for wp, _ in plan):
            return False
        self._set_plan(plan)
        self.reroutes += 1
        self._log(f'reroute around ({blocker.x:.1f},{blocker.y:.1f}), {len(plan)} waypoints')
        return True

    def _memory_maneuvers(self):
        for c in self.constraints:
            mid = c['memory_id']
            d = c.get('_d')
            if mid in self.handled or d is None or c['action'] == 'slow':
                continue
            loc = self.carla.Location(c['x'], c['y'], self.vehicle.get_location().z)
            if c['action'] == 'reroute' and 30.0 < d < self.cfg.reroute_trigger_m:
                self.handled.add(mid)
                if self.plan_reroute(loc):
                    self.maneuver = 'reroute'
                elif self.plan_bypass(loc, c.get('preferred_lane', '')):
                    self.maneuver = 'bypass'
            elif c['action'] in ('lane_change', 'reroute') and 12.0 < d < self.cfg.bypass_trigger_m:
                self.handled.add(mid)
                if self.plan_bypass(loc, c.get('preferred_lane', '')):
                    self.maneuver = 'bypass'

    def _update_bypass(self):
        if self.bypass_blocker is None:
            return
        tf = self.vehicle.get_transform()
        f = tf.get_forward_vector()
        dx = self.bypass_blocker.x - tf.location.x
        dy = self.bypass_blocker.y - tf.location.y
        if dx * f.x + dy * f.y < -15.0:
            self.bypass_blocker = None
            if hasattr(self.agent, 'ignore_vehicles'):
                self.agent.ignore_vehicles(False)

    # ----------------------------------------------------------------- step
    def run_step(self, dt: float, obstacle=None):
        """One control step. ``obstacle`` is a perception.Obstacle (CARLA coords) or None."""
        speed = self.vehicle.get_velocity()
        v = math.sqrt(speed.x ** 2 + speed.y ** 2)
        route = RoutePolyline(self.path_points(250.0))
        cap_kmh, cap_src = self._speed_cap(route)
        self._memory_maneuvers()
        self._update_bypass()
        if self.maneuver in ('bypass', 'reroute', 'recover') and self.bypass_blocker is None:
            self.maneuver = 'cruise'

        target = min(self.cfg.cruise_speed_kmh, cap_kmh, self._curve_limit(route))
        target = max(self.cfg.min_speed_kmh, target) if cap_kmh > self.cfg.min_speed_kmh else target
        beh = self.agent._behavior
        beh.max_speed = target
        beh.speed_lim_dist = -200.0         # cruise speed, not the map speed limit, governs
        control = self.agent.run_step()

        state = self.maneuver if self.maneuver != 'cruise' else (
            'capped' if cap_src is not None and cap_kmh < self.cfg.cruise_speed_kmh else 'cruise')

        # perception-based braking (the agent itself ignores static props)
        if obstacle is not None and obstacle.closing_speed > 0.0:
            gap = obstacle.distance - self.cfg.stop_gap_m
            a_req = obstacle.closing_speed ** 2 / (2.0 * max(0.3, gap))
            if gap <= 0.3 or a_req > 1.0:
                brake = 1.0 if gap <= 0.3 else min(1.0, 1.15 * a_req / self.cfg.emergency_max_decel)
                if brake > control.brake:
                    control.throttle = 0.0
                    control.brake = max(0.15, brake)
                if a_req > 0.4 * 9.81 or gap <= 0.3:
                    state = 'emergency_brake'

        # recovery when stuck behind a static obstacle or a stalled vehicle
        if (obstacle is not None and v < 0.5 and obstacle.stationary and obstacle.distance < 12.0
                and obstacle.kind in ('static', 'vehicle')):
            self.stuck_time += dt
            wait = self.cfg.recover_after_s if obstacle.kind == 'static' else self.cfg.recover_vehicle_after_s
            if self.stuck_time > wait and obstacle.actor_id not in self.recovered:
                self.recovered.add(obstacle.actor_id)
                loc = self.carla.Location(obstacle.x, obstacle.y, self.vehicle.get_location().z)
                if self.plan_bypass(loc) or self.plan_reroute(loc):
                    self.maneuver = 'recover'
                self.stuck_time = 0.0
        else:
            self.stuck_time = 0.0

        info = {
            'target_kmh': target,
            'cap_kmh': cap_kmh if cap_src is not None else 0.0,
            'cap_memory_id': cap_src['memory_id'] if cap_src is not None else '',
            'cap_xy': (cap_src['x'], cap_src['y']) if cap_src is not None else None,
            'maneuver': state,
        }
        return control, info

    def done(self) -> bool:
        return self.agent.done()

    def replan_to_destination(self):
        try:
            self.agent.set_destination(self.destination)
        except Exception as exc:
            self._log(f'replan failed: {exc}')

    def _log(self, msg):
        if self.log is not None:
            self.log.info(f'[driver] {msg}')
