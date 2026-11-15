"""Scenario runner: CARLA world, hazards, the stream of N cars, vehicle I/O and metrics.

This node is the single CARLA client. It owns the world in synchronous mode
and publishes ``/clock``. It places the hazards, then sends cars 1..N one after
another along the same route. For each car ``ego_k`` it:

* spawns the vehicle with a front camera, an IMU and a collision sensor;
* drives it with ``EgoDriver`` (CARLA BehaviorAgent + Fleet-Memory hooks);
* publishes ``/ego_k/{vehicle_state, odometry, imu, camera/image, obstacle,
  collision, planned_route}`` and TF ``map -> ego_k/base_link``;
* applies ``/ego_k/driving_advice`` from the memory reader;
* records metrics, trajectories and the memory records the car wrote.

This replaces the generic CARLA-ROS bridge with a small Python-API bridge
behind the same kind of per-vehicle ROS 2 interfaces. That is the fallback
listed in the proposal's risk table, and it removes the bridge-version
problems.

Results go to ``<results_dir>/<run_name>/``.
"""

import array
import csv
import json
import math
import os
import queue
import random
import threading
import time
from datetime import datetime

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry, Path
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Image, Imu
from std_msgs.msg import Int32
from std_srvs.srv import Trigger
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

from fleet_memory.conversions import to_time_msg
from fleet_msgs.msg import (CollisionEvent, DrivingAdvice, HazardMemory, MemoryEvent,
                            ObstacleInfo, VehicleState)
from fleet_msgs.srv import WriteMemory

from .carla_utils import (actor_kind, carla_to_ros_xy, carla_yaw_to_ros, ros_to_carla_xy,
                          setup_carla_path, weather_for_seed, weather_text, yaw_to_quaternion)
from .ego_driver import DriverConfig, EgoDriver
from .hazards import (HAZARD_TYPES, HazardSpec, build_hazards, free_positions,
                      plan_placements, route_with_s, same_dir_adjacent)
from .metrics import MetricsConfig, MetricsRecorder
from .perception import GroundTruthPerception, PerceptionConfig

LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                     reliability=ReliabilityPolicy.RELIABLE)


class CsvLog:
    def __init__(self, path):
        self.path = path
        self.fields = None

    def write(self, rows):
        if isinstance(rows, dict):
            rows = [rows]
        if not rows:
            return
        new = not os.path.exists(self.path)
        if self.fields is None:
            if new:
                self.fields = list(rows[0].keys())
            else:
                with open(self.path) as f:
                    self.fields = next(csv.reader(f), list(rows[0].keys()))
        for r in rows:                                    # tolerate new columns
            for k in r:
                if k not in self.fields:
                    self.fields.append(k)
        with open(self.path, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=self.fields, extrasaction='ignore')
            if new:
                w.writeheader()
            w.writerows(rows)


class CarIO:
    """Per-vehicle ROS publishers / subscribers (namespace ``/ego_k``)."""

    def __init__(self, node: Node, ns: str, cb_group):
        self.node, self.ns = node, ns
        mk = node.create_publisher
        self.state_pub = mk(VehicleState, f'/{ns}/vehicle_state', 20)
        self.odom_pub = mk(Odometry, f'/{ns}/odometry', 20)
        self.imu_pub = mk(Imu, f'/{ns}/imu', 20)
        self.cam_pub = mk(Image, f'/{ns}/camera/image', 5)
        self.obs_pub = mk(ObstacleInfo, f'/{ns}/obstacle', 20)
        self.col_pub = mk(CollisionEvent, f'/{ns}/collision', 20)
        self.route_pub = mk(Path, f'/{ns}/planned_route', 2)
        self.lock = threading.Lock()
        self.advice = None
        self.pending = 0
        self.records = []
        self.subs = [
            node.create_subscription(DrivingAdvice, f'/{ns}/driving_advice', self._on_advice, 10,
                                     callback_group=cb_group),
            node.create_subscription(Int32, f'/{ns}/writer_pending', self._on_pending, LATCHED,
                                     callback_group=cb_group),
            node.create_subscription(HazardMemory, f'/{ns}/hazard_records', self._on_record, 10,
                                     callback_group=cb_group),
        ]
        self.pubs = [self.state_pub, self.odom_pub, self.imu_pub, self.cam_pub, self.obs_pub,
                     self.col_pub, self.route_pub]

    def _on_advice(self, msg):
        with self.lock:
            self.advice = msg

    def _on_pending(self, msg):
        with self.lock:
            self.pending = msg.data

    def _on_record(self, msg):
        with self.lock:
            self.records.append(msg)

    def take_records(self):
        with self.lock:
            out, self.records = self.records, []
        return out

    def destroy(self):
        for s in self.subs:
            self.node.destroy_subscription(s)
        for p in self.pubs:
            self.node.destroy_publisher(p)


class ScenarioRunner(Node):
    def __init__(self):
        super().__init__('scenario_runner')
        p = self.declare_parameter
        # CARLA / world
        p('carla_host', 'localhost')
        p('carla_port', 2000)
        p('carla_timeout_s', 30.0)
        p('tm_port', 8000)
        p('scenario_file', '')
        p('town', '')                            # overrides scenario town if set
        p('fixed_delta_s', 0.05)
        p('no_rendering', False)
        p('realtime_factor', 1.0)                # <= 0: as fast as possible
        p('weather', 'auto')                     # auto (by seed) or a WeatherParameters preset
        p('traffic_lights', 'green')             # normal | green | ignore
        p('num_background_vehicles', 0)
        p('follow_spectator', True)
        # experiment
        p('num_cars', 10)
        p('seed', 0)
        p('memory_mode', 'vlm')                  # vlm | template | oracle | none (for bookkeeping)
        p('reset_memory_on_start', True)
        p('false_memories', 0)                   # E3: inject N false memories before car 1
        p('max_car_time_s', 240.0)
        p('stuck_timeout_s', 60.0)
        p('inter_car_gap_s', 30.0)               # simulated time between cars
        p('writer_flush_timeout_s', 120.0)       # wall time to wait for VLM writes
        p('results_dir', '~/fleet_memory_results')
        p('run_name', '')
        p('config_name', '')                     # label used by plot_results to group runs
        # ego
        p('ego_blueprint', 'vehicle.tesla.model3')   # single gear: no shift jolts
        p('ego_low_coast_drag', True)              # no ~0.45 g engine drag at zero throttle
        p('speed_pid_max_brake', 0.15)             # agent speed PID brake limit (CARLA: 0.3)
        p('camera_width', 800)
        p('camera_height', 450)
        p('camera_fov', 90.0)
        p('camera_rate_hz', 10.0)
        p('record_video', False)
        p('video_width', 1280)
        p('video_height', 720)
        # driver
        p('behavior', 'normal')
        p('cruise_speed_kmh', 45.0)
        p('a_comfort', 2.0)
        p('bypass_trigger_m', 90.0)
        p('bypass_past_m', 50.0)
        # perception
        p('perception_range_m', 45.0)
        p('corridor_half_width_m', 1.4)

        self.g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.cb = ReentrantCallbackGroup()
        self.clock_pub = self.create_publisher(Clock, '/clock', 10)
        self.tf = TransformBroadcaster(self)
        self.road_pub = self.create_publisher(MarkerArray, '/fleet_experiments/road_network', LATCHED)
        self.gt_pub = self.create_publisher(MarkerArray, '/fleet_experiments/gt_hazards', LATCHED)
        self.route_marker_pub = self.create_publisher(Marker, '/fleet_experiments/route', LATCHED)
        self.write_cli = self.create_client(WriteMemory, '/fleet_memory/write', callback_group=self.cb)
        self.reset_cli = self.create_client(Trigger, '/fleet_memory/reset', callback_group=self.cb)
        self.create_subscription(MemoryEvent, '/fleet_memory/events', self._on_memory_event, 100,
                                 callback_group=self.cb)
        self.current_car = 0
        self.mem_events = []
        self.mem_lock = threading.Lock()
        self.t_sim = 0.0
        self.actors_to_cleanup = []

    # ================================================================ helpers
    def log(self, msg):
        self.get_logger().info(msg)

    def _on_memory_event(self, msg):
        with self.mem_lock:
            r = msg.record
            self.mem_events.append({
                'car': self.current_car, 't': round(self.t_sim, 2), 'event': msg.event,
                'id': r.id, 'hazard_type': r.hazard_type, 'confidence': round(r.confidence, 3),
                'count': r.count, 'refutations': r.refutations, 'source': r.source,
                'reporter': r.reporter_id, 'x': round(r.x, 2), 'y': round(r.y, 2),
                'description': r.description})

    def call(self, cli, req, timeout=10.0):
        if not cli.wait_for_service(timeout_sec=timeout):
            raise RuntimeError(f'service {cli.srv_name} not available')
        fut = cli.call_async(req)
        t0 = time.time()
        while not fut.done():
            if time.time() - t0 > timeout:
                raise RuntimeError(f'service {cli.srv_name} timed out')
            time.sleep(0.01)
        return fut.result()

    def publish_clock(self, t):
        self.clock_pub.publish(Clock(clock=to_time_msg(t)))

    def tick(self, n=1):
        for _ in range(n):
            self.world.tick()
            self.t_sim = self.world.get_snapshot().timestamp.elapsed_seconds
            self.publish_clock(self.t_sim)

    # ============================================================== scenario
    def load_scenario(self):
        path = self.g('scenario_file')
        if not path:
            from ament_index_python.packages import get_package_share_directory
            path = os.path.join(get_package_share_directory('fleet_experiments'),
                                'config', 'scenarios', 'town03_default.yaml')
        with open(os.path.expanduser(path)) as f:
            sc = yaml.safe_load(f)
        sc['_path'] = path
        return sc

    def select_route(self, sc):
        carla, cmap, grp = self.carla, self.cmap, self.grp
        specs = [HazardSpec.from_dict(h) for h in sc.get('hazards', [])]
        min_gap = sc.get('min_hazard_gap_m', 70.0)
        r = sc.get('route', {}) or {}
        mode = r.get('mode', 'auto')
        spawns = cmap.get_spawn_points()

        def try_pair(start_tf, end_loc):
            route = grp.trace_route(start_tf.location, end_loc)
            if not route:
                return None
            route_s = route_with_s(route)
            placements = plan_placements(carla, route_s, specs, min_gap)
            return (route, route_s, placements) if placements else None

        if mode == 'spawn_points':
            start_tf, end_loc = spawns[r['start_spawn']], spawns[r['end_spawn']].location
            res = try_pair(start_tf, end_loc)
            if res is None:
                raise RuntimeError('hazards cannot be placed on the configured route; '
                                   'use route.mode: auto or change after_m values')
            return start_tf, end_loc, res, (r['start_spawn'], r['end_spawn'])
        if mode == 'coordinates':
            s, e = r['start'], r['end']
            start_tf = cmap.get_waypoint(carla.Location(x=s[0], y=s[1], z=s[2] if len(s) > 2 else 0.5)).transform
            end_loc = cmap.get_waypoint(carla.Location(x=e[0], y=e[1], z=e[2] if len(e) > 2 else 0.5)).transform.location
            start_tf.location.z += 0.5
            res = try_pair(start_tf, end_loc)
            if res is None:
                raise RuntimeError('hazards cannot be placed on the configured route')
            return start_tf, end_loc, res, None
        # auto: deterministic search (route_seed) so every seed / config uses the same route
        rng = random.Random(r.get('route_seed', 7))
        pairs = [(i, j) for i in range(len(spawns)) for j in range(len(spawns)) if i != j]
        rng.shuffle(pairs)
        lo, hi = r.get('min_length_m', 500.0), r.get('max_length_m', 1400.0)
        tried = 0
        for i, j in pairs:
            if tried >= r.get('attempts', 400):
                break
            d = spawns[i].location.distance(spawns[j].location)
            if d < lo * 0.4 or d > hi:
                continue
            tried += 1
            route = grp.trace_route(spawns[i].location, spawns[j].location)
            if not route:
                continue
            route_s = route_with_s(route)
            if not lo <= route_s[-1][0] <= hi:
                continue
            placements = plan_placements(carla, route_s, specs, min_gap)
            if placements:
                self.log(f'Auto route: spawn {i} -> {j}, {route_s[-1][0]:.0f} m '
                         f'(pin with route.mode: spawn_points)')
                return spawns[i], spawns[j].location, (route, route_s, placements), (i, j)
        raise RuntimeError('no route found that fits all hazards; relax the scenario')

    # ============================================================= markers
    def publish_static_markers(self):
        arr = MarkerArray()
        m = Marker()
        m.header.frame_id = 'map'
        m.ns, m.id, m.type, m.action = 'roads', 0, Marker.LINE_LIST, Marker.ADD
        m.scale.x = 0.3
        m.color.r = m.color.g = m.color.b = 0.6
        m.color.a = 0.6
        m.pose.orientation.w = 1.0
        from geometry_msgs.msg import Point
        for wp in self.cmap.generate_waypoints(4.0):
            for nxt in wp.next(4.0)[:1]:
                a, b = wp.transform.location, nxt.transform.location
                ax, ay = carla_to_ros_xy(a.x, a.y)
                bx, by = carla_to_ros_xy(b.x, b.y)
                m.points.append(Point(x=ax, y=ay, z=0.0))
                m.points.append(Point(x=bx, y=by, z=0.0))
        arr.markers.append(m)
        self.road_pub.publish(arr)

        rm = Marker()
        rm.header.frame_id = 'map'
        rm.ns, rm.id, rm.type, rm.action = 'route', 0, Marker.LINE_STRIP, Marker.ADD
        rm.scale.x = 1.0
        rm.color.g, rm.color.b, rm.color.a = 0.8, 1.0, 0.8
        rm.pose.orientation.w = 1.0
        for wp, _ in self.route:
            x, y = carla_to_ros_xy(wp.transform.location.x, wp.transform.location.y)
            rm.points.append(Point(x=x, y=y, z=0.2))
        self.route_marker_pub.publish(rm)
        self.publish_gt_markers()

    def publish_gt_markers(self):
        arr = MarkerArray()
        clear = Marker()
        clear.header.frame_id = 'map'
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)
        for i, h in enumerate(self.hazards):
            x, y = h.ros_xy()
            m = Marker()
            m.header.frame_id = 'map'
            m.ns, m.id, m.type, m.action = 'gt', i, Marker.SPHERE, Marker.ADD
            m.pose.position.x, m.pose.position.y, m.pose.position.z = x, y, 0.5
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 2.5
            m.color.r, m.color.a = 1.0, 0.9 if h.active else 0.25
            arr.markers.append(m)
            t = Marker()
            t.header.frame_id = 'map'
            t.ns, t.id, t.type, t.action = 'gt_text', i, Marker.TEXT_VIEW_FACING, Marker.ADD
            t.pose.position.x, t.pose.position.y, t.pose.position.z = x, y, 3.0
            t.pose.orientation.w = 1.0
            t.scale.z = 1.8
            t.color.r = t.color.g = t.color.a = 1.0
            t.text = f'GT {h.name} ({h.type}){"" if h.active else " [inactive]"}'
            arr.markers.append(t)
        self.gt_pub.publish(arr)

    # ============================================================== memory
    def setup_memory(self):
        if self.g('reset_memory_on_start'):
            try:
                self.call(self.reset_cli, Trigger.Request(), timeout=30.0)
                self.log('Fleet memory reset')
            except RuntimeError as exc:
                self.get_logger().warning(f'could not reset memory: {exc}')
        if self.g('memory_mode') == 'oracle':
            for h in self.hazards:
                if not h.is_active_for(1):
                    continue
                x, y = h.ros_xy()
                self._write_record(h.type, h.oracle_text(), x, y, h.ros_heading(), 4, 1.0,
                                   'oracle', h.lane)
            self.log(f'Oracle memory seeded with {len(self.hazards)} hazards')
        n_false = int(self.g('false_memories'))
        if n_false > 0:
            rng = random.Random(1000 + self.g('seed'))
            for x, y, hdg, s in free_positions(self.route_s, self.hazards, n_false, rng):
                htype = rng.choice(HAZARD_TYPES)
                self._write_record(htype, f'{htype.replace("_", " ")} reported here', x, y, hdg,
                                   4, 0.0, 'injected', 'unknown')
                self.false_memory_log.write({'x': x, 'y': y, 'route_s': round(s, 1),
                                             'hazard_type': htype})
            self.log(f'Injected {n_false} false memories')

    def _write_record(self, htype, desc, x, y, heading, severity, conf, source, lane):
        m = HazardMemory()
        m.hazard_type, m.description, m.lane = htype, desc, lane
        m.severity, m.confidence, m.source, m.trigger = severity, conf, source, source
        m.x, m.y, m.heading = x, y, heading
        m.reporter_id = source
        m.count = 1
        m.first_seen = m.last_confirmed = to_time_msg(self.t_sim)
        return self.call(self.write_cli, WriteMemory.Request(record=m, allow_merge=False))

    # ============================================================= traffic
    def spawn_background(self, n, rng):
        if n <= 0:
            return
        lib = self.world.get_blueprint_library().filter('vehicle.*')
        lib = [b for b in lib if int(b.get_attribute('number_of_wheels')) == 4] \
            if lib and lib[0].has_attribute('number_of_wheels') else list(lib)
        spawns = self.cmap.get_spawn_points()
        rng.shuffle(spawns)
        start = self.start_tf.location
        count = 0
        for sp in spawns:
            if count >= n:
                break
            if sp.location.distance(start) < 30.0:
                continue
            bp = rng.choice(lib)
            if bp.has_attribute('role_name'):
                bp.set_attribute('role_name', 'fleet_background')
            a = self.world.try_spawn_actor(bp, sp)
            if a is None:
                continue
            a.set_autopilot(True, self.g('tm_port'))
            self.actors_to_cleanup.append(a)
            count += 1
        self.log(f'Spawned {count} background vehicles')

    # ================================================================= run
    def run(self):
        setup_carla_path()
        import carla
        from agents.navigation.global_route_planner import GlobalRoutePlanner
        self.carla = carla
        seed = int(self.g('seed'))
        rng = random.Random(seed)
        sc = self.load_scenario()
        town = self.g('town') or sc.get('town', 'Town03')

        client = carla.Client(self.g('carla_host'), int(self.g('carla_port')))
        client.set_timeout(float(self.g('carla_timeout_s')))
        world = client.get_world()
        if not world.get_map().name.split('/')[-1].startswith(town):
            self.log(f'Loading {town} ...')
            world = client.load_world(town)
        self.client, self.world = client, world
        self.original_settings = world.get_settings()
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = float(self.g('fixed_delta_s'))
        settings.no_rendering_mode = bool(self.g('no_rendering'))
        world.apply_settings(settings)
        tm = client.get_trafficmanager(int(self.g('tm_port')))
        tm.set_synchronous_mode(True)
        tm.set_random_device_seed(seed)
        for a in world.get_actors():
            if a.attributes.get('role_name', '').startswith('fleet_'):
                a.destroy()
        self.tick()

        preset, weather = weather_for_seed(carla, seed, self.g('weather'))
        world.set_weather(weather)
        self.weather_text = weather_text(preset)
        if self.g('traffic_lights') == 'green':
            for tl in world.get_actors().filter('traffic.traffic_light*'):
                tl.set_state(carla.TrafficLightState.Green)
                tl.freeze(True)

        self.cmap = world.get_map()
        self.grp = GlobalRoutePlanner(self.cmap, 2.0)
        self.start_tf, self.dest, (self.route, self.route_s, placements), spawn_pair = \
            self.select_route(sc)
        self.hazards = build_hazards(carla, world, self.route_s, placements, random.Random(seed + 17))
        for h in self.hazards:
            h.prepare_for_car(1)
        self.tick(10)

        # ---- results directory
        run_name = self.g('run_name') or \
            f'{datetime.now():%Y%m%d_%H%M%S}_{self.g("memory_mode")}_s{seed}'
        self.out_dir = os.path.join(os.path.expanduser(self.g('results_dir')), run_name)
        os.makedirs(os.path.join(self.out_dir, 'trajectories'), exist_ok=True)
        self.cars_log = CsvLog(os.path.join(self.out_dir, 'cars.csv'))
        self.hazards_log = CsvLog(os.path.join(self.out_dir, 'hazards.csv'))
        self.advice_log = CsvLog(os.path.join(self.out_dir, 'advice.csv'))
        self.records_log = CsvLog(os.path.join(self.out_dir, 'written_records.csv'))
        self.events_log = CsvLog(os.path.join(self.out_dir, 'memory_events.csv'))
        self.false_memory_log = CsvLog(os.path.join(self.out_dir, 'false_memories.csv'))
        params = {n: self.get_parameter(n).value for n in self._parameters}
        with open(os.path.join(self.out_dir, 'config.json'), 'w') as f:
            json.dump({'params': params, 'scenario': {k: v for k, v in sc.items()},
                       'config_name': self.g('config_name') or self.g('memory_mode'),
                       'town': town, 'weather': preset, 'spawn_pair': spawn_pair,
                       'route_length_m': self.route_s[-1][0],
                       'hazards': [h.info() for h in self.hazards]}, f, indent=2, default=str)
        self.log(f'Writing results to {self.out_dir}')
        self.log('Hazards: ' + ', '.join(f'{h.name}@{h.s:.0f}m' for h in self.hazards))

        self.publish_static_markers()
        self.setup_memory()
        self.spawn_background(int(self.g('num_background_vehicles')), rng)

        try:
            for k in range(1, int(self.g('num_cars')) + 1):
                if not rclpy.ok():
                    break
                self.current_car = k
                self.run_car(k, rng)
                self.flush_events()
                gap_ticks = int(self.g('inter_car_gap_s') / self.g('fixed_delta_s'))
                self.tick(gap_ticks)
        finally:
            self.flush_events()
            self.cleanup()
        self.log(f'Experiment finished. Results in {self.out_dir}')

    def flush_events(self):
        with self.mem_lock:
            ev, self.mem_events = self.mem_events, []
        self.events_log.write(ev)

    def cleanup(self):
        for h in getattr(self, 'hazards', []):
            h.destroy()
        for a in self.actors_to_cleanup:
            try:
                a.destroy()
            except RuntimeError:
                pass
        try:
            self.world.apply_settings(self.original_settings)
            self.client.get_trafficmanager(int(self.g('tm_port'))).set_synchronous_mode(False)
        except Exception:
            pass

    # ================================================================ car
    def gt_for_car(self, k):
        out = []
        for h in self.hazards:
            x, y = h.ros_xy()
            out.append({'name': h.name, 'type': h.type, 'x': x, 'y': y, 'active': h.is_active_for(k)})
        return out

    def spawn_ego(self, ns):
        carla = self.carla
        lib = self.world.get_blueprint_library()
        found = lib.filter(self.g('ego_blueprint'))
        bp = found[0] if len(found) else lib.filter('vehicle.tesla.model3')[0]
        bp.set_attribute('role_name', f'fleet_{ns}')
        tf = carla.Transform(carla.Location(self.start_tf.location.x, self.start_tf.location.y,
                                            self.start_tf.location.z + 0.5), self.start_tf.rotation)
        for _ in range(100):
            ego = self.world.try_spawn_actor(bp, tf)
            if ego is not None:
                # With zero throttle CARLA applies a strong engine drag
                # (damping_rate_zero_throttle_*), so every throttle-off tick of the agent's
                # PID decelerated the car at ~0.45 g, which the trigger counted as hard brakes.
                # Coast with the full-throttle damping instead. (Gearboxed cars such as the MKZ
                # also jolt at every downshift, hence the single-gear Tesla as default ego.)
                if self.g('ego_low_coast_drag'):
                    try:
                        pc = ego.get_physics_control()
                        pc.damping_rate_zero_throttle_clutch_engaged = pc.damping_rate_full_throttle
                        pc.damping_rate_zero_throttle_clutch_disengaged = pc.damping_rate_full_throttle
                        ego.apply_physics_control(pc)
                    except RuntimeError as exc:
                        self.log(f'could not change ego physics: {exc}')
                return ego
            self.tick(10)                              # start blocked by traffic: wait
        raise RuntimeError('could not spawn ego vehicle at the route start')

    def attach_sensors(self, ego, k):
        carla = self.carla
        lib = self.world.get_blueprint_library()
        self.cam_q, self.col_q, self.imu_latest = queue.Queue(maxsize=4), queue.Queue(), None
        sensors = []
        cam_bp = lib.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', str(self.g('camera_width')))
        cam_bp.set_attribute('image_size_y', str(self.g('camera_height')))
        cam_bp.set_attribute('fov', str(self.g('camera_fov')))
        cam_bp.set_attribute('sensor_tick', str(1.0 / self.g('camera_rate_hz')))
        cam = self.world.spawn_actor(cam_bp, carla.Transform(carla.Location(x=1.5, z=1.7)),
                                     attach_to=ego)
        cam.listen(lambda img: self._put(self.cam_q, img))
        sensors.append(cam)
        col = self.world.spawn_actor(lib.find('sensor.other.collision'), carla.Transform(),
                                     attach_to=ego)
        col.listen(lambda ev: self.col_q.put(ev))
        sensors.append(col)
        imu = self.world.spawn_actor(lib.find('sensor.other.imu'), carla.Transform(), attach_to=ego)
        imu.listen(lambda m: setattr(self, 'imu_latest', m))
        sensors.append(imu)
        self.video = None
        if self.g('record_video'):
            vbp = lib.find('sensor.camera.rgb')
            vbp.set_attribute('image_size_x', str(self.g('video_width')))
            vbp.set_attribute('image_size_y', str(self.g('video_height')))
            vbp.set_attribute('sensor_tick', '0.1')
            vcam = self.world.spawn_actor(
                vbp, carla.Transform(carla.Location(x=-7.0, z=3.5), carla.Rotation(pitch=-15.0)),
                attach_to=ego)
            self.video = VideoWriter(os.path.join(self.out_dir, 'video', f'car_{k:02d}'))
            vcam.listen(self.video.on_image)
            sensors.append(vcam)
        return sensors

    @staticmethod
    def _put(q, item):
        try:
            q.put_nowait(item)
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                pass
            q.put_nowait(item)

    def wait_for_subscribers(self, ns, timeout=10.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.count_subscribers(f'/{ns}/vehicle_state') >= 1 and \
                    self.count_publishers(f'/{ns}/driving_advice') >= 1:
                return True
            time.sleep(0.1)
        self.get_logger().warning(f'no writer/reader nodes found for /{ns}; driving without memory')
        return False

    def context_text(self, ego_wp, path_wps):
        parts = []
        if ego_wp.is_junction:
            parts.append('in intersection')
        elif any(w.is_junction for w in path_wps):
            parts.append('approaching intersection')
        else:
            parts.append('straight road' if len(path_wps) < 2 else 'road segment')
        right = same_dir_adjacent(self.carla, ego_wp, 'right')
        left = same_dir_adjacent(self.carla, ego_wp, 'left')
        if right is None and left is None:
            parts.append('single lane')
        elif right is None:
            parts.append('right lane')
        elif left is None:
            parts.append('left lane')
        else:
            parts.append('middle lane')
        parts.append(self.weather_text)
        return ', '.join(parts)

    def run_car(self, k, rng):
        carla = self.carla
        ns = f'ego_{k}'
        dt = float(self.g('fixed_delta_s'))
        for h in self.hazards:
            h.prepare_for_car(k)
        self.publish_gt_markers()
        self.tick(5)
        ego = self.spawn_ego(ns)
        sensors = self.attach_sensors(ego, k)
        io = CarIO(self, ns, self.cb)
        self.wait_for_subscribers(ns)
        self.tick(5)

        drv_cfg = DriverConfig(behavior=self.g('behavior'), cruise_speed_kmh=self.g('cruise_speed_kmh'),
                               a_comfort=self.g('a_comfort'), bypass_trigger_m=self.g('bypass_trigger_m'),
                               bypass_past_m=self.g('bypass_past_m'),
                               ignore_traffic_lights=self.g('traffic_lights') == 'ignore',
                               speed_pid_max_brake=self.g('speed_pid_max_brake'))
        driver = EgoDriver(carla, self.world, self.cmap, self.grp, ego, list(self.route), self.dest,
                           drv_cfg, self.get_logger())
        perception = GroundTruthPerception(self.world, ego, PerceptionConfig(
            corridor_half_width_m=self.g('corridor_half_width_m'),
            default_range_m=self.g('perception_range_m')))
        metrics = MetricsRecorder(k, self.gt_for_car(k), MetricsConfig())
        traj = CsvLog(os.path.join(self.out_dir, 'trajectories', f'car_{k:02d}.csv'))
        traj_rows = []
        self.log(f'=== Car {k}/{self.g("num_cars")} ({ns}) starts ===')

        status, completed = 'running', False
        t_start = self.t_sim
        prev_v = None
        speed = 0.0
        t_end = self.t_sim
        obstacle = None
        slow_time = 0.0
        replans = 0
        settle = 0
        last_advice_stamp = None
        info = {'cap_kmh': 0.0, 'maneuver': 'cruise', 'target_kmh': 0.0, 'cap_xy': None}
        wall0, sim0 = time.time(), self.t_sim
        step = 0
        while rclpy.ok():
            step += 1
            # ---------------- advice -> driver
            with io.lock:
                adv = io.advice
            constraints_ros = []
            if adv is not None:
                for c in adv.constraints:
                    constraints_ros.append({'memory_id': c.memory_id, 'x': c.x, 'y': c.y})
                driver.set_constraints([{
                    'memory_id': c.memory_id, 'x': ros_to_carla_xy(c.x, c.y)[0],
                    'y': ros_to_carla_xy(c.x, c.y)[1], 'speed_cap_kmh': c.speed_cap_kmh,
                    'action': c.action, 'preferred_lane': c.preferred_lane,
                    'hazard_type': c.hazard_type, 'description': c.description}
                    for c in adv.constraints])
                if adv.header.stamp != last_advice_stamp and adv.constraints:
                    last_advice_stamp = adv.header.stamp
                    loc = ego.get_location()
                    ex, ey = carla_to_ros_xy(loc.x, loc.y)
                    metrics.advice(ex, ey, constraints_ros)
                    self.advice_log.write([{
                        'car': k, 't': round(self.t_sim - t_start, 2), 'memory_id': c.memory_id,
                        'hazard_type': c.hazard_type, 'route_distance': round(c.route_distance, 1),
                        'score': round(c.score, 3), 'confidence': round(c.confidence, 3),
                        'speed_cap_kmh': c.speed_cap_kmh, 'action': c.action,
                        'adapter': adv.adapter, 'description': c.description}
                        for c in adv.constraints])

            # ---------------- control
            if settle > 0:
                # gentle stop (~2 m/s^2) so arriving does not look like a hard brake
                control = carla.VehicleControl(throttle=0.0, brake=0.25 if speed > 0.3 else 1.0)
            else:
                control, info = driver.run_step(dt, obstacle)
            ego.apply_control(control)
            for h in self.hazards:
                if h.active:
                    h.tick(ego, dt)
            self.tick()
            t = self.t_sim

            # ---------------- state
            tf = ego.get_transform()
            vel = ego.get_velocity()
            fwd = tf.get_forward_vector()
            v_fwd = vel.x * fwd.x + vel.y * fwd.y + vel.z * fwd.z
            speed = math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)
            accel = 0.0 if prev_v is None else (v_fwd - prev_v) / dt
            prev_v = v_fwd
            path_pts = driver.path_points(80.0)
            obstacle = perception.detect(path_pts, {aid: av for h in self.hazards if h.active
                                                    for aid, av in h.perceivable.items()})
            x, y = carla_to_ros_xy(tf.location.x, tf.location.y)
            yaw = carla_yaw_to_ros(tf.rotation.yaw)
            stamp = to_time_msg(t)

            if step % 10 == 1 or not hasattr(self, '_ctx'):
                ego_wp = self.cmap.get_waypoint(tf.location)
                ahead = [w for w, _ in driver.remaining_plan()[:12]]
                self._ctx = (ego_wp, self.context_text(ego_wp, ahead))
            ego_wp, ctx = self._ctx

            st = VehicleState()
            st.header.stamp, st.header.frame_id = stamp, 'map'
            st.vehicle_id = ns
            st.x, st.y, st.z, st.heading = x, y, tf.location.z, yaw
            st.speed, st.accel_long = speed, accel
            st.road_id, st.lane_id, st.is_junction = ego_wp.road_id, ego_wp.lane_id, ego_wp.is_junction
            st.target_speed_kmh = float(info.get('target_kmh', 0.0))
            st.speed_cap_kmh = float(info.get('cap_kmh', 0.0))
            st.maneuver = info.get('maneuver', 'cruise')
            st.context = ctx
            io.state_pub.publish(st)
            self._publish_odom_tf(io, ns, stamp, tf, x, y, yaw, v_fwd)
            self._publish_imu(io, ns, stamp)
            self._publish_obstacle(io, stamp, obstacle)
            self._publish_camera(io, ns)
            if step % 20 == 1:
                self._publish_route(io, stamp, path_pts_full=driver.path_points(300.0, step_m=3.0))
            while not self.col_q.empty():
                ev = self.col_q.get_nowait()
                other = ev.other_actor
                oid = other.id if other is not None else 0
                if any(oid in h.ignored_collision_ids for h in self.hazards):
                    continue
                kind = actor_kind(other) if other is not None else 'other'
                imp = ev.normal_impulse
                intensity = math.sqrt(imp.x ** 2 + imp.y ** 2 + imp.z ** 2)
                io.col_pub.publish(CollisionEvent(header=st.header, other_actor_id=oid,
                                                  other_type=kind, intensity=intensity))
                metrics.collision(t, x, y, kind)
                self.log(f'Car {k} collision with {kind} ({getattr(other, "type_id", "?")})')
            obs_d = None
            if obstacle is not None:
                ox, oy = carla_to_ros_xy(obstacle.x, obstacle.y)
                obs_d = {'detected': True, 'x': ox, 'y': oy, 'distance': obstacle.distance,
                         'closing_speed': obstacle.closing_speed, 'ttc': obstacle.ttc}
            cap_xy = None
            if info.get('cap_xy') is not None:
                cap_xy = carla_to_ros_xy(*info['cap_xy'])
            if settle == 0:
                metrics.update(t, x, y, speed, accel, obs_d, info.get('cap_kmh', 0.0), cap_xy)
            traj_rows.append({'t': round(t - t_start, 2), 'x': round(x, 2), 'y': round(y, 2),
                              'speed_kmh': round(speed * 3.6, 2), 'accel': round(accel, 3),
                              'target_kmh': round(info.get('target_kmh', 0.0), 1),
                              'cap_kmh': round(info.get('cap_kmh', 0.0), 1),
                              'maneuver': info.get('maneuver', ''),
                              'obstacle_d': round(obstacle.distance, 2) if obstacle else '',
                              'ttc': round(obstacle.ttc, 2) if obstacle and obstacle.ttc < 1e8 else ''})
            if self.video is not None:
                self.video.overlay = self._overlay_lines(k, speed, info, adv)
            if self.g('follow_spectator') and step % 2 == 0:
                spec_tf = carla.Transform(
                    carla.Location(tf.location.x - 10.0 * fwd.x, tf.location.y - 10.0 * fwd.y,
                                   tf.location.z + 6.0),
                    carla.Rotation(pitch=-20.0, yaw=tf.rotation.yaw))
                self.world.get_spectator().set_transform(spec_tf)

            # ---------------- termination
            if settle > 0:
                settle -= 1
                if settle == 0:
                    break
            else:
                d_dest = tf.location.distance(self.dest)
                if d_dest < 6.0 or (driver.done() and d_dest < 20.0):
                    status, completed = 'arrived', True
                elif driver.done() and replans < 3:
                    replans += 1
                    driver.replan_to_destination()
                elif t - t_start > self.g('max_car_time_s'):
                    status = 'timeout'
                else:
                    slow_time = slow_time + dt if speed < 0.1 else 0.0
                    if slow_time > self.g('stuck_timeout_s'):
                        status = 'stuck'
                if status != 'running':
                    t_end = t
                    settle = int(2.0 / dt)              # let the writer finish its event window
            # ---------------- pacing
            rtf = float(self.g('realtime_factor'))
            if rtf > 0:
                ahead = (self.t_sim - sim0) / rtf - (time.time() - wall0)
                if ahead > 0:
                    time.sleep(ahead)

        # ------------------------------------------------------------ wrap-up
        traj.write(traj_rows)
        for s in sensors:
            try:
                s.stop()
                s.destroy()
            except RuntimeError:
                pass
        if self.video is not None:
            self.video.close()
        ego.destroy()
        self.tick(2)
        self.wait_writer(io)
        recs = io.take_records()
        gts = self.gt_for_car(k)
        for r in recs:
            best = min(gts, key=lambda g: math.hypot(g['x'] - r.x, g['y'] - r.y), default=None)
            dist = math.hypot(best['x'] - r.x, best['y'] - r.y) if best else 1e9
            match = best if best is not None and dist < 40.0 else None
            self.records_log.write({
                'car': k, 'id': r.id, 'source': r.source, 'trigger': r.trigger,
                'pred_type': r.hazard_type, 'gt_hazard': match['name'] if match else '',
                'gt_type': match['type'] if match else 'none',
                'correct': int(match is not None and match['type'] == r.hazard_type),
                'loc_error_m': round(dist, 2), 'severity': r.severity, 'lane': r.lane,
                'description': r.description, 'advice': r.advice})
        row, hrows = metrics.finish(t_end if status != 'running' else self.t_sim, completed, status,
                                    extra={'lane_changes': driver.lane_changes,
                                           'reroutes': driver.reroutes,
                                           'records_written': len(recs),
                                           'memory_mode': self.g('memory_mode'),
                                           'seed': self.g('seed')})
        self.cars_log.write(row)
        self.hazards_log.write(hrows)
        io.destroy()
        self.log(f'=== Car {k} done: {status}, {row["arrival_time_s"]} s, near-misses '
                 f'{row["near_misses"]}/{row["hazard_encounters"]}, hard brakes {row["hard_brakes"]}, '
                 f'collisions {row["collisions"]}, wrote {len(recs)} memories ===')

    def wait_writer(self, io):
        t0 = time.time()
        time.sleep(0.5)
        while time.time() - t0 < self.g('writer_flush_timeout_s'):
            with io.lock:
                if io.pending <= 0:
                    break
            time.sleep(0.2)
        time.sleep(0.5)                                  # let hazard_records arrive

    # ============================================================ publishing
    def _publish_odom_tf(self, io, ns, stamp, tf, x, y, yaw, v_fwd):
        qx, qy, qz, qw = yaw_to_quaternion(yaw)
        od = Odometry()
        od.header.stamp, od.header.frame_id = stamp, 'map'
        od.child_frame_id = f'{ns}/base_link'
        od.pose.pose.position.x, od.pose.pose.position.y, od.pose.pose.position.z = x, y, tf.location.z
        od.pose.pose.orientation.x, od.pose.pose.orientation.y = qx, qy
        od.pose.pose.orientation.z, od.pose.pose.orientation.w = qz, qw
        od.twist.twist.linear.x = v_fwd
        io.odom_pub.publish(od)
        ts = TransformStamped()
        ts.header = od.header
        ts.child_frame_id = od.child_frame_id
        ts.transform.translation.x, ts.transform.translation.y = x, y
        ts.transform.translation.z = tf.location.z
        ts.transform.rotation = od.pose.pose.orientation
        self.tf.sendTransform(ts)

    def _publish_imu(self, io, ns, stamp):
        m = self.imu_latest
        if m is None:
            return
        imu = Imu()
        imu.header.stamp, imu.header.frame_id = stamp, f'{ns}/base_link'
        imu.linear_acceleration.x = m.accelerometer.x
        imu.linear_acceleration.y = -m.accelerometer.y
        imu.linear_acceleration.z = m.accelerometer.z
        imu.angular_velocity.x = -m.gyroscope.x
        imu.angular_velocity.y = m.gyroscope.y
        imu.angular_velocity.z = -m.gyroscope.z
        imu.orientation.w = 1.0
        imu.orientation_covariance[0] = -1.0
        io.imu_pub.publish(imu)

    def _publish_obstacle(self, io, stamp, obstacle):
        o = ObstacleInfo()
        o.header.stamp, o.header.frame_id = stamp, 'map'
        if obstacle is not None:
            o.detected = True
            o.obstacle_type = obstacle.kind
            o.actor_id = int(obstacle.actor_id)
            o.distance = float(obstacle.distance)
            o.closing_speed = float(obstacle.closing_speed)
            o.ttc = float(min(obstacle.ttc, 1e9))
            o.x, o.y = carla_to_ros_xy(obstacle.x, obstacle.y)
        else:
            o.ttc = 1e9
        io.obs_pub.publish(o)

    def _publish_camera(self, io, ns):
        img = None
        while True:
            try:
                img = self.cam_q.get_nowait()
            except queue.Empty:
                break
        if img is None:
            return
        msg = Image()
        msg.header.stamp = to_time_msg(img.timestamp)
        msg.header.frame_id = f'{ns}/camera'
        msg.height, msg.width = img.height, img.width
        msg.encoding = 'bgra8'
        msg.is_bigendian = 0
        msg.step = img.width * 4
        data = array.array('B')
        data.frombytes(bytes(img.raw_data))
        msg.data = data
        io.cam_pub.publish(msg)

    def _publish_route(self, io, stamp, path_pts_full):
        path = Path()
        path.header.stamp, path.header.frame_id = stamp, 'map'
        for cx, cy in path_pts_full:
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x, ps.pose.position.y = carla_to_ros_xy(cx, cy)
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)
        io.route_pub.publish(path)

    @staticmethod
    def _overlay_lines(k, speed, info, adv):
        lines = [f'Car {k}   {speed * 3.6:5.1f} km/h   target {info.get("target_kmh", 0):4.0f}'
                 f'   {info.get("maneuver", "")}']
        if adv is not None:
            for c in list(adv.constraints)[:3]:
                lines.append(f'MEMORY {c.hazard_type} in {c.route_distance:.0f} m '
                             f'(c={c.confidence:.2f}) -> {c.action} <= {c.speed_cap_kmh:.0f} km/h')
                if c.description:
                    lines.append(f'   "{c.description[:90]}"')
        return lines


class VideoWriter:
    """Saves chase-camera frames with the retrieved-memory overlay as JPEGs."""

    def __init__(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.overlay = []
        self.q = queue.Queue(maxsize=200)
        self.n = 0
        self.th = threading.Thread(target=self._loop, daemon=True)
        self.th.start()

    def on_image(self, img):
        try:
            arr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape(img.height, img.width, 4)
            self.q.put_nowait((arr[:, :, :3].copy(), list(self.overlay)))
        except queue.Full:
            pass

    def _loop(self):
        import cv2
        while True:
            item = self.q.get()
            if item is None:
                return
            frame, lines = item
            y = 30
            for i, line in enumerate(lines):
                color = (255, 255, 255) if i == 0 else (0, 200, 255)
                cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4,
                            cv2.LINE_AA)
                cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
                            cv2.LINE_AA)
                y += 30
            cv2.imwrite(os.path.join(self.out_dir, f'frame_{self.n:05d}.jpg'), frame)
            self.n += 1

    def close(self):
        self.q.put(None)
        self.th.join(timeout=30.0)


def main(args=None):
    rclpy.init(args=args)
    node = ScenarioRunner()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    spin = threading.Thread(target=executor.spin, daemon=True)
    spin.start()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        node.get_logger().error(f'Scenario runner failed: {exc!r}')
        import traceback
        traceback.print_exc()
        try:
            node.cleanup()
        except Exception:
            pass
        raise
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
