"""Per-vehicle hazard writer: surprise trigger + VLM event writer.

Runs inside the vehicle namespace (e.g. ``/ego_3``).

Subscribes (relative names)
  vehicle_state    fleet_msgs/VehicleState
  obstacle         fleet_msgs/ObstacleInfo
  collision        fleet_msgs/CollisionEvent
  camera/image     sensor_msgs/Image (rgb8 / bgr8 / bgra8)

Publishes
  surprise_events  fleet_msgs/SurpriseEvent   (strong + weak events)
  hazard_records   fleet_msgs/HazardMemory    (records sent to the memory)
  writer_pending   std_msgs/Int32 (latched)   (number of VLM jobs not yet written)

Calls
  /fleet_memory/write  fleet_msgs/srv/WriteMemory
"""

import json
import math
import os
import queue
import threading
import time
import uuid

import numpy as np
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Int32

from fleet_memory.conversions import to_time_msg
from fleet_memory.llm_client import ChatClient
from fleet_msgs.msg import (CollisionEvent, HazardMemory, ObstacleInfo, SurpriseEvent,
                            VehicleState)
from fleet_msgs.srv import WriteMemory

from .ring_buffer import FrameBuffer
from .trigger import Detection, SurpriseDetector, TriggerConfig, most_important, severity_from
from .vlm import VLMDescriber, encode_jpeg


def stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def image_to_rgb(msg: Image) -> np.ndarray:
    enc = msg.encoding.lower()
    ch = {'rgb8': 3, 'bgr8': 3, 'rgba8': 4, 'bgra8': 4, 'mono8': 1}.get(enc)
    if ch is None:
        raise ValueError(f'unsupported encoding {msg.encoding}')
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(msg.height, msg.step)
    img = buf[:, :msg.width * ch].reshape(msg.height, msg.width, ch)
    if enc == 'mono8':
        return np.repeat(img, 3, axis=2)
    if enc.startswith('bgr'):
        img = img[:, :, 2::-1]
    else:
        img = img[:, :, :3]
    return np.ascontiguousarray(img)


class HazardWriterNode(Node):
    def __init__(self):
        super().__init__('hazard_writer')
        p = self.declare_parameter
        p('vehicle_id', '')
        p('writer_mode', 'vlm')                 # vlm | template | off
        p('vlm_base_url', 'http://localhost:11434/v1')
        p('vlm_model', 'qwen2.5vl:7b')
        p('vlm_api_key_env', 'OPENAI_API_KEY')
        p('vlm_timeout_s', 90.0)
        p('vlm_max_tokens', 300)
        p('max_image_width', 640)
        p('buffer_s', 3.0)
        p('post_trigger_s', 1.0)
        p('num_keyframes', 4)
        p('keyframe_before_s', 2.0)
        p('keyframe_after_s', 0.5)
        p('cooldown_s', 5.0)
        p('hard_brake_g', 0.4)
        p('ttc_threshold_s', 1.5)
        p('jerk_threshold', 25.0)
        p('bump_jerk_threshold', 8.0)
        p('obstacle_evidence_range_m', 25.0)
        p('memory_service', '/fleet_memory/write')
        p('dataset_dir', '')                     # save keyframes + records here if set

        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.vehicle_id = g('vehicle_id') or self.get_namespace().strip('/') or 'ego'
        self.mode = g('writer_mode')
        self.post_trigger_s = g('post_trigger_s')
        self.cooldown_s = g('cooldown_s')
        self.num_keyframes = g('num_keyframes')
        self.kf_before, self.kf_after = g('keyframe_before_s'), g('keyframe_after_s')
        self.dataset_dir = os.path.expanduser(g('dataset_dir') or '')
        self.max_image_width = g('max_image_width')

        self.detector = SurpriseDetector(TriggerConfig(
            hard_brake_g=g('hard_brake_g'), ttc_threshold_s=g('ttc_threshold_s'),
            jerk_threshold=g('jerk_threshold'), bump_jerk_threshold=g('bump_jerk_threshold'),
            obstacle_evidence_range_m=g('obstacle_evidence_range_m')))
        self.frames = FrameBuffer(max_age_s=g('buffer_s') + self.post_trigger_s + 1.0)

        client = None
        if self.mode == 'vlm':
            client = ChatClient(base_url=g('vlm_base_url'), model=g('vlm_model'),
                                api_key_env=g('vlm_api_key_env'), timeout_s=g('vlm_timeout_s'),
                                max_tokens=g('vlm_max_tokens'))
        self.describer = VLMDescriber(client, self.max_image_width, self.get_logger())

        self.lock = threading.Lock()
        self.state: VehicleState = None
        self.state_history = []                 # (t, VehicleState) for the last few seconds
        self.obstacle: ObstacleInfo = None
        self.obstacle_t = -1e9
        self.pending_event = None
        self.last_event_t = -1e9
        self.jobs: queue.Queue = queue.Queue()
        self.pending_jobs = 0

        sensors_cb = MutuallyExclusiveCallbackGroup()
        self.cam_cb = MutuallyExclusiveCallbackGroup()
        self.srv_cb = ReentrantCallbackGroup()
        self.create_subscription(VehicleState, 'vehicle_state', self.on_state, 50,
                                 callback_group=sensors_cb)
        self.create_subscription(ObstacleInfo, 'obstacle', self.on_obstacle, 50,
                                 callback_group=sensors_cb)
        self.create_subscription(CollisionEvent, 'collision', self.on_collision, 20,
                                 callback_group=sensors_cb)
        self.create_subscription(Image, 'camera/image', self.on_image, 5,
                                 callback_group=self.cam_cb)
        self.event_pub = self.create_publisher(SurpriseEvent, 'surprise_events', 50)
        self.record_pub = self.create_publisher(HazardMemory, 'hazard_records', 10)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.pending_pub = self.create_publisher(Int32, 'writer_pending', latched)
        self.write_cli = self.create_client(WriteMemory, g('memory_service'),
                                            callback_group=self.srv_cb)
        self._publish_pending()

        self.worker = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker.start()
        self.get_logger().info(f'Hazard writer for {self.vehicle_id} ready (mode={self.mode})')

    # --------------------------------------------------------------- inputs
    def on_image(self, msg: Image):
        try:
            img = image_to_rgb(msg)
        except Exception as exc:
            self.get_logger().warning(f'bad image: {exc}', throttle_duration_sec=10.0)
            return
        if img.shape[1] > self.max_image_width:
            step = int(math.ceil(img.shape[1] / self.max_image_width))
            img = np.ascontiguousarray(img[::step, ::step])
        self.frames.append(stamp_to_sec(msg.header.stamp), img)

    def on_obstacle(self, msg: ObstacleInfo):
        t = stamp_to_sec(msg.header.stamp)
        with self.lock:
            self.obstacle, self.obstacle_t = msg, t
        dets = self.detector.update_obstacle(
            t, msg.detected, msg.distance, msg.closing_speed, msg.ttc, msg.obstacle_type,
            msg.actor_id, (msg.x, msg.y))
        self._handle(dets)

    def on_collision(self, msg: CollisionEvent):
        t = stamp_to_sec(msg.header.stamp)
        self._handle([self.detector.collision(t, msg.other_type, msg.intensity)])

    def on_state(self, msg: VehicleState):
        t = stamp_to_sec(msg.header.stamp)
        with self.lock:
            if self.state is not None and t + 1.0 < stamp_to_sec(self.state.header.stamp):
                # simulation clock went backwards (new experiment) -> reset everything
                self.detector.reset()
                self.frames.clear()
                self.pending_event = None
                self.last_event_t = -1e9
                self.state_history.clear()
            self.state = msg
            self.state_history.append((t, msg))
            while self.state_history and t - self.state_history[0][0] > 5.0:
                self.state_history.pop(0)
        self._handle(self.detector.update_state(t, msg.accel_long))
        self._maybe_finalize(t)

    # -------------------------------------------------------------- events
    def _state_at(self, t: float):
        with self.lock:
            best = None
            for ts, st in self.state_history:
                if best is None or abs(ts - t) < abs(best[0] - t):
                    best = (ts, st)
            return best[1] if best else self.state

    def _hazard_xy(self, det: Detection, st: VehicleState):
        if det.obstacle_xy is not None and det.obstacle_type != 'none':
            return det.obstacle_xy
        with self.lock:
            obs, obs_t = self.obstacle, self.obstacle_t
        if obs is not None and obs.detected and abs(obs_t - det.t) < 0.5 and obs.distance < 40.0:
            return obs.x, obs.y
        if det.kind == 'collision':
            return (st.x + 2.5 * math.cos(st.heading), st.y + 2.5 * math.sin(st.heading))
        return st.x, st.y

    def _handle(self, dets):
        for det in dets:
            st = self._state_at(det.t)
            if st is None:
                continue
            if det.obstacle_type == 'none' and det.kind == 'collision':
                det.obstacle_type = 'other'
            hx, hy = self._hazard_xy(det, st)
            ev = SurpriseEvent()
            ev.header.stamp = to_time_msg(det.t)
            ev.header.frame_id = 'map'
            ev.vehicle_id = self.vehicle_id
            ev.kind = det.kind
            ev.is_surprise = det.is_surprise
            ev.magnitude = float(det.magnitude)
            ev.x, ev.y, ev.heading, ev.speed = st.x, st.y, st.heading, st.speed
            ev.obstacle_type = det.obstacle_type or 'none'
            ev.hazard_x, ev.hazard_y = float(hx), float(hy)
            self.event_pub.publish(ev)
            if not det.is_surprise or self.mode == 'off':
                continue
            self.get_logger().info(f'Surprise: {det.kind} mag={det.magnitude:.2f} '
                                   f'obstacle={det.obstacle_type} at ({hx:.1f},{hy:.1f})')
            with self.lock:
                if self.pending_event is not None:
                    self.pending_event['dets'].append((det, ev))
                elif det.t - self.last_event_t >= self.cooldown_s:
                    self.pending_event = {'t0': det.t, 'dets': [(det, ev)]}
                    self.last_event_t = det.t

    def _maybe_finalize(self, t: float):
        with self.lock:
            ev = self.pending_event
            if ev is None or t < ev['t0'] + self.post_trigger_s:
                return
            self.pending_event = None
            self.pending_jobs += 1
        self._publish_pending()
        self.jobs.put(ev)

    def _publish_pending(self):
        self.pending_pub.publish(Int32(data=int(self.pending_jobs)))

    # -------------------------------------------------------------- worker
    def _worker_loop(self):
        while rclpy.ok():
            try:
                ev = self.jobs.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._process(ev)
            except Exception as exc:  # never kill the worker
                self.get_logger().error(f'failed to write memory: {exc!r}')
            finally:
                with self.lock:
                    self.pending_jobs -= 1
                self._publish_pending()

    def _process(self, ev):
        main = most_important([d for d, _ in ev['dets']])
        det = main
        # hazard position: prefer an event that localised an obstacle
        sev_msg = next((e for d, e in ev['dets'] if d.obstacle_type not in ('', 'none', 'other')),
                       next(e for d, e in ev['dets'] if d is main))
        st =self._state_at(ev['t0'] - 0.3)      # state just before the reaction
        with self.lock:
            obs = self.obstacle
        obstacle_type = next((d.obstacle_type for d, _ in ev['dets']
                              if d.obstacle_type not in ('', 'none')), 'none')
        ctx = {
            'trigger': det.kind,
            'magnitude': float(det.magnitude),
            'speed_kmh': st.speed * 3.6,
            'obstacle_type': obstacle_type,
            'obstacle_distance': float(obs.distance) if obs is not None and obs.detected else 0.0,
            'obstacle_stationary': (obs is not None and obs.detected
                                    and abs(obs.closing_speed - st.speed) < 1.5),
            'context': st.context,
        }
        severity = severity_from(det)
        frames = self.frames.keyframes(ev['t0'], self.num_keyframes, self.kf_before, self.kf_after)
        imgs = [f for _, f in frames]
        t_call = time.time()
        rec, source, raw = self.describer.describe(imgs, ctx, severity)
        latency = time.time() - t_call

        m = HazardMemory()
        m.id = uuid.uuid4().hex[:12]
        m.hazard_type = rec['hazard_type']
        m.description = rec['description']
        m.lane = rec['lane']
        m.severity = int(rec['severity'])
        m.advice = rec['advice']
        m.x, m.y, m.z = sev_msg.hazard_x, sev_msg.hazard_y, st.z
        m.heading = st.heading
        m.road_id, m.lane_id = st.road_id, st.lane_id
        m.first_seen = m.last_confirmed = to_time_msg(ev['t0'])
        m.reporter_id = self.vehicle_id
        m.confidence = 0.0                       # memory default
        m.count = 1
        m.source = source
        m.trigger = det.kind

        if self.dataset_dir:
            self._save_example(m, frames, ctx, raw, latency)

        if not self.write_cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('fleet memory write service unavailable; record dropped')
            return
        req = WriteMemory.Request(record=m, allow_merge=True)
        done = threading.Event()
        fut = self.write_cli.call_async(req)
        fut.add_done_callback(lambda _: done.set())
        if not done.wait(timeout=10.0) or fut.result() is None:
            self.get_logger().error('fleet memory write timed out')
            return
        res = fut.result()
        m.id = res.id
        self.record_pub.publish(m)
        self.get_logger().info(
            f'Wrote memory {res.id} ({res.message}, c={res.confidence:.2f}, src={source}, '
            f'vlm {latency:.1f}s): {m.hazard_type} sev={m.severity} "{m.description}"')

    def _save_example(self, m: HazardMemory, frames, ctx, raw, latency):
        d = os.path.join(self.dataset_dir, f'{self.vehicle_id}_{m.id}')
        os.makedirs(d, exist_ok=True)
        for i, (t, f) in enumerate(frames):
            with open(os.path.join(d, f'frame_{i}_{t:.2f}.jpg'), 'wb') as fh:
                fh.write(encode_jpeg(f, self.max_image_width))
        with open(os.path.join(d, 'record.json'), 'w') as fh:
            json.dump({'id': m.id, 'vehicle_id': self.vehicle_id, 'hazard_type': m.hazard_type,
                       'description': m.description, 'lane': m.lane, 'severity': int(m.severity),
                       'advice': m.advice, 'x': m.x, 'y': m.y, 'source': m.source,
                       'trigger': m.trigger, 'context': ctx, 'raw_reply': raw,
                       'latency_s': latency}, fh, indent=2)


def main(args=None):
    rclpy.init(args=args)
    node = HazardWriterNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
