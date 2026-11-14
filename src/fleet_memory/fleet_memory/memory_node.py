"""Shared fleet memory node (plays the role of the fleet's cloud server).

Services
  /fleet_memory/write     fleet_msgs/srv/WriteMemory
  /fleet_memory/query     fleet_msgs/srv/QueryMemory
  /fleet_memory/feedback  fleet_msgs/srv/MemoryFeedback
  /fleet_memory/reset     std_srvs/srv/Trigger
  /fleet_memory/save      std_srvs/srv/Trigger   (writes ``persist_path``)

Topics
  /fleet_memory/events    fleet_msgs/msg/MemoryEvent   (insert/merge/confirm/refute/prune/reset)
  /fleet_memory/markers   visualization_msgs/msg/MarkerArray (RViz)
"""

import os

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray

from fleet_msgs.msg import MemoryEvent
from fleet_msgs.srv import MemoryFeedback, QueryMemory, WriteMemory

from .conversions import msg_to_record, record_to_msg, to_time_msg
from .embeddings import make_embedder
from .store import FleetMemoryStore, MemoryConfig

TYPE_COLORS = {
    'pedestrian_crossing': (1.0, 0.55, 0.0),
    'lane_closure': (1.0, 0.9, 0.0),
    'road_damage': (0.6, 0.3, 0.1),
    'stalled_vehicle': (0.9, 0.1, 0.1),
    'other': (0.6, 0.6, 0.6),
}


class FleetMemoryNode(Node):
    def __init__(self):
        super().__init__('fleet_memory')
        p = self.declare_parameter
        p('merge_radius_m', 20.0)
        p('merge_text_similarity', 0.3)
        p('merge_heading_cos', 0.0)
        p('initial_confidence', 0.6)
        p('confirm_gain', 0.5)
        p('refute_factor', 0.5)
        p('decay_lambda', 1.0 / 3600.0)
        p('prune_threshold', 0.1)
        p('prune_period_s', 5.0)
        p('embedding_backend', 'auto')
        p('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2')
        p('persist_path', '')
        p('load_on_start', False)
        p('save_on_exit', False)
        p('frame_id', 'map')
        p('marker_period_s', 1.0)

        g = lambda n: self.get_parameter(n).value  # noqa: E731
        cfg = MemoryConfig(
            merge_radius_m=g('merge_radius_m'), merge_text_similarity=g('merge_text_similarity'),
            merge_heading_cos=g('merge_heading_cos'), initial_confidence=g('initial_confidence'),
            confirm_gain=g('confirm_gain'), refute_factor=g('refute_factor'),
            decay_lambda=g('decay_lambda'), prune_threshold=g('prune_threshold'))
        embedder = make_embedder(g('embedding_backend'), g('embedding_model'), self.get_logger())
        self.store = FleetMemoryStore(cfg, embedder)
        self.frame_id = g('frame_id')
        self.persist_path = os.path.expanduser(g('persist_path') or '')

        if self.persist_path and g('load_on_start') and os.path.exists(self.persist_path):
            n = self.store.load(self.persist_path, now=self._now())
            self.get_logger().info(f'Loaded {n} memories from {self.persist_path}')

        self.events_pub = self.create_publisher(MemoryEvent, '/fleet_memory/events', 50)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.marker_pub = self.create_publisher(MarkerArray, '/fleet_memory/markers', latched)

        self.create_service(WriteMemory, '/fleet_memory/write', self.on_write)
        self.create_service(QueryMemory, '/fleet_memory/query', self.on_query)
        self.create_service(MemoryFeedback, '/fleet_memory/feedback', self.on_feedback)
        self.create_service(Trigger, '/fleet_memory/reset', self.on_reset)
        self.create_service(Trigger, '/fleet_memory/save', self.on_save)

        self.create_timer(g('prune_period_s'), self.on_prune_timer)
        self.create_timer(g('marker_period_s'), self.publish_markers)
        self.get_logger().info(f'Fleet memory ready (embedder={embedder.name})')

    # ----------------------------------------------------------------- helpers
    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _emit(self, event: str, rec):
        msg = MemoryEvent()
        msg.stamp = to_time_msg(self._now())
        msg.event = event
        if rec is not None:
            msg.record = record_to_msg(rec)
        self.events_pub.publish(msg)

    # --------------------------------------------------------------- services
    def on_write(self, req, resp):
        rec = msg_to_record(req.record)
        now = self._now()
        stored, merged = self.store.write(rec, now, allow_merge=req.allow_merge)
        resp.success = True
        resp.merged = merged
        resp.id = stored.id
        resp.confidence = float(stored.confidence)
        resp.message = 'merged' if merged else 'inserted'
        self._emit('merge' if merged else 'insert', stored)
        self.get_logger().info(
            f'[{resp.message}] {stored.id} {stored.hazard_type} sev={stored.severity} '
            f'c={stored.confidence:.2f} n={stored.count} from {rec.reporter_id}: '
            f'"{stored.description[:80]}"')
        return resp

    def on_query(self, req, resp):
        hits = self.store.query(req.x, req.y, req.radius, self._now(), req.query_text,
                                int(req.max_results), float(req.min_confidence))
        resp.records = [record_to_msg(r) for r, _, _ in hits]
        resp.similarities = [float(s) for _, s, _ in hits]
        resp.distances = [float(d) for _, _, d in hits]
        return resp

    def on_feedback(self, req, resp):
        rec, pruned = self.store.feedback(req.id, req.confirm, self._now())
        if rec is None:
            resp.success = False
            return resp
        resp.success = True
        resp.confidence = float(rec.confidence)
        resp.pruned = pruned
        self._emit('confirm' if req.confirm else 'refute', rec)
        if pruned:
            self._emit('prune', rec)
        self.get_logger().info(
            f'[{"confirm" if req.confirm else "refute"}] {rec.id} by {req.reporter_id} '
            f'-> c={rec.confidence:.2f}{" (pruned)" if pruned else ""}')
        return resp

    def on_reset(self, req, resp):
        self.store.reset()
        self._emit('reset', None)
        resp.success = True
        resp.message = 'memory cleared'
        self.get_logger().info('Memory reset')
        return resp

    def on_save(self, req, resp):
        if not self.persist_path:
            resp.success = False
            resp.message = 'persist_path not set'
            return resp
        self.store.save(self.persist_path)
        resp.success = True
        resp.message = f'saved {len(self.store)} records to {self.persist_path}'
        return resp

    # ------------------------------------------------------------------ timers
    def on_prune_timer(self):
        for rec in self.store.prune(self._now()):
            self._emit('prune', rec)
            self.get_logger().info(f'[prune] {rec.id} {rec.hazard_type} c={rec.confidence:.3f}')

    def publish_markers(self):
        # confidences change continuously with decay, so always refresh
        self.store.refresh(self._now())
        arr = MarkerArray()
        clear = Marker()
        clear.header.frame_id = self.frame_id
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)
        stamp = self.get_clock().now().to_msg()
        for i, rec in enumerate(self.store.all_records()):
            r, g, b = TYPE_COLORS.get(rec.hazard_type, TYPE_COLORS['other'])
            cyl = Marker()
            cyl.header.frame_id = self.frame_id
            cyl.header.stamp = stamp
            cyl.ns = 'memory'
            cyl.id = 2 * i
            cyl.type = Marker.CYLINDER
            cyl.action = Marker.ADD
            h = 1.0 + 6.0 * rec.confidence
            cyl.pose.position.x, cyl.pose.position.y, cyl.pose.position.z = rec.x, rec.y, h / 2
            cyl.pose.orientation.w = 1.0
            cyl.scale.x = cyl.scale.y = 2.0 + rec.severity
            cyl.scale.z = h
            cyl.color.r, cyl.color.g, cyl.color.b = r, g, b
            cyl.color.a = 0.3 + 0.6 * rec.confidence
            arr.markers.append(cyl)
            txt = Marker()
            txt.header = cyl.header
            txt.ns = 'memory_text'
            txt.id = 2 * i + 1
            txt.type = Marker.TEXT_VIEW_FACING
            txt.action = Marker.ADD
            txt.pose.position.x, txt.pose.position.y, txt.pose.position.z = rec.x, rec.y, h + 2.0
            txt.pose.orientation.w = 1.0
            txt.scale.z = 1.5
            txt.color.r = txt.color.g = txt.color.b = txt.color.a = 1.0
            txt.text = (f'{rec.hazard_type} c={rec.confidence:.2f} n={rec.count}\n'
                        f'{rec.description[:60]}')
            arr.markers.append(txt)
        self.marker_pub.publish(arr)

    def shutdown(self):
        if self.persist_path and self.get_parameter('save_on_exit').value:
            self.store.save(self.persist_path)
            self.get_logger().info(f'Saved memory to {self.persist_path}')


def main(args=None):
    rclpy.init(args=args)
    node = FleetMemoryNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
