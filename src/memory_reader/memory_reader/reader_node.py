"""Per-vehicle memory reader: route look-ahead retrieval, re-ranking and policy adapter.

Runs inside the vehicle namespace (e.g. ``/ego_3``).

Subscribes (relative)
  vehicle_state     fleet_msgs/VehicleState
  planned_route     nav_msgs/Path            (remaining route of the driving agent)
  surprise_events   fleet_msgs/SurpriseEvent (from the hazard writer; evidence for feedback)

Publishes
  driving_advice    fleet_msgs/DrivingAdvice

Calls
  /fleet_memory/query     (at ``query_rate_hz``)
  /fleet_memory/feedback  (confirm / refute after passing a retrieved memory)
"""

import math

import rclpy
from nav_msgs.msg import Path
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from fleet_memory.llm_client import ChatClient
from fleet_msgs.msg import DrivingAdvice, HazardConstraint, SurpriseEvent, VehicleState
from fleet_msgs.srv import MemoryFeedback, QueryMemory

from .feedback import PassTracker
from .geometry import RoutePolyline, route_distance
from .policy import LLMAdapter, PolicyConfig, describe_actions, rule_adapter
from .scoring import Candidate, ScoringConfig, rank_candidates


def stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


class MemoryReaderNode(Node):
    def __init__(self):
        super().__init__('memory_reader')
        p = self.declare_parameter
        p('vehicle_id', '')
        p('adapter', 'rules')                    # rules | llm | none
        p('query_rate_hz', 2.0)
        p('min_confidence', 0.25)
        # scoring
        p('lookahead_m', 150.0)
        p('corridor_m', 6.0)
        p('behind_margin_m', 15.0)
        p('w_distance', 0.2)
        p('w_heading', 0.2)
        p('w_confidence', 0.2)
        p('w_recency', 0.2)
        p('w_similarity', 0.2)
        p('use_heading', True)
        p('tau_s', 1800.0)
        p('s_min', 0.45)
        p('top_k', 3)
        # policy
        p('a_comfort', 2.0)
        p('brake_buffer_m', 8.0)
        p('enable_lane_change', True)
        p('enable_reroute', True)
        p('reroute_min_severity', 5)
        p('reroute_min_confidence', 0.8)
        p('cruise_speed_kmh', 45.0)
        # LLM adapter
        p('llm_base_url', 'http://localhost:11434/v1')
        p('llm_model', 'qwen2.5:7b')
        p('llm_api_key_env', 'OPENAI_API_KEY')
        p('llm_timeout_s', 30.0)
        # feedback
        p('feedback_enabled', True)
        p('pass_radius_m', 15.0)
        p('evidence_radius_m', 25.0)

        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.vehicle_id = g('vehicle_id') or self.get_namespace().strip('/') or 'ego'
        self.adapter = g('adapter')
        self.min_conf = g('min_confidence')
        self.cruise_kmh = g('cruise_speed_kmh')
        self.scoring = ScoringConfig(
            lookahead_m=g('lookahead_m'), corridor_m=g('corridor_m'),
            behind_margin_m=g('behind_margin_m'), w_distance=g('w_distance'),
            w_heading=g('w_heading'), w_confidence=g('w_confidence'), w_recency=g('w_recency'),
            w_similarity=g('w_similarity'), use_heading=g('use_heading'), tau_s=g('tau_s'),
            s_min=g('s_min'), top_k=g('top_k'))
        self.policy = PolicyConfig(
            a_comfort=g('a_comfort'), brake_buffer_m=g('brake_buffer_m'),
            enable_lane_change=g('enable_lane_change'), enable_reroute=g('enable_reroute'),
            reroute_min_severity=g('reroute_min_severity'),
            reroute_min_confidence=g('reroute_min_confidence'))
        self.llm = None
        if self.adapter == 'llm':
            self.llm = LLMAdapter(ChatClient(g('llm_base_url'), g('llm_model'), g('llm_api_key_env'),
                                             timeout_s=g('llm_timeout_s'), max_tokens=400),
                                  self.policy, self.get_logger())
        self.feedback_enabled = g('feedback_enabled') and self.adapter != 'none'
        self.tracker = PassTracker(pass_radius_m=g('pass_radius_m'),
                                   evidence_radius_m=g('evidence_radius_m'))

        self.state: VehicleState = None
        self.route: RoutePolyline = RoutePolyline([])
        self.pending_query = None
        self.pending_since = 0.0
        self.last_t = None

        self.create_subscription(VehicleState, 'vehicle_state', self.on_state, 20)
        self.create_subscription(Path, 'planned_route', self.on_route, 5)
        self.create_subscription(SurpriseEvent, 'surprise_events', self.on_event, 50)
        self.advice_pub = self.create_publisher(DrivingAdvice, 'driving_advice', 10)
        self.query_cli = self.create_client(QueryMemory, '/fleet_memory/query')
        self.feedback_cli = self.create_client(MemoryFeedback, '/fleet_memory/feedback')
        self.create_timer(1.0 / max(0.1, g('query_rate_hz')), self.on_timer)
        self.get_logger().info(f'Memory reader for {self.vehicle_id} ready (adapter={self.adapter})')

    # ---------------------------------------------------------------- inputs
    def on_state(self, msg: VehicleState):
        t = stamp_to_sec(msg.header.stamp)
        if self.last_t is not None and t + 1.0 < self.last_t:
            self.tracker.reset()                 # simulation restarted
        self.last_t = t
        self.state = msg
        if self.feedback_enabled:
            for mid, decision in self.tracker.update(msg.x, msg.y, msg.heading, t):
                self._send_feedback(mid, decision)

    def on_route(self, msg: Path):
        self.route = RoutePolyline([(ps.pose.position.x, ps.pose.position.y) for ps in msg.poses])

    def on_event(self, msg: SurpriseEvent):
        self.tracker.add_evidence(stamp_to_sec(msg.header.stamp), msg.hazard_x, msg.hazard_y,
                                  msg.is_surprise, msg.kind)

    # ----------------------------------------------------------------- query
    def on_timer(self):
        st = self.state
        if st is None:
            return
        if self.adapter == 'none':
            self._publish([], 'memory disabled')
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.pending_query is not None and now - self.pending_since < 2.0:
            return
        if not self.query_cli.service_is_ready():
            self.get_logger().warning('fleet memory not available', throttle_duration_sec=10.0)
            return
        req = QueryMemory.Request()
        req.x, req.y = st.x, st.y
        req.radius = float(self.scoring.lookahead_m + self.scoring.behind_margin_m)
        req.query_text = st.context or 'hazard ahead on my route'
        req.min_confidence = float(self.min_conf)
        self.pending_since = now
        self.pending_query = self.query_cli.call_async(req)
        self.pending_query.add_done_callback(lambda fut, st=st: self.on_query_result(fut, st))

    def _route_distance(self, st: VehicleState, x: float, y: float):
        if len(self.route) >= 2:
            return route_distance(self.route, (st.x, st.y), (x, y), self.scoring.corridor_m)
        # no route yet: straight-line approximation along the heading
        dx, dy = x - st.x, y - st.y
        along = dx * math.cos(st.heading) + dy * math.sin(st.heading)
        lateral = abs(-dx * math.sin(st.heading) + dy * math.cos(st.heading))
        return along if lateral <= self.scoring.corridor_m else None

    def on_query_result(self, fut, st: VehicleState):
        self.pending_query = None
        try:
            res = fut.result()
        except Exception as exc:
            self.get_logger().warning(f'query failed: {exc}')
            return
        if res is None:
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        cands = []
        for rec, sim in zip(res.records, res.similarities):
            d = self._route_distance(st, rec.x, rec.y)
            if d is None:
                continue
            age = now - stamp_to_sec(rec.last_confirmed)
            cands.append(Candidate(memory=rec, route_distance=d,
                                   heading_cos=math.cos(st.heading - rec.heading),
                                   age_s=age, similarity=float(sim)))
        ranked = rank_candidates(cands, self.scoring)
        actions = rule_adapter(ranked, st.speed, self.policy)
        why = 'rules'
        if self.llm is not None and actions:
            actions, why, _ = self.llm.decide(
                actions, {'speed_kmh': st.speed * 3.6, 'cruise_kmh': self.cruise_kmh,
                          'context': st.context}, st.speed)
        for a in actions:
            self.tracker.observe_memory(a.memory_id, a.x, a.y, stamp_to_sec(st.header.stamp))
        if actions:
            self.get_logger().info(f'{describe_actions(actions)} ({why})', throttle_duration_sec=2.0)
        self._publish(actions, why)

    def _publish(self, actions, why: str):
        msg = DrivingAdvice()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.vehicle_id = self.vehicle_id
        msg.adapter = self.adapter
        msg.justification = why
        for a in actions:
            c = HazardConstraint()
            c.memory_id = a.memory_id
            c.hazard_type = a.hazard_type
            c.description = a.description
            c.severity = int(a.severity)
            c.confidence = float(a.confidence)
            c.score = float(a.score)
            c.x, c.y = float(a.x), float(a.y)
            c.route_distance = float(a.route_distance)
            c.speed_cap_kmh = float(a.speed_cap_kmh)
            c.brake_distance_m = float(a.brake_distance_m)
            c.action = a.action
            c.preferred_lane = a.preferred_lane
            msg.constraints.append(c)
        self.advice_pub.publish(msg)

    # -------------------------------------------------------------- feedback
    def _send_feedback(self, mem_id: str, decision: str):
        if decision == 'skip':
            self.get_logger().info(f'Passed memory {mem_id}: surprise observed (writer confirms)')
            return
        if not self.feedback_cli.service_is_ready():
            return
        req = MemoryFeedback.Request(id=mem_id, confirm=(decision == 'confirm'),
                                     reporter_id=self.vehicle_id)
        self.feedback_cli.call_async(req)
        self.get_logger().info(f'Passed memory {mem_id}: {decision}')


def main(args=None):
    rclpy.init(args=args)
    node = MemoryReaderNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
