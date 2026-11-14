"""Conversions between ``fleet_msgs`` messages and the pure-Python store types."""

from builtin_interfaces.msg import Time
from fleet_msgs.msg import HazardMemory

from .store import MemoryRecord


def to_time_msg(t: float) -> Time:
    t = max(0.0, float(t))
    sec = int(t)
    return Time(sec=sec, nanosec=int(round((t - sec) * 1e9)) % 1_000_000_000)


def from_time_msg(t: Time) -> float:
    return float(t.sec) + float(t.nanosec) * 1e-9


def record_to_msg(rec: MemoryRecord) -> HazardMemory:
    m = HazardMemory()
    m.id = rec.id
    m.hazard_type = rec.hazard_type
    m.description = rec.description
    m.lane = rec.lane
    m.severity = int(max(0, min(255, rec.severity)))
    m.advice = rec.advice
    m.x, m.y, m.z = float(rec.x), float(rec.y), float(rec.z)
    m.heading = float(rec.heading)
    m.road_id = int(rec.road_id)
    m.lane_id = int(rec.lane_id)
    m.first_seen = to_time_msg(rec.first_seen)
    m.last_confirmed = to_time_msg(rec.last_confirmed)
    m.reporter_id = rec.reporter_id
    m.confidence = float(rec.confidence)
    m.count = int(rec.count)
    m.refutations = int(rec.refutations)
    m.source = rec.source
    m.trigger = rec.trigger
    return m


def msg_to_record(m: HazardMemory) -> MemoryRecord:
    return MemoryRecord(
        id=m.id, hazard_type=m.hazard_type or 'other', description=m.description,
        lane=m.lane or 'unknown', severity=int(m.severity) or 3, advice=m.advice,
        x=m.x, y=m.y, z=m.z, heading=m.heading, road_id=m.road_id, lane_id=m.lane_id,
        first_seen=from_time_msg(m.first_seen), last_confirmed=from_time_msg(m.last_confirmed),
        reporter_id=m.reporter_id, confidence=float(m.confidence), count=max(1, int(m.count)),
        refutations=int(m.refutations), source=m.source or 'vlm', trigger=m.trigger)
