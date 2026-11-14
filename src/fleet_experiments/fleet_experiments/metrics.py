"""Per-car metrics (Section 4.1 of the proposal). No ROS / CARLA dependency.

Per car: near-miss rate, hard-braking events (per km and per hazard encounter),
collisions, arrival time, comfort (peak jerk / deceleration near hazards),
unnecessary slow-downs, and memory quality (retrieval precision / recall, lead
distance).
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

G = 9.81


@dataclass
class MetricsConfig:
    hard_brake_g: float = 0.4
    hard_brake_release_g: float = 0.3
    ttc_threshold_s: float = 1.5
    near_miss_min_closing_mps: float = 2.0
    near_miss_distance_m: float = 1.0
    zone_radius_m: float = 35.0          # hazard encounter window (euclidean)
    match_radius_m: float = 30.0         # memory <-> ground-truth hazard matching
    accel_smoothing: float = 0.35


@dataclass
class HazardEncounter:
    name: str
    type: str
    x: float
    y: float
    active: bool
    entered: bool = False
    min_ttc: float = 1e9
    min_distance: float = 1e9
    min_distance_closing: float = 1e9    # min distance while still approaching
    max_decel: float = 0.0
    max_jerk: float = 0.0
    min_speed_kmh: float = 1e9
    closest_dist: float = 1e9
    speed_at_closest_kmh: float = 0.0
    hard_brakes: int = 0
    collisions: int = 0
    retrieved: bool = False              # matching memory retrieved before reaching it
    lead_distance_m: float = 0.0         # distance at which the car started adapting
    near_miss: bool = False

    def row(self) -> Dict:
        return {
            'hazard': self.name, 'hazard_type': self.type, 'active': int(self.active),
            'encountered': int(self.entered), 'near_miss': int(self.near_miss),
            'collisions': self.collisions, 'hard_brakes': self.hard_brakes,
            'min_ttc_s': round(self.min_ttc, 3) if self.min_ttc < 1e8 else '',
            'min_distance_m': round(self.min_distance, 2) if self.min_distance < 1e8 else '',
            'max_decel_mps2': round(self.max_decel, 3), 'max_jerk_mps3': round(self.max_jerk, 2),
            'min_speed_kmh': round(self.min_speed_kmh, 2) if self.min_speed_kmh < 1e8 else '',
            'speed_at_hazard_kmh': round(self.speed_at_closest_kmh, 2),
            'retrieved': int(self.retrieved), 'lead_distance_m': round(self.lead_distance_m, 1),
        }


class MetricsRecorder:
    def __init__(self, car_index: int, hazards: List[Dict], cfg: Optional[MetricsConfig] = None):
        """``hazards``: [{'name', 'type', 'x', 'y', 'active'}] in the ROS map frame."""
        self.car = car_index
        self.cfg = cfg or MetricsConfig()
        self.enc = [HazardEncounter(h['name'], h['type'], h['x'], h['y'], h.get('active', True))
                    for h in hazards]
        self.t0: Optional[float] = None
        self.t_last = 0.0
        self.distance_m = 0.0
        self.last_xy: Optional[Tuple[float, float]] = None
        self._acc_s: Optional[float] = None
        self._braking = False
        self.hard_brakes = 0
        self.collisions: List[Tuple[float, str]] = []
        self.peak_decel = 0.0
        self.peak_jerk = 0.0
        self.unnecessary_slow_s = 0.0
        self.capped_s = 0.0
        self.retrieved: Dict[str, Tuple[float, float, bool]] = {}   # id -> (x, y, true_positive)
        self.speed_sum = 0.0
        self.n = 0

    # ------------------------------------------------------------- helpers
    def _nearest_active(self, x: float, y: float, radius: float) -> Optional[HazardEncounter]:
        best, bd = None, radius
        for e in self.enc:
            if not e.active:
                continue
            d = math.hypot(e.x - x, e.y - y)
            if d <= bd:
                best, bd = e, d
        return best

    # ------------------------------------------------------------- updates
    def update(self, t: float, x: float, y: float, speed: float, accel_long: float,
               obstacle: Optional[Dict] = None, cap_kmh: float = 0.0,
               cap_xy: Optional[Tuple[float, float]] = None):
        c = self.cfg
        if self.t0 is None:
            self.t0, self.t_last = t, t
            self._acc_s = accel_long
            self.last_xy = (x, y)
            return
        dt = max(1e-6, t - self.t_last)
        self.t_last = t
        self.distance_m += math.hypot(x - self.last_xy[0], y - self.last_xy[1])
        self.last_xy = (x, y)
        self.speed_sum += speed
        self.n += 1

        prev = self._acc_s
        a = prev + c.accel_smoothing * (accel_long - prev)
        jerk = abs(a - prev) / dt
        self._acc_s = a
        self.peak_decel = max(self.peak_decel, -a)
        self.peak_jerk = max(self.peak_jerk, jerk)
        new_brake = False
        if not self._braking and a < -c.hard_brake_g * G:
            self._braking = True
            self.hard_brakes += 1
            new_brake = True
        elif self._braking and a > -c.hard_brake_release_g * G:
            self._braking = False

        if cap_kmh > 0 and cap_xy is not None:
            self.capped_s += dt
            if self._nearest_active(cap_xy[0], cap_xy[1], 40.0) is None:
                self.unnecessary_slow_s += dt

        for e in self.enc:
            if not e.active:
                continue
            d = math.hypot(e.x - x, e.y - y)
            if d > c.zone_radius_m:
                continue
            e.entered = True
            e.max_decel = max(e.max_decel, -a)
            e.max_jerk = max(e.max_jerk, jerk)
            e.min_speed_kmh = min(e.min_speed_kmh, speed * 3.6)
            if d < e.closest_dist:
                e.closest_dist = d
                e.speed_at_closest_kmh = speed * 3.6
            if new_brake:
                e.hard_brakes += 1
            if obstacle and obstacle.get('detected'):
                ox, oy = obstacle['x'], obstacle['y']
                if math.hypot(ox - e.x, oy - e.y) <= c.zone_radius_m:
                    e.min_distance = min(e.min_distance, obstacle['distance'])
                    if obstacle['closing_speed'] > c.near_miss_min_closing_mps:
                        e.min_ttc = min(e.min_ttc, obstacle['ttc'])
                        e.min_distance_closing = min(e.min_distance_closing, obstacle['distance'])

    def collision(self, t: float, x: float, y: float, other_type: str):
        if self.collisions and t - self.collisions[-1][0] < 1.0:
            return                               # same contact
        self.collisions.append((t, other_type))
        e = self._nearest_active(x, y, self.cfg.zone_radius_m)
        if e is not None:
            e.collisions += 1

    def advice(self, ego_x: float, ego_y: float, constraints: List[Dict]):
        """Record retrieved memories (``constraints`` in ROS frame with memory_id, x, y)."""
        for c in constraints:
            mid = c['memory_id']
            gt = self._nearest_active(c['x'], c['y'], self.cfg.match_radius_m)
            if mid not in self.retrieved:
                self.retrieved[mid] = (c['x'], c['y'], gt is not None)
            if gt is not None and not gt.retrieved and gt.closest_dist > 15.0:
                gt.retrieved = True
                gt.lead_distance_m = math.hypot(gt.x - ego_x, gt.y - ego_y)

    # ------------------------------------------------------------- results
    def finish(self, t_end: float, completed: bool, status: str, extra: Optional[Dict] = None):
        c = self.cfg
        for e in self.enc:
            e.near_miss = e.entered and (
                e.min_ttc < c.ttc_threshold_s or e.collisions > 0 or
                (e.min_distance_closing < c.near_miss_distance_m))
        active = [e for e in self.enc if e.active and e.entered]
        n_enc = max(1, len(active))
        tp = sum(1 for v in self.retrieved.values() if v[2])
        km = max(1e-3, self.distance_m / 1000.0)
        row = {
            'car': self.car,
            'status': status,
            'completed': int(completed),
            'arrival_time_s': round((t_end - self.t0) if self.t0 is not None else 0.0, 2),
            'distance_m': round(self.distance_m, 1),
            'mean_speed_kmh': round(3.6 * self.speed_sum / max(1, self.n), 2),
            'hazard_encounters': len(active),
            'near_misses': sum(e.near_miss for e in active),
            'near_miss_rate': round(sum(e.near_miss for e in active) / n_enc, 3),
            'hard_brakes': self.hard_brakes,
            'hard_brakes_per_km': round(self.hard_brakes / km, 3),
            'hard_brakes_per_encounter': round(sum(e.hard_brakes for e in active) / n_enc, 3),
            'collisions': len(self.collisions),
            'peak_decel_mps2': round(self.peak_decel, 3),
            'peak_jerk_mps3': round(self.peak_jerk, 2),
            'peak_decel_near_hazards': round(max([e.max_decel for e in active], default=0.0), 3),
            'peak_jerk_near_hazards': round(max([e.max_jerk for e in active], default=0.0), 2),
            'capped_time_s': round(self.capped_s, 2),
            'unnecessary_slowdown_s': round(self.unnecessary_slow_s, 2),
            'retrieved_memories': len(self.retrieved),
            'retrieval_precision': round(tp / len(self.retrieved), 3) if self.retrieved else '',
            'retrieval_recall': round(sum(e.retrieved for e in active) / len(active), 3) if active else '',
            'mean_lead_distance_m': round(
                sum(e.lead_distance_m for e in active if e.retrieved) /
                max(1, sum(e.retrieved for e in active)), 1),
        }
        if extra:
            row.update(extra)
        hazard_rows = [dict(car=self.car, **e.row()) for e in self.enc]
        return row, hazard_rows
