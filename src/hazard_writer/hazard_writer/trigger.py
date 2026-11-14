"""Surprise trigger (Section 3.4 of the proposal), independent of ROS.

Strong ("surprise") events, which make the writer create a memory:
  * hard_brake  - smoothed longitudinal deceleration above ``hard_brake_g`` * g
  * near_miss   - time-to-collision below ``ttc_threshold_s`` while closing fast
  * collision   - any collision reported by the collision sensor
  * jerk        - jerk spike while the acceleration is also large (e.g. a pothole hit at speed)

Weak evidence, used by the memory reader to confirm memories (never written):
  * bump          - small jerk spike (e.g. a pothole driven over slowly)
  * obstacle_seen - an obstacle detected in the path within ``obstacle_evidence_range_m``
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

G = 9.81

# Higher number = more important when several triggers fire in one event window.
KIND_PRIORITY = {'collision': 4, 'near_miss': 3, 'hard_brake': 2, 'jerk': 1}


@dataclass
class TriggerConfig:
    hard_brake_g: float = 0.4
    hard_brake_release_g: float = 0.3
    ttc_threshold_s: float = 1.5
    near_miss_min_closing_mps: float = 2.0
    near_miss_max_distance_m: float = 30.0
    jerk_threshold: float = 25.0            # m/s^3
    jerk_min_accel_g: float = 0.25           # jerk only counts with large accel too
    bump_jerk_threshold: float = 8.0         # m/s^3, weak evidence
    accel_smoothing: float = 0.35            # EMA factor for accel (per 20 Hz sample)
    obstacle_evidence_range_m: float = 25.0
    weak_event_period_s: float = 2.0


@dataclass
class Detection:
    kind: str
    t: float
    magnitude: float
    is_surprise: bool
    obstacle_type: str = 'none'
    obstacle_xy: Optional[tuple] = None


class SurpriseDetector:
    def __init__(self, cfg: Optional[TriggerConfig] = None):
        self.cfg = cfg or TriggerConfig()
        self.reset()

    def reset(self):
        self._acc_s: Optional[float] = None
        self._last_t: Optional[float] = None
        self._braking = False
        self._near_miss_active = False
        self._jerk_active = False
        self._last_weak: Dict[str, float] = {}
        self.last_accel = 0.0
        self.last_jerk = 0.0

    # ----------------------------------------------------------------------
    def _weak_ok(self, key: str, t: float) -> bool:
        last = self._last_weak.get(key)
        if last is not None and t - last < self.cfg.weak_event_period_s:
            return False
        self._last_weak[key] = t
        return True

    def update_state(self, t: float, accel_long: float) -> List[Detection]:
        """Feed one longitudinal-acceleration sample (m/s^2, negative = braking)."""
        out: List[Detection] = []
        c = self.cfg
        if self._acc_s is None or self._last_t is None or t <= self._last_t:
            self._acc_s, self._last_t = accel_long, t
            return out
        dt = t - self._last_t
        prev = self._acc_s
        a = prev + c.accel_smoothing * (accel_long - prev)
        jerk = (a - prev) / dt
        self._acc_s, self._last_t = a, t
        self.last_accel, self.last_jerk = a, jerk

        # hard brake with hysteresis -> one detection per braking episode
        if not self._braking and a < -c.hard_brake_g * G:
            self._braking = True
            out.append(Detection('hard_brake', t, -a, True))
        elif self._braking and a > -c.hard_brake_release_g * G:
            self._braking = False

        big_jerk = abs(jerk) > c.jerk_threshold and abs(a) > c.jerk_min_accel_g * G
        if big_jerk and not self._jerk_active:
            self._jerk_active = True
            out.append(Detection('jerk', t, abs(jerk), True))
        elif not big_jerk and abs(jerk) < c.jerk_threshold * 0.5:
            self._jerk_active = False
        if not big_jerk and abs(jerk) > c.bump_jerk_threshold and self._weak_ok('bump', t):
            out.append(Detection('bump', t, abs(jerk), False))
        return out

    def update_obstacle(self, t: float, detected: bool, distance: float, closing_speed: float,
                        ttc: float, obstacle_type: str = 'static', actor_id: int = 0,
                        xy: Optional[tuple] = None) -> List[Detection]:
        out: List[Detection] = []
        c = self.cfg
        if not detected:
            self._near_miss_active = False
            return out
        near = (ttc < c.ttc_threshold_s and closing_speed > c.near_miss_min_closing_mps
                and distance < c.near_miss_max_distance_m)
        if near and not self._near_miss_active:
            self._near_miss_active = True
            out.append(Detection('near_miss', t, ttc, True, obstacle_type, xy))
        elif not near:
            self._near_miss_active = False
        if distance < c.obstacle_evidence_range_m and self._weak_ok(f'obs{actor_id}', t):
            out.append(Detection('obstacle_seen', t, distance, False, obstacle_type, xy))
        return out

    def collision(self, t: float, other_type: str, intensity: float) -> Detection:
        return Detection('collision', t, intensity, True, other_type)


def most_important(dets: List[Detection]) -> Detection:
    return max(dets, key=lambda d: (KIND_PRIORITY.get(d.kind, 0), d.magnitude
                                    if d.kind != 'near_miss' else -d.magnitude))


def severity_from(det: Detection) -> int:
    """Heuristic severity (1..5) from the triggering detection."""
    if det.kind == 'collision':
        return 5
    if det.kind == 'near_miss':
        return 5 if det.magnitude < 0.75 else 4
    if det.kind == 'hard_brake':
        return 4 if det.magnitude > 0.6 * G else 3
    if det.kind == 'jerk':
        return 4 if det.magnitude > 60 else 3
    return 2
