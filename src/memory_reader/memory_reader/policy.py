"""Policy adapters that turn retrieved memories into driving constraints.

* ``rule_adapter`` (main method): hazard type + severity -> target-speed cap and
  earliest-braking distance; lane preference for lane closures / stalled
  vehicles; reroute for high-severity, high-confidence blockages.
* ``LLMAdapter`` (stretch goal): an LLM picks one action per memory, with a
  short justification. Runs asynchronously and falls back to the rules.
"""

import json
import threading
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional

from fleet_memory.llm_client import ChatClient, extract_json

from .scoring import Candidate

# Target speed (km/h) at the hazard, indexed by severity 1..5.
DEFAULT_SPEED_CAPS_KMH: Dict[str, List[float]] = {
    'pedestrian_crossing': [30, 25, 20, 18, 15],
    'lane_closure': [35, 30, 30, 25, 20],
    'road_damage': [30, 25, 20, 15, 12],
    'stalled_vehicle': [35, 30, 25, 20, 20],
    'other': [40, 35, 30, 25, 20],
}
BLOCKAGE_TYPES = ('lane_closure', 'stalled_vehicle')
ACTIONS = ('slow', 'lane_change', 'reroute')


@dataclass
class PolicyConfig:
    a_comfort: float = 2.0               # m/s^2 used to compute the earliest braking distance
    brake_buffer_m: float = 8.0          # reach the cap this far before the hazard
    low_confidence: float = 0.4          # below this, relax the cap
    low_confidence_relax_kmh: float = 10.0
    reroute_min_severity: int = 5
    reroute_min_confidence: float = 0.8
    enable_lane_change: bool = True
    enable_reroute: bool = True
    speed_caps: Dict[str, List[float]] = field(
        default_factory=lambda: {k: list(v) for k, v in DEFAULT_SPEED_CAPS_KMH.items()})


@dataclass
class Action:
    memory_id: str
    hazard_type: str
    description: str
    severity: int
    confidence: float
    score: float
    x: float
    y: float
    route_distance: float
    speed_cap_kmh: float
    brake_distance_m: float
    action: str = 'slow'
    preferred_lane: str = ''


def brake_distance(v0_mps: float, cap_kmh: float, a: float, buffer_m: float) -> float:
    v1 = cap_kmh / 3.6
    if v0_mps <= v1:
        return buffer_m
    return (v0_mps ** 2 - v1 ** 2) / (2.0 * max(0.1, a)) + buffer_m


def lane_preference(memory_lane: str) -> str:
    """Prefer the lane opposite to the one reported as affected."""
    return {'right': 'left', 'left': 'right'}.get((memory_lane or '').lower(), '')


def speed_cap_for(hazard_type: str, severity: int, confidence: float, cfg: PolicyConfig) -> float:
    caps = cfg.speed_caps.get(hazard_type, cfg.speed_caps['other'])
    cap = float(caps[max(1, min(5, int(severity))) - 1])
    if confidence < cfg.low_confidence:
        cap += cfg.low_confidence_relax_kmh
    return cap


def rule_adapter(ranked: List[Candidate], ego_speed_mps: float, cfg: PolicyConfig) -> List[Action]:
    out = []
    for c in ranked:
        m = c.memory
        sev = int(m.severity) or 3
        conf = float(m.confidence)
        cap = speed_cap_for(m.hazard_type, sev, conf, cfg)
        action, lane = 'slow', ''
        if m.hazard_type in BLOCKAGE_TYPES:
            full_block = (m.lane or '').lower() == 'all'
            strong = sev >= cfg.reroute_min_severity and conf >= cfg.reroute_min_confidence
            if cfg.enable_reroute and (full_block or strong):
                action = 'reroute'
            elif cfg.enable_lane_change:
                action, lane = 'lane_change', lane_preference(m.lane)
        out.append(Action(
            memory_id=m.id, hazard_type=m.hazard_type, description=m.description, severity=sev,
            confidence=conf, score=c.score, x=m.x, y=m.y, route_distance=c.route_distance,
            speed_cap_kmh=cap,
            brake_distance_m=brake_distance(ego_speed_mps, cap, cfg.a_comfort, cfg.brake_buffer_m),
            action=action, preferred_lane=lane))
    return out


# --------------------------------------------------------------------------- LLM
LLM_SYSTEM_PROMPT = (
    'You are the tactical decision module of an autonomous car in a fleet. The fleet memory '
    'returned hazard reports written by other cars that drove the same road. For EACH memory '
    'choose one action: "ignore" (memory looks irrelevant or stale), "slow" (reduce speed before '
    'the hazard), "lane_change" (move to the other lane before the hazard) or "reroute" (avoid the '
    'road). Also choose the speed in km/h at which to pass the hazard. Be safe but do not be '
    'overly cautious: arrival time also matters. Reply with ONE JSON object: '
    '{"decisions": [{"memory_id": str, "action": str, "speed_cap_kmh": number, '
    '"preferred_lane": "left"|"right"|""}], "justification": "one short sentence"}'
)


def build_llm_prompt(state: dict, actions: List[Action]) -> str:
    lines = [f"Ego: speed {state.get('speed_kmh', 0):.0f} km/h, cruise {state.get('cruise_kmh', 0):.0f}"
             f" km/h, context: {state.get('context', '')}", 'Retrieved memories:']
    for a in actions:
        lines.append(json.dumps({
            'memory_id': a.memory_id, 'hazard_type': a.hazard_type, 'description': a.description,
            'severity': a.severity, 'confidence': round(a.confidence, 2),
            'distance_ahead_m': round(a.route_distance, 1),
            'rule_based_suggestion': {'action': a.action, 'speed_cap_kmh': a.speed_cap_kmh}}))
    return '\n'.join(lines)


def parse_llm_decisions(text: str, base: List[Action], ego_speed_mps: float,
                        cfg: PolicyConfig) -> Optional[tuple]:
    obj = extract_json(text)
    if not obj or not isinstance(obj.get('decisions'), list):
        return None
    by_id = {a.memory_id: a for a in base}
    out = []
    for d in obj['decisions']:
        if not isinstance(d, dict) or d.get('memory_id') not in by_id:
            continue
        a = by_id[d['memory_id']]
        act = str(d.get('action', 'slow')).lower()
        if act == 'ignore':
            continue
        if act not in ACTIONS:
            act = 'slow'
        try:
            cap = float(d.get('speed_cap_kmh', a.speed_cap_kmh))
        except (TypeError, ValueError):
            cap = a.speed_cap_kmh
        cap = max(5.0, min(60.0, cap))
        lane = str(d.get('preferred_lane', '') or '').lower()
        out.append(replace(a, action=act, speed_cap_kmh=cap,
                           preferred_lane=lane if lane in ('left', 'right') else '',
                           brake_distance_m=brake_distance(ego_speed_mps, cap, cfg.a_comfort,
                                                           cfg.brake_buffer_m)))
    return out, str(obj.get('justification', ''))[:300]


class LLMAdapter:
    """Asynchronous LLM adapter with a per-memory-set cache and rule fallback."""

    def __init__(self, client: ChatClient, cfg: PolicyConfig, logger=None):
        self.client = client
        self.cfg = cfg
        self.logger = logger
        self._cache: Dict[frozenset, tuple] = {}
        self._busy = False
        self._lock = threading.Lock()

    def decide(self, actions: List[Action], state: dict, ego_speed_mps: float):
        """Return (actions, justification, used_llm). Never blocks."""
        if not actions:
            return [], '', False
        key = frozenset(a.memory_id for a in actions)
        with self._lock:
            cached = self._cache.get(key)
            start = cached is None and not self._busy
            if start:
                self._busy = True
        if start:
            threading.Thread(target=self._run, args=(key, actions, state, ego_speed_mps),
                             daemon=True).start()
        if cached is None:
            return actions, 'rules (LLM pending)', False
        decisions, why = cached
        # refresh distances of the cached decisions with the current ones
        cur = {a.memory_id: a for a in actions}
        merged = [replace(d, route_distance=cur[d.memory_id].route_distance,
                          brake_distance_m=brake_distance(ego_speed_mps, d.speed_cap_kmh,
                                                          self.cfg.a_comfort, self.cfg.brake_buffer_m))
                  for d in decisions if d.memory_id in cur]
        return merged, why, True

    def _run(self, key, actions, state, ego_speed_mps):
        try:
            reply = self.client.chat(LLM_SYSTEM_PROMPT, build_llm_prompt(state, actions))
            parsed = parse_llm_decisions(reply, actions, ego_speed_mps, self.cfg)
            if parsed is None:
                raise ValueError(f'unparseable reply {reply[:200]!r}')
            with self._lock:
                self._cache[key] = parsed
            if self.logger:
                self.logger.info(f'LLM adapter: {[(a.memory_id, a.action, a.speed_cap_kmh) for a in parsed[0]]}'
                                 f' - {parsed[1]}')
        except Exception as exc:
            if self.logger:
                self.logger.warning(f'LLM adapter failed ({exc}); using rules for this set')
            with self._lock:
                self._cache[key] = (actions, 'rules (LLM failed)')
        finally:
            with self._lock:
                self._busy = False


def describe_actions(actions: List[Action]) -> str:
    return '; '.join(f'{a.hazard_type}@{a.route_distance:.0f}m->{a.action}<={a.speed_cap_kmh:.0f}'
                     for a in actions)

