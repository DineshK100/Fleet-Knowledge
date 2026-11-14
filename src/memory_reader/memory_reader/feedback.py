"""Confirm / refute logic for retrieved memories (no ROS dependency).

A memory retrieved by a car is evaluated once the car has passed it:

* a strong surprise event near the memory  -> ``skip`` (the hazard writer will
  write a new report which the memory merges, i.e. confirms, by itself);
* only weak evidence (obstacle seen, small bump) -> ``confirm``;
* nothing observed while passing             -> ``refute`` (clean pass).
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class _Tracked:
    x: float
    y: float
    first_t: float
    min_dist: float = float('inf')
    done: bool = False


@dataclass
class PassTracker:
    pass_radius_m: float = 15.0
    evidence_radius_m: float = 25.0
    evidence_max_age_s: float = 120.0
    tracked: Dict[str, _Tracked] = field(default_factory=dict)
    evidence: List[Tuple[float, float, float, bool, str]] = field(default_factory=list)

    def reset(self):
        self.tracked.clear()
        self.evidence.clear()

    def observe_memory(self, mem_id: str, x: float, y: float, t: float):
        tr = self.tracked.get(mem_id)
        if tr is None:
            self.tracked[mem_id] = _Tracked(x, y, t)
        elif not tr.done:
            tr.x, tr.y = x, y

    def add_evidence(self, t: float, hx: float, hy: float, is_surprise: bool, kind: str):
        self.evidence.append((t, hx, hy, is_surprise, kind))
        self.evidence = [e for e in self.evidence if t - e[0] <= self.evidence_max_age_s]

    def update(self, ego_x: float, ego_y: float, ego_heading: float, t: float
               ) -> List[Tuple[str, str]]:
        """Returns a list of (memory_id, 'confirm' | 'refute' | 'skip') for memories just passed."""
        out = []
        ch, sh = math.cos(ego_heading), math.sin(ego_heading)
        for mid, tr in self.tracked.items():
            if tr.done:
                continue
            dx, dy = tr.x - ego_x, tr.y - ego_y
            d = math.hypot(dx, dy)
            tr.min_dist = min(tr.min_dist, d)
            behind = dx * ch + dy * sh < 0
            if tr.min_dist <= self.pass_radius_m and d > self.pass_radius_m and behind:
                tr.done = True
                out.append((mid, self._decide(tr, t)))
        return out

    def _decide(self, tr: _Tracked, t: float) -> str:
        strong = weak = False
        for (te, hx, hy, is_surprise, _kind) in self.evidence:
            if te < tr.first_t - 1.0 or te > t:
                continue
            if math.hypot(hx - tr.x, hy - tr.y) > self.evidence_radius_m:
                continue
            if is_surprise:
                strong = True
            else:
                weak = True
        if strong:
            return 'skip'
        return 'confirm' if weak else 'refute'
