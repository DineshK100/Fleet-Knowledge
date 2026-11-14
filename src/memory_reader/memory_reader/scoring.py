"""Retrieval scoring of fleet memories (Section 3.6 of the proposal).

    s = w_d f(distance) + w_h cos(dheading) + w_c c + w_r recency + w_s sim(q, text)

* ``f(d) = max(0, 1 - d / R)`` with ``d`` measured along the planned route,
* ``cos(dheading)`` between the ego heading and the reporter heading,
* ``c`` the memory confidence,
* ``recency = exp(-dt / tau)`` with ``dt`` the time since the last confirmation,
* ``sim`` the cosine similarity between the context query and the description.

Weights are non-negative and normalised to sum to 1.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ScoringConfig:
    lookahead_m: float = 150.0
    corridor_m: float = 6.0
    behind_margin_m: float = 15.0      # keep acting on a hazard until this far past it
    w_distance: float = 0.2
    w_heading: float = 0.2
    w_confidence: float = 0.2
    w_recency: float = 0.2
    w_similarity: float = 0.2
    use_heading: bool = True
    tau_s: float = 1800.0
    s_min: float = 0.45
    top_k: int = 3

    def weights(self) -> Dict[str, float]:
        w = {'d': self.w_distance, 'h': self.w_heading if self.use_heading else 0.0,
             'c': self.w_confidence, 'r': self.w_recency, 's': self.w_similarity}
        w = {k: max(0.0, v) for k, v in w.items()}
        total = sum(w.values())
        if total <= 0:
            return {k: 0.2 for k in w}
        return {k: v / total for k, v in w.items()}


@dataclass
class Candidate:
    memory: Any                     # object with id, hazard_type, confidence, heading, ...
    route_distance: float
    heading_cos: float
    age_s: float
    similarity: float
    score: float = 0.0
    terms: Dict[str, float] = field(default_factory=dict)


def distance_term(d: float, R: float) -> float:
    if d < 0:
        return 1.0                  # we are passing the hazard right now
    return max(0.0, 1.0 - d / R) if R > 0 else 0.0


def score_candidate(c: Candidate, cfg: ScoringConfig) -> float:
    w = cfg.weights()
    terms = {
        'd': distance_term(c.route_distance, cfg.lookahead_m),
        'h': c.heading_cos,
        'c': float(c.memory.confidence),
        'r': math.exp(-max(0.0, c.age_s) / cfg.tau_s) if cfg.tau_s > 0 else 1.0,
        's': max(0.0, min(1.0, c.similarity)),
    }
    c.terms = terms
    c.score = sum(w[k] * terms[k] for k in w)
    return c.score


def rank_candidates(cands: List[Candidate], cfg: ScoringConfig) -> List[Candidate]:
    """Score, filter (look-ahead window, s_min) and return the top-k candidates."""
    kept = []
    for c in cands:
        if c.route_distance > cfg.lookahead_m or c.route_distance < -cfg.behind_margin_m:
            continue
        score_candidate(c, cfg)
        if c.score >= cfg.s_min:
            kept.append(c)
    kept.sort(key=lambda c: -c.score)
    return kept[:cfg.top_k] if cfg.top_k > 0 else kept
