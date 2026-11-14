import math
from types import SimpleNamespace

from memory_reader.feedback import PassTracker
from memory_reader.geometry import RoutePolyline, route_distance
from memory_reader.policy import (PolicyConfig, brake_distance, lane_preference, parse_llm_decisions,
                                  rule_adapter)
from memory_reader.scoring import Candidate, ScoringConfig, distance_term, rank_candidates


def mem(**kw):
    base = dict(id='m1', hazard_type='pedestrian_crossing', description='ped', severity=4,
                confidence=0.8, lane='right', x=50.0, y=0.0, heading=0.0)
    base.update(kw)
    return SimpleNamespace(**base)


def test_route_projection_and_distance():
    r = RoutePolyline([(0, 0), (100, 0), (100, 100)])
    assert r.length == 200
    s, lat, _ = r.project(50, 3)
    assert abs(s - 50) < 1e-9 and abs(lat - 3) < 1e-9
    assert abs(route_distance(r, (10, 0), (100, 50)) - 140) < 1e-9
    assert route_distance(r, (10, 0), (50, 20), corridor_m=6) is None
    assert route_distance(r, (10, 0), (5, 0)) < 0
    assert r.max_curvature(80, 120) > 0.0
    assert abs(r.heading_at(150) - math.pi / 2) < 1e-9


def test_scoring_terms_and_ranking():
    cfg = ScoringConfig()
    assert distance_term(0, 150) == 1 and distance_term(150, 150) == 0 and distance_term(-5, 150) == 1
    near = Candidate(mem(id='a'), 20.0, 1.0, 10.0, 0.5)
    far = Candidate(mem(id='b'), 140.0, 1.0, 10.0, 0.5)
    oncoming = Candidate(mem(id='c'), 20.0, -1.0, 10.0, 0.5)
    out_of_range = Candidate(mem(id='d'), 400.0, 1.0, 10.0, 0.5)
    ranked = rank_candidates([far, oncoming, near, out_of_range], cfg)
    ids = [c.memory.id for c in ranked]
    assert ids[0] == 'a' and 'd' not in ids and 'c' not in ids
    assert abs(sum(cfg.weights().values()) - 1.0) < 1e-9
    w = ScoringConfig(use_heading=False).weights()
    assert w['h'] == 0 and abs(sum(w.values()) - 1) < 1e-9


def test_rule_adapter():
    cfg = PolicyConfig()
    ped = Candidate(mem(), 60.0, 1.0, 0.0, 0.3, score=0.8)
    closure = Candidate(mem(id='m2', hazard_type='lane_closure', severity=3, lane='right'), 80.0,
                        1.0, 0.0, 0.3, score=0.7)
    blocked = Candidate(mem(id='m3', hazard_type='lane_closure', severity=5, confidence=0.9,
                            lane='right'), 80.0, 1.0, 0.0, 0.3, score=0.7)
    acts = {a.memory_id: a for a in rule_adapter([ped, closure, blocked], 12.5, cfg)}
    assert acts['m1'].action == 'slow' and acts['m1'].speed_cap_kmh == 18
    assert acts['m2'].action == 'lane_change' and acts['m2'].preferred_lane == 'left'
    assert acts['m3'].action == 'reroute'
    assert acts['m1'].brake_distance_m > cfg.brake_buffer_m
    low = Candidate(mem(confidence=0.3), 60.0, 1.0, 0.0, 0.3)
    assert rule_adapter([low], 10, cfg)[0].speed_cap_kmh == 18 + cfg.low_confidence_relax_kmh
    assert lane_preference('left') == 'right' and lane_preference('unknown') == ''
    assert brake_distance(5.0, 30.0, 2.0, 8.0) == 8.0


def test_llm_decision_parsing():
    cfg = PolicyConfig()
    base = rule_adapter([Candidate(mem(), 60.0, 1.0, 0.0, 0.3, score=0.8),
                         Candidate(mem(id='m2'), 90.0, 1.0, 0.0, 0.3, score=0.6)], 12.0, cfg)
    acts, why = parse_llm_decisions(
        '{"decisions": [{"memory_id": "m1", "action": "slow", "speed_cap_kmh": 22},'
        ' {"memory_id": "m2", "action": "ignore"}, {"memory_id": "zz", "action": "slow"}],'
        ' "justification": "ok"}', base, 12.0, cfg)
    assert [a.memory_id for a in acts] == ['m1'] and acts[0].speed_cap_kmh == 22 and why == 'ok'
    assert parse_llm_decisions('nope', base, 12.0, cfg) is None


def test_pass_tracker():
    tr = PassTracker(pass_radius_m=10.0, evidence_radius_m=20.0)
    tr.observe_memory('a', 50.0, 0.0, t=0.0)
    tr.observe_memory('b', 100.0, 0.0, t=0.0)
    tr.observe_memory('c', 150.0, 0.0, t=0.0)
    tr.add_evidence(5.0, 98.0, 1.0, False, 'obstacle_seen')
    tr.add_evidence(9.0, 152.0, 0.0, True, 'hard_brake')
    decisions = {}
    for i in range(0, 200, 2):
        for mid, dec in tr.update(float(i), 0.0, 0.0, t=i / 10.0):
            decisions[mid] = dec
    assert decisions == {'a': 'refute', 'b': 'confirm', 'c': 'skip'}
