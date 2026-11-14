import math

from fleet_memory.embeddings import HashingEmbedder
from fleet_memory.llm_client import extract_json
from fleet_memory.store import FleetMemoryStore, MemoryConfig, MemoryRecord, SpatialGrid


def rec(**kw):
    base = dict(hazard_type='pedestrian_crossing', description='pedestrian stepped out behind a parked van',
                x=0.0, y=0.0, heading=0.0, severity=4, reporter_id='ego_1')
    base.update(kw)
    return MemoryRecord(**base)


def test_insert_and_query():
    s = FleetMemoryStore()
    r, merged = s.write(rec(), now=0.0)
    assert not merged and r.id and r.confidence == s.cfg.initial_confidence
    hits = s.query(5.0, 0.0, 10.0, now=0.0, query_text='pedestrian near intersection')
    assert len(hits) == 1 and hits[0][0].id == r.id
    assert abs(hits[0][2] - 5.0) < 1e-6
    assert hits[0][1] > 0.0
    assert s.query(100.0, 0.0, 10.0, now=0.0) == []


def test_merge_same_hazard_increases_confidence():
    s = FleetMemoryStore()
    a, _ = s.write(rec(), now=0.0)
    b, merged = s.write(rec(x=5.0, description='a pedestrian came out from behind the van'), now=10.0)
    assert merged and b.id == a.id and b.count == 2
    assert b.confidence > s.cfg.initial_confidence
    assert 0.0 < b.x < 5.0
    assert len(s) == 1


def test_no_merge_for_other_type_direction_or_far():
    s = FleetMemoryStore()
    s.write(rec(), now=0.0)
    _, m1 = s.write(rec(hazard_type='road_damage', description='pothole'), now=0.0)
    _, m2 = s.write(rec(heading=math.pi), now=0.0)            # opposite direction
    _, m3 = s.write(rec(x=100.0), now=0.0)
    assert not (m1 or m2 or m3)
    assert len(s) == 4


def test_allow_merge_false_inserts():
    s = FleetMemoryStore()
    s.write(rec(), now=0.0)
    _, merged = s.write(rec(), now=0.0, allow_merge=False)
    assert not merged and len(s) == 2


def test_decay_and_prune():
    cfg = MemoryConfig(decay_lambda=0.01, prune_threshold=0.1)
    s = FleetMemoryStore(cfg)
    r, _ = s.write(rec(), now=0.0)
    hits = s.query(0, 0, 10, now=100.0)
    assert abs(hits[0][0].confidence - 0.6 * math.exp(-1.0)) < 1e-6
    pruned = s.prune(now=1000.0)
    assert [p.id for p in pruned] == [r.id] and len(s) == 0


def test_oracle_does_not_decay():
    s = FleetMemoryStore(MemoryConfig(decay_lambda=0.1))
    s.write(rec(source='oracle', confidence=1.0), now=0.0)
    assert s.query(0, 0, 10, now=1000.0)[0][0].confidence == 1.0


def test_feedback_confirm_refute_prune():
    s = FleetMemoryStore(MemoryConfig(decay_lambda=0.0))
    r, _ = s.write(rec(), now=0.0)
    r2, pruned = s.feedback(r.id, True, now=1.0)
    assert not pruned and r2.confidence == 0.6 + 0.5 * 0.4 and r2.count == 2
    s.feedback(r.id, False, now=2.0)
    s.feedback(r.id, False, now=3.0)
    _, pruned = s.feedback(r.id, False, now=4.0)            # 0.8 -> .4 -> .2 -> .1 -> .05
    _, pruned = s.feedback(r.id, False, now=5.0)
    assert pruned and len(s) == 0
    assert s.feedback('missing', True, now=0.0) == (None, False)


def test_min_confidence_and_max_results():
    s = FleetMemoryStore()
    for i in range(5):
        s.write(rec(x=float(i * 30), confidence=0.1 + 0.2 * i), now=0.0, allow_merge=False)
    assert len(s.query(0, 0, 0, now=0.0, min_confidence=0.4)) == 3
    assert len(s.query(0, 0, 0, now=0.0, max_results=2)) == 2


def test_save_load(tmp_path):
    s = FleetMemoryStore()
    s.write(rec(), now=0.0)
    p = tmp_path / 'mem.json'
    s.save(str(p))
    s2 = FleetMemoryStore()
    assert s2.load(str(p), now=50.0) == 1
    assert s2.query(0, 0, 5, now=50.0)[0][0].first_seen == 50.0


def test_spatial_grid():
    g = SpatialGrid(10.0)
    g.insert('a', 0, 0)
    g.insert('b', 25, 0)
    g.insert('a', 50, 50)                                    # move
    assert g.query(0, 0, 30) == [('b', 25.0)]
    g.remove('b')
    assert g.query(0, 0, 30) == []


def test_hashing_embedder_similarity():
    e = HashingEmbedder()
    a = e.embed('pedestrian stepped out behind parked van')
    b = e.embed('a person came out behind the van')
    c = e.embed('traffic cones closing the right lane')
    assert abs(float(a @ a) - 1.0) < 1e-5
    assert float(a @ b) > float(a @ c)


def test_extract_json():
    assert extract_json('```json\n{"a": 1}\n```') == {'a': 1}
    assert extract_json('Sure! {"a": {"b": "}"}} trailing') == {'a': {'b': '}'}}
    assert extract_json('no json') is None
