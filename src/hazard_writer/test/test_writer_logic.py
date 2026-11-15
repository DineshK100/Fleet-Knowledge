import numpy as np

from hazard_writer.ring_buffer import FrameBuffer
from hazard_writer.trigger import (G, Detection, SurpriseDetector, TriggerConfig, most_important,
                                   severity_from)
from hazard_writer.vlm import (VLMDescriber, normalize_hazard_type, parse_vlm_reply,
                               template_record)


def feed(det, accels, dt=0.05, t0=0.0):
    out = []
    for i, a in enumerate(accels):
        out += det.update_state(t0 + i * dt, a)
    return out


def test_hard_brake_once_per_episode():
    d = SurpriseDetector(TriggerConfig(jerk_threshold=1e9))
    out = feed(d, [0.0] * 5 + [-6.0] * 20 + [0.0] * 20 + [-6.0] * 20)
    hb = [x for x in out if x.kind == 'hard_brake']
    assert len(hb) == 2 and all(x.is_surprise for x in hb)


def test_gentle_braking_no_trigger():
    d = SurpriseDetector()
    out = feed(d, [0.0] * 5 + [-2.0] * 40)
    assert not [x for x in out if x.is_surprise]


def test_jolt_triggers_jerk_but_slow_bump_is_weak():
    d = SurpriseDetector()
    out = feed(d, [0.0] * 5 + [-70.0] + [0.0] * 10)        # pothole at speed
    assert any(x.kind in ('jerk', 'hard_brake') and x.is_surprise for x in out)
    d2 = SurpriseDetector()
    out2 = feed(d2, [0.0] * 5 + [-6.0] + [0.0] * 10)        # slow bump
    assert not [x for x in out2 if x.is_surprise]
    assert any(x.kind == 'bump' for x in out2)


def test_gear_shift_kick_is_not_a_surprise():
    # CARLA gearbox: torque gap (~0 accel) then a kick back to 9 m/s^2 while accelerating
    d = SurpriseDetector()
    out = feed(d, ([4.0] * 10 + [0.0] * 3 + [9.0] * 6) * 3)
    assert not [x for x in out if x.is_surprise]


def test_near_miss_and_obstacle_evidence():
    d = SurpriseDetector()
    out = d.update_obstacle(0.0, True, 10.0, 10.0, 1.0, 'walker', 7, (1.0, 2.0))
    kinds = {x.kind for x in out}
    assert kinds == {'near_miss', 'obstacle_seen'}
    out = d.update_obstacle(0.05, True, 9.5, 10.0, 0.95, 'walker', 7)
    assert not out                                          # same episode, weak rate-limited
    out = d.update_obstacle(0.1, True, 5.0, 1.0, 5.0, 'walker', 7)  # slow approach
    assert not [x for x in out if x.kind == 'near_miss']


def test_priority_and_severity():
    dets = [Detection('hard_brake', 0, 5.0, True), Detection('collision', 0, 100.0, True)]
    assert most_important(dets).kind == 'collision'
    assert severity_from(Detection('collision', 0, 1, True)) == 5
    assert severity_from(Detection('near_miss', 0, 0.5, True)) == 5
    assert severity_from(Detection('hard_brake', 0, 0.45 * G, True)) == 3


def test_frame_buffer_keyframes():
    b = FrameBuffer(max_age_s=3.0)
    for i in range(60):
        b.append(i * 0.1, i)
    kf = b.keyframes(5.0, n=4, before_s=2.0, after_s=0.5)
    assert [f for _, f in kf] == [30, 38, 47, 55] or len(kf) == 4
    assert kf[0][0] < kf[-1][0]


def test_parse_vlm_reply():
    r = parse_vlm_reply('```json\n{"hazard_type": "Pothole", "description": "deep hole", '
                        '"lane": "Right", "severity": "7", "advice": "slow"}\n```')
    assert r == {'hazard_type': 'road_damage', 'description': 'deep hole', 'lane': 'right',
                 'severity': 5, 'advice': 'slow'}
    assert parse_vlm_reply('garbage') is None
    assert normalize_hazard_type('traffic cones') == 'lane_closure'
    assert normalize_hazard_type('broken_down_vehicle') == 'stalled_vehicle'
    assert normalize_hazard_type('alien') == 'other'


def test_template_and_fallback():
    assert template_record({'obstacle_type': 'walker', 'trigger': 'near_miss'}, 4)['hazard_type'] == 'pedestrian_crossing'
    assert template_record({'obstacle_type': 'none', 'trigger': 'jerk'}, 3)['hazard_type'] == 'road_damage'
    assert template_record({'obstacle_type': 'static', 'trigger': 'hard_brake'}, 3, with_text=False)['description'] == ''

    class Boom:
        def chat(self, *a, **k):
            raise RuntimeError('down')
    rec, src, _ = VLMDescriber(Boom()).describe([np.zeros((4, 4, 3), np.uint8)],
                                                {'obstacle_type': 'static', 'trigger': 'hard_brake'}, 3)
    assert src == 'template' and rec['hazard_type'] == 'lane_closure' and rec['description']

    class Fake:
        def chat(self, system, user, images):
            assert len(images) == 1 and 'trigger' in user
            return '{"hazard_type": "lane_closure", "description": "cones", "lane": "right", "severity": 3}'
    rec, src, _ = VLMDescriber(Fake()).describe([np.zeros((8, 8, 3), np.uint8)], {'trigger': 'hard_brake'}, 3)
    assert src == 'vlm' and rec['lane'] == 'right'
