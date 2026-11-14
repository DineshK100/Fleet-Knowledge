from fleet_experiments.metrics import MetricsRecorder
from fleet_experiments.plot_results import mean_ci


def drive(m, decel_at=None, obstacle_ttc=None):
    t, x, v = 0.0, 0.0, 12.0
    for i in range(400):
        a = 0.0
        if decel_at is not None and decel_at <= x <= decel_at + 15:
            a = -6.0
        v = max(1.0, v + a * 0.05)
        x += v * 0.05
        t += 0.05
        obs = None
        if obstacle_ttc is not None and 80 <= x <= 95:
            obs = {'detected': True, 'x': 100.0, 'y': 0.0, 'distance': 100 - x,
                   'closing_speed': v, 'ttc': obstacle_ttc}
        m.update(t, x, 0.0, v, a, obs)
    return t


def test_learning_metrics():
    hz = [{'name': 'ped', 'type': 'pedestrian_crossing', 'x': 100.0, 'y': 0.0, 'active': True},
          {'name': 'off', 'type': 'lane_closure', 'x': 2000.0, 'y': 0.0, 'active': True}]
    m1 = MetricsRecorder(1, hz)
    t = drive(m1, decel_at=85, obstacle_ttc=1.0)
    m1.collision(t, 100.0, 0.0, 'walker')
    row, hrows = m1.finish(t, True, 'arrived')
    assert row['hard_brakes'] == 1 and row['near_misses'] == 1 and row['collisions'] == 1
    assert row['hazard_encounters'] == 1 and row['near_miss_rate'] == 1.0
    assert hrows[0]['near_miss'] == 1 and hrows[1]['encountered'] == 0

    m2 = MetricsRecorder(2, hz)
    m2.advice(0.0, 0.0, [{'memory_id': 'a', 'x': 101.0, 'y': 0.0},
                          {'memory_id': 'b', 'x': 500.0, 'y': 0.0}])
    t = drive(m2, obstacle_ttc=4.0)
    row2, h2 = m2.finish(t, True, 'arrived')
    assert row2['near_misses'] == 0 and row2['hard_brakes'] == 0
    assert row2['retrieval_precision'] == 0.5 and row2['retrieval_recall'] == 1.0
    assert h2[0]['retrieved'] == 1 and h2[0]['lead_distance_m'] > 90


def test_mean_ci():
    mu, ci, n = mean_ci([1.0, 2.0, 3.0])
    assert mu == 2.0 and n == 3 and ci > 0
    assert mean_ci([])[2] == 0
