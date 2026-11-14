"""Hazard placement on a fake CARLA road (no simulator needed)."""
import math
from types import SimpleNamespace

from fleet_experiments.hazards import HazardSpec, plan_placements, route_with_s

carla = SimpleNamespace(LaneType=SimpleNamespace(Driving='Driving', Sidewalk='Sidewalk'))


class Loc:
    def __init__(self, x, y):
        self.x, self.y, self.z = x, y, 0.0

    def distance(self, o):
        return math.hypot(self.x - o.x, self.y - o.y)


class WP:
    def __init__(self, x, y, yaw, junction=False, multilane=False, lane_id=-1):
        self.transform = SimpleNamespace(location=Loc(x, y), rotation=SimpleNamespace(yaw=yaw))
        self.is_junction = junction
        self.lane_id = lane_id
        self.lane_type = 'Driving'
        self.lane_width = 3.5
        self.road_id = 1
        self._multi = multilane

    def get_left_lane(self):
        return WP(self.transform.location.x, self.transform.location.y, 0, lane_id=-2) if self._multi else None

    def get_right_lane(self):
        return None


def make_route():
    """0-200 m straight multi-lane, junction at 200-215, 215-400 single lane, then a curve."""
    route = []
    for i in range(0, 400, 2):
        x = float(i)
        route.append((WP(x, 0.0, 0.0, junction=200 <= x <= 215, multilane=x < 200), None))
    for k in range(1, 40):                                  # 90 deg curve, then straight
        a = math.radians(min(90, k * 4))
        route.append((WP(400 + 30 * math.sin(a), 30 - 30 * math.cos(a), math.degrees(a)), None))
    for j in range(1, 60):
        route.append((WP(430.0, 30.0 + 2 * j, 90.0, multilane=True), None))
    return route


def test_placements_respect_constraints():
    rs = route_with_s(make_route())
    specs = [HazardSpec('ped', 'pedestrian_crossing', after_m=60),
             HazardSpec('closure', 'lane_closure', after_m=40),
             HazardSpec('pothole', 'road_damage', after_m=230)]
    pl = plan_placements(carla, rs, specs, min_gap_m=50)
    assert pl is not None
    where = {spec.name: rs[i] for spec, i in pl}
    s_closure, wp_closure, _ = where['closure']
    assert 40 <= s_closure and wp_closure.get_left_lane() is not None and not wp_closure.is_junction
    s_ped, wp_ped, _ = where['ped']
    assert 185 <= s_ped < 200 and not wp_ped.is_junction        # just before the junction
    assert where['pothole'][0] >= 230


def test_impossible_placement_returns_none():
    rs = route_with_s(make_route()[:50])                      # 100 m, no junction
    assert plan_placements(carla, rs, [HazardSpec('ped', 'pedestrian_crossing', after_m=10)]) is None


def test_stalled_after_curve():
    rs = route_with_s(make_route())
    pl = plan_placements(carla, rs, [HazardSpec('stall', 'stalled_vehicle', after_m=300,
                                                params={'min_curve_deg': 30.0})])
    assert pl is not None
    s, wp, _ = rs[pl[0][1]]
    assert wp.transform.rotation.yaw == 90.0                  # on the straight after the curve
