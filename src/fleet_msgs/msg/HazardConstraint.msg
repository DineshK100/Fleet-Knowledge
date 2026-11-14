# One retrieved memory turned into a driving constraint.
string memory_id
string hazard_type
string description
uint8 severity
float32 confidence
float32 score
float64 x
float64 y
float64 route_distance         # along the planned route from the ego, m (negative = just passed)
float32 speed_cap_kmh          # target speed when reaching the hazard
float32 brake_distance_m       # earliest braking distance (start slowing at least this far ahead)
string action                  # slow | lane_change | reroute
string preferred_lane          # left | right | '' (no preference)
