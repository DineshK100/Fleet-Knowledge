# Compact per-vehicle state published by the vehicle interface (20 Hz).
std_msgs/Header header
string vehicle_id
float64 x
float64 y
float64 z
float64 heading                # rad, ROS convention
float64 speed                  # m/s
float64 accel_long             # m/s^2 (negative = braking)
int32 road_id
int32 lane_id
bool is_junction
float32 target_speed_kmh       # speed the driving agent is currently aiming for
float32 speed_cap_kmh          # active cap from the fleet memory (<= 0: none)
string maneuver                # cruise | capped | bypass | reroute | emergency_brake | recover
string context                 # short text context, e.g. "approaching intersection, right lane, clear day"
