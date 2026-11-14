# Nearest obstacle in the planned path of the vehicle (simulated perception).
std_msgs/Header header
bool detected
string obstacle_type           # walker | vehicle | static
uint32 actor_id
float64 distance               # free distance along the route (bumper to object), m
float64 closing_speed          # m/s, > 0 when approaching
float64 ttc                    # s; 1e9 when not closing
float64 x                      # obstacle position (map frame)
float64 y
