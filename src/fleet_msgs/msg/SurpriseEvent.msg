# Event detected on board a vehicle by the surprise trigger.
std_msgs/Header header
string vehicle_id
string kind                    # hard_brake | near_miss | collision | jerk | bump | obstacle_seen
bool is_surprise               # true: strong event that triggers a memory write; false: weak evidence
float32 magnitude              # decel (m/s^2) | TTC (s) | jerk (m/s^3) | collision intensity
float64 x                      # ego position
float64 y
float64 heading
float64 speed                  # m/s
string obstacle_type           # walker | vehicle | static | none
float64 hazard_x               # best estimate of the hazard position
float64 hazard_y
