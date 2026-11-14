# One record of the shared fleet memory.
# All positions are in the ROS "map" frame (CARLA coordinates with y flipped), metres / radians.

string id                      # unique id (assigned by the memory if empty)
string hazard_type             # pedestrian_crossing | lane_closure | road_damage | stalled_vehicle | other
string description             # natural-language description written by the VLM
string lane                    # left | right | center | all | unknown
uint8 severity                 # 1 (minor) .. 5 (critical)
string advice                  # free-text advice, e.g. "approach below 20 km/h"

float64 x
float64 y
float64 z
float64 heading                # heading of the reporting vehicle (rad)
int32 road_id
int32 lane_id

builtin_interfaces/Time first_seen
builtin_interfaces/Time last_confirmed

string reporter_id             # vehicle that first reported the hazard
float32 confidence             # [0, 1]; on write, <= 0 means "use the memory default"
uint32 count                   # number of reports / confirmations merged into this record
uint32 refutations
string source                  # vlm | template | oracle | injected
string trigger                 # hard_brake | near_miss | collision | jerk | oracle | injected
