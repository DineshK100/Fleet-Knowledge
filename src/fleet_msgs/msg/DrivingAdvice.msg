# Output of the memory reader / policy adapter for one vehicle.
std_msgs/Header header
string vehicle_id
string adapter                 # rules | llm | none
fleet_msgs/HazardConstraint[] constraints
string justification
