cmake_minimum_required(VERSION 3.8)
project(fleet_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/HazardMemory.msg"
  "msg/MemoryEvent.msg"
  "msg/SurpriseEvent.msg"
  "msg/VehicleState.msg"
  "msg/ObstacleInfo.msg"
  "msg/CollisionEvent.msg"
  "msg/HazardConstraint.msg"
  "msg/DrivingAdvice.msg"
  "srv/WriteMemory.srv"
  "srv/QueryMemory.srv"
  "srv/MemoryFeedback.srv"
  DEPENDENCIES builtin_interfaces std_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
