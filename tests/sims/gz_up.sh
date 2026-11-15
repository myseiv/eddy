#!/usr/bin/env bash
# Inside the eddy-sims image: start Gazebo (no screen) on a test world and bridge its topics
# to ROS 2.  Usage: gz_up.sh diff_drive|ackermann_steering plain|namespaced|steering
source /opt/ros/humble/setup.bash
python3 /eddy/tests/sims/make_world.py "$1" > /tmp/world.sdf
ign gazebo -s -r --headless-rendering /tmp/world.sdf >/tmp/gz.log 2>&1 &
blue=/model/vehicle_blue
T="geometry_msgs/msg/Twist]ignition.msgs.Twist"; O="nav_msgs/msg/Odometry[ignition.msgs.Odometry"; S="sensor_msgs/msg/LaserScan[ignition.msgs.LaserScan"
case "$2" in
  plain)        # one vehicle on /cmd_vel, /odom and /scan
    ros2 run ros_gz_bridge parameter_bridge "$blue/cmd_vel@$T" "$blue/odometry@$O" "$blue/scan@$S" --ros-args \
      -r "$blue/cmd_vel:=/cmd_vel" -r "$blue/odometry:=/odom" -r "$blue/scan:=/scan" >/tmp/bridge.log 2>&1 & ;;
  namespaced)   # every vehicle under /model/<name>/..., as Gazebo names them
    bridges=()
    for model in $(grep -o "model name='vehicle_[a-z]*'" /tmp/world.sdf | cut -d"'" -f2); do
      bridges+=("/model/$model/cmd_vel@$T" "/model/$model/odometry@$O" "/model/$model/scan@$S")
    done
    ros2 run ros_gz_bridge parameter_bridge "${bridges[@]}" >/tmp/bridge.log 2>&1 & ;;
  steering)     # one vehicle that takes speed and steering angle on /drive (see ackermann_shim.py)
    ros2 run ros_gz_bridge parameter_bridge "$blue/odometry@$O" "$blue/scan@$S" --ros-args \
      -r "$blue/odometry:=/odom" -r "$blue/scan:=/scan" >/tmp/bridge.log 2>&1 &
    ROS_DOMAIN_ID=1 ros2 run ros_gz_bridge parameter_bridge "$blue/cmd_vel@$T" >/tmp/bridge1.log 2>&1 &
    python3 /eddy/tests/sims/ackermann_shim.py >/tmp/shim.log 2>&1 & ;;
esac
# Wait until the laser is publishing, which means rendering has started.
for i in $(seq 60); do
  timeout 3 ign topic -e -t /model/vehicle_blue/scan -n 1 >/dev/null 2>&1 && break
  sleep 1
done
echo "gazebo up ($1, $2): $(ros2 topic list | grep -c -E 'cmd_vel|odom|scan|drive') robot topics"
