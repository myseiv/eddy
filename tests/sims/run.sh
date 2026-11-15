#!/usr/bin/env bash
# Run Eddy against public simulators and controllers (needs Docker; builds a 4 GB image the
# first time). From the repo root:  ./tests/sims/run.sh [panda|diff|two|car|steer|drone ...]
#   panda  Franka Panda arm on stock ros2_control controllers, positions checked against TF
#   diff   Gazebo's stock differential-drive vehicle, checked against the simulator's true pose
#   two    two of those vehicles on one ROS graph, driving only the one asked for
#   car    Gazebo's stock Ackermann (car-steered) vehicle, commanded by velocity
#   steer  the same vehicle commanded by speed and steering angle (ackermann_msgs)
#   drone  Gazebo's stock quadcopter, flown by velocity commands
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
docker image inspect eddy-sims >/dev/null 2>&1 || docker build -q -t eddy-sims "$HERE/tests/sims" >/dev/null
status=0
for what in "${@:-panda diff two car steer drone}"; do for one in $what; do
  case "$one" in
    panda) cmd='bash /eddy/tests/sims/panda_up.sh >/dev/null; python3 /eddy/tests/sims/check_panda.py' ;;
    diff)  cmd='bash /eddy/tests/sims/gz_up.sh diff_drive plain; python3 /eddy/tests/sims/check_gazebo.py diff_drive vehicle_blue ros2' ;;
    two)   cmd='bash /eddy/tests/sims/gz_up.sh diff_drive namespaced; python3 /eddy/tests/sims/check_gazebo.py diff_drive vehicle_green ros2:/model/vehicle_green 4.7 vehicle_blue' ;;
    car)   cmd='bash /eddy/tests/sims/gz_up.sh ackermann_steering plain; python3 /eddy/tests/sims/check_gazebo.py ackermann_steering vehicle_blue ros2' ;;
    steer) cmd='bash /eddy/tests/sims/gz_up.sh ackermann_steering steering; python3 /eddy/tests/sims/check_gazebo.py ackermann_steering vehicle_blue ros2' ;;
    drone) cmd='bash /eddy/tests/sims/gz_up.sh multicopter_velocity_control drone; python3 /eddy/tests/sims/check_drone.py' ;;
    *) echo "unknown test: $one"; exit 2 ;;
  esac
  echo "=================== $one ==================="
  docker run --rm -v "$HERE":/eddy:ro -e PYTHONUNBUFFERED=1 eddy-sims bash -c \
    "source /opt/ros/humble/setup.bash; export PYTHONPATH=/eddy/src:\$PYTHONPATH; $cmd" || status=1
done; done
exit $status
