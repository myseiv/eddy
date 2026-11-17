#!/usr/bin/env bash
# The drive-and-arm checks (battery.py) on robots that drive and have an arm, one at a time
# (needs Docker). From the repo root:  ./tests/manip/run.sh [mock] [landerpi] [gzpanda]
#   mock      the ROS 2 Jazzy mock mobile manipulator (tests/ros2/mock_robot.py)
#   landerpi  the LanderPi stand-in on ROS 2 Humble (tests/ros2/mock_hiwonder.py)
#   gzpanda   Gazebo's stock vehicle and a stock Franka Panda on one ROS 2 graph (tests/sims image)
# Do not run other Docker ROS tests at the same time: containers see each other's topics.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
status=0
for one in ${@:-mock landerpi gzpanda}; do
  echo "=================== $one ==================="
  case "$one" in
    mock) docker run --rm -v "$HERE":/eddy:ro -e PYTHONPATH=/eddy/src -e PYTHONUNBUFFERED=1 ros:jazzy-ros-base bash -c '
            source /opt/ros/$ROS_DISTRO/setup.bash
            python3 /eddy/tests/ros2/mock_robot.py & sleep 3
            python3 /eddy/tests/manip/battery.py mock' || status=1 ;;
    landerpi) docker run --rm -v "$HERE":/eddy:ro -e PYTHONPATH=/eddy/src -e PYTHONUNBUFFERED=1 ros:humble-ros-base bash -c '
            source /opt/ros/$ROS_DISTRO/setup.bash
            python3 -c "import control_msgs" 2>/dev/null || { apt-get update -qq && apt-get install -y -qq ros-$ROS_DISTRO-control-msgs; } >/dev/null 2>&1
            EDDY_MOCK_KIND=landerpi python3 /eddy/tests/ros2/mock_hiwonder.py & sleep 4
            python3 /eddy/tests/manip/battery.py landerpi' || status=1 ;;
    gzpanda) docker image inspect eddy-sims >/dev/null 2>&1 || docker build -q -t eddy-sims "$HERE/tests/sims" >/dev/null
            docker run --rm -v "$HERE":/eddy:ro -e PYTHONUNBUFFERED=1 eddy-sims bash -c "
            source /opt/ros/humble/setup.bash; export PYTHONPATH=/eddy/src:\$PYTHONPATH
            bash /eddy/tests/sims/gz_up.sh diff_drive plain; bash /eddy/tests/sims/panda_up.sh >/dev/null
            python3 /eddy/tests/manip/battery.py gzpanda" || status=1 ;;
    *) echo "unknown robot: $one"; exit 2 ;;
  esac
done
exit $status
