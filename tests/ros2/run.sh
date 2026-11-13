#!/usr/bin/env bash
# Test Eddy against real ROS 2 (needs Docker). From the repo root:  ./tests/ros2/run.sh
# Starts a mock ROS 2 robot in a ROS container and lets Eddy discover and drive it.
set -euo pipefail
# The project folder, mounted read-only into the container at /eddy.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${EDDY_ROS_IMAGE:-ros:jazzy-ros-base}"

docker run --rm -v "$HERE":/eddy:ro -e PYTHONPATH=/eddy/src -e PYTHONUNBUFFERED=1 "$IMAGE" bash -c '
  source /opt/ros/$ROS_DISTRO/setup.bash
  # Start the mock robot in the background and give it a moment to come up.
  python3 /eddy/tests/ros2/mock_robot.py &
  sleep 3
  echo "--- eddy scan"; python3 -m eddy scan
  echo "--- end-to-end"; python3 /eddy/tests/ros2/check_ros.py
  status=$?
  echo "--- eddy do (CLI on the live graph)"
  python3 -m eddy do -r ros2 "turn right 45 degrees" "status" || status=1
  exit $status
'
