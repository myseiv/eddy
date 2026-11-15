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
  echo "--- faults: sensors going quiet, the robot dying, and how fast stop takes effect"
  python3 /eddy/tests/ros2/check_faults.py || status=1
  # Second pass: the same robot with its arm and gripper driven by actions instead of topics.
  # Needs the control_msgs package, which the base image lacks; skipped if it cannot be installed.
  if python3 -c "import control_msgs" 2>/dev/null || { apt-get update -qq && apt-get install -y -qq ros-$ROS_DISTRO-control-msgs; } >/dev/null 2>&1; then
    kill %1; wait %1 2>/dev/null
    EDDY_MOCK_ACTIONS=1 python3 /eddy/tests/ros2/mock_robot.py &
    sleep 3
    echo "--- end-to-end again, controllers as actions"; python3 /eddy/tests/ros2/check_ros.py || status=1
  else
    echo "--- SKIPPED the action-controller pass: control_msgs could not be installed"
  fi
  exit $status
'
