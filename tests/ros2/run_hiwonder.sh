#!/usr/bin/env bash
# Run Eddy against stand-ins for the two real robots it was tried on (needs Docker).
# From the repo root:  ./tests/ros2/run_hiwonder.sh
# Uses ROS 2 Humble, the version on those robots. See mock_hiwonder.py for what is reproduced.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${EDDY_ROS_IMAGE:-ros:humble-ros-base}"

docker run --rm -v "$HERE":/eddy:ro -e PYTHONPATH=/eddy/src -e PYTHONUNBUFFERED=1 "$IMAGE" bash -c '
  source /opt/ros/$ROS_DISTRO/setup.bash
  python3 -c "import control_msgs" 2>/dev/null || { apt-get update -qq && apt-get install -y -qq ros-$ROS_DISTRO-control-msgs; } >/dev/null 2>&1
  status=0
  for kind in landerpi car; do
    echo "=================== $kind ==================="
    EDDY_MOCK_KIND=$kind python3 /eddy/tests/ros2/mock_hiwonder.py &
    sleep 4
    EDDY_MOCK_KIND=$kind python3 /eddy/tests/ros2/battery_hiwonder.py || status=1
    kill %% 2>/dev/null; wait 2>/dev/null
    sleep 1
  done
  exit $status
'
