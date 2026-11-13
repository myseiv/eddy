#!/usr/bin/env bash
# Two robots on two separate (container) machines find each other and take commands from a third.
#   alpha: Eddy on the simulator        bravo: Eddy on a ROS 2 robot (the mock)
# Needs Docker. From the repo root:  ./tests/peers/run.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${EDDY_ROS_IMAGE:-ros:jazzy-ros-base}"
# A private Docker network: each container on it is a separate machine with its own address.
NET=eddy-test-net
# The common part of every docker run: on that network, with the project mounted read-only.
RUN=(docker run --rm --network "$NET" -v "$HERE":/eddy:ro -e PYTHONPATH=/eddy/src -e PYTHONUNBUFFERED=1)
# Remove the containers and the network, both before starting and on exit.
cleanup() { docker rm -f eddy-alpha eddy-bravo >/dev/null 2>&1 || true; docker network rm "$NET" >/dev/null 2>&1 || true; }
trap cleanup EXIT
cleanup
docker network create "$NET" >/dev/null

# Machine 1: alpha, an Eddy on the simulator.
"${RUN[@]}" -d --name eddy-alpha "$IMAGE" python3 -m eddy run -r sim --peers --name alpha --key demo-key --obey >/dev/null
# Machine 2: bravo, an Eddy on the mock ROS 2 robot.
"${RUN[@]}" -d --name eddy-bravo "$IMAGE" bash -c '
  source /opt/ros/$ROS_DISTRO/setup.bash
  python3 /eddy/tests/ros2/mock_robot.py & sleep 3
  exec python3 -m eddy run -r ros2 --peers --name bravo --key demo-key --obey' >/dev/null
# Give both robots time to start, then run the controller on a third machine.
sleep 8
status=0
"${RUN[@]}" "$IMAGE" python3 /eddy/tests/peers/controller.py || status=$?
echo "--- alpha log"; docker logs eddy-alpha 2>&1 | tail -6
echo "--- bravo log"; docker logs eddy-bravo 2>&1 | tail -6
exit $status
