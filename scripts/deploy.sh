#!/usr/bin/env bash
# Copy this folder to another machine over SSH and install Eddy there.
#
#   ./scripts/deploy.sh pi@raspberrypi.local              core install
#   ./scripts/deploy.sh pi@192.168.1.42 --all             with serial, voice and vision
#   ./scripts/deploy.sh pi@robot --service "run -r ros2"  and start at boot
#
# Everything after the host is passed to install.sh on the other machine.
# Extra ssh options (port, key) go in EDDY_SSH_OPTS, e.g. EDDY_SSH_OPTS="-p 2222 -i ~/.ssh/robot".
set -euo pipefail

# The first argument is the machine; everything after it is kept for install.sh.
HOST="${1:?usage: deploy.sh user@host [install.sh options]}"
shift
# HERE is the project folder (the parent of scripts/), wherever the script is run from.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_DIR="${EDDY_REMOTE_DIR:-eddy_ai}"
# shellcheck disable=SC2206
SSH=(ssh ${EDDY_SSH_OPTS:-})

echo "==> Copying Eddy to $HOST:~/$REMOTE_DIR"
# tar over ssh: works even where rsync is not installed on the robot.
tar -C "$HERE" --exclude .venv --exclude .git --exclude __pycache__ --exclude '*.egg-info' \
    --exclude .pytest_cache --exclude models -czf - . \
  | "${SSH[@]}" "$HOST" "mkdir -p '$REMOTE_DIR' && tar -xzf - -C '$REMOTE_DIR'"

# Quote each remaining argument so it survives the trip through ssh intact.
ARGS=""
for a in "$@"; do ARGS="$ARGS $(printf '%q' "$a")"; done
echo "==> Installing on $HOST"
"${SSH[@]}" "$HOST" "cd '$REMOTE_DIR' && bash scripts/install.sh$ARGS"
