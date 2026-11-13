#!/usr/bin/env bash
# Install Eddy AI on this machine (Raspberry Pi OS, Ubuntu, Debian, macOS).
#
#   ./scripts/install.sh                 core only: simulator, ROS 2, network robots, CLI
#   ./scripts/install.sh --serial        + microcontroller robots over USB serial
#   ./scripts/install.sh --voice         + listening (downloads a ~40 MB offline speech model)
#   ./scripts/install.sh --vision        + camera and hand signs
#   ./scripts/install.sh --all           everything
#   ./scripts/install.sh --service "run -r ros2 --speak"
#                                        also start Eddy at boot with those arguments
#
# Installs into ~/.eddy (override with EDDY_HOME) and links `eddy` into ~/.local/bin.
# Safe to run again: it upgrades in place.
set -euo pipefail

# HERE is the project folder (the parent of scripts/), wherever the script is run from.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EDDY_HOME="${EDDY_HOME:-$HOME/.eddy}"
EXTRAS=()          # optional parts to install: serial, voice, vision
SERVICE_ARGS=""
VOICE=0
VISION=0

# Read the command-line options.
while [ $# -gt 0 ]; do
  case "$1" in
    --serial) EXTRAS+=(serial) ;;
    --voice)  EXTRAS+=(voice); VOICE=1 ;;
    --vision) EXTRAS+=(vision); VISION=1 ;;
    --all)    EXTRAS+=(serial voice vision); VOICE=1; VISION=1 ;;
    --service) shift; SERVICE_ARGS="${1:?--service needs the arguments for 'eddy', e.g. \"run -r ros2\"}" ;;
    -h|--help) sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

# Print a step heading.
say() { printf '\n==> %s\n' "$*"; }

# Python 3.9 or newer is required.
PY="${PYTHON:-python3}"
command -v "$PY" >/dev/null || { echo "python3 is not installed (sudo apt install python3 python3-venv)" >&2; exit 1; }
"$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' \
  || { echo "Eddy needs Python 3.9 or newer; this is $("$PY" -V)" >&2; exit 1; }

# System packages (only on apt systems, and only if we can get root without a prompt).
if command -v apt-get >/dev/null; then
  NEED=()          # packages that are missing
  "$PY" -c 'import ensurepip' 2>/dev/null || NEED+=(python3-venv)
  "$PY" -c 'import numpy' 2>/dev/null || NEED+=(python3-numpy)
  if [ "$VOICE" = 1 ]; then
    command -v espeak-ng >/dev/null || NEED+=(espeak-ng)
    command -v unzip >/dev/null || NEED+=(unzip)
    command -v curl >/dev/null || NEED+=(curl ca-certificates)
    dpkg -s libportaudio2 >/dev/null 2>&1 || NEED+=(libportaudio2)
    dpkg -s libatomic1 >/dev/null 2>&1 || NEED+=(libatomic1)        # the speech engine needs it on ARM
  fi
  if [ "$VISION" = 1 ]; then                                        # OpenCV (pulled in by MediaPipe) needs these
    command -v curl >/dev/null || NEED+=(curl ca-certificates)
    dpkg -s libgl1 >/dev/null 2>&1 || NEED+=(libgl1)
    dpkg -s libegl1 >/dev/null 2>&1 || NEED+=(libegl1)
    dpkg -s libgles2 >/dev/null 2>&1 || NEED+=(libgles2)
    dpkg -s libglib2.0-0 >/dev/null 2>&1 || dpkg -s libglib2.0-0t64 >/dev/null 2>&1 || NEED+=(libglib2.0-0)
  fi
  if [ ${#NEED[@]} -gt 0 ]; then
    # Use sudo unless already root; -n means fail instead of stopping to ask for a password.
    SUDO=""
    [ "$(id -u)" = 0 ] || SUDO="sudo -n"
    say "Installing system packages: ${NEED[*]}"
    if ! $SUDO apt-get install -y --no-install-recommends "${NEED[@]}"; then
      echo "Could not install them automatically. Run:  sudo apt install ${NEED[*]}   then run this script again." >&2
      exit 1
    fi
  fi
fi

say "Creating Eddy's Python environment in $EDDY_HOME/venv"
mkdir -p "$EDDY_HOME"
# --system-site-packages lets Eddy see apt-installed numpy and ROS 2's Python packages.
[ -x "$EDDY_HOME/venv/bin/python" ] || "$PY" -m venv --system-site-packages "$EDDY_HOME/venv"

# What pip installs: this folder, plus "[serial,voice,vision]" for the chosen extras.
TARGET="$HERE"
if [ ${#EXTRAS[@]} -gt 0 ]; then
  TARGET="$HERE[$(IFS=,; echo "${EXTRAS[*]}")]"
fi
say "Installing Eddy${EXTRAS[*]:+ with: ${EXTRAS[*]}}"
"$EDDY_HOME/venv/bin/python" -m pip install --quiet --upgrade "$TARGET"

# Make the `eddy` command available from the user's PATH.
mkdir -p "$HOME/.local/bin"
ln -sf "$EDDY_HOME/venv/bin/eddy" "$HOME/.local/bin/eddy"

# The speech model is downloaded once and kept in ~/.eddy.
if [ "$VOICE" = 1 ] && [ ! -d "$EDDY_HOME/vosk-model" ]; then
  say "Downloading the offline speech model"
  MODEL=vosk-model-small-en-us-0.15
  TMP="$(mktemp -d)"
  curl -fsSL "https://alphacephei.com/vosk/models/$MODEL.zip" -o "$TMP/model.zip"
  unzip -q "$TMP/model.zip" -d "$TMP"
  mv "$TMP/$MODEL" "$EDDY_HOME/vosk-model"
  rm -rf "$TMP"
fi

# So is the hand-tracking model.
if [ "$VISION" = 1 ] && [ ! -f "$EDDY_HOME/hand_landmarker.task" ]; then
  say "Downloading the hand-tracking model"
  curl -fsSL -o "$EDDY_HOME/hand_landmarker.task" \
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" \
    || echo "note: could not download it now; hand signs will say how to get it"
fi

# --service: write a systemd user service that starts Eddy at boot and restarts it if it fails.
if [ -n "$SERVICE_ARGS" ]; then
  command -v systemctl >/dev/null || { echo "--service needs systemd" >&2; exit 1; }
  say "Installing the boot service: eddy $SERVICE_ARGS"
  # If ROS 2 is installed, the service sources it first so the ros2 adapter works.
  ROS_SETUP="$(ls /opt/ros/*/setup.bash 2>/dev/null | tail -1 || true)"
  UNIT_DIR="$HOME/.config/systemd/user"
  mkdir -p "$UNIT_DIR"
  cat > "$UNIT_DIR/eddy.service" <<EOF
[Unit]
Description=Eddy AI robot controller
After=network-online.target

[Service]
ExecStart=/bin/bash -c '${ROS_SETUP:+source $ROS_SETUP; }exec $EDDY_HOME/venv/bin/eddy $SERVICE_ARGS'
Restart=on-failure
RestartSec=3
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=default.target
EOF
  systemctl --user daemon-reload
  systemctl --user enable --now eddy.service
  # "Linger" lets a user service run without that user being logged in.
  loginctl enable-linger "$(id -un)" 2>/dev/null || echo "note: run 'sudo loginctl enable-linger $(id -un)' so Eddy starts at boot without a login"
  echo "Service installed. Logs:  journalctl --user -u eddy -f"
fi

# Finish by proving the install works on this machine.
say "Checking this machine"
"$EDDY_HOME/venv/bin/eddy" doctor
say "Running the self-test"
"$EDDY_HOME/venv/bin/eddy" selftest --quiet && echo "Self-test passed."

# Remind the user to add ~/.local/bin to PATH if it is not there yet.
case ":$PATH:" in
  *":$HOME/.local/bin:"*) ;;
  *) echo; echo "Add Eddy to your PATH:  echo 'export PATH=\"\$HOME/.local/bin:\$PATH\"' >> ~/.bashrc && source ~/.bashrc" ;;
esac
echo
echo "Eddy is installed. Try:  eddy scan   |   eddy shell -r sim   |   eddy shell -r ros2"
