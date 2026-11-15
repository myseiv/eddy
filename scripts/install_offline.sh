#!/usr/bin/env bash
# Install Eddy on a robot with no internet, from a bundle made by make_offline_bundle.sh.
# Run it from inside the unpacked bundle folder:   ./install_offline.sh
#
#   ./install_offline.sh              a private Python environment in ~/.eddy/venv (the normal way)
#   ./install_offline.sh --no-venv    no environment: packages go in ~/.eddy/lib and a small `eddy`
#                                     launcher adds them to PYTHONPATH. Use this inside a ROS container
#                                     that has no python3-venv, or when you must keep the system's numpy.
#
# Installs into ~/.eddy (override with EDDY_HOME) and links `eddy` into ~/.local/bin.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EDDY_HOME="${EDDY_HOME:-$HOME/.eddy}"
USE_VENV=1
while [ $# -gt 0 ]; do
  case "$1" in
    --no-venv) USE_VENV=0 ;;
    -h|--help) sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

say() { printf '\n==> %s\n' "$*"; }
[ -d "$HERE/wheels" ] && [ -f "$HERE/extras.txt" ] || { echo "run this from inside an Eddy offline bundle folder" >&2; exit 1; }
EXTRAS="$(cat "$HERE/extras.txt")"
WANT_ARCH="$(sed -n 's/^arch=//p' "$HERE/target.txt")"
WANT_PY="$(sed -n 's/^python=//p' "$HERE/target.txt")"

PY="${PYTHON:-python3}"
command -v "$PY" >/dev/null || { echo "python3 is not installed on this robot" >&2; exit 1; }
HAVE_PY="$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
HAVE_ARCH="$(uname -m)"
# arm64 and aarch64 are the same thing.
[ "$HAVE_ARCH" = "arm64" ] && HAVE_ARCH=aarch64
if [ "$HAVE_PY" != "$WANT_PY" ] || [ "$HAVE_ARCH" != "$WANT_ARCH" ]; then
  echo "This bundle was made for $WANT_ARCH with Python $WANT_PY, but this machine is $HAVE_ARCH with Python $HAVE_PY." >&2
  echo "Its packages will be refused. Make a new bundle:  make_offline_bundle.sh --arch $HAVE_ARCH --python $HAVE_PY" >&2
  exit 1
fi

# System libraries cannot be bundled. Say which are missing, and carry on: some images already have them.
if command -v dpkg >/dev/null; then
  MISSING=()
  case ",$EXTRAS," in *,vision,*)
    for p in libgl1 libegl1 libgles2; do dpkg -s "$p" >/dev/null 2>&1 || MISSING+=("$p"); done
    dpkg -s libglib2.0-0 >/dev/null 2>&1 || dpkg -s libglib2.0-0t64 >/dev/null 2>&1 || MISSING+=(libglib2.0-0) ;;
  esac
  case ",$EXTRAS," in *,voice,*)
    for p in libportaudio2 libatomic1; do dpkg -s "$p" >/dev/null 2>&1 || MISSING+=("$p"); done ;;
  esac
  if [ ${#MISSING[@]} -gt 0 ]; then
    echo "note: these system libraries are not installed: ${MISSING[*]}"
    echo "      they come from apt, which needs internet. Install them from .deb files (apt-get download on a"
    echo "      connected machine of the same type, then dpkg -i), or the camera/voice part may fail to load."
  fi
fi

mkdir -p "$EDDY_HOME" "$HOME/.local/bin"
if [ "$USE_VENV" = 1 ]; then
  say "Creating Eddy's Python environment in $EDDY_HOME/venv"
  if [ ! -x "$EDDY_HOME/venv/bin/python" ]; then
    "$PY" -m venv --system-site-packages "$EDDY_HOME/venv" 2>/dev/null \
      || { echo "could not create a Python environment (python3-venv missing?). Run again with --no-venv." >&2; exit 1; }
  fi
  say "Installing Eddy with: $EXTRAS (from this folder, no internet)"
  "$EDDY_HOME/venv/bin/python" -m pip install --quiet --no-index --find-links "$HERE/wheels" --upgrade "eddy-ai[$EXTRAS]"
  ln -sf "$EDDY_HOME/venv/bin/eddy" "$HOME/.local/bin/eddy"
  EDDY="$EDDY_HOME/venv/bin/eddy"
else
  say "Installing Eddy into $EDDY_HOME/lib (no environment)"
  # The robot's own numpy (ROS uses it) must not be replaced, so numpy is left out when the robot has one.
  HAVE_NUMPY=0
  "$PY" -c 'import numpy' 2>/dev/null && HAVE_NUMPY=1
  WHEELS=()
  for w in "$HERE"/wheels/*.whl; do
    if [ "$HAVE_NUMPY" = 1 ]; then case "$(basename "$w")" in numpy-*) continue ;; esac; fi
    WHEELS+=("$w")
  done
  "$PY" -m pip install --quiet --no-index --no-deps --upgrade --target "$EDDY_HOME/lib" "${WHEELS[@]}"
  cat > "$HOME/.local/bin/eddy" <<LAUNCH
#!/bin/sh
export PYTHONPATH="$EDDY_HOME/lib\${PYTHONPATH:+:\$PYTHONPATH}"
exec $PY -c 'import sys; from eddy.cli import main; sys.exit(main())' "\$@"
LAUNCH
  chmod +x "$HOME/.local/bin/eddy"
  EDDY="$HOME/.local/bin/eddy"
fi

# Put the models where Eddy looks for them.
if [ -f "$HERE/models/hand_landmarker.task" ]; then cp "$HERE/models/hand_landmarker.task" "$EDDY_HOME/hand_landmarker.task"; fi
if [ -d "$HERE/models/vosk-model" ]; then rm -rf "$EDDY_HOME/vosk-model"; cp -R "$HERE/models/vosk-model" "$EDDY_HOME/vosk-model"; fi

say "Checking this machine"
"$EDDY" doctor
say "Running the self-test"
"$EDDY" selftest --quiet && echo "Self-test passed."

case ":$PATH:" in
  *":$HOME/.local/bin:"*) ;;
  *) echo; echo "Add Eddy to your PATH:  echo 'export PATH=\"\$HOME/.local/bin:\$PATH\"' >> ~/.bashrc && source ~/.bashrc" ;;
esac
echo
echo "Eddy is installed with: $EXTRAS. Try:  eddy run -r ros2 --gestures"
