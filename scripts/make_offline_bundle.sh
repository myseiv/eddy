#!/usr/bin/env bash
# Build an offline install bundle for a robot that has no internet.
#
# Run this on a computer WITH internet. It downloads everything the chosen extras need, for the
# robot's own CPU and Python, and writes one folder (and a .tar.gz) to copy to the robot.
# On the robot you then run ./install_offline.sh from inside it. No internet is needed there.
#
#   ./scripts/make_offline_bundle.sh --arch aarch64 --python 3.10 --all
#   ./scripts/make_offline_bundle.sh --arch aarch64 --python 3.10 --vision
#   ./scripts/make_offline_bundle.sh                       (asks, if run in a terminal)
#
#   --arch aarch64|x86_64   the robot's CPU: run `uname -m` on it (a Raspberry Pi 5 is aarch64)
#   --python X.Y            the robot's Python: run `python3 --version` on it (inside its container, if it has one)
#   --serial --voice --vision --all    which extras to include (the core needs nothing and is not downloaded)
#   --out DIR               where to write the bundle (default: ./dist)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARCH=aarch64
PYVER=3.10
OUT="$HERE/dist"
EXTRAS=()
VOICE=0
VISION=0

while [ $# -gt 0 ]; do
  case "$1" in
    --arch)   shift; ARCH="${1:?--arch needs aarch64 or x86_64}" ;;
    --python) shift; PYVER="${1:?--python needs a version like 3.10}" ;;
    --out)    shift; OUT="${1:?--out needs a folder}" ;;
    --serial) EXTRAS+=(serial) ;;
    --voice)  EXTRAS+=(voice); VOICE=1 ;;
    --vision) EXTRAS+=(vision); VISION=1 ;;
    --all)    EXTRAS+=(serial voice vision); VOICE=1; VISION=1 ;;
    -h|--help) sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

case "$ARCH" in aarch64|x86_64) ;; *) echo "--arch must be aarch64 or x86_64 (got $ARCH)" >&2; exit 2 ;; esac
case "$PYVER" in [0-9].[0-9]|[0-9].[0-9][0-9]) ;; *) echo "--python must look like 3.10 (got $PYVER)" >&2; exit 2 ;; esac

# No extras on the command line and a person at the keyboard: ask which version they want.
if [ ${#EXTRAS[@]} -eq 0 ] && [ -t 0 ]; then
  echo "Which Eddy do you want on the robot?"
  echo "  1) Core only            (drive, arm, typed commands, ROS 2: nothing to bundle)"
  echo "  2) + hand signs         (camera: opencv, mediapipe and the hand model)"
  echo "  3) + voice              (microphone: vosk, sounddevice and the speech model)"
  echo "  4) Everything           (hand signs, voice and USB serial boards)"
  read -r -p "Choose 1-4: " CHOICE
  case "$CHOICE" in
    1) ;;
    2) EXTRAS=(vision); VISION=1 ;;
    3) EXTRAS=(voice); VOICE=1 ;;
    4) EXTRAS=(serial voice vision); VOICE=1; VISION=1 ;;
    *) echo "please choose 1, 2, 3 or 4" >&2; exit 2 ;;
  esac
fi
if [ ${#EXTRAS[@]} -eq 0 ]; then
  echo "Core only needs nothing downloaded: copy this folder to the robot and set PYTHONPATH (see docs)." >&2
  echo "Pass --vision, --voice, --serial or --all to build a bundle with extras." >&2
  exit 0
fi

PY="${PYTHON:-python3}"
command -v "$PY" >/dev/null || { echo "python3 is needed on this computer" >&2; exit 1; }
command -v curl >/dev/null || { echo "curl is needed on this computer" >&2; exit 1; }

NAME="eddy-offline-$ARCH-py$PYVER"
BUNDLE="$OUT/$NAME"
rm -rf "$BUNDLE"
mkdir -p "$BUNDLE/wheels" "$BUNDLE/models"
TAG="cp${PYVER/./}"
EXTRAS_CSV="$(IFS=,; echo "${EXTRAS[*]}")"

# 1. Eddy itself, as a wheel, so the robot never has to build anything.
echo "==> Building Eddy's own wheel"
"$PY" -m pip wheel --quiet --no-deps -w "$BUNDLE/wheels" "$HERE"

# vosk needs `srt`, which is published only as source. It is pure Python, so a wheel built here
# works on any robot, and pip can then find it while it downloads the binary packages below.
if [ "$VOICE" = 1 ]; then
  echo "==> Building the source-only package srt into a wheel"
  "$PY" -m pip wheel --quiet --no-deps -w "$BUNDLE/wheels" srt
fi

# 2. Everything the extras need, for the robot's CPU and Python (binary wheels only).
echo "==> Downloading packages for $ARCH, Python $PYVER (extras: $EXTRAS_CSV)"
PLATFORMS=()
for glibc in manylinux2014 manylinux_2_17 manylinux_2_28; do PLATFORMS+=(--platform "${glibc}_${ARCH}"); done
"$PY" -m pip download --quiet --dest "$BUNDLE/wheels" --only-binary=:all: \
  --python-version "$PYVER" --implementation cp --abi "$TAG" "${PLATFORMS[@]}" \
  --find-links "$BUNDLE/wheels" "eddy-ai[$EXTRAS_CSV]"

# 3. The two model files the packages need but do not contain.
if [ "$VISION" = 1 ]; then
  echo "==> Downloading the hand-tracking model"
  curl -fsSL -o "$BUNDLE/models/hand_landmarker.task" \
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
fi
if [ "$VOICE" = 1 ]; then
  echo "==> Downloading the offline speech model"
  MODEL=vosk-model-small-en-us-0.15
  TMP="$(mktemp -d)"
  curl -fsSL "https://alphacephei.com/vosk/models/$MODEL.zip" -o "$TMP/model.zip"
  "$PY" -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$TMP/model.zip" "$TMP"
  mv "$TMP/$MODEL" "$BUNDLE/models/vosk-model"
  rm -rf "$TMP"
fi

# 4. What the robot side needs to know.
printf '%s\n' "$EXTRAS_CSV" > "$BUNDLE/extras.txt"
printf 'arch=%s\npython=%s\n' "$ARCH" "$PYVER" > "$BUNDLE/target.txt"
cp "$HERE/scripts/install_offline.sh" "$BUNDLE/install_offline.sh"
chmod +x "$BUNDLE/install_offline.sh"

# No macOS metadata in the archive (it makes GNU tar on the robot print a warning per file).
TAR_FLAGS=""
tar --help 2>&1 | grep -q -- "--no-xattrs" && TAR_FLAGS="--no-xattrs"
( cd "$OUT" && COPYFILE_DISABLE=1 tar $TAR_FLAGS -czf "$NAME.tar.gz" "$NAME" )
echo
echo "Bundle ready: $BUNDLE  ($(du -sh "$BUNDLE" | cut -f1))"
echo "To send it:   scp $OUT/$NAME.tar.gz <user>@<robot>:~   (or copy the .tar.gz on a USB stick)"
echo "On the robot: tar xzf $NAME.tar.gz && cd $NAME && ./install_offline.sh"
