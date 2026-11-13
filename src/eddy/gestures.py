"""Hand signs: from 21 hand landmarks to a named gesture to a robot command.

Landmarks use the MediaPipe Hands layout (0 wrist, 4 thumb tip, 8 index tip,
12 middle tip, 16 ring tip, 20 pinky tip), as (x, y) or (x, y, z) with image
coordinates: x to the right, y DOWN. The maths here needs no camera and no
MediaPipe, so it can be tested with made-up hands.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

from .model import Drive, Stop

WRIST = 0
THUMB = (1, 2, 3, 4)
FINGERS = {"index": (5, 6, 7, 8), "middle": (9, 10, 11, 12), "ring": (13, 14, 15, 16), "pinky": (17, 18, 19, 20)}

# What each sign tells the robot to do. Change this table to change the vocabulary.
DEFAULT_ACTIONS = {
    "open_palm": lambda: Stop(),
    "fist": lambda: Stop(),
    "point": lambda: Drive(linear=0.2, duration=0.6),
    "victory": lambda: Drive(linear=-0.2, duration=0.6),
    "thumb_left": lambda: Drive(angular=0.8, duration=0.6),
    "thumb_right": lambda: Drive(angular=-0.8, duration=0.6),
}


def _dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def fingers_up(lm: Sequence[Sequence[float]]) -> Dict[str, bool]:
    """Which fingers are extended."""
    wrist = lm[WRIST]
    palm = _dist(wrist, lm[9]) or 1e-6           # wrist to middle knuckle: the hand's size
    up = {}
    for name, (mcp, pip, dip, tip) in FINGERS.items():
        up[name] = _dist(lm[tip], wrist) > _dist(lm[pip], wrist) * 1.15 and _dist(lm[tip], lm[mcp]) > 0.6 * palm
    # The thumb is out when its tip is clear of the index and middle fingers (a tucked thumb rests
    # on them) and further from the pinky knuckle than its own middle joint is.
    clearance = min(_dist(lm[4], lm[j]) for j in (5, 6, 7, 9, 10, 11))
    up["thumb"] = clearance > 0.45 * palm and _dist(lm[4], lm[17]) > _dist(lm[3], lm[17])
    return up


def classify(lm: Sequence[Sequence[float]]) -> Optional[str]:
    """Name the hand sign, or None if it is not one Eddy knows."""
    if lm is None or len(lm) != 21:
        return None
    up = fingers_up(lm)
    four = [up["index"], up["middle"], up["ring"], up["pinky"]]
    if all(four):
        return "open_palm"
    if four == [True, False, False, False]:
        return "point"
    if four == [True, True, False, False]:
        return "victory"
    if not any(four):
        if not up["thumb"]:
            return "fist"
        dx, dy = lm[4][0] - lm[2][0], lm[4][1] - lm[2][1]
        if abs(dx) > abs(dy):
            # Named from the robot's point of view: its camera looks forward, so the right of
            # the image is the robot's right. "Thumb that way" means "turn that way".
            return "thumb_right" if dx > 0 else "thumb_left"
        return "thumb_up" if dy < 0 else "thumb_down"
    return None


class GestureFilter:
    """Only report a sign after it has been seen for several frames in a row, and only once."""

    def __init__(self, hold_frames: int = 5, repeat_frames: int = 15):
        self.hold_frames, self.repeat_frames = hold_frames, repeat_frames
        self.current: Optional[str] = None
        self.count = 0

    def update(self, gesture: Optional[str]) -> Optional[str]:
        if gesture != self.current:
            self.current, self.count = gesture, 1
        else:
            self.count += 1
        if gesture is None:
            return None
        if self.count == self.hold_frames:
            return gesture
        # A held sign repeats, so "keep pointing" keeps the robot moving.
        if self.count > self.hold_frames and (self.count - self.hold_frames) % self.repeat_frames == 0:
            return gesture
        return None


def to_command(gesture: Optional[str], actions=None):
    make = (actions or DEFAULT_ACTIONS).get(gesture)
    return make() if make else None


def fake_hand(extended: Sequence[str] = (), thumb_dir: Optional[str] = None, scale: float = 0.1,
              origin=(0.5, 0.7)) -> List[List[float]]:
    """Build a synthetic upright hand for tests and demos.

    extended: any of index/middle/ring/pinky. thumb_dir: None (tucked) or up/down/left/right
    as seen in the image.
    """
    ox, oy = origin
    lm = [[ox, oy]] * 21
    lm = [list(p) for p in lm]
    columns = {"index": -0.35, "middle": -0.1, "ring": 0.15, "pinky": 0.4}
    for name, (mcp, pip, dip, tip) in FINGERS.items():
        x = ox + columns[name] * scale
        lm[mcp] = [x, oy - 1.0 * scale]
        if name in extended:
            lm[pip], lm[dip], lm[tip] = [x, oy - 1.4 * scale], [x, oy - 1.7 * scale], [x, oy - 2.0 * scale]
        else:   # curled back toward the palm
            lm[pip], lm[dip], lm[tip] = [x, oy - 1.3 * scale], [x, oy - 1.1 * scale], [x, oy - 0.9 * scale]
    base = [ox - 0.45 * scale, oy - 0.35 * scale]
    lm[1] = base
    if thumb_dir is None:
        lm[2] = [ox - 0.5 * scale, oy - 0.6 * scale]
        lm[3] = [ox - 0.4 * scale, oy - 0.8 * scale]
        lm[4] = [ox - 0.25 * scale, oy - 0.9 * scale]
    else:
        ux, uy = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}[thumb_dir]
        lm[2] = [base[0] + ux * 0.4 * scale - 0.15 * scale, base[1] + uy * 0.4 * scale]
        lm[3] = [lm[2][0] + ux * 0.5 * scale, lm[2][1] + uy * 0.5 * scale]
        lm[4] = [lm[3][0] + ux * 0.5 * scale, lm[3][1] + uy * 0.5 * scale]
    return lm
