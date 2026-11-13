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

# Landmark numbers. Each finger has four points, from the knuckle out to the tip:
# (mcp = knuckle, pip = middle joint, dip = last joint, tip).
WRIST = 0
THUMB = (1, 2, 3, 4)
FINGERS = {"index": (5, 6, 7, 8), "middle": (9, 10, 11, 12), "ring": (13, 14, 15, 16), "pinky": (17, 18, 19, 20)}

# What each sign tells the robot to do. Change this table to change the vocabulary.
# Each entry is a function, so every use gets a fresh command object.
# (thumb_up and thumb_down are recognised but have no action yet.)
DEFAULT_ACTIONS = {
    "open_palm": lambda: Stop(),
    "fist": lambda: Stop(),
    "point": lambda: Drive(linear=0.2, duration=0.6),           # forward a little
    "victory": lambda: Drive(linear=-0.2, duration=0.6),        # back a little
    "thumb_left": lambda: Drive(angular=0.8, duration=0.6),     # turn left
    "thumb_right": lambda: Drive(angular=-0.8, duration=0.6),   # turn right
}


def _dist(a, b) -> float:
    """Distance between two landmarks in the image plane (z is ignored)."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def fingers_up(lm: Sequence[Sequence[float]]) -> Dict[str, bool]:
    """Which fingers are extended."""
    wrist = lm[WRIST]
    # All thresholds are fractions of the hand's size, so they hold at any distance from the camera.
    palm = _dist(wrist, lm[9]) or 1e-6           # wrist to middle knuckle: the hand's size
    up = {}
    for name, (mcp, pip, dip, tip) in FINGERS.items():
        # Extended = the tip is clearly further from the wrist than the middle joint is, and
        # well away from its own knuckle. A curled finger fails both tests.
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
        # All four fingers curled: a fist, or a thumb sign if the thumb sticks out.
        if not up["thumb"]:
            return "fist"
        # Which way the thumb points: from its base joint to its tip.
        dx, dy = lm[4][0] - lm[2][0], lm[4][1] - lm[2][1]
        if abs(dx) > abs(dy):
            # Named from the robot's point of view: its camera looks forward, so the right of
            # the image is the robot's right. "Thumb that way" means "turn that way".
            return "thumb_right" if dx > 0 else "thumb_left"
        return "thumb_up" if dy < 0 else "thumb_down"       # y grows downward in an image
    return None     # any other combination of fingers is not a sign Eddy knows


class GestureFilter:
    """Only report a sign after it has been seen for several frames in a row, and only once."""

    def __init__(self, hold_frames: int = 5, repeat_frames: int = 15):
        """hold_frames: frames a sign must last before it counts.
        repeat_frames: frames between repeats while it is still being held."""
        self.hold_frames, self.repeat_frames = hold_frames, repeat_frames
        self.current: Optional[str] = None      # the sign seen in the latest frame
        self.count = 0                          # how many frames in a row it has been seen

    def update(self, gesture: Optional[str]) -> Optional[str]:
        """Feed the sign seen in one frame; returns a sign only on the frames where it should act."""
        if gesture != self.current:
            self.current, self.count = gesture, 1       # a different sign: start counting again
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
    """The command for a sign, or None if the sign has no action."""
    make = (actions or DEFAULT_ACTIONS).get(gesture)
    return make() if make else None


def fake_hand(extended: Sequence[str] = (), thumb_dir: Optional[str] = None, scale: float = 0.1,
              origin=(0.5, 0.7)) -> List[List[float]]:
    """Build a synthetic upright hand for tests and demos.

    extended: any of index/middle/ring/pinky. thumb_dir: None (tucked) or up/down/left/right
    as seen in the image.
    """
    ox, oy = origin     # the wrist
    # Start with all 21 points at the wrist, then place each one.
    lm = [[ox, oy]] * 21
    lm = [list(p) for p in lm]
    # Fingers: four columns above the wrist ("above" is smaller y).
    columns = {"index": -0.35, "middle": -0.1, "ring": 0.15, "pinky": 0.4}
    for name, (mcp, pip, dip, tip) in FINGERS.items():
        x = ox + columns[name] * scale
        lm[mcp] = [x, oy - 1.0 * scale]
        if name in extended:
            lm[pip], lm[dip], lm[tip] = [x, oy - 1.4 * scale], [x, oy - 1.7 * scale], [x, oy - 2.0 * scale]
        else:   # curled back toward the palm
            lm[pip], lm[dip], lm[tip] = [x, oy - 1.3 * scale], [x, oy - 1.1 * scale], [x, oy - 0.9 * scale]
    # Thumb: starts at the side of the palm.
    base = [ox - 0.45 * scale, oy - 0.35 * scale]
    lm[1] = base
    if thumb_dir is None:
        # Tucked: lying across the palm, tip resting near the index knuckle.
        lm[2] = [ox - 0.5 * scale, oy - 0.6 * scale]
        lm[3] = [ox - 0.4 * scale, oy - 0.8 * scale]
        lm[4] = [ox - 0.25 * scale, oy - 0.9 * scale]
    else:
        # Sticking out in a straight line in the requested direction.
        ux, uy = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}[thumb_dir]
        lm[2] = [base[0] + ux * 0.4 * scale - 0.15 * scale, base[1] + uy * 0.4 * scale]
        lm[3] = [lm[2][0] + ux * 0.5 * scale, lm[2][1] + uy * 0.5 * scale]
        lm[4] = [lm[3][0] + ux * 0.5 * scale, lm[3][1] + uy * 0.5 * scale]
    return lm
