"""Turn a sentence (typed or heard) into a robot command.

Deliberately rule-based: it runs instantly on a Raspberry Pi with no network and
never invents an action. Anything it cannot understand returns None.
"""
from __future__ import annotations

import math
import re
from typing import Optional

from .model import Drive, Gripper, Home, MoveTo, Report, Say, Stop, Wander

_WORDS = {"zero": 0, "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
          "seven": 7, "eight": 8, "nine": 9, "ten": 10, "twenty": 20, "thirty": 30, "forty": 40,
          "forty-five": 45, "fifty": 50, "sixty": 60, "ninety": 90, "half": 0.5, "quarter": 0.25,
          "hundred": 100, "one-eighty": 180,
          # what speech recognisers write when someone says "two" / "four" before a unit
          "to": 2, "too": 2, "for": 4}

# Every word Eddy can act on. The speech recogniser is restricted to these, which makes it far
# more accurate than open dictation ("open the gripper" instead of a random similar-sounding phrase).
VOCABULARY = (
    "eddy eddie stop halt freeze go drive move forward ahead straight back backward backwards reverse "
    "turn rotate spin left right around clockwise slow slowly fast quickly "
    "open close gripper hand grab release home reach arm the wander explore patrol status report say hello "
    "zero one two three four five six seven eight nine ten twenty thirty forty fifty sixty ninety hundred half "
    "meter meters centimeter centimeters foot feet inch inches second seconds minute minutes degree degrees "
    "for please"
)
_NUM = r"(-?\d+(?:\.\d+)?|" + "|".join(sorted(_WORDS, key=len, reverse=True)) + r")"
_LENGTH = {"m": 1.0, "meter": 1.0, "meters": 1.0, "metre": 1.0, "metres": 1.0,
           "cm": 0.01, "centimeter": 0.01, "centimeters": 0.01, "centimetre": 0.01, "centimetres": 0.01,
           "mm": 0.001, "foot": 0.3048, "feet": 0.3048, "ft": 0.3048, "inch": 0.0254, "inches": 0.0254}
_TIME = {"s": 1.0, "sec": 1.0, "secs": 1.0, "second": 1.0, "seconds": 1.0, "minute": 60.0, "minutes": 60.0}


def _num(token: str) -> float:
    return float(_WORDS[token]) if token in _WORDS else float(token)


def _amount(text: str, units: dict) -> Optional[float]:
    m = re.search(_NUM + r"\s*(" + "|".join(sorted(units, key=len, reverse=True)) + r")\b", text)
    return _num(m.group(1)) * units[m.group(2)] if m else None


def parse(text: str, speed: float = 0.2, turn_speed: float = 0.8):
    """Return a command, or None if the sentence is not a command Eddy knows."""
    raw = text.strip()
    t = re.sub(r"[^\w\s.\-,]", " ", raw.lower())
    t = re.sub(r"\b(please|eddy|hey|robot|now|can you|could you)\b", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return None

    m = re.match(r"(?:say|speak|tell me)\s+(.+)", t)
    if m:
        return Say(raw[re.search(r"(?i)say|speak|tell me", raw).end():].strip(" :,"))

    if re.search(r"\b(stop|halt|freeze|whoa|brake|abort|don't move|do not move|stay)\b", t):
        return Stop()

    if re.search(r"\b(status|where are you|what are you|who are you|report|what can you do)\b", t):
        return Report()

    if re.search(r"\b(open)\b.*\b(gripper|hand|claw)\b|\b(release|let go|drop)\b", t):
        return Gripper(1.0)
    if re.search(r"\b(close)\b.*\b(gripper|hand|claw)\b|\b(grab|grasp|grip|pinch)\b", t):
        return Gripper(0.0)

    if re.search(r"\b(home|rest position|park|stow)\b", t):
        return Home()

    # "reach 0.2 0 0.3", "move arm to x 0.2 y 0 z 0.3", "arm to 20 0 30 cm"
    if re.search(r"\b(reach|arm|point at|tool|hand to)\b", t):
        nums = re.findall(r"-?\d+(?:\.\d+)?", t)
        if len(nums) >= 3:
            scale = 0.01 if re.search(r"\bcm\b|centimet", t) else (0.001 if re.search(r"\bmm\b", t) else 1.0)
            x, y, z = (float(v) * scale for v in nums[:3])
            return MoveTo(x, y, z)
        return None

    if re.search(r"\b(wander|explore|roam|look around|patrol)\b", t):
        return Wander(_amount(t, _TIME) or 10.0)

    slow = 0.5 if re.search(r"\b(slow|slowly|careful|carefully)\b", t) else 1.0
    fast = 1.75 if re.search(r"\b(fast|quick|quickly)\b", t) else 1.0
    v, w = speed * slow * fast, turn_speed * slow * fast

    turn = re.search(r"\b(turn|rotate|spin|face|left|right|around)\b", t)
    straight = re.search(r"\b(forward|forwards|ahead|straight|back|backward|backwards|reverse)\b", t)
    if turn and not straight:
        if re.search(r"\b(around|about)\b", t) and not re.search(_NUM + r"\s*(deg|degree|degrees)\b", t):
            return Drive(angular=w, angle=math.pi)
        direction = -1.0 if re.search(r"\b(right|clockwise)\b", t) and not re.search(r"counter|anti", t) else 1.0
        deg = re.search(_NUM + r"\s*(deg|degree|degrees)\b", t)
        duration = _amount(t, _TIME)
        if deg:
            return Drive(angular=direction * w, angle=math.radians(_num(deg.group(1))))
        if duration:
            return Drive(angular=direction * w, duration=duration)
        if re.search(r"\b(left|right|clockwise)\b", t):
            return Drive(angular=direction * w, angle=math.pi / 2)
        if re.search(r"\bspin\b", t):
            return Drive(angular=w, angle=2 * math.pi)
        return None

    if straight or re.search(r"\b(go|drive|move|come|advance|roll)\b", t):
        direction = -1.0 if re.search(r"\b(back|backward|backwards|reverse)\b", t) else 1.0
        distance, duration = _amount(t, _LENGTH), _amount(t, _TIME)
        if distance is not None:
            return Drive(linear=direction * v, distance=distance)
        if duration is not None:
            return Drive(linear=direction * v, duration=duration)
        if straight:
            return Drive(linear=direction * v, distance=0.5)   # "go forward" with no amount: half a metre
    return None
