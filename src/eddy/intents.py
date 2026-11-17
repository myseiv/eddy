"""Turn a sentence (typed or heard) into a robot command.

Deliberately rule-based: it runs instantly on a Raspberry Pi with no network and
never invents an action. Anything it cannot understand returns None.
"""
from __future__ import annotations

import math
import re
from typing import Optional

from .model import Climb, Drive, Gripper, Home, Land, MoveBy, MoveTo, TakeOff, Report, Say, SetJoints, Slide, Stop, Wander

# Number words and the value each stands for, so "two meters" works as well as "2 meters".
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
    "open close gripper hand grab release home reach arm up down raise lower take off land fly climb descend the wander explore patrol status report say hello "
    "zero one two three four five six seven eight nine ten twenty thirty forty fifty sixty ninety hundred half "
    "meter meters centimeter centimeters foot feet inch inches second seconds minute minutes degree degrees "
    "for please "
    # Words that change or cancel the meaning of a command. The recogniser must be able to
    # hear them, or "don't drive forward" would arrive as "drive forward".
    "don't do not never no or unless if and then cancel wait "
    "joint set to"
)
# Regex matching one number: digits ("1.5", "-2") or a number word. Longest words first, so
# "forty-five" is not cut short at "forty".
_NUM = r"(-?\d+(?:\.\d+)?|" + "|".join(sorted(_WORDS, key=len, reverse=True)) + r")"
# Units and how many metres (or seconds) one of each is.
_LENGTH = {"m": 1.0, "meter": 1.0, "meters": 1.0, "metre": 1.0, "metres": 1.0,
           "cm": 0.01, "centimeter": 0.01, "centimeters": 0.01, "centimetre": 0.01, "centimetres": 0.01,
           "mm": 0.001, "foot": 0.3048, "feet": 0.3048, "ft": 0.3048, "inch": 0.0254, "inches": 0.0254}
_TIME = {"s": 1.0, "sec": 1.0, "secs": 1.0, "second": 1.0, "seconds": 1.0, "minute": 60.0, "minutes": 60.0}


def _num(token: str) -> float:
    """Turn a number token ("2", "0.5", "two") into a float."""
    return float(_WORDS[token]) if token in _WORDS else float(token)


_ANGLE = {"deg": 1.0, "degree": 1.0, "degrees": 1.0}
# Number words, for spotting an amount Eddy has not used ("a", "an", "to" and "for" are left out:
# they are ordinary words far more often than numbers).
_NUMBER_WORDS = re.compile(r"\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|twenty|thirty|forty|"
                           r"forty-five|fifty|sixty|ninety|hundred|one-eighty|half|quarter)\b")


def _take(text: str, units: dict):
    """Find "<number> <unit>" in the text. Returns (the amount in base units, the text without it),
    or (None, the text) if there is none."""
    # "half a meter" and "a quarter of a second" are a half and a quarter, not "a meter".
    text = re.sub(r"\b(?:a\s+|one\s+)?(half|quarter)\s+(?:of\s+)?an?\s+", r"\1 ", text)
    names = "|".join(sorted(units, key=len, reverse=True))
    # "one and a half meters": the number plus a half.
    m = re.search(_NUM + r"\s+and\s+a\s+half\s*(" + names + r")\b", text)
    if m:
        return (_num(m.group(1)) + 0.5) * units[m.group(2)], text[:m.start()] + " " + text[m.end():]
    m = re.search(_NUM + r"\s*(" + names + r")\b", text)
    if m:
        return _num(m.group(1)) * units[m.group(2)], text[:m.start()] + " " + text[m.end():]
    return None, text


def _amount(text: str, units: dict) -> Optional[float]:
    """Find "<number> <unit>" in the text and return it in base units (metres or seconds)."""
    return _take(text, units)[0]


def _numbers_left(text: str) -> bool:
    """True if a number is still in the text once the amounts a command used are taken out.
    Such a number is an amount Eddy would otherwise drop ("turn left 30 cm" turning 90 degrees)."""
    return bool(re.search(r"\d", text) or _NUMBER_WORDS.search(text))


def hint(text: str) -> str:
    """A short suggestion for a sentence Eddy could not use, or "" if it has none."""
    t = text.lower()
    if "joint" in t:
        return ' To move one joint, say where it should go: "move joint1 to 20 degrees".'
    if re.search(r"\b(turn|rotate|spin)\b", t) and _amount(t, _LENGTH) is not None:
        return (' A turn is given in degrees: "turn left 30 degrees". To turn and drive, say each in order: '
                '"turn left 30 degrees and drive forward 20 cm".')
    if re.search(r"\d", t):
        return " Give each amount a unit Eddy knows (cm, meters, degrees or seconds), one amount per step."
    return ""


def parse(text: str, speed: float = 0.2, turn_speed: float = 0.8):
    """Return a command, or None if the sentence is not a command Eddy knows.

    speed and turn_speed are the normal driving speeds (m/s and rad/s) that "slowly" and
    "fast" scale. The checks below run in priority order: the first rule that matches wins.
    """
    # Clean up: lower case, drop punctuation, drop filler words, collapse spaces.
    raw = text.strip()
    t = re.sub(r"[^\w\s.\-,]", " ", raw.lower())
    t = re.sub(r"\b(please|eddy|hey|robot|now|can you|could you)\b", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return None

    # "say ..." comes first so the words after it are never read as a command.
    # The text to speak is taken from the original sentence, keeping its capitals.
    m = re.match(r"(?:say|speak|tell me)\s+(.+)", t)
    if m:
        return Say(raw[re.search(r"(?i)say|speak|tell me", raw).end():].strip(" :,"))

    # Stop is checked before any motion, so "stop driving forward" stops.
    # (Apostrophes were removed above, so "don't" reads "don t" here.)
    if re.search(r"\b(stop|halt|freeze|whoa|brake|abort|cancel|don t move|dont move|do not move|stay)\b", t):
        return Stop()

    # A sentence that says what NOT to do, or offers a choice, is not an instruction to act.
    # ("don't drive forward" must never be read as "drive forward".)
    if re.search(r"\b(don t|dont|do not|never|not|no|without|won t|shouldn t|can t|cannot|or|unless|if)\b", t):
        return None

    if re.search(r"\b(status|where are you|what are you|who are you|report|what can you do)\b", t):
        return Report()

    # One joint on its own: "set joint1 to 0.3", "move joint elbow to 20 degrees".
    # The value is radians unless the sentence says degrees.
    # "joint 1" is joint1, never a joint called "joint" set to 1.
    value = r"\s+(?:to\s+)?(-?\d+(?:\.\d+)?)\s*(deg|degree|degrees)?\b"
    m = re.search(r"\bjoint\s*(\d+)" + value, t)
    named = None
    if m:
        named, amount, unit = "joint" + m.group(1), m.group(2), m.group(3)
    else:
        m = re.search(r"\b(?:joint\s+(?!to\b)([a-z_]\w*)|(\w*joint\w+|\w+joint\w*))" + value, t)
        if m:
            named, amount, unit = m.group(1) or m.group(2), m.group(3), m.group(4)
    if m:
        if _numbers_left(t[:m.start()] + " " + t[m.end():]):
            return None             # "move joint1 to 20 degrees for 2 seconds": not guessed
        return SetJoints({named: math.radians(float(amount)) if unit else float(amount)})
    if "joint" in t:
        return None                 # about a joint, but not where it should go ("move joint1", "joint1 up")

    # The arm, relative to where it is: "move the arm up 5 cm", "raise the arm", "arm left 10 cm".
    # With no amount it moves 5 cm.
    if not re.search(r"\b(open|close)\b", t):
        m = (re.search(r"\b(?:arm|hand|tool|gripper)\b.*\b(up|down|forward|forwards|back|backward|backwards|left|right)\b", t)
             or re.search(r"\b(raise|lift|lower)\b.*\b(?:arm|hand|tool|gripper)\b", t))
        if m:
            step = _amount(t, _LENGTH) or 0.05
            axis = {"up": (0, 0, 1), "raise": (0, 0, 1), "lift": (0, 0, 1), "down": (0, 0, -1), "lower": (0, 0, -1),
                    "forward": (1, 0, 0), "forwards": (1, 0, 0), "back": (-1, 0, 0), "backward": (-1, 0, 0),
                    "backwards": (-1, 0, 0), "left": (0, 1, 0), "right": (0, -1, 0)}[m.group(1)]
            return MoveBy(*(step * a for a in axis))

    # Gripper: "open the gripper", "let go", "grab", ...
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
            # The three numbers are metres unless the sentence says cm or mm.
            scale = 0.01 if re.search(r"\bcm\b|centimet", t) else (0.001 if re.search(r"\bmm\b", t) else 1.0)
            x, y, z = (float(v) * scale for v in nums[:3])
            return MoveTo(x, y, z)
        return None

    # Sideways, for bases that can: "slide left 20 cm", "strafe right 30 cm", "sidestep left".
    # With no amount it moves 20 cm. The direction must be said.
    if re.search(r"\b(slide|strafe|sidestep|sideways|crab)\b", t):
        side = re.search(r"\b(left|right)\b", t)
        if not side or re.search(r"(?<![\w.])-\d", t):
            return None
        distance, rest = _take(t, _LENGTH)
        if _numbers_left(rest) or _amount(rest, _TIME) is not None:
            return None             # "slide left 20", "slide left 30 degrees": not guessed
        slow = 0.5 if re.search(r"\b(slow|slowly|careful|carefully)\b", t) else 1.0
        return Slide((1.0 if side.group(1) == "left" else -1.0) * (distance or 0.2), 0.15 * slow)

    # Flying: "take off", "take off to 2 meters", "land", "go up 50 cm", "descend 1 meter".
    if re.search(r"\b(take off|takeoff|lift off|liftoff|launch)\b", t):
        return TakeOff(_amount(t, _LENGTH) or 1.0)
    if re.search(r"\b(land|touch down|touchdown)\b", t):
        return Land()
    m = re.search(r"\b(climb|ascend|descend)\b|\b(?:go|fly|move|come)\s+(up|down|higher|lower)\b", t)
    if m:
        word = m.group(1) or m.group(2)
        if re.search(r"(?<![\w.])-\d", t) or (_amount(t, _LENGTH) is None and re.search(r"\d", t)):
            return None             # a negative amount, or a number with no unit: not guessed
        return Climb((-1.0 if word in ("descend", "down", "lower") else 1.0) * (_amount(t, _LENGTH) or 0.5))

    if re.search(r"\b(wander|explore|roam|look around|patrol)\b", t):
        return Wander(_amount(t, _TIME) or 10.0)

    # Driving. "slowly" halves the speed, "fast" raises it by 75%.
    slow = 0.5 if re.search(r"\b(slow|slowly|careful|carefully)\b", t) else 1.0
    fast = 1.75 if re.search(r"\b(fast|quick|quickly)\b", t) else 1.0
    v, w = speed * slow * fast, turn_speed * slow * fast

    turn = re.search(r"\b(turn|rotate|spin|face|left|right|around)\b", t)
    straight = re.search(r"\b(forward|forwards|ahead|straight|back|backward|backwards|reverse)\b", t)
    if turn and not straight:
        if re.search(r"(?<![\w.])-\d", t):
            return None             # a negative amount is ambiguous: say the direction instead
        if _amount(t, _LENGTH) is not None:
            return None             # "turn left 30 cm": a turn is in degrees or seconds, a distance is not guessed into one
        angle, rest = _take(t, _ANGLE)
        duration, rest = _take(rest, _TIME)
        if angle is not None and duration is not None:
            return None             # both an angle and a time: not guessed
        # A bare number after a turn ("turn left 45") is degrees.
        bare = re.search(r"(?<![\w.])(\d+(?:\.\d+)?)\s*$", rest) if angle is None and duration is None else None
        if bare:
            angle, rest = float(bare.group(1)), rest[:bare.start()]
        if _numbers_left(rest):
            return None             # an amount Eddy would otherwise drop
        # "turn around" with no amount means 180 degrees.
        if angle is None and duration is None and re.search(r"\b(around|about)\b", t):
            return Drive(angular=w, angle=math.pi)
        # Positive angular speed is left (counter-clockwise); right and clockwise are negative.
        direction = -1.0 if re.search(r"\b(right|clockwise)\b", t) and not re.search(r"counter|anti", t) else 1.0
        if angle is not None:
            return Drive(angular=direction * w, angle=math.radians(angle))
        if duration:
            return Drive(angular=direction * w, duration=duration)
        if re.search(r"\b(left|right|clockwise)\b", t):
            return Drive(angular=direction * w, angle=math.pi / 2)     # "turn left": a quarter turn
        if re.search(r"\bspin\b", t):
            return Drive(angular=w, angle=2 * math.pi)                 # "spin": one full turn
        return None

    if straight or re.search(r"\b(go|drive|move|come|advance|roll|fly)\b", t):
        direction = -1.0 if re.search(r"\b(back|backward|backwards|reverse)\b", t) else 1.0
        if re.search(r"(?<![\w.])-\d", t):
            return None             # a negative amount is ambiguous: say the direction instead
        distance, rest = _take(t, _LENGTH)
        duration, rest = _take(rest, _TIME)
        if distance is not None and duration is not None:
            return None             # "drive forward 1 meter for 5 seconds": which one? Not guessed.
        if _numbers_left(rest):
            return None             # "go forward 5" (five what?), "go back 80 degrees for 20 cm": not guessed
        if turn and re.search(r"\b(turn|rotate|spin)\b", t):
            return None             # "turn left go forward": two commands run together; not guessed
        if distance is not None:
            return Drive(linear=direction * v, distance=distance)
        if duration is not None:
            return Drive(linear=direction * v, duration=duration)
        if straight:
            return Drive(linear=direction * v, distance=0.5)   # "go forward" with no amount: half a metre
    # "go" or "move" with no direction and no amount is too vague to act on.
    return None


def parse_all(text: str, **speeds) -> list:
    """Return the commands in a sentence, in order: usually one, several for
    "turn left and go forward 50 cm". An empty list means nothing was understood.

    A sentence is only split when every piece is a command on its own. A sentence that
    contains "stop" is just Stop, and "say ..." is never split.
    """
    whole = parse(text, **speeds)
    if isinstance(whole, (Stop, Say)):
        return [whole]
    pieces = [p for p in re.split(r"(?i)\s*(?:;|,|\band then\b|\bthen\b|\band\b)\s*", text) if p.strip()]
    if len(pieces) > 1:
        commands = [parse(p, **speeds) for p in pieces]
        if all(c is not None for c in commands):
            return commands
    return [whole] if whole is not None else []
