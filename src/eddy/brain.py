"""Eddy itself: takes commands from anywhere and carries them out safely on any robot.

Commands can come from typed text, speech, hand signs, a program, or another
robot. Eddy checks each against what the attached robot can actually do, applies
the safety limits, and drives the adapter. A newer command always replaces the
one in progress, and Stop always wins.
"""
from __future__ import annotations

import dataclasses
import math
import threading
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

from . import gestures, intents, runlog
from .adapters.base import Adapter
from .model import (Climb, Drive, Gripper, Home, Land, MoveBy, MoveTo, TakeOff, Refused, Report, RobotModel, Say, SetJoints, Slide, Stop,
                    Unsupported, Wander)
from .senses.speak import Speaker


@dataclass
class Safety:
    """The limits Eddy applies to every motion, whatever asked for it."""
    min_range: float = 0.25          # never drive forward with something closer than this (metres)
    max_drive_seconds: float = 30.0  # no single drive command runs longer than this
    speed_scale: float = 1.0         # turn everything down with one number
    tick: float = 0.05               # how often commands are re-sent and sensors re-checked
    peer_hold: float = 10.0          # after the robot's own operator says stop, seconds it ignores commands from peers


@dataclass
class Result:
    """What happened to a command: whether it worked, and a sentence saying so."""
    ok: bool
    message: str
    command: object = None

    def __str__(self) -> str:
        return self.message


CAR = "car-like"    # DriveSpec.kind of a robot found unable to turn on the spot
CREEP = 0.12        # m/s: forward speed used to turn such a robot
NO_ROOM = "this robot has to roll forward to turn, and something is too close in front of it"
FLYING = "flying"   # DriveSpec.kind of a robot found to change height on command
CLIMB = 0.3         # m/s: climb and descent speed
BLIND = 1.0         # seconds without a range reading after which the last one is not trusted
STALL = 2.0         # seconds a drive may show no progress at all before Eddy gives up on it
SHORT = math.radians(12)    # a laser can be this far out on a good turn, so less than this is not a jam
AGREE = 0.85        # the share of the odometry's turn the laser must confirm
# Driving straight means holding the heading the drive started with. The correction is never
# exactly zero while moving: a real car's driver (Hiwonder MentorPi) leaves its steering wherever
# the last turn put it when told to turn at exactly 0, so "straight" drives veered 7 to 42 degrees.
HOLD = 0.3          # rad/s: the largest correction
HOLD_GAIN = 1.5     # rad/s of correction per radian off course
HOLD_MIN = 1e-3     # rad/s: the smallest turn rate sent while moving (far too small to see)
# The tightest a car-like robot is asked to steer: turn rate over speed (1/m, a 0.17 m turning
# circle). The same car's driver clamps any steering angle past 45 degrees to +45 whatever its
# sign, which turns a too-tight right turn into a left one.
CURVE = 6.0
SIDE = 1.0          # metres: the most Eddy slides sideways in one command, since it cannot see to the side


def _wrap(a: float) -> float:
    """Bring an angle into -pi..pi, so heading differences across the +/-180 line stay small."""
    return math.atan2(math.sin(a), math.cos(a))


def _cap(angular: float, linear: float) -> float:
    """The turn rate for a car-like robot rolling at `linear`, no tighter than CURVE allows."""
    return math.copysign(min(abs(angular), CURVE * abs(linear)), angular)


class Eddy:
    """The controller. Give it an adapter (a robot), then call tell(), sign() or execute()."""

    def __init__(self, adapter: Adapter, speaker: Optional[Speaker] = None, safety: Optional[Safety] = None,
                 log: Optional[runlog.RunLog] = None):
        """log: a run log to record every command in (see eddy.runlog), or None to keep none."""
        self.adapter = adapter
        self.log = log
        self._doing: Optional[str] = None     # who asked for the motion now running (None: nothing running)
        self._own_stop = float("-inf")        # when the robot's own operator last said stop
        self.turn_check: Optional[dict] = None    # the last turn as the odometry and the laser each saw it (radians)
        self.speaker = speaker or Speaker("print")
        self.safety = safety or Safety()
        self.model: Optional[RobotModel] = None     # filled in by start()
        self.listeners: List[Callable[[Result], None]] = []    # called with every Result (used by the peer link)
        self._epoch = 0                       # bumped by every new command; a running one notices and yields
        self._epoch_lock = threading.Lock()
        self._motion = threading.Lock()       # one motion at a time
        self._sign_filter = gestures.GestureFilter()

    # ---- lifecycle -----------------------------------------------------------

    def start(self) -> RobotModel:
        """Connect to the robot and learn what it can do."""
        self.model = self.adapter.connect()
        if self.log is not None:
            self.log.write("start", robot=self.model.name, kind=self.model.kind,
                           capabilities=self.model.capabilities(), details=self.model.details)
        return self.model

    def close(self) -> None:
        """Stop the robot and disconnect."""
        self.stop()
        self.adapter.close()
        if self.log is not None:
            self.log.close()

    # Support `with Eddy(adapter) as eddy:` so the robot is always stopped and released.
    def __enter__(self) -> "Eddy":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---- entry points --------------------------------------------------------

    def tell(self, text: str, source: str = "typed") -> Result:
        """Carry out a sentence. source says where it came from ("typed", "voice", "web",
        "peer:<name>"): it is kept in the run log, and a peer's commands give way to the
        robot's own operator."""
        commands = intents.parse_all(text)
        if not commands:
            result = Result(False, f"I don't know how to \"{text.strip()}\"." + intents.hint(text))
            if self.log is not None:
                self.log.write("command", source=source, said=text, command=None, ok=False, message=result.message)
            return self._finish(result)
        if len(commands) == 1:
            return self.execute(commands[0], source, text)
        # Several commands in one sentence ("turn left and go forward 50 cm"): run them in
        # order, and do not carry on past one that failed or was interrupted.
        results = []
        for command in commands:
            results.append(self.execute(command, source, text))
            if not results[-1].ok or results[-1].message.startswith(("Interrupted", "Stopped early")):
                break
        message = " Then: ".join(r.message for r in results)
        if len(results) < len(commands):
            message += f" I did not do the remaining {len(commands) - len(results)} step(s)."
        return Result(results[-1].ok, message, results[-1].command)

    def sign(self, landmarks) -> Optional[Result]:
        """Feed one camera frame's hand landmarks. Acts when a sign has been held steadily."""
        # classify() names the sign in this frame; the filter only passes it on once it has
        # been seen for several frames in a row, so a passing hand does not move the robot.
        gesture = self._sign_filter.update(gestures.classify(landmarks) if landmarks else None)
        command = gestures.to_command(gesture)
        if command is None:
            return None
        result = self.execute(command, "sign", gesture)
        result.message = f"[{gesture}] {result.message}"
        return result

    def stop(self, source: str = "program") -> Result:
        """Stop the robot now."""
        return self.execute(Stop(), source)

    def peer_refusal(self) -> Optional[str]:
        """Why a command from another robot may not be carried out right now, or None if it may.
        The robot's own operator comes first: a peer cannot take over a move the operator
        asked for, and cannot restart the robot straight after the operator stopped it."""
        doing = self._doing
        if doing is not None and not doing.startswith("peer"):
            return "it is busy with a command from its own operator"
        wait = self.safety.peer_hold - (time.monotonic() - self._own_stop)
        if wait > 0:
            return f"its own operator stopped it; it takes no commands from peers for another {wait:.0f} s"
        return None

    def execute(self, command, source: str = "program", said: Optional[str] = None) -> Result:
        """Carry out one command. Never raises for an unsupported or unsafe command.
        source: who asked (see tell). said: the sentence or sign it came from, for the run log."""
        if self.model is None:
            self.start()
        if isinstance(command, Stop) and not source.startswith("peer"):
            self._own_stop = time.monotonic()
        if self.log is None:
            return self._execute(command, source)
        # Keep a record: where the robot was, what was asked, what happened, where it ended up.
        # (No reading before a stop: nothing may delay a stop.)
        before = None if isinstance(command, Stop) else self._snapshot()
        began = time.monotonic()
        result = self._execute(command, source)
        self.log.write("command", source=source, said=said, command=type(command).__name__,
                       args=dataclasses.asdict(command) if dataclasses.is_dataclass(command) else None,
                       ok=result.ok, message=result.message, seconds=round(time.monotonic() - began, 3),
                       before=before, after=self._snapshot())
        return result

    def _snapshot(self) -> Optional[dict]:
        """The robot's state for the run log, or None if it cannot be read just now."""
        try:
            return runlog.snapshot(self.adapter.state())
        except Exception:
            return None

    def _execute(self, command, source: str) -> Result:
        """Carry out one command (see execute)."""
        # Take a new epoch number. Any motion still running holds an older number, sees the
        # change on its next tick and gives way. Say and Report do not move the robot, so
        # they leave a running motion alone.
        with self._epoch_lock:
            if not isinstance(command, (Say, Report)):
                self._epoch += 1
            mine = self._epoch
        try:
            if isinstance(command, Stop):
                # Stop goes straight to the adapter without waiting for the motion lock.
                self.adapter.stop()
                message = "Stopped."
            elif isinstance(command, Say):
                self.speaker.say(command.text)
                message = f"Said: {command.text}"
            elif isinstance(command, Report):
                message = self.describe()
                self.speaker.say(message)
            else:
                # Motions run one at a time. While we waited for the lock, a newer command
                # may have arrived; if so this one is already out of date.
                with self._motion:
                    if self._epoch != mine:
                        return self._finish(Result(False, "Cancelled before it started.", command))
                    self._doing = source
                    try:
                        message = self._move(command, mine)
                    finally:
                        self._doing = None
            return self._finish(Result(True, message, command))
        except (Unsupported, Refused) as exc:
            # Expected "no": the robot lacks the ability, or the command is unsafe.
            return self._finish(Result(False, f"I can't do that: {exc}", command))
        except Exception as exc:
            # Anything unexpected: make sure the robot is not left moving.
            try:
                self.adapter.stop()
            except Exception:
                pass
            if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
                # The link to the robot went quiet. Be plain about what is and is not known.
                return self._finish(Result(False, f"I lost contact with the robot ({exc}). I sent a stop, but I "
                                                  "cannot confirm it arrived: check the robot.", command))
            return self._finish(Result(False, f"That failed: {type(exc).__name__}: {exc}", command))

    def _tock(self) -> float:
        """Seconds since the last call (or since the motion began). Control loops add this up
        instead of counting ticks, because each exchange with the robot takes real time too:
        over a slow serial link a loop of 50 ms ticks can take twice as long as it counts."""
        now = self.adapter.now()
        passed, self._mark = now - self._mark, now
        return passed

    def _finish(self, result: Result) -> Result:
        """Tell every listener about a result, then hand it back. A broken listener is ignored."""
        for listener in list(self.listeners):
            try:
                listener(result)
            except Exception:
                pass
        return result

    def describe(self) -> str:
        """A spoken-style status report: what the robot is, where it is, what is ahead."""
        m, s = self.model, self.adapter.state()
        parts = [f"I am Eddy, running {m.name}. I can: {', '.join(m.capabilities()) or 'nothing yet'}."]
        if s.has_odometry:
            parts.append(f"I am at x {s.x:.2f}, y {s.y:.2f}, facing {math.degrees(s.heading):.0f} degrees.")
            if self.model.drive is not None and self.model.drive.kind == FLYING:
                parts.append(f"My height is {s.z:.2f} metres.")
        if s.range_ahead is not None:
            parts.append(f"The nearest thing ahead is {s.range_ahead:.2f} metres away.")
        return " ".join(parts)

    # ---- motions -------------------------------------------------------------

    def _move(self, command, mine: int) -> str:
        """Send a motion command to the method that knows how to do it."""
        if isinstance(command, Drive):
            return self._drive(command, mine)
        if isinstance(command, Slide):
            return self._slide(command, mine)
        if isinstance(command, Wander):
            return self._wander(command, mine)
        if isinstance(command, TakeOff):
            return self._climb(abs(command.height), mine, "Took off")
        if isinstance(command, Climb):
            return self._climb(command.dz, mine, "Climbed" if command.dz > 0 else "Descended")
        if isinstance(command, Land):
            return self._land(mine)
        if isinstance(command, MoveTo):
            return self._reach(command, mine)
        if isinstance(command, MoveBy):
            # Relative move: work out where the tool is now, then reach to that plus the step.
            chain = self.model.chain
            if chain is None:
                raise Unsupported("I don't know this robot's arm geometry, so I can't move the arm that way")
            now = self.adapter.state().joints
            x, y, z = chain.position([now.get(n, 0.0) for n in chain.joint_names])
            return self._reach(MoveTo(float(x) + command.dx, float(y) + command.dy, float(z) + command.dz), mine)
        if isinstance(command, SetJoints):
            return self._joints(command.positions, mine, strict=True)
        if isinstance(command, Home):
            # Home is "every arm joint to zero". Use the arm chain's joints if the geometry
            # is known, otherwise every joint the robot has.
            names = self.model.chain.joint_names if self.model.chain is not None else [j.name for j in self.model.joints]
            if not names:
                raise Unsupported("this robot has no joints to send home")
            return self._joints({n: 0.0 for n in names}, mine, label="Home")
        if isinstance(command, Gripper):
            return self._grip(command, mine)
        raise Unsupported(f"unknown command {type(command).__name__}")

    def _grip(self, command: Gripper, mine: int) -> str:
        """Open or close the gripper, and say only what the robot shows really happened."""
        word = "open" if command.opening >= 0.5 else "closed"
        before = self.adapter.gripper_feedback()
        self.adapter.set_gripper(command.opening)
        # Watch the gripper's reported position until it has moved and settled (at most 2 s;
        # a newer command cuts the wait short). Without a reading, give it the second it is allowed.
        waited, moved, still, last = 0.0, False, 0.0, before
        while waited < (2.0 if before is not None else 1.0) and self._epoch == mine:
            self.adapter.wait(self.safety.tick)
            waited += self.safety.tick
            if before is None:
                continue
            now = self.adapter.gripper_feedback() or {}
            changed = any(abs(now.get(k, v) - v) > 0.002 for k, v in before.items())
            stepping = any(abs(now.get(k, v) - v) > 0.0005 for k, v in (last or {}).items())
            moved = moved or changed
            still = still + self.safety.tick if not stepping else 0.0
            last = now
            if moved and still >= 0.2:
                break
        if before is None:
            return (f"Gripper command sent ({word}). This robot does not report where its gripper is, "
                    "so I cannot confirm it moved.")
        if self._epoch != mine and not moved:
            return "Interrupted before the gripper moved."
        if moved:
            self._grip_state = word
            return f"Gripper {word}."
        already = self.adapter.gripper_at(command.opening)
        if already or (already is None and getattr(self, "_grip_state", None) == word):
            self._grip_state = word
            return f"The gripper is already {word}."
        raise Refused(f"I sent the {'open' if word == 'open' else 'close'} command, but the gripper did not move: "
                      "its joints read the same before and after. It may already have been "
                      f"{word}, or the gripper is not responding. Check it before relying on it.")

    def _limits(self, linear: float, angular: float):
        """Apply the speed scale, then clamp to the robot's own maximum speeds."""
        d, k = self.model.drive, self.safety.speed_scale
        return (max(-d.max_linear, min(d.max_linear, linear * k)),
                max(-d.max_angular, min(d.max_angular, angular * k)))

    def _blocked(self, linear: float) -> bool:
        """True if we are about to drive forward into something the range sensor sees."""
        s = self.adapter.state()
        if linear != 0 and s.range_age is not None and s.range_age > BLIND:
            # The range sensor was working and has gone quiet. Its last reading says nothing
            # about now, so the robot must not be driven on the strength of it.
            raise Refused(f"my range sensor has not reported for {s.range_age:.1f} s, so I will not drive blind")
        if linear < 0:
            # Reversing: only a sensor that also looks behind (a 360-degree laser) can say.
            return s.range_behind is not None and s.range_behind < self.safety.min_range
        if linear == 0:
            return False        # turning on the spot
        return s.range_ahead is not None and s.range_ahead < self.safety.min_range

    def _hold(self, heading0: Optional[float], s=None) -> float:
        """The turn rate that keeps a straight move on `heading0` (None: no heading to hold).
        Never exactly zero (see HOLD_MIN)."""
        if heading0 is not None:
            s = s if s is not None else self.adapter.state()
            if s.has_odometry:
                turn = max(-HOLD, min(HOLD, HOLD_GAIN * _wrap(heading0 - s.heading)))
                if abs(turn) >= HOLD_MIN:
                    return turn
                return math.copysign(HOLD_MIN, turn) if turn else HOLD_MIN
        return HOLD_MIN

    def _drive(self, cmd: Drive, mine: int, trim: bool = True) -> str:
        """Drive a distance, turn an angle, or drive for a time."""
        self._mark = self.adapter.now()     # for _tock(): real time, not a count of ticks
        if self.model.drive is None:
            raise Unsupported("this robot cannot drive")
        # Pick default speeds when only a distance or an angle was given, then limit them.
        lin, ang = cmd.linear, cmd.angular
        if cmd.distance is not None and lin == 0:
            lin = 0.2
        if cmd.angle is not None and ang == 0:
            ang = 0.8
        lin, ang = self._limits(lin, ang)
        # goal: how far to go (metres or radians). budget: how long that should take.
        if cmd.distance is not None:
            goal, budget = abs(cmd.distance), abs(cmd.distance) / max(abs(lin), 1e-6)
        elif cmd.angle is not None:
            goal, budget = abs(cmd.angle), abs(cmd.angle) / max(abs(ang), 1e-6)
        else:
            goal, budget = None, cmd.duration if cmd.duration is not None else 1.0
        # A move that could not finish within the time Eddy allows one command is refused now,
        # rather than driven for a minute and then abandoned somewhere.
        longest = 2 * self.safety.max_drive_seconds
        if budget > longest:
            asked = (f"{goal:.0f} m" if cmd.distance is not None else f"{math.degrees(goal):.0f} degrees") \
                if goal is not None else f"{budget:.0f} s"
            raise Refused(f"{asked} would take about {budget:.0f} s, and I do moves of up to {longest:.0f} s at a "
                          "time. Ask for less, or give it in steps.")
        budget = min(budget, self.safety.max_drive_seconds)
        # A turn with no forward speed asks the robot to turn on the spot. A robot that steers
        # like a car cannot; once Eddy has seen that (below) it creeps forward while turning.
        spot_turn = abs(ang) > 0 and lin == 0
        arc = spot_turn and self.model.drive.kind == CAR
        if arc:
            lin = self._limits(CREEP, 0.0)[0]
            ang = _cap(ang, lin)
        if self._blocked(lin):
            raise Refused(NO_ROOM if arc else
                          "something is too close " + ("behind me" if lin < 0 else "in front of me"))

        start = self.adapter.state()
        # With odometry we measure progress and stop on arrival; without it we just time the move.
        odometry = start.has_odometry and goal is not None
        deadline = budget * 2 + 2 if odometry else budget   # with odometry, time is only a backstop
        if arc and odometry:
            deadline = min(goal / 0.03 + 6, 2 * self.safety.max_drive_seconds)   # turning while rolling is slower
        tick = self.safety.tick
        last_heading, turned, travelled, elapsed, why = start.heading, 0.0, 0.0, 0.0, "done"
        last_x, last_y, path = start.x, start.y, 0.0    # distance along the path actually driven
        small = 0.005 if cmd.distance is not None else math.radians(2)   # less than this is "no progress"
        mark_time, mark_progress = 0.0, 0.0     # progress at the start of the current stall window
        # For a straight drive, the laser reading in the direction of travel at the start of
        # the window: if the odometry says we moved but that distance did not shrink, the
        # wheels are slipping (odometry from wheel rotation cannot tell).
        straight = ang == 0 and lin != 0

        def seen(state):
            if state.range_age is not None and state.range_age > BLIND:
                return None                     # the laser has gone quiet: it cannot be used as a check
            r = state.axis_ahead if lin > 0 else state.axis_behind
            nothing = r is None or (state.range_max is not None and r >= 0.98 * state.range_max)
            return None if nothing else r
        mark_range = seen(start) if straight else None

        def wide(state):
            """The laser's wide view in the direction of travel (the nearest thing in it)."""
            if state.range_age is not None and state.range_age > BLIND:
                return None
            r = state.range_ahead if lin > 0 else state.range_behind
            nothing = r is None or (state.range_max is not None and r >= 0.98 * state.range_max)
            return None if nothing else r
        mark_wide = wide(start) if straight else None
        sent = 0.0                              # the speed last commanded along the goal (m/s or rad/s)
        rate, rate_from = 0.0, None             # the speed the odometry actually shows, and (progress, time) it was last seen at
        mark_scan = start.scan_turn             # the laser's own count of the turn, at the window start
        try:
            # Control loop: every tick, check for a newer command and for obstacles, measure
            # progress, and re-send the velocity (robots with a command timeout need that).
            while elapsed < deadline - 1e-9:
                if self._epoch != mine:
                    why = "interrupted"
                    break
                if self._blocked(lin):
                    why = "blocked"
                    break
                scale = 1.0
                if odometry:
                    s = self.adapter.state()
                    travelled = math.hypot(s.x - start.x, s.y - start.y)
                    # Add up the turn in small steps so turns past 180 degrees count correctly.
                    turned += abs(_wrap(s.heading - last_heading))
                    last_heading = s.heading
                    path += math.hypot(s.x - last_x, s.y - last_y)
                    last_x, last_y = s.x, s.y
                    progress = travelled if cmd.distance is not None else turned
                    # Slow odometry (some robots update it about once a second): between updates,
                    # assume the robot kept doing what it was last told, so it stops on time.
                    # The speed used is the one the odometry itself has shown (never more than
                    # was commanded): a car steering at its limit turns slower than asked.
                    if s.odom_age is not None:
                        at = elapsed - s.odom_age
                        if rate_from is None:
                            rate_from = (progress, at)
                        elif progress != rate_from[0] and at - rate_from[1] > 1e-3:
                            rate, rate_from = min(sent, (progress - rate_from[0]) / (at - rate_from[1])), (progress, at)
                    left = goal - progress - rate * min(s.odom_age or 0.0, 2.0)
                    if left <= 0.004:
                        break
                    # Every STALL seconds, check that the robot got somewhere in that time.
                    if elapsed - mark_time >= STALL:
                        if progress - mark_progress < small:
                            if progress < small and spot_turn and not arc and cmd.angle is not None:
                                # A turn on the spot did nothing at all. Perhaps the robot steers
                                # like a car: try again, rolling forward while turning.
                                if self._blocked(CREEP):
                                    why = "no room"
                                    break
                                arc, lin = True, self._limits(CREEP, 0.0)[0]
                                ang = _cap(ang, lin)
                                mark_time = elapsed
                                deadline = elapsed + min(goal / 0.03 + 6, 2 * self.safety.max_drive_seconds)
                                continue
                            why = "stalled"
                            break
                        if (cmd.angle is not None and not arc and mark_scan is not None and s.scan_turn is not None
                                and (s.range_age is None or s.range_age <= BLIND)):
                            # The same idea for turns: the odometry says we turned, but the laser
                            # scan has not rotated to match. The body is jammed against something.
                            claimed = progress - mark_progress
                            if claimed >= math.radians(20) and abs(s.scan_turn - mark_scan) < 0.3 * claimed:
                                why = "jammed"
                                break
                        mark_scan = s.scan_turn
                        if straight:
                            now_range, now_wide = seen(s), wide(s)
                            covered = progress - mark_progress
                            # Slipping only if neither the narrow beam nor the wide view came closer:
                            # on the real car one narrow beam stayed put while the robot really moved.
                            closed_in = (mark_wide is not None and now_wide is not None
                                         and mark_wide - now_wide >= 0.3 * covered)
                            if (mark_range is not None and now_range is not None and covered >= 0.15
                                    and mark_range - now_range < 0.3 * covered and not closed_in):
                                why = "slipping"
                                break
                            mark_range, mark_wide = now_range, now_wide
                        mark_time, mark_progress = elapsed, progress
                    if arc and self.model.drive.kind != CAR and turned >= math.radians(5):
                        self.model.drive.kind = CAR     # it worked: remember for the next turn
                    ease = 0.15 if cmd.distance is not None else 0.35   # slow down near the goal
                    scale = max(0.2, min(1.0, left / ease))
                # Slowing down near the goal: a robot turning while it rolls keeps its rolling speed
                # and eases only the steering, or it would crawl and never finish.
                turn = ang * scale
                if straight:
                    # Hold the heading where the position is being read anyway; a timed drive only
                    # times the move (an extra reading per tick would slow it on a slow link).
                    turn = self._hold(start.heading, s) if odometry else HOLD_MIN
                self.adapter.set_velocity(lin if arc else lin * scale, turn)
                sent = abs(lin if cmd.distance is not None else ang) * scale
                step = min(tick, deadline - elapsed)
                self.adapter.wait(step)
                elapsed += self._tock()
            else:
                # The loop ran out of time. With odometry that means the goal was never
                # reached (the robot is stuck, lifted, or its odometry is not registering).
                if odometry:
                    why = "timeout"
        finally:
            self.adapter.stop()     # whatever happened, never leave the robot moving
        # Work out what actually happened, for the report. Where the odometry is slow, wait for
        # one more update so the report is the final position, not one from before the stop.
        end = self.adapter.state()
        if odometry:
            # Let the robot come to rest first (a drone or a heavy base coasts on after the
            # stop), so the report is where it ended up, not where it was when told to stop.
            waited, still = 0.0, 0.0
            while waited < 2.0 and still < 0.3:
                self.adapter.wait(tick)
                waited += tick
                now = self.adapter.state()
                moving = (math.hypot(now.x - end.x, now.y - end.y) > 0.002
                          or abs(_wrap(now.heading - end.heading)) > math.radians(0.3))
                still = 0.0 if moving else still + tick
                end = now
        if end.odom_age is not None and odometry:
            waited = 0.0
            while waited < 1.5 and self.adapter.state().odom_age >= max(end.odom_age, 0.25) + waited - 0.05:
                self.adapter.wait(0.05)
                waited += 0.05
            end = self.adapter.state()
        if start.has_odometry:
            moved = math.hypot(end.x - start.x, end.y - start.y)
            if odometry:
                turned += abs(_wrap(end.heading - last_heading))    # include any coasting after the stop
            else:
                turned = abs(_wrap(end.heading - start.heading))
            what = (f"turned {math.degrees(turned):.0f} degrees" if spot_turn
                    else f"moved {moved:.2f} m")
            if not odometry:
                path = moved        # a timed move is not tracked tick by tick
            if arc and self.model.drive.kind == CAR:
                what += f", rolling forward {path:.2f} m because this robot cannot turn on the spot"
        else:
            what = f"{'turned' if spot_turn else 'drove'} for {elapsed:.1f} s (this robot reports no position)"
        if why == "blocked":
            self.speaker.say("Something is in my way.")
            return f"Stopped early, something is in the way ({what})."
        asked = None if goal is None else (f"{goal:.2f} m" if cmd.distance is not None
                                           else f"{math.degrees(goal):.0f} degrees")
        if why == "no room":
            raise Refused(NO_ROOM)
        # A finished turn is checked once more against the laser, over the whole turn: a robot
        # that jammed part of the way round would otherwise be reported as having arrived.
        seen_turn = None
        # (Only for turns on the spot: while the robot also travels, the scan changes for that
        # reason too and cannot be read as a pure rotation.)
        if (cmd.angle is not None and not arc and start.scan_turn is not None and end.scan_turn is not None
                and (end.range_age is None or end.range_age <= BLIND)):
            seen_turn = abs(end.scan_turn - start.scan_turn)
            self.turn_check = {"odometry": turned, "laser": seen_turn}
            # Jammed means the laser saw clearly less turn than the odometry claims: at least
            # SHORT less, and under AGREE of it. Both numbers come from measuring in Gazebo
            # (tests/sims/measure_turns.py): on good turns the laser was within 10 degrees and
            # 5 percent of the truth; a turn that jammed part of the way came out at 79 percent.
            if why == "done" and turned - seen_turn > SHORT and seen_turn < AGREE * turned:
                why = "jammed"
        if why == "jammed":
            really = f" (about {math.degrees(seen_turn):.0f} degrees by the laser)" if seen_turn is not None else ""
            raise Refused(f"I stopped: my odometry says I {what}, but the laser shows the robot did not turn "
                          f"that far{really}. It is probably jammed against something; back away before turning.")
        if why == "slipping":
            where = "ahead" if lin > 0 else "behind"
            raise Refused(f"I stopped: my odometry says I {what}, but the laser shows the distance {where} "
                          "has not changed. The wheels are probably slipping against something.")
        if why == "stalled":
            if mark_progress >= small:
                # It was moving and then stopped getting anywhere: most likely pushing on something.
                raise Refused(f"I stopped because the robot was no longer getting anywhere: I only {what} "
                              f"of the {asked} asked for. Something may be blocking it.")
            if spot_turn:
                raise Refused("the turn had no effect: the heading did not change, on the spot or while "
                              "rolling forward. Either the wheels are not turning the robot, or its "
                              "odometry does not report its heading.")
            raise Refused("the command had no effect: the robot's odometry shows no movement. "
                          "Is it free to move, and is this the right velocity topic?")
        if why == "timeout":
            raise Refused(f"I ran out of time: I only {what} of the {asked} asked for. "
                          "Is the robot free to move, and is its odometry registering the motion?")
        if why == "interrupted":
            return f"Interrupted ({what})."
        # A robot that is slow to respond (a drone, a heavy base, slow odometry) ends a turn on
        # the spot past the angle asked for. Turn back the difference once, slowly.
        over = turned - goal if (trim and cmd.angle is not None and odometry and not arc) else 0.0
        # Not short enough to call a jam, but the laser saw clearly less than the odometry: say so,
        # so the figure reported is not taken as certain.
        doubt = ""
        if seen_turn is not None and turned - seen_turn > SHORT:
            doubt = (f" The laser saw only about {math.degrees(seen_turn):.0f} degrees, so the robot may have "
                     "turned less than its odometry says.")
        if over > math.radians(4) and self._epoch == mine:
            before = self.adapter.state().heading
            try:
                self._drive(Drive(angular=-math.copysign(min(abs(ang), 0.3), ang), angle=over), mine, trim=False)
            except Refused:
                pass            # the correction could not be made: report what the turn really was
            turned -= abs(_wrap(self.adapter.state().heading - before))
            what = f"turned {math.degrees(turned):.0f} degrees"
        return f"Done: {what}.{doubt}"

    def _slide(self, cmd: Slide, mine: int) -> str:
        """Move sideways by a distance without turning, measured by the robot's own position.

        Nothing on a robot says whether its base can move sideways (the speed message has a
        sideways field on every robot; most bases ignore it). So Eddy finds out the way it finds
        out that a robot cannot turn on the spot: it tries, watches the position, and remembers.
        """
        self._mark = self.adapter.now()     # for _tock(): real time, not a count of ticks
        drive = self.model.drive
        if drive is None:
            raise Unsupported("this robot cannot drive")
        cannot = "this robot cannot move sideways (only bases with mecanum or omni wheels can)"
        if drive.kind == CAR or drive.slides is False:
            raise Refused(cannot + ("; it steers like a car" if drive.kind == CAR else "; I tried earlier and it did not"))
        goal, side = abs(cmd.distance), (1.0 if cmd.distance > 0 else -1.0)
        if goal > SIDE + 1e-9:
            raise Refused(f"I cannot see to the side, so I slide at most {SIDE:.0f} m at a time. Ask for less.")
        start = self.adapter.state()
        if not start.has_odometry:
            raise Refused("I will not move sideways without a position reading: I could not tell how far the robot went")
        speed = min(abs(cmd.speed) * self.safety.speed_scale, drive.max_linear)
        deadline = min(goal / max(speed, 1e-6) * 2 + 3, 2 * self.safety.max_drive_seconds)
        sin0, cos0 = math.sin(start.heading), math.cos(start.heading)

        def moved():
            """How far the robot has gone sideways (in the direction asked) and forwards since the start."""
            s = self.adapter.state()
            dx, dy = s.x - start.x, s.y - start.y
            return side * (dy * cos0 - dx * sin0), dx * cos0 + dy * sin0

        elapsed, done, along, mark_time, mark_done, why = 0.0, 0.0, 0.0, 0.0, 0.0, "timeout"
        try:
            while elapsed < deadline:
                if self._epoch != mine:
                    why = "interrupted"
                    break
                done, along = moved()
                if goal - done <= 0.004:
                    why = "done"
                    break
                if abs(along) > 0.05 and abs(along) > 2 * abs(done):
                    why = "wrong way"       # it is rolling forwards or backwards instead
                    break
                if elapsed - mark_time >= STALL:
                    if done - mark_done < 0.005:
                        why = "stalled"
                        break
                    mark_time, mark_done = elapsed, done
                scale = max(0.25, min(1.0, (goal - done) / 0.12))      # slow down near the distance asked for
                self.adapter.set_velocity(0.0, 0.0, lateral=side * speed * scale)
                self.adapter.wait(self.safety.tick)
                elapsed += self._tock()
        finally:
            self.adapter.stop()
        self.adapter.wait(0.3)              # let it come to rest before measuring
        done, along = moved()
        way = "left" if side > 0 else "right"
        if done >= 0.03 and why != "wrong way":
            drive.slides = True             # it moved sideways on command: remember that it can
        if why == "wrong way":
            drive.slides = False
            raise Refused(f"I stopped: asked to slide {way}, the robot rolled {abs(along):.2f} m "
                          f"{'forwards' if along > 0 else 'backwards'} instead. It does not move sideways the way I expected.")
        if why == "stalled" and done < 0.03:
            if drive.slides:
                raise Refused(f"the robot is not moving {way}: something may be blocking it on that side")
            drive.slides = False
            raise Refused("commanding the robot sideways changed nothing. It does not seem to be able to move "
                          "sideways (only bases with mecanum or omni wheels can).")
        if why in ("stalled", "timeout"):
            raise Refused(f"I stopped because the robot was no longer getting anywhere: I only slid {done:.2f} m "
                          f"of the {goal:.2f} m asked for. Something may be blocking it on that side.")
        if why == "interrupted":
            return f"Interrupted (slid {done:.2f} m {way})."
        return f"Done: slid {done:.2f} m to the {way}."

    def _climb(self, dz: float, mine: int, label: str) -> str:
        """Go straight up or down by dz metres, measured by the robot's own height reading.

        Nothing on a robot says whether it flies, so Eddy finds out the way it finds out that
        a robot cannot turn on the spot: it tries, and if the height does not change it says so.
        """
        self._mark = self.adapter.now()     # for _tock(): real time, not a count of ticks
        if self.model.drive is None:
            raise Unsupported("this robot cannot move")
        start = self.adapter.state()
        if not start.has_odometry:
            raise Refused("I will not fly without a position reading that tells me my height")
        goal = abs(dz)
        speed = min(CLIMB, self.model.drive.max_linear) * self.safety.speed_scale * (1 if dz > 0 else -1)
        deadline = min(goal / abs(speed) * 2 + 4, 2 * self.safety.max_drive_seconds)
        elapsed, moved, mark_time, mark_moved, why = 0.0, 0.0, 0.0, 0.0, "timeout"
        try:
            while elapsed < deadline:
                if self._epoch != mine:
                    why = "interrupted"
                    break
                moved = abs(self.adapter.state().z - start.z)
                if goal - moved <= 0.01:
                    why = "done"
                    break
                if elapsed - mark_time >= STALL:
                    if moved - mark_moved < 0.02:
                        why = "stalled"
                        break
                    mark_time, mark_moved = elapsed, moved
                scale = max(0.25, min(1.0, (goal - moved) / 0.3))      # slow down near the height asked for
                self.adapter.set_velocity(0.0, 0.0, speed * scale)
                self.adapter.wait(self.safety.tick)
                elapsed += self._tock()
        finally:
            self.adapter.stop()         # zero velocity: a flying robot hovers
        end = self.adapter.state()
        moved = abs(end.z - start.z)
        if moved >= 0.05:
            self.model.drive.kind = FLYING      # it changed height on command: remember that it flies
        if why == "stalled" and moved < 0.02:
            if dz < 0:
                return "I am not going any lower: I seem to be on the ground already."
            raise Refused("commanding the robot upwards changed nothing. It does not seem to be a flying robot "
                          "(or its motors are not enabled).")
        if why in ("stalled", "timeout"):
            raise Refused(f"I only changed height by {moved:.2f} m of the {goal:.2f} m asked for; my height is now {end.z:.2f} m")
        if why == "interrupted":
            return f"Interrupted (height now {end.z:.2f} m)."
        return f"{label} {moved:.2f} m. My height is now {end.z:.2f} m."

    def _land(self, mine: int) -> str:
        """Descend until the height stops falling, then stop."""
        self._mark = self.adapter.now()     # for _tock(): real time, not a count of ticks
        if self.model.drive is None:
            raise Unsupported("this robot cannot move")
        start = self.adapter.state()
        if not start.has_odometry:
            raise Refused("I will not fly without a position reading that tells me my height")
        speed = -min(CLIMB, self.model.drive.max_linear) * self.safety.speed_scale
        elapsed, mark_time, mark_z = 0.0, 0.0, start.z
        try:
            while elapsed < 2 * self.safety.max_drive_seconds:
                if self._epoch != mine:
                    return f"Interrupted (height now {self.adapter.state().z:.2f} m)."
                z = self.adapter.state().z
                if elapsed - mark_time >= STALL / 2:
                    if mark_z - z < 0.02:
                        break                    # not going down any more: on the ground
                    mark_time, mark_z = elapsed, z
                self.adapter.set_velocity(0.0, 0.0, speed)
                self.adapter.wait(self.safety.tick)
                elapsed += self._tock()
        finally:
            self.adapter.stop()
        end = self.adapter.state()
        if start.z - end.z < 0.02:
            return "I am already on the ground."
        return f"Landed: came down {start.z - end.z:.2f} m."

    def _wander(self, cmd: Wander, mine: int) -> str:
        """Simple autonomy: go forward, and turn away whenever something is close ahead."""
        self._mark = self.adapter.now()     # for _tock(): real time, not a count of ticks
        if self.model.drive is None:
            raise Unsupported("this robot cannot drive")
        if "range" not in self.model.sensors:
            raise Refused("I will not wander without a distance sensor facing forward")
        duration = min(cmd.duration, 600.0)                 # never wander more than ten minutes
        lin, ang = self._limits(0.2, 0.8)
        caution = max(self.safety.min_range * 2, 0.5)       # start turning well before the hard limit
        elapsed, turns, tick = 0.0, 0, self.safety.tick
        turning, turn_done, turn_time = False, 0.0, 0.0     # the turn in progress, if any
        backing = True                                      # a car-like robot starts its turn by backing away
        away = math.pi * 0.6                                # turn about 110 degrees away
        last_heading = self.adapter.state().heading
        ahead_from = None                                   # the heading the current straight stretch holds
        try:
            while elapsed < duration:
                if self._epoch != mine:
                    return f"Interrupted after wandering {elapsed:.0f} s."
                s = self.adapter.state()
                if s.range_age is not None and s.range_age > BLIND:
                    raise Refused(f"my range sensor went quiet after {elapsed:.0f} s of wandering, so I stopped")
                car = self.model.drive.kind == CAR
                if turning:
                    turn_done += abs(_wrap(s.heading - last_heading))
                    if (turn_done >= away) if s.has_odometry else (turn_time >= away / ang):
                        turning, turns = False, turns + 1
                    elif s.has_odometry and turn_time >= STALL and turn_done < math.radians(2):
                        if car:
                            raise Refused(f"I got stuck after wandering {elapsed:.0f} s: I could not turn "
                                          "away from what is in front of me")
                        # Turning on the spot did nothing: treat the robot as car-like from now on.
                        self.model.drive.kind, turn_time = CAR, 0.0
                    elif turn_time > away / 0.03 + 6:
                        raise Refused(f"I got stuck after wandering {elapsed:.0f} s: the turn never finished")
                    elif car and backing and s.range_behind is not None and s.range_behind < self.safety.min_range:
                        # No room left behind: carry on with the turn rolling forwards instead
                        # (a three-point turn), unless that way is blocked too.
                        if s.range_ahead is not None and s.range_ahead < self.safety.min_range:
                            raise Refused(f"I got stuck after wandering {elapsed:.0f} s: no room in front or behind to turn")
                        backing = False
                    elif car and not backing and s.range_ahead is not None and s.range_ahead < self.safety.min_range:
                        backing = True
                    else:
                        # A car-like robot backs away while steering; others turn on the spot.
                        creep = self._limits(CREEP, 0.0)[0]
                        self.adapter.set_velocity((-creep if backing else creep) if car else 0.0,
                                                  _cap(ang, creep) if car else ang)
                        turn_time += tick
                elif s.range_ahead is not None and s.range_ahead < caution:
                    turning, turn_done, turn_time, backing = True, 0.0, 0.0, True   # something is close: turn away
                    ahead_from = None
                else:
                    if ahead_from is None:
                        ahead_from = s.heading
                    # Path is clear: go forward, holding the heading this stretch began on.
                    self.adapter.set_velocity(lin, self._hold(ahead_from if s.has_odometry else None, s))
                last_heading = s.heading
                self.adapter.wait(tick)
                elapsed += self._tock()
        finally:
            self.adapter.stop()
        return f"Wandered for {elapsed:.0f} s and turned away from obstacles {turns} times."

    def _reach(self, cmd: MoveTo, mine: int) -> str:
        """Move the arm's tool to a point, using inverse kinematics."""
        chain = self.model.chain
        if chain is None:
            raise Unsupported("I don't know this robot's arm geometry, so I can't reach to a point")
        target = (cmd.x, cmd.y, cmd.z)
        # Cheap check first: a point further away than the arm is long can never be reached.
        if math.sqrt(sum(v * v for v in target)) > chain.reach() + 1e-6:
            raise Refused(f"that point is beyond my reach of {chain.reach():.2f} m")
        # Solve starting from where the arm is now, so it takes a nearby solution.
        now = self.adapter.state().joints
        q0 = [now.get(n, 0.0) for n in chain.joint_names]
        result = chain.ik(target, q0=q0)
        if not result.success:
            raise Refused(f"I can't reach that point (closest I can get is {result.error * 100:.1f} cm away)")
        done = self._joints(result.as_dict(chain), mine)
        if done.startswith("Interrupted"):
            # A stop or a newer command cut the move short: say where the arm really is, not "reached".
            now = self.adapter.state().joints
            gap = math.dist(target, chain.position([now.get(n, 0.0) for n in chain.joint_names]))
            return (f"Interrupted: the arm stopped {gap * 100:.0f} cm short of x {cmd.x:.2f}, y {cmd.y:.2f}, "
                    f"z {cmd.z:.2f} and is holding there.")
        return f"Reached x {cmd.x:.2f}, y {cmd.y:.2f}, z {cmd.z:.2f}."

    def _joints(self, positions: dict, mine: int, label: str = "Joints", strict: bool = False) -> str:
        """Move named joints to target positions and wait until they get there."""
        self._mark = self.adapter.now()     # for _tock(): real time, not a count of ticks
        if not self.model.joints:
            raise Unsupported("this robot has no joints")
        # Check every name and clamp every value to that joint's limits.
        targets = {}
        for name, value in positions.items():
            spec = self.model.find_joint(name)
            if spec is None:
                known = ", ".join(j.name for j in self.model.joints)
                raise Refused(f"there is no single joint called {name} (this robot has: {known})")
            if strict and not spec.lower - 1e-6 <= float(value) <= spec.upper + 1e-6:
                # A person asked for a position this joint cannot take: say so instead of
                # quietly going somewhere else.
                unit, k = ("m", 1.0) if spec.kind == "prismatic" else ("degrees", 180 / math.pi)
                raise Refused(f"{spec.name} can only go from {spec.lower * k:.3g} to {spec.upper * k:.3g} {unit}, "
                              f"and {float(value) * k:.3g} was asked for")
            targets[spec.name] = max(spec.lower, min(spec.upper, float(value)))
        # Give the move enough time that the joint with furthest to go stays under 1 rad/s.
        now = self.adapter.state().joints
        biggest = max((abs(targets[n] - now.get(n, 0.0)) for n in targets), default=0.0)
        duration = max(0.5, biggest / 1.0)      # no faster than 1 rad/s
        self.adapter.set_joints(targets, duration)
        # With joint feedback, wait until the joints arrive (or give up after a time limit).
        # Without it, just wait for the planned duration.
        feedback = "joint_state" in self.model.sensors
        elapsed, limit = 0.0, duration * 2 + 1.0
        while elapsed < (limit if feedback else duration):
            if self._epoch != mine:
                # A newer command arrived: stop the joints where they are, and wait for the arm to
                # come to rest, so what is reported is where it really stopped.
                self.adapter.hold_joints(list(targets))
                if feedback:
                    last, still, waited = self.adapter.state().joints, 0.0, 0.0
                    while waited < 0.6 and still < 0.1:
                        self.adapter.wait(self.safety.tick)
                        waited += self.safety.tick
                        now_j = self.adapter.state().joints
                        moving = any(abs(now_j.get(n, 0.0) - last.get(n, 0.0)) > 0.002 for n in targets)
                        still = 0.0 if moving else still + self.safety.tick
                        last = now_j
                return f"Interrupted ({label.lower()} move stopped part way; the arm is holding where it is)."
            if feedback:
                current = self.adapter.state().joints
                if all(abs(current.get(n, 1e9) - targets[n]) < 0.02 for n in targets):
                    break
            self.adapter.wait(self.safety.tick)
            elapsed += self._tock()
        else:
            # The loop ran out of time without a break, i.e. the joints never arrived.
            if feedback:
                # Say what each joint actually did, so it is clear whether the arm never
                # moved, moved part of the way, or arrived but outside the tolerance.
                current = self.adapter.state().joints
                detail = "; ".join(
                    f"{n} went from {math.degrees(now.get(n, 0.0)):.1f} to "
                    f"{math.degrees(current.get(n, float('nan'))):.1f} degrees, wanted {math.degrees(targets[n]):.1f}"
                    for n in targets if abs(current.get(n, 1e9) - targets[n]) >= 0.02)
                raise Refused(f"the joints did not reach their targets in time ({detail})")
        return f"{label} set."
