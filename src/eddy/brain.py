"""Eddy itself: takes commands from anywhere and carries them out safely on any robot.

Commands can come from typed text, speech, hand signs, a program, or another
robot. Eddy checks each against what the attached robot can actually do, applies
the safety limits, and drives the adapter. A newer command always replaces the
one in progress, and Stop always wins.
"""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Callable, List, Optional

from . import gestures, intents
from .adapters.base import Adapter
from .model import (Drive, Gripper, Home, MoveTo, Refused, Report, RobotModel, Say, SetJoints, Stop,
                    Unsupported, Wander)
from .senses.speak import Speaker


@dataclass
class Safety:
    """The limits Eddy applies to every motion, whatever asked for it."""
    min_range: float = 0.25          # never drive forward with something closer than this (metres)
    max_drive_seconds: float = 30.0  # no single drive command runs longer than this
    speed_scale: float = 1.0         # turn everything down with one number
    tick: float = 0.05               # how often commands are re-sent and sensors re-checked


@dataclass
class Result:
    """What happened to a command: whether it worked, and a sentence saying so."""
    ok: bool
    message: str
    command: object = None

    def __str__(self) -> str:
        return self.message


def _wrap(a: float) -> float:
    """Bring an angle into -pi..pi, so heading differences across the +/-180 line stay small."""
    return math.atan2(math.sin(a), math.cos(a))


class Eddy:
    """The controller. Give it an adapter (a robot), then call tell(), sign() or execute()."""

    def __init__(self, adapter: Adapter, speaker: Optional[Speaker] = None, safety: Optional[Safety] = None):
        self.adapter = adapter
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
        return self.model

    def close(self) -> None:
        """Stop the robot and disconnect."""
        self.stop()
        self.adapter.close()

    # Support `with Eddy(adapter) as eddy:` so the robot is always stopped and released.
    def __enter__(self) -> "Eddy":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---- entry points --------------------------------------------------------

    def tell(self, text: str) -> Result:
        """Carry out a sentence."""
        command = intents.parse(text)
        if command is None:
            return self._finish(Result(False, f"I don't know how to \"{text.strip()}\"."))
        return self.execute(command)

    def sign(self, landmarks) -> Optional[Result]:
        """Feed one camera frame's hand landmarks. Acts when a sign has been held steadily."""
        # classify() names the sign in this frame; the filter only passes it on once it has
        # been seen for several frames in a row, so a passing hand does not move the robot.
        gesture = self._sign_filter.update(gestures.classify(landmarks) if landmarks else None)
        command = gestures.to_command(gesture)
        if command is None:
            return None
        result = self.execute(command)
        result.message = f"[{gesture}] {result.message}"
        return result

    def stop(self) -> Result:
        """Stop the robot now."""
        return self.execute(Stop())

    def execute(self, command) -> Result:
        """Carry out one command. Never raises for an unsupported or unsafe command."""
        if self.model is None:
            self.start()
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
                    message = self._move(command, mine)
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
            return self._finish(Result(False, f"That failed: {type(exc).__name__}: {exc}", command))

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
        if s.range_ahead is not None:
            parts.append(f"The nearest thing ahead is {s.range_ahead:.2f} metres away.")
        return " ".join(parts)

    # ---- motions -------------------------------------------------------------

    def _move(self, command, mine: int) -> str:
        """Send a motion command to the method that knows how to do it."""
        if isinstance(command, Drive):
            return self._drive(command, mine)
        if isinstance(command, Wander):
            return self._wander(command, mine)
        if isinstance(command, MoveTo):
            return self._reach(command, mine)
        if isinstance(command, SetJoints):
            return self._joints(command.positions, mine)
        if isinstance(command, Home):
            # Home is "every arm joint to zero". Use the arm chain's joints if the geometry
            # is known, otherwise every joint the robot has.
            names = self.model.chain.joint_names if self.model.chain is not None else [j.name for j in self.model.joints]
            if not names:
                raise Unsupported("this robot has no joints to send home")
            return self._joints({n: 0.0 for n in names}, mine, label="Home")
        if isinstance(command, Gripper):
            self.adapter.set_gripper(command.opening)
            return "Gripper open." if command.opening >= 0.5 else "Gripper closed."
        raise Unsupported(f"unknown command {type(command).__name__}")

    def _limits(self, linear: float, angular: float):
        """Apply the speed scale, then clamp to the robot's own maximum speeds."""
        d, k = self.model.drive, self.safety.speed_scale
        return (max(-d.max_linear, min(d.max_linear, linear * k)),
                max(-d.max_angular, min(d.max_angular, angular * k)))

    def _blocked(self, linear: float) -> bool:
        """True if we are about to drive forward into something the range sensor sees."""
        if linear <= 0:
            return False        # reversing or turning on the spot: the forward sensor says nothing
        ahead = self.adapter.state().range_ahead
        return ahead is not None and ahead < self.safety.min_range

    def _drive(self, cmd: Drive, mine: int) -> str:
        """Drive a distance, turn an angle, or drive for a time."""
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
        budget = min(budget, self.safety.max_drive_seconds)
        if self._blocked(lin):
            raise Refused("something is too close in front of me")

        start = self.adapter.state()
        # With odometry we measure progress and stop on arrival; without it we just time the move.
        odometry = start.has_odometry and goal is not None
        deadline = budget * 2 + 2 if odometry else budget   # with odometry, time is only a backstop
        tick = self.safety.tick
        last_heading, turned, travelled, elapsed, why = start.heading, 0.0, 0.0, 0.0, "done"
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
                    left = goal - (travelled if cmd.distance is not None else turned)
                    if left <= 0.004:
                        break
                    ease = 0.15 if cmd.distance is not None else 0.35   # slow down near the goal
                    scale = max(0.2, min(1.0, left / ease))
                self.adapter.set_velocity(lin * scale, ang * scale)
                step = min(tick, deadline - elapsed)
                self.adapter.wait(step)
                elapsed += step
            else:
                # The loop ran out of time. With odometry that means the goal was never
                # reached (the robot is stuck, lifted, or its odometry is not registering).
                if odometry:
                    why = "timeout"
        finally:
            self.adapter.stop()     # whatever happened, never leave the robot moving
        # Work out what actually happened, for the report.
        end = self.adapter.state()
        if start.has_odometry:
            moved = math.hypot(end.x - start.x, end.y - start.y)
            if odometry:
                turned += abs(_wrap(end.heading - last_heading))    # include any coasting after the stop
            else:
                turned = abs(_wrap(end.heading - start.heading))
            what = (f"turned {math.degrees(turned):.0f} degrees" if abs(ang) > 0 and abs(lin) == 0
                    else f"moved {moved:.2f} m")
        else:
            what = f"drove for {elapsed:.1f} s"
        if why == "blocked":
            self.speaker.say("Something is in my way.")
            return f"Stopped early, something is in the way ({what})."
        if why == "timeout":
            asked = f"{goal:.2f} m" if cmd.distance is not None else f"{math.degrees(goal):.0f} degrees"
            raise Refused(f"I ran out of time: I only {what} of the {asked} asked for. "
                          "Is the robot free to move, and is its odometry registering the motion?")
        if why == "interrupted":
            return f"Interrupted ({what})."
        return f"Done: {what}."

    def _wander(self, cmd: Wander, mine: int) -> str:
        """Simple autonomy: go forward, and turn away whenever something is close ahead."""
        if self.model.drive is None:
            raise Unsupported("this robot cannot drive")
        if "range" not in self.model.sensors:
            raise Refused("I will not wander without a distance sensor facing forward")
        duration = min(cmd.duration, 600.0)                 # never wander more than ten minutes
        lin, ang = self._limits(0.2, 0.8)
        caution = max(self.safety.min_range * 2, 0.5)       # start turning well before the hard limit
        elapsed, turns = 0.0, 0
        turning_until = -1.0                                # time at which the current turn ends
        try:
            while elapsed < duration:
                if self._epoch != mine:
                    return f"Interrupted after wandering {elapsed:.0f} s."
                ahead = self.adapter.state().range_ahead
                if elapsed < turning_until:
                    # In the middle of a turn: keep turning.
                    self.adapter.set_velocity(0.0, ang)
                elif ahead is not None and ahead < caution:
                    # Something is close: start a new turn.
                    turns += 1
                    turning_until = elapsed + (math.pi * 0.6) / ang    # turn about 110 degrees away
                    self.adapter.set_velocity(0.0, ang)
                else:
                    # Path is clear: go forward.
                    self.adapter.set_velocity(lin, 0.0)
                self.adapter.wait(self.safety.tick)
                elapsed += self.safety.tick
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
        self._joints(result.as_dict(chain), mine)
        return f"Reached x {cmd.x:.2f}, y {cmd.y:.2f}, z {cmd.z:.2f}."

    def _joints(self, positions: dict, mine: int, label: str = "Joints") -> str:
        """Move named joints to target positions and wait until they get there."""
        if not self.model.joints:
            raise Unsupported("this robot has no joints")
        # Check every name and clamp every value to that joint's limits.
        targets = {}
        for name, value in positions.items():
            spec = self.model.joint(name)
            if spec is None:
                raise Refused(f"there is no joint called {name}")
            targets[name] = max(spec.lower, min(spec.upper, float(value)))
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
                # A newer command arrived: command the joints to stay where they are now.
                current = self.adapter.state().joints
                self.adapter.set_joints({n: current.get(n, targets[n]) for n in targets}, 0.2)   # hold still
                return f"{label} move interrupted."
            if feedback:
                current = self.adapter.state().joints
                if all(abs(current.get(n, 1e9) - targets[n]) < 0.02 for n in targets):
                    break
            self.adapter.wait(self.safety.tick)
            elapsed += self.safety.tick
        else:
            # The loop ran out of time without a break, i.e. the joints never arrived.
            if feedback:
                raise Refused("the joints did not reach their targets in time")
        return f"{label} set."
