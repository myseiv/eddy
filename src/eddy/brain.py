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
    min_range: float = 0.25          # never drive forward with something closer than this (metres)
    max_drive_seconds: float = 30.0  # no single drive command runs longer than this
    speed_scale: float = 1.0         # turn everything down with one number
    tick: float = 0.05               # how often commands are re-sent and sensors re-checked


@dataclass
class Result:
    ok: bool
    message: str
    command: object = None

    def __str__(self) -> str:
        return self.message


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


class Eddy:
    def __init__(self, adapter: Adapter, speaker: Optional[Speaker] = None, safety: Optional[Safety] = None):
        self.adapter = adapter
        self.speaker = speaker or Speaker("print")
        self.safety = safety or Safety()
        self.model: Optional[RobotModel] = None
        self.listeners: List[Callable[[Result], None]] = []
        self._epoch = 0                       # bumped by every new command; a running one notices and yields
        self._epoch_lock = threading.Lock()
        self._motion = threading.Lock()       # one motion at a time
        self._sign_filter = gestures.GestureFilter()

    # ---- lifecycle -----------------------------------------------------------

    def start(self) -> RobotModel:
        self.model = self.adapter.connect()
        return self.model

    def close(self) -> None:
        self.stop()
        self.adapter.close()

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
        gesture = self._sign_filter.update(gestures.classify(landmarks) if landmarks else None)
        command = gestures.to_command(gesture)
        if command is None:
            return None
        result = self.execute(command)
        result.message = f"[{gesture}] {result.message}"
        return result

    def stop(self) -> Result:
        return self.execute(Stop())

    def execute(self, command) -> Result:
        """Carry out one command. Never raises for an unsupported or unsafe command."""
        if self.model is None:
            self.start()
        with self._epoch_lock:
            if not isinstance(command, (Say, Report)):
                self._epoch += 1
            mine = self._epoch
        try:
            if isinstance(command, Stop):
                self.adapter.stop()
                message = "Stopped."
            elif isinstance(command, Say):
                self.speaker.say(command.text)
                message = f"Said: {command.text}"
            elif isinstance(command, Report):
                message = self.describe()
                self.speaker.say(message)
            else:
                with self._motion:
                    if self._epoch != mine:
                        return self._finish(Result(False, "Cancelled before it started.", command))
                    message = self._move(command, mine)
            return self._finish(Result(True, message, command))
        except (Unsupported, Refused) as exc:
            return self._finish(Result(False, f"I can't do that: {exc}", command))
        except Exception as exc:
            try:
                self.adapter.stop()
            except Exception:
                pass
            return self._finish(Result(False, f"That failed: {type(exc).__name__}: {exc}", command))

    def _finish(self, result: Result) -> Result:
        for listener in list(self.listeners):
            try:
                listener(result)
            except Exception:
                pass
        return result

    def describe(self) -> str:
        m, s = self.model, self.adapter.state()
        parts = [f"I am Eddy, running {m.name}. I can: {', '.join(m.capabilities()) or 'nothing yet'}."]
        if s.has_odometry:
            parts.append(f"I am at x {s.x:.2f}, y {s.y:.2f}, facing {math.degrees(s.heading):.0f} degrees.")
        if s.range_ahead is not None:
            parts.append(f"The nearest thing ahead is {s.range_ahead:.2f} metres away.")
        return " ".join(parts)

    # ---- motions -------------------------------------------------------------

    def _move(self, command, mine: int) -> str:
        if isinstance(command, Drive):
            return self._drive(command, mine)
        if isinstance(command, Wander):
            return self._wander(command, mine)
        if isinstance(command, MoveTo):
            return self._reach(command, mine)
        if isinstance(command, SetJoints):
            return self._joints(command.positions, mine)
        if isinstance(command, Home):
            names = self.model.chain.joint_names if self.model.chain is not None else [j.name for j in self.model.joints]
            if not names:
                raise Unsupported("this robot has no joints to send home")
            return self._joints({n: 0.0 for n in names}, mine, label="Home")
        if isinstance(command, Gripper):
            self.adapter.set_gripper(command.opening)
            return "Gripper open." if command.opening >= 0.5 else "Gripper closed."
        raise Unsupported(f"unknown command {type(command).__name__}")

    def _limits(self, linear: float, angular: float):
        d, k = self.model.drive, self.safety.speed_scale
        return (max(-d.max_linear, min(d.max_linear, linear * k)),
                max(-d.max_angular, min(d.max_angular, angular * k)))

    def _blocked(self, linear: float) -> bool:
        if linear <= 0:
            return False
        ahead = self.adapter.state().range_ahead
        return ahead is not None and ahead < self.safety.min_range

    def _drive(self, cmd: Drive, mine: int) -> str:
        if self.model.drive is None:
            raise Unsupported("this robot cannot drive")
        lin, ang = cmd.linear, cmd.angular
        if cmd.distance is not None and lin == 0:
            lin = 0.2
        if cmd.angle is not None and ang == 0:
            ang = 0.8
        lin, ang = self._limits(lin, ang)
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
        odometry = start.has_odometry and goal is not None
        deadline = budget * 2 + 2 if odometry else budget   # with odometry, time is only a backstop
        tick = self.safety.tick
        last_heading, turned, travelled, elapsed, why = start.heading, 0.0, 0.0, 0.0, "done"
        try:
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
        finally:
            self.adapter.stop()
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
        if why == "interrupted":
            return f"Interrupted ({what})."
        return f"Done: {what}."

    def _wander(self, cmd: Wander, mine: int) -> str:
        """Simple autonomy: go forward, and turn away whenever something is close ahead."""
        if self.model.drive is None:
            raise Unsupported("this robot cannot drive")
        if "range" not in self.model.sensors:
            raise Refused("I will not wander without a distance sensor facing forward")
        duration = min(cmd.duration, 600.0)
        lin, ang = self._limits(0.2, 0.8)
        caution = max(self.safety.min_range * 2, 0.5)
        elapsed, turns = 0.0, 0
        turning_until = -1.0
        try:
            while elapsed < duration:
                if self._epoch != mine:
                    return f"Interrupted after wandering {elapsed:.0f} s."
                ahead = self.adapter.state().range_ahead
                if elapsed < turning_until:
                    self.adapter.set_velocity(0.0, ang)
                elif ahead is not None and ahead < caution:
                    turns += 1
                    turning_until = elapsed + (math.pi * 0.6) / ang    # turn about 110 degrees away
                    self.adapter.set_velocity(0.0, ang)
                else:
                    self.adapter.set_velocity(lin, 0.0)
                self.adapter.wait(self.safety.tick)
                elapsed += self.safety.tick
        finally:
            self.adapter.stop()
        return f"Wandered for {elapsed:.0f} s and turned away from obstacles {turns} times."

    def _reach(self, cmd: MoveTo, mine: int) -> str:
        chain = self.model.chain
        if chain is None:
            raise Unsupported("I don't know this robot's arm geometry, so I can't reach to a point")
        target = (cmd.x, cmd.y, cmd.z)
        if math.sqrt(sum(v * v for v in target)) > chain.reach() + 1e-6:
            raise Refused(f"that point is beyond my reach of {chain.reach():.2f} m")
        now = self.adapter.state().joints
        q0 = [now.get(n, 0.0) for n in chain.joint_names]
        result = chain.ik(target, q0=q0)
        if not result.success:
            raise Refused(f"I can't reach that point (closest I can get is {result.error * 100:.1f} cm away)")
        self._joints(result.as_dict(chain), mine)
        return f"Reached x {cmd.x:.2f}, y {cmd.y:.2f}, z {cmd.z:.2f}."

    def _joints(self, positions: dict, mine: int, label: str = "Joints") -> str:
        if not self.model.joints:
            raise Unsupported("this robot has no joints")
        targets = {}
        for name, value in positions.items():
            spec = self.model.joint(name)
            if spec is None:
                raise Refused(f"there is no joint called {name}")
            targets[name] = max(spec.lower, min(spec.upper, float(value)))
        now = self.adapter.state().joints
        biggest = max((abs(targets[n] - now.get(n, 0.0)) for n in targets), default=0.0)
        duration = max(0.5, biggest / 1.0)      # no faster than 1 rad/s
        self.adapter.set_joints(targets, duration)
        feedback = "joint_state" in self.model.sensors
        elapsed, limit = 0.0, duration * 2 + 1.0
        while elapsed < (limit if feedback else duration):
            if self._epoch != mine:
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
            if feedback:
                raise Refused("the joints did not reach their targets in time")
        return f"{label} set."
