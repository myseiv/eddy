"""A simulated robot: a two-wheel rover carrying a 4-joint arm, inside a square room.

It needs no hardware and no extra packages, so every part of Eddy can be exercised
on any machine. By default it runs on its own clock (waiting costs no real time).
"""
from __future__ import annotations

import math
import threading
import time
from typing import Dict

from ..kinematics import demo_arm
from ..model import DriveSpec, JointSpec, RobotModel, RobotState
from .base import Adapter


class SimAdapter(Adapter):
    kind = "sim"

    def __init__(self, name: str = "sim-rover", room: float = 2.0, realtime: bool = False,
                 arm: bool = True, drive: bool = True):
        self.name = name
        self.room = room            # walls at +/- room metres
        self.radius = 0.15
        self.realtime = realtime
        self.has_arm, self.has_drive = arm, drive
        self.chain = demo_arm() if arm else None
        self.x = self.y = self.heading = 0.0
        self.v = self.w = 0.0
        self.clock = 0.0
        self.bumped = False
        self.joints: Dict[str, float] = {n: 0.0 for n in (self.chain.joint_names if arm else [])}
        self.targets = dict(self.joints)
        self.gripper = 1.0
        self.connected = False
        self._lock = threading.RLock()
        self._live = None            # background clock thread, when running live

    def connect(self) -> RobotModel:
        self.connected = True
        joints = [JointSpec(j.name, j.lower, j.upper, j.kind, j.max_velocity)
                  for j in (self.chain.active if self.chain else [])]
        sensors = (["odometry", "range"] if self.has_drive else []) + (["joint_state"] if self.has_arm else [])
        return RobotModel(
            name=self.name, kind=self.kind,
            drive=DriveSpec("differential", 0.5, 1.5) if self.has_drive else None,
            joints=joints, chain=self.chain, gripper=self.has_arm, sensors=sensors,
            details={"room": f"{2 * self.room:.1f} m square"},
        )

    def close(self) -> None:
        self.stop()
        self.connected = False
        self._live = None

    def go_live(self, rate: float = 50.0) -> "SimAdapter":
        """Advance the simulation with the wall clock, like a real robot would move."""
        def loop():
            last = time.monotonic()
            me = threading.current_thread()
            while self._live is me:
                time.sleep(1.0 / rate)
                now = time.monotonic()
                self._advance(now - last)
                last = now
        self._live = threading.Thread(target=loop, daemon=True)
        self._live.start()
        return self

    def set_velocity(self, linear: float, angular: float) -> None:
        if not self.has_drive:
            return super().set_velocity(linear, angular)
        self.v, self.w = float(linear), float(angular)

    def set_joints(self, positions: Dict[str, float], duration: float = 1.0) -> None:
        if not self.has_arm:
            return super().set_joints(positions, duration)
        for name, value in positions.items():
            if name not in self.targets:
                raise KeyError(f"no joint named '{name}'")
            self.targets[name] = float(value)

    def set_gripper(self, opening: float) -> None:
        if not self.has_arm:
            return super().set_gripper(opening)
        self.gripper = min(1.0, max(0.0, float(opening)))

    def _range_ahead(self) -> float:
        dx, dy = math.cos(self.heading), math.sin(self.heading)
        hits = []
        for pos, d in ((self.x, dx), (self.y, dy)):
            if d > 1e-9:
                hits.append((self.room - pos) / d)
            elif d < -1e-9:
                hits.append((-self.room - pos) / d)
        return max(0.0, min(hits) - self.radius)

    def state(self) -> RobotState:
        return RobotState(self.x, self.y, self.heading, dict(self.joints),
                          self._range_ahead() if self.has_drive else None,
                          self.gripper if self.has_arm else None, has_odometry=self.has_drive)

    def wait(self, seconds: float) -> None:
        if self._live is not None:
            time.sleep(seconds)
            return
        if self.realtime:
            time.sleep(seconds)
        self._advance(seconds)

    def _advance(self, seconds: float) -> None:
        with self._lock:
            self._advance_locked(seconds)

    def _advance_locked(self, seconds: float) -> None:
        remaining = seconds
        while remaining > 1e-9:
            dt = min(0.01, remaining)
            remaining -= dt
            self.clock += dt
            self.heading = math.atan2(math.sin(self.heading + self.w * dt), math.cos(self.heading + self.w * dt))
            nx = self.x + self.v * math.cos(self.heading) * dt
            ny = self.y + self.v * math.sin(self.heading) * dt
            limit = self.room - self.radius
            if abs(nx) > limit or abs(ny) > limit:
                self.bumped = True          # hit a wall: the wheels spin but it does not move
            else:
                self.x, self.y = nx, ny
            for name, target in self.targets.items():
                step = 1.5 * dt
                delta = target - self.joints[name]
                self.joints[name] += max(-step, min(step, delta))
