"""The one interface every robot has to fit behind.

An adapter wraps whatever already runs on the robot (ROS 2 nodes, microcontroller
firmware, a simulator) and exposes the same handful of calls. Eddy never talks to
robot code directly, only to an adapter.
"""
from __future__ import annotations

import time
from typing import Dict

from ..model import RobotModel, RobotState, Unsupported


class Adapter:
    kind = "base"

    def connect(self) -> RobotModel:
        """Reach the robot, work out what it can do, and return its model."""
        raise NotImplementedError

    def close(self) -> None:
        pass

    def state(self) -> RobotState:
        return RobotState()

    def set_velocity(self, linear: float, angular: float) -> None:
        raise Unsupported("this robot cannot drive")

    def set_joints(self, positions: Dict[str, float], duration: float = 1.0) -> None:
        raise Unsupported("this robot has no joints Eddy can command")

    def set_gripper(self, opening: float) -> None:
        raise Unsupported("this robot has no gripper")

    def stop(self) -> None:
        """Must always be safe to call, in any state."""
        try:
            self.set_velocity(0.0, 0.0)
        except Unsupported:
            pass

    def wait(self, seconds: float) -> None:
        """Let time pass. The simulator overrides this to advance its own clock."""
        time.sleep(seconds)
