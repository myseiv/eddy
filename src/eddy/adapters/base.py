"""The one interface every robot has to fit behind.

An adapter wraps whatever already runs on the robot (ROS 2 nodes, microcontroller
firmware, a simulator) and exposes the same handful of calls. Eddy never talks to
robot code directly, only to an adapter.
"""
from __future__ import annotations

import time
from typing import Dict, Optional

from ..model import RobotModel, RobotState, Unsupported


class Adapter:
    """Base class for all adapters. A new kind of robot is supported by subclassing this
    and overriding whichever calls that robot can actually do; the rest stay "unsupported"."""
    kind = "base"       # short name of the adapter type, shown in the robot's summary

    def connect(self) -> RobotModel:
        """Reach the robot, work out what it can do, and return its model."""
        raise NotImplementedError

    def close(self) -> None:
        """Disconnect. Nothing to do by default."""
        pass

    def state(self) -> RobotState:
        """Where the robot is right now. By default: an empty state with no odometry."""
        return RobotState()

    def set_velocity(self, linear: float, angular: float, vertical: float = 0.0) -> None:
        """Drive the base: forward speed in m/s, turn rate in rad/s (positive is left), and
        for robots that fly, climb rate in m/s (positive is up)."""
        raise Unsupported("this robot cannot drive")

    def set_joints(self, positions: Dict[str, float], duration: float = 1.0) -> None:
        """Move the named joints to these positions, taking about `duration` seconds."""
        raise Unsupported("this robot has no joints Eddy can command")

    def set_gripper(self, opening: float) -> None:
        """Set the gripper: 0 is closed, 1 is fully open."""
        raise Unsupported("this robot has no gripper")

    # Why there is no camera picture, for adapters that have none (shown on the web page).
    camera_note = "Eddy cannot read a camera on this kind of robot."

    def camera_frame(self) -> Optional[bytes]:
        """The camera's latest picture as JPEG bytes, or None if there is none (yet)."""
        return None

    def stop(self) -> None:
        """Must always be safe to call, in any state."""
        try:
            self.set_velocity(0.0, 0.0)
        except Unsupported:
            pass        # a robot that cannot drive has nothing to stop

    def now(self) -> float:
        """The time, in seconds, on the clock this robot runs by. (The simulator has its own.)"""
        return time.monotonic()

    def wait(self, seconds: float) -> None:
        """Let time pass. The simulator overrides this to advance its own clock."""
        time.sleep(seconds)
