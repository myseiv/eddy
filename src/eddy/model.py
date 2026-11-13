"""What Eddy knows about a robot: its abilities, its state, and the commands it accepts.

Every adapter describes its robot with a RobotModel. Everything above the adapter
(voice, hand signs, autonomy, the CLI, other robots) only ever speaks in Commands,
so the same instruction works on any robot that has the needed ability.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class JointSpec:
    name: str
    lower: float = -3.14159
    upper: float = 3.14159
    kind: str = "revolute"
    max_velocity: float = 1.5


@dataclass
class DriveSpec:
    kind: str = "differential"
    max_linear: float = 0.5   # m/s
    max_angular: float = 1.5  # rad/s


@dataclass
class RobotModel:
    name: str
    kind: str = "unknown"                      # which adapter produced it
    drive: Optional[DriveSpec] = None
    joints: List[JointSpec] = field(default_factory=list)
    chain: object = None                       # eddy.kinematics.Chain, when the arm geometry is known
    gripper: bool = False
    sensors: List[str] = field(default_factory=list)   # odometry, range, joint_state, camera
    details: Dict[str, object] = field(default_factory=dict)  # adapter-specific notes (topics, ports)

    def capabilities(self) -> List[str]:
        caps = []
        if self.drive:
            caps.append("drive")
        if self.joints:
            caps.append("joints")
        if self.chain is not None:
            caps.append("reach")
        if self.gripper:
            caps.append("gripper")
        return caps + list(self.sensors)

    def joint(self, name: str) -> Optional[JointSpec]:
        for j in self.joints:
            if j.name == name:
                return j
        return None

    def summary(self) -> str:
        lines = [f"robot: {self.name}  (via {self.kind})",
                 f"can:   {', '.join(self.capabilities()) or 'nothing detected'}"]
        if self.drive:
            lines.append(f"drive: {self.drive.kind}, up to {self.drive.max_linear:.2f} m/s, "
                         f"{self.drive.max_angular:.2f} rad/s")
        for j in self.joints:
            lines.append(f"joint: {j.name:<20} {j.kind:<10} [{j.lower:+.2f}, {j.upper:+.2f}]")
        for key, value in self.details.items():
            lines.append(f"{key}: {value}")
        return "\n".join(lines)


@dataclass
class RobotState:
    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0                       # radians, counter-clockwise from +x
    joints: Dict[str, float] = field(default_factory=dict)
    range_ahead: Optional[float] = None        # metres to the nearest thing in front
    gripper: Optional[float] = None            # 0 closed .. 1 open
    has_odometry: bool = False


# ---- Commands -----------------------------------------------------------------

@dataclass
class Drive:
    """Drive at a speed. Give one of distance (m), angle (rad) or duration (s)."""
    linear: float = 0.0
    angular: float = 0.0
    distance: Optional[float] = None
    angle: Optional[float] = None
    duration: Optional[float] = None


@dataclass
class Stop:
    pass


@dataclass
class MoveTo:
    """Put the arm's tool at x, y, z (metres, robot base frame) using inverse kinematics."""
    x: float
    y: float
    z: float


@dataclass
class SetJoints:
    positions: Dict[str, float]


@dataclass
class Home:
    pass


@dataclass
class Gripper:
    opening: float  # 0 closed .. 1 open


@dataclass
class Say:
    text: str


@dataclass
class Report:
    """Ask Eddy to say what the robot is and where it is."""
    pass


@dataclass
class Wander:
    duration: float = 10.0


class Unsupported(Exception):
    """The robot does not have the ability a command needs."""


class Refused(Exception):
    """Eddy declined a command for safety reasons."""
