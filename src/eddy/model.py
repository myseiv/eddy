"""What Eddy knows about a robot: its abilities, its state, and the commands it accepts.

Every adapter describes its robot with a RobotModel. Everything above the adapter
(voice, hand signs, autonomy, the CLI, other robots) only ever speaks in Commands,
so the same instruction works on any robot that has the needed ability.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---- Description of a robot -----------------------------------------------------

@dataclass
class JointSpec:
    """One joint the robot lets us command, with the limits Eddy must respect."""
    name: str
    lower: float = -3.14159          # lowest allowed position (radians, or metres if prismatic)
    upper: float = 3.14159           # highest allowed position
    kind: str = "revolute"           # revolute (turns), continuous (turns forever) or prismatic (slides)
    max_velocity: float = 1.5        # rad/s (or m/s)


@dataclass
class DriveSpec:
    """How the robot's base drives, and how fast Eddy is allowed to push it."""
    kind: str = "differential"
    max_linear: float = 0.5   # m/s
    max_angular: float = 1.5  # rad/s


@dataclass
class RobotModel:
    """Everything an adapter found out about the robot it connected to.

    Fields left empty (drive=None, no joints, chain=None, gripper=False) mean the robot does
    not have that ability, and Eddy will refuse commands that need it.
    """
    name: str
    kind: str = "unknown"                      # which adapter produced it
    drive: Optional[DriveSpec] = None
    joints: List[JointSpec] = field(default_factory=list)
    chain: object = None                       # eddy.kinematics.Chain, when the arm geometry is known
    gripper: bool = False
    sensors: List[str] = field(default_factory=list)   # odometry, range, joint_state, camera
    details: Dict[str, object] = field(default_factory=dict)  # adapter-specific notes (topics, ports)

    def capabilities(self) -> List[str]:
        """List what this robot can do, as short words ("drive", "reach", "range", ...)."""
        caps = []
        if self.drive:
            caps.append("drive")
        if self.joints:
            caps.append("joints")
        # "reach" (move the tool to a point) needs the arm's geometry, not just its joints.
        if self.chain is not None:
            caps.append("reach")
        if self.gripper:
            caps.append("gripper")
        return caps + list(self.sensors)

    def joint(self, name: str) -> Optional[JointSpec]:
        """Look up a joint by name; None if the robot has no such joint."""
        for j in self.joints:
            if j.name == name:
                return j
        return None

    def find_joint(self, name: str) -> Optional[JointSpec]:
        """Look up a joint the way a person would name it: exact name first, then ignoring
        capitals, then "1" for "joint1", then any single joint whose name contains the word.
        None if nothing matches, or if more than one joint would."""
        exact = self.joint(name)
        if exact is not None:
            return exact
        wanted = name.lower()
        for candidates in ([j for j in self.joints if j.name.lower() == wanted],
                           [j for j in self.joints if j.name.lower() == "joint" + wanted],
                           [j for j in self.joints if wanted in j.name.lower()]):
            if len(candidates) == 1:
                return candidates[0]
            if candidates:
                # Several match ("joint1" on an arm that also has "finger_joint1"): take the
                # one on the arm itself if that settles it, otherwise do not guess.
                arm = set(getattr(self.chain, "joint_names", []) or [])
                on_arm = [j for j in candidates if j.name in arm]
                return on_arm[0] if len(on_arm) == 1 else None
        return None

    def summary(self) -> str:
        """A multi-line, human-readable description (what `eddy info` prints)."""
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
    """A snapshot of where the robot is right now, as reported by its adapter."""
    x: float = 0.0                             # metres, in the odometry frame
    y: float = 0.0
    heading: float = 0.0                       # radians, counter-clockwise from +x
    joints: Dict[str, float] = field(default_factory=dict)   # joint name -> current position
    range_ahead: Optional[float] = None        # metres to the nearest thing in front
    gripper: Optional[float] = None            # 0 closed .. 1 open
    has_odometry: bool = False                 # False means x, y, heading cannot be trusted
    range_behind: Optional[float] = None       # metres to the nearest thing behind, if the sensor looks there
    range_max: Optional[float] = None          # the range sensor's limit: a reading this large means "nothing seen"
    axis_ahead: Optional[float] = None         # metres to whatever is exactly straight ahead (a narrow beam)
    axis_behind: Optional[float] = None        # the same, exactly straight behind
    scan_turn: Optional[float] = None          # radians turned so far as seen by a 360-degree laser (None: unknown)
    odom_age: Optional[float] = None           # seconds since the position was last updated (None: always fresh)


# ---- Commands -----------------------------------------------------------------
# These are the only things Eddy's brain accepts. Every input (text, speech, hand
# signs, another robot) is turned into one of these before anything moves.

@dataclass
class Drive:
    """Drive at a speed. Give one of distance (m), angle (rad) or duration (s)."""
    linear: float = 0.0                # m/s, negative is backwards
    angular: float = 0.0               # rad/s, positive is counter-clockwise (left)
    distance: Optional[float] = None
    angle: Optional[float] = None
    duration: Optional[float] = None


@dataclass
class Stop:
    """Stop all motion now. Always obeyed, and interrupts whatever is running."""
    pass


@dataclass
class MoveTo:
    """Put the arm's tool at x, y, z (metres, robot base frame) using inverse kinematics."""
    x: float
    y: float
    z: float


@dataclass
class MoveBy:
    """Move the arm's tool by this much from where it is now (metres, robot base frame)."""
    dx: float = 0.0
    dy: float = 0.0
    dz: float = 0.0


@dataclass
class SetJoints:
    """Move named joints straight to the given positions (no inverse kinematics)."""
    positions: Dict[str, float]


@dataclass
class Home:
    """Send every arm joint to zero."""
    pass


@dataclass
class Gripper:
    """Open or close the gripper."""
    opening: float  # 0 closed .. 1 open


@dataclass
class Say:
    """Speak a piece of text out loud."""
    text: str


@dataclass
class Report:
    """Ask Eddy to say what the robot is and where it is."""
    pass


@dataclass
class Wander:
    """Drive around on its own for `duration` seconds, turning away from obstacles."""
    duration: float = 10.0


# ---- Reasons a command is not carried out -----------------------------------------

class Unsupported(Exception):
    """The robot does not have the ability a command needs."""


class Refused(Exception):
    """Eddy declined a command for safety reasons."""
