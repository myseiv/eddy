"""ROS 2 adapter: Eddy sits on top of a robot's existing ROS 2 nodes.

Nothing on the robot is changed. Eddy joins the ROS graph as one more node, reads
every topic the robot already exposes, works out which ones mean "drive", "arm",
"odometry", "laser" and so on, and then uses them.

`classify_graph` is pure Python (no ROS needed) so the detection rules can be
tested anywhere. Everything else needs rclpy, which comes with ROS 2.
"""
from __future__ import annotations

import math
import threading
import time
import xml.etree.ElementTree as ET
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ..model import DriveSpec, JointSpec, RobotModel, RobotState, Unsupported
from .base import Adapter

# The ROS message types Eddy looks for. A topic's type is what tells Eddy what it is for.
TWIST = "geometry_msgs/msg/Twist"                       # velocity command
TWIST_STAMPED = "geometry_msgs/msg/TwistStamped"        # the same, with a timestamp (newer robots)
ODOM = "nav_msgs/msg/Odometry"                          # where the robot thinks it is
SCAN = "sensor_msgs/msg/LaserScan"                      # laser range finder
JOINT_STATE = "sensor_msgs/msg/JointState"              # current joint positions
TRAJECTORY = "trajectory_msgs/msg/JointTrajectory"      # joint targets for a controller
IMAGES = ("sensor_msgs/msg/Image", "sensor_msgs/msg/CompressedImage")
STRING = "std_msgs/msg/String"                          # used for the robot's URDF
FOLLOW = "control_msgs/action/FollowJointTrajectory"    # joint targets for a controller, as an action
ACKERMANN = ("ackermann_msgs/msg/AckermannDriveStamped", "ackermann_msgs/msg/AckermannDrive")   # speed + steering angle
GRIP = "control_msgs/action/GripperCommand"             # open/close a gripper to a position, as an action
CLOUD = "sensor_msgs/msg/PointCloud2"                   # 3D lidar or depth camera points
POSES = ("geometry_msgs/msg/PoseWithCovarianceStamped", "geometry_msgs/msg/PoseStamped")   # a position estimate


def classify_graph(topics: Iterable[Tuple[str, Sequence[str]]],
                   sub_counts: Optional[Dict[str, int]] = None,
                   pub_counts: Optional[Dict[str, int]] = None,
                   actions: Optional[Iterable[Tuple[str, Sequence[str]]]] = None) -> Dict[str, object]:
    """Decide what each topic on a robot is for.

    topics: (name, [types]) pairs as returned by node.get_topic_names_and_types().
    sub_counts / pub_counts: how many nodes listen to / publish each topic, when known.
    A command topic nobody listens to is ranked last; so is a sensor topic nobody publishes.
    actions: (name, [types]) pairs for the ROS actions on the graph, when known.
    """
    sub_counts, pub_counts = sub_counts or {}, pub_counts or {}
    # Group topic names by message type.
    by_type: Dict[str, List[str]] = {}
    for name, types in topics:
        for t in types:
            by_type.setdefault(t, []).append(name)

    def pick(names: List[str], preferred: str, counts: Dict[str, int], known: bool,
             used: Optional[Dict[str, int]] = None) -> Optional[str]:
        """Choose the most likely topic among several of the same type.
        used: for a command topic, how many of the robot's own nodes already publish to it."""
        if not names:
            return None

        def rank(n: str):
            """Sort key: lower is better. False sorts before True."""
            last = n.rsplit("/", 1)[-1]
            return (known and counts.get(n, 0) == 0,   # nobody on the other end
                    bool(used) and used.get(n, 0) == 0,   # the robot's own software does not use it
                    last != preferred,                 # not the conventional name
                    preferred not in n,
                    n.count("/"), len(n), n)           # then shallowest, shortest, alphabetical
        return sorted(names, key=rank)[0]

    profile: Dict[str, object] = {}
    # Velocity command: either Twist flavour. Remember which, because it changes what we publish.
    twists = [(n, TWIST) for n in by_type.get(TWIST, [])] + [(n, TWIST_STAMPED) for n in by_type.get(TWIST_STAMPED, [])]
    if twists:
        types = dict(twists)
        # Only topics that look like velocity commands, unless there is exactly one Twist topic.
        named = [n for n in types if "cmd_vel" in n or "cmd" in n.rsplit("/", 1)[-1]]
        pool = named or (list(types) if len(types) == 1 else [])
        # Where two topics both reach the motors, take the one the robot's own programs publish
        # to. (Seen on a real robot: a second topic meant for a phone app, with speeds scaled down.)
        best = pick(pool, "cmd_vel", sub_counts, bool(sub_counts), pub_counts)
        if best:
            profile["cmd_vel"] = (best, types[best])
    if "cmd_vel" not in profile:
        # No velocity command, but a speed-and-steering-angle one (car-like robots, racing
        # platforms). Only one that something is listening to.
        cars = {n: t for t in ACKERMANN for n in by_type.get(t, []) if not sub_counts or sub_counts.get(n, 0) > 0}
        best = pick(list(cars), "drive", sub_counts, bool(sub_counts), pub_counts)
        if best:
            profile["cmd_vel"] = (best, cars[best])
    # Sensors: one topic each for odometry, laser scan and joint states.
    for key, type_name, preferred, counts in (
            ("odom", ODOM, "odom", pub_counts), ("scan", SCAN, "scan", pub_counts),
            ("joint_states", JOINT_STATE, "joint_states", pub_counts)):
        best = pick(by_type.get(type_name, []), preferred, counts, bool(pub_counts))
        if best:
            profile[key] = best
    # A 3D point cloud can stand in for a laser scan, and a published pose for odometry.
    clouds = [n for n in by_type.get(CLOUD, []) if not pub_counts or pub_counts.get(n, 0) > 0]
    best = pick(clouds, "points", pub_counts, bool(pub_counts))
    if best:
        profile["cloud"] = best
        # Up to three candidates: which one is the sensor's own output is decided from the
        # messages themselves once they arrive (see Ros2Adapter._wire).
        profile["clouds"] = [best] + sorted(n for n in clouds if n != best)[:2]
    # Pose topics are also used to send goals and to reset a robot's position. Only one that
    # something publishes, and that is not named like a command, counts as "where the robot is".
    poses = {n: t for t in POSES for n in by_type.get(t, [])
             if (not pub_counts or pub_counts.get(n, 0) > 0)
             and not any(w in n.rsplit("/", 1)[-1].lower() for w in ("goal", "initial", "target", "set", "waypoint", "cmd"))}
    best = pick(list(poses), "pose", pub_counts, bool(pub_counts))
    if best:
        profile["pose"] = (best, poses[best])
    # Trajectory topics a controller is actually listening on; ones named like a gripper are grippers.
    trajectories = sorted(n for n in by_type.get(TRAJECTORY, []) if not sub_counts or sub_counts.get(n, 0) > 0)
    grippers = [n for n in trajectories if any(w in n.lower() for w in ("gripper", "hand", "claw"))]
    profile["arm_trajectories"] = [n for n in trajectories if n not in grippers]
    profile["gripper_trajectories"] = grippers
    # Controllers that take their trajectory as an action rather than on a topic (many real
    # arms do). A controller that offers both is used through its topic, so it is not listed twice.
    topic_owners = {n.rsplit("/", 1)[0] for n in trajectories}
    follow = sorted(n for n, types in (actions or [])
                    if FOLLOW in types and n.rsplit("/", 1)[0] not in topic_owners)
    gripper_actions = [n for n in follow if any(w in n.lower() for w in ("gripper", "hand", "claw"))]
    profile["arm_actions"] = [n for n in follow if n not in gripper_actions]
    profile["gripper_actions"] = gripper_actions
    # Grippers that take a single "go to this opening" command instead of a trajectory.
    profile["gripper_commands"] = sorted(n for n, types in (actions or []) if GRIP in types)
    profile["cameras"] = sorted(n for t in IMAGES for n in by_type.get(t, []))
    # The order Eddy tries them in for a picture to show: compressed ones first (they are already
    # JPEG, so nothing has to be converted), colour before depth, published before silent.
    def camera_rank(item):
        name, kind = item
        low = name.lower()
        return (pub_counts.get(name, 1) == 0, any(w in low for w in ("depth", "disparity", "/ir", "infra", "theora")),
                kind != IMAGES[1], len(name), name)
    profile["camera_feeds"] = sorted(((n, t) for t in IMAGES for n in by_type.get(t, [])), key=camera_rank)
    # The URDF is published as a String on a topic called robot_description.
    descriptions = [n for n in by_type.get(STRING, []) if n.rsplit("/", 1)[-1] == "robot_description"]
    if descriptions:
        profile["description"] = sorted(descriptions, key=len)[0]
    return profile


def robot_namespaces(topics: Iterable[Tuple[str, Sequence[str]]]) -> List[str]:
    """The namespaces that each hold a whole robot, when several robots share one ROS graph.

    A robot's namespace is the deepest prefix of a drive-command topic that also contains one
    of that robot's sensors, e.g. "/tb1" for /tb1/cmd_vel and /tb1/odom. A robot with no
    namespace gives "". (A second command topic such as /controller/cmd_vel on a single
    robot is not a namespace: no sensor lives under /controller.)
    """
    topics = list(topics)
    commands = [n for n, types in topics if any(t in (TWIST, TWIST_STAMPED) + ACKERMANN for t in types)
                and ("cmd" in n.rsplit("/", 1)[-1] or "drive" in n.rsplit("/", 1)[-1])]
    sensors = [n for n, types in topics if any(t in (ODOM, SCAN, JOINT_STATE, CLOUD) + POSES for t in types)]
    found = []
    for name in commands:
        parts = name.split("/")[1:-1]
        ns = ""
        for depth in range(len(parts), 0, -1):          # deepest ancestor first
            prefix = "/" + "/".join(parts[:depth])
            if any(s.startswith(prefix + "/") for s in sensors):
                ns = prefix
                break
        if ns not in found:
            found.append(ns)
    return sorted(found)


def urdf_limits(urdf: str) -> Dict[str, Tuple[float, float, str]]:
    """Read every joint's (lower limit, upper limit, type) out of a URDF."""
    out = {}
    for el in ET.fromstring(urdf).findall("joint"):
        kind = el.get("type", "fixed")
        limit = el.find("limit")
        lower = float(limit.get("lower", -math.pi)) if limit is not None else -math.pi
        upper = float(limit.get("upper", math.pi)) if limit is not None else math.pi
        if kind == "continuous":
            lower, upper = -math.pi, math.pi        # no limits: one full turn covers it
        out[el.get("name")] = (lower, upper, kind)
    return out


def urdf_mimics(urdf: str) -> Dict[str, str]:
    """Joints that only follow another joint (URDF <mimic>), as {follower: the joint it follows}.
    Such joints cannot be commanded on their own: the joint they follow is the real one."""
    out = {}
    for el in ET.fromstring(urdf).findall("joint"):
        mimic = el.find("mimic")
        if mimic is not None and mimic.get("joint"):
            out[el.get("name")] = mimic.get("joint")
    return out


def front_range(ranges: Sequence[float], angle_min: float, angle_increment: float,
                range_min: float, range_max: float, half_width: float = 0.26,
                center: float = 0.0) -> Optional[float]:
    """Nearest valid laser return within +/- half_width radians of straight ahead
    (or of `center`: pi looks straight behind)."""
    best, seen = None, False
    for i, r in enumerate(ranges):
        # The angle of beam i, wrapped to -pi..pi (some lasers report 0..2pi).
        a = angle_min + i * angle_increment - center
        a = math.atan2(math.sin(a), math.cos(a))
        if abs(a) > half_width:
            continue
        seen = True
        # Skip readings the laser itself marks as invalid.
        if r is None or math.isnan(r) or math.isinf(r) or r < range_min or r > range_max:
            continue
        best = r if best is None else min(best, r)
    if best is None and seen:
        return range_max   # beams point ahead and nothing is within range
    return best            # None if the laser does not look ahead at all


def scan_rotation(previous, current, increment: float, limit: float = 0.45, far: float = 10.0) -> float:
    """How far the robot turned between two full-circle laser scans, in radians (left positive).

    A turn shifts the whole scan sideways, so this finds the shift (up to `limit` radians)
    that makes the new scan line up best with the old one. It is independent of the wheels,
    which is the point: wheel odometry cannot tell when the robot is jammed and slipping.
    """
    import numpy as np
    a = np.nan_to_num(np.asarray(previous, dtype=float), nan=far, posinf=far, neginf=far).clip(0.0, far)
    b = np.nan_to_num(np.asarray(current, dtype=float), nan=far, posinf=far, neginf=far).clip(0.0, far)
    if len(a) != len(b) or len(a) < 30 or increment <= 0:
        return 0.0
    most = max(1, int(limit / increment))
    shifts = range(-most, most + 1)
    # After the robot turns left by k beams, what was at beam i + k is now at beam i.
    errors = [np.abs(b - np.roll(a, -k)).mean() for k in shifts]
    return shifts[int(np.argmin(errors))] * increment


def cloud_xyz(msg):
    """The x, y, z columns of a PointCloud2 message as three numpy arrays (sensor frame)."""
    import numpy as np
    fields = {f.name: f for f in msg.fields}
    if not all(k in fields for k in "xyz") or not msg.point_step:
        return None
    kinds = {7: "f4", 8: "f8"}      # PointCloud2 codes for 32- and 64-bit floats
    if any(fields[k].datatype not in kinds for k in "xyz"):
        return None
    order = ">" if msg.is_bigendian else "<"
    dtype = np.dtype({"names": list("xyz"), "formats": [order + kinds[fields[k].datatype] for k in "xyz"],
                      "offsets": [fields[k].offset for k in "xyz"], "itemsize": msg.point_step})
    points = np.frombuffer(bytes(msg.data), dtype=dtype, count=len(msg.data) // msg.point_step)
    return points["x"], points["y"], points["z"]


def ground_plane(x, y, z, cell: float = 1.0):
    """Fit the ground as a plane z = a*x + b*y + c from the lowest point in each cell of a grid
    over the area in front of the sensor. Returns (a, b, c), or None if too little ground is seen.

    This is what lets a 3D sensor of unknown height and tilt tell the floor from things on it.
    """
    import numpy as np
    area = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & (x > 0.5) & (x < 15.0) & (np.abs(y) < 5.0)
    if area.sum() < 50:
        return None
    gx, gy, gz = x[area], y[area], z[area]
    # Lowest point per grid cell: sort by cell then height, and keep the first of each cell.
    cells = np.floor(gx / cell).astype(np.int64) * 1000 + np.floor(gy / cell).astype(np.int64)
    order = np.lexsort((gz, cells))
    first = np.concatenate(([True], cells[order][1:] != cells[order][:-1]))
    lx, ly, lz = gx[order][first], gy[order][first], gz[order][first]
    keep = np.ones(len(lz), dtype=bool)
    plane = None
    for _ in range(4):
        if keep.sum() < 6:
            return None
        A = np.column_stack((lx[keep], ly[keep], np.ones(keep.sum())))
        plane = np.linalg.lstsq(A, lz[keep], rcond=None)[0]
        # Cells whose lowest point is well above the fit hold only an object, not ground.
        residual = lz - (plane[0] * lx + plane[1] * ly + plane[2])
        keep = residual < 0.10
    return tuple(float(v) for v in plane)


def cloud_front_range(x, y, z, half_width: float = 0.4, below: float = 0.2, above: float = 1.5,
                      nearest: float = 0.3, clearance: float = 0.15) -> Optional[float]:
    """Distance to the nearest thing in a corridor straight ahead of a 3D sensor.

    Assumes the usual ROS axes for the sensor (x forward, y left, z up). The corridor is
    half_width to each side. Where the ground can be found, a "thing" is a point between
    `clearance` and `above` over it; otherwise points more than `below` under the sensor are
    taken to be the ground. Points closer than `nearest` are taken to be the robot itself.
    None if nothing is there.
    """
    import numpy as np
    x, y, z = (np.asarray(v, dtype=float) for v in (x, y, z))
    d = np.hypot(x, y)
    keep = np.isfinite(d) & np.isfinite(z) & (x > 0) & (np.abs(y) <= half_width) & (d >= nearest)
    plane = ground_plane(x, y, z)
    if plane is not None:
        height = z - (plane[0] * x + plane[1] * y + plane[2])
        keep &= (height >= clearance) & (height <= above)
    else:
        keep &= (z >= -below) & (z <= above)
    return float(d[keep].min()) if keep.any() else None


def hold_target(position: float, velocity: float, age: float, period: float, limits) -> float:
    """Where to hold a joint that is told to stop while it moves: where it will be when the command
    arrives (its last reading carried on at its last speed for the reading's age plus one update),
    not where it was. Holding at the old reading makes the arm swing back to it.
    A joint that is barely moving, or a reading too old to trust, is held where it was read."""
    if abs(velocity) < 0.05 or age > 0.5:
        return position
    lead = min(0.3, max(0.0, age) + max(0.0, period))
    lower, upper = limits
    return max(lower, min(upper, position + velocity * lead))


class Ros2Adapter(Adapter):
    """Controls a ROS 2 robot through the topics its own nodes already provide."""
    kind = "ros2"

    def __init__(self, discover_seconds: float = 3.0, cmd_vel: Optional[str] = None,
                 node_name: str = "eddy", max_linear: float = 0.5, max_angular: float = 1.5,
                 wheelbase: float = 0.33):
        """discover_seconds: how long to watch the graph. cmd_vel: force this velocity topic
        instead of the detected one. max_linear / max_angular: speed limits, since ROS does
        not say how fast a robot may go."""
        self.discover_seconds = discover_seconds
        self.cmd_vel_override = cmd_vel           # a velocity topic to force, or a robot's namespace
        self.wheelbase = wheelbase                # metres, used only to turn a turn rate into a steering angle
        self.namespace: Optional[str] = None      # the robot chosen, when several share the graph
        self.other_robots: List[str] = []
        self.node_name = node_name
        self.max_linear, self.max_angular = max_linear, max_angular
        self.node = None
        self.profile: Dict[str, object] = {}      # what classify_graph decided
        self._state = RobotState()                # latest values from the subscriptions
        self._lock = threading.Lock()             # guards _state and _urdf (written by the ROS thread)
        self._urdf: Optional[str] = None
        self._seen = set()                        # which sensors have delivered a first message
        self._odom_time: Optional[float] = None   # when the last position message arrived
        self._scan_prev = None                    # the previous full-circle laser scan
        self._range_time: Optional[float] = None  # when the range sensor (laser or point cloud) last reported
        self._pose_counts = {"odom": 0, "pose": 0}   # messages from each source of position
        self._pose_source: Optional[str] = None   # which of them is used, once decided
        self._cloud_subs: Dict[str, object] = {}  # point cloud topic -> subscription
        self._cloud_fields: Dict[str, int] = {}   # point cloud topic -> fields per point, once seen
        self._cloud_choice: Optional[str] = None  # the point cloud used for range, once decided
        self._controllers: Dict[str, dict] = {}   # topic or action name -> {"pub" or "client", "joints", "gripper"}
        self._limits: Dict[str, Tuple[float, float, str]] = {}
        self._mimics: Dict[str, str] = {}       # joints that only follow another joint
        self._we_started_rclpy = False
        self._jtrack: Dict[str, dict] = {}        # joint -> last position, when it came, speed, time between readings
        self._jpeg: Optional[bytes] = None        # the camera's latest picture
        self._camera_sub = None                   # the camera subscription, made the first time a picture is asked for
        self._camera_try = 0                      # which of the robot's camera feeds is being tried
        self._camera_since = 0.0                  # when that feed was subscribed to
        self._camera_encoded = 0.0                # when a raw picture was last converted to JPEG
        self.camera_note = "This robot has no camera topic."

    # ---- connection ----------------------------------------------------------

    def connect(self) -> RobotModel:
        """Join the ROS graph, find the robot's interfaces, and return its model."""
        try:
            import rclpy
            from rclpy.executors import SingleThreadedExecutor
        except ImportError as exc:
            raise RuntimeError("ROS 2 is not available in this Python. Source your ROS 2 setup "
                               "(e.g. `source /opt/ros/jazzy/setup.bash`) and try again.") from exc
        # Start ROS only if the surrounding program has not, and remember so close() undoes it.
        if not rclpy.ok():
            rclpy.init()
            self._we_started_rclpy = True
        self._rclpy = rclpy
        # Our node, with a background thread delivering incoming messages to the callbacks.
        self.node = rclpy.create_node(self.node_name)
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self.node)
        self._spin = threading.Thread(target=self._executor.spin, daemon=True)
        self._spin.start()

        # Read the graph and count who is on the other end of each topic. This is done before
        # we create our own publishers and subscribers, so we never count ourselves.
        topics = self._discover()
        me = self.node.get_name()
        subs = {n: self.node.count_subscribers(n) for n, _ in topics}
        pubs = {n: self.node.count_publishers(n) for n, _ in topics}
        from rclpy.action import get_action_names_and_types
        actions = get_action_names_and_types(self.node)
        # Several robots on one graph: work with one of them only, so that one robot's
        # commands are never paired with another robot's sensors.
        robots = robot_namespaces(topics)
        wanted = (self.cmd_vel_override or "").rstrip("/")
        if wanted and wanted in robots:
            self.namespace, self.cmd_vel_override = wanted, None
        elif len(robots) > 1 and not self.cmd_vel_override:
            self.namespace = robots[0]
        if self.namespace is not None and len(robots) > 1:
            self.other_robots = [r or "/" for r in robots if r != self.namespace]
            inside = (lambda n: n.startswith(self.namespace + "/")) if self.namespace else \
                (lambda n: not any(n.startswith(r + "/") for r in robots if r))
            topics = [(n, t) for n, t in topics if inside(n)]
            actions = [(n, t) for n, t in actions if inside(n)]
        self.profile = classify_graph(topics, subs, pubs, actions)
        if self.cmd_vel_override:
            types = dict((n, t[0]) for n, t in topics)
            self.profile["cmd_vel"] = (self.cmd_vel_override, types.get(self.cmd_vel_override, TWIST))
        self._wire()
        nodes = sorted({(ns.rstrip("/") + "/" + n) for n, ns in self.node.get_node_names_and_namespaces() if n != me})
        return self._build_model(nodes, len(topics))

    def _discover(self) -> List[Tuple[str, List[str]]]:
        """Wait until the list of topics stops changing."""
        start = time.time()
        deadline = start + self.discover_seconds
        # On a busy computer the robot's topics can take longer than that to show up at all.
        # While nothing but the two topics every node has is visible, keep looking for longer.
        patience = start + max(self.discover_seconds * 4, 12.0)
        last, stable_since = None, time.time()
        while True:
            topics = sorted(self.node.get_topic_names_and_types())
            now = time.time()
            if topics != last:
                last, stable_since = topics, now
            # More than the two topics every node has, and unchanged for a second: done.
            elif len(topics) > 2 and now - stable_since > 1.0:
                break
            if now >= (deadline if len(topics) > 2 else patience):
                break
            time.sleep(0.2)
        return last or []

    def _wire(self) -> None:
        """Create the publishers and subscriptions for everything the profile found."""
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
        from geometry_msgs.msg import Twist, TwistStamped
        node, p = self.node, self.profile
        if "cmd_vel" in p:
            topic, type_name = p["cmd_vel"]
            self._twist_stamped = type_name == TWIST_STAMPED
            self._ackermann = type_name if type_name in ACKERMANN else None
            if self._ackermann:
                from ackermann_msgs.msg import AckermannDrive, AckermannDriveStamped
                self._cmd_pub = node.create_publisher(
                    AckermannDriveStamped if type_name.endswith("Stamped") else AckermannDrive, topic, 10)
            else:
                self._cmd_pub = node.create_publisher(TwistStamped if self._twist_stamped else Twist, topic, 10)
        # Sensor subscriptions use the "sensor data" quality of service, which matches how
        # robots publish fast sensor streams (latest value wins, nothing is re-sent).
        if "odom" in p:
            from nav_msgs.msg import Odometry
            node.create_subscription(Odometry, p["odom"], self._on_odom, qos_profile_sensor_data)
        if "pose" in p:
            from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
            topic, type_name = p["pose"]
            node.create_subscription(PoseStamped if type_name.endswith("PoseStamped") else PoseWithCovarianceStamped,
                                     topic, self._on_pose, qos_profile_sensor_data)
        if "scan" in p:
            from sensor_msgs.msg import LaserScan
            node.create_subscription(LaserScan, p["scan"], self._on_scan, qos_profile_sensor_data)
        elif "cloud" in p:
            # No laser scan: use the 3D point cloud to see what is ahead.
            from sensor_msgs.msg import PointCloud2
            for topic in p.get("clouds", [p["cloud"]]):
                self._cloud_subs[topic] = node.create_subscription(
                    PointCloud2, topic, lambda msg, topic=topic: self._on_cloud(msg, topic), qos_profile_sensor_data)
        if "joint_states" in p:
            from sensor_msgs.msg import JointState
            node.create_subscription(JointState, p["joint_states"], self._on_joints, qos_profile_sensor_data)
        if "description" in p:
            from std_msgs.msg import String
            # The URDF is published once and kept for late joiners ("latched"); we must ask
            # for it with the matching settings or we would never receive it.
            latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                                 reliability=ReliabilityPolicy.RELIABLE)
            node.create_subscription(String, p["description"], self._on_description, latched)
        # One publisher per trajectory controller, plus the list of joints it drives.
        from trajectory_msgs.msg import JointTrajectory
        for topic in list(p["arm_trajectories"]) + list(p["gripper_trajectories"]):
            self._controllers[topic] = {
                "pub": node.create_publisher(JointTrajectory, topic, 10),
                "joints": self._controller_joints(topic),
                "gripper": topic in p["gripper_trajectories"],
            }
        # And one action client per controller that is driven by an action instead.
        actions = list(p["arm_actions"]) + list(p["gripper_actions"])
        if actions:
            try:
                from control_msgs.action import FollowJointTrajectory
                from rclpy.action import ActionClient
            except ImportError:
                actions = []        # the action type is not installed here: leave those controllers alone
            for name in actions:
                self._controllers[name] = {
                    "client": ActionClient(node, FollowJointTrajectory, name),
                    "joints": self._controller_joints(name),
                    "gripper": name in p["gripper_actions"],
                }
        if p.get("gripper_commands"):
            try:
                from control_msgs.action import GripperCommand
                from rclpy.action import ActionClient
                for name in p["gripper_commands"]:
                    # This kind of controller names its one joint in a `joint` parameter.
                    joint = self._param(name.rsplit("/", 1)[0] or "/", "joint")
                    self._controllers[name] = {"grip": ActionClient(node, GripperCommand, name),
                                               "joints": [joint] if isinstance(joint, str) and joint else [],
                                               "gripper": True}
            except ImportError:
                pass
        # Give the first message of everything we subscribed to a moment to arrive. Without
        # this, a command sent straight after connecting would act before the first laser
        # scan or odometry reading, i.e. with no obstacle check and no position.
        deadline = time.time() + 2.0
        while time.time() < deadline:
            with self._lock:
                have_desc = "description" not in p or self._urdf is not None
                have_joints = "joint_states" not in p or bool(self._state.joints)
                where = not ({"odom", "pose"} & set(p)) or bool({"odom", "pose"} & self._seen)
                ahead = (not ({"scan", "cloud"} & set(p)) or "scan" in self._seen
                         or (bool(self._cloud_subs) and len(self._cloud_fields) == len(self._cloud_subs)))
            if have_desc and have_joints and where and ahead:
                break
            time.sleep(0.05)
        # A robot may publish both odometry and a pose. Use whichever is actually arriving
        # (odometry if both are); mixing two sources would make the position jump.
        with self._lock:
            c = self._pose_counts
            self._pose_source = "pose" if c["pose"] > c["odom"] else "odom"
            # Several point clouds: keep the one whose points carry the most fields. A sensor's
            # own output has per-point extras (ring, line, tag, time) that processed copies
            # tend to drop, and processed copies may hold points that are not real surfaces.
            if self._cloud_fields:
                order = list(self._cloud_subs)
                p["cloud"] = max(self._cloud_fields, key=lambda t: (self._cloud_fields[t], -order.index(t)))
            self._cloud_choice = p.get("cloud")
        for topic, sub in list(self._cloud_subs.items()):
            if topic != self._cloud_choice:
                node.destroy_subscription(sub)
                del self._cloud_subs[topic]
        # No URDF on a topic: ask robot_state_publisher, which holds it as a parameter.
        if self._urdf is None:
            self._urdf = self._param("/robot_state_publisher", "robot_description")
        if self._urdf:
            try:
                self._limits = urdf_limits(self._urdf)
                self._mimics = urdf_mimics(self._urdf)
            except ET.ParseError:
                self._urdf = None       # unreadable URDF: carry on without arm geometry

    def _param(self, node_name: str, param: str, timeout: float = 1.5):
        """Read one parameter from another node, or None."""
        from rcl_interfaces.srv import GetParameters
        client = self.node.create_client(GetParameters, f"{node_name}/get_parameters")
        try:
            if not client.wait_for_service(timeout_sec=timeout):
                return None
            # Call the service and wait for the answer (our spin thread delivers it).
            future = client.call_async(GetParameters.Request(names=[param]))
            deadline = time.time() + timeout
            while not future.done() and time.time() < deadline:
                time.sleep(0.02)
            if not future.done() or not future.result().values:
                return None
            v = future.result().values[0]
            # ROS parameter type codes: 4 is a string, 9 is a list of strings.
            return {4: v.string_value, 9: list(v.string_array_value)}.get(v.type)
        finally:
            self.node.destroy_client(client)

    def _controller_joints(self, topic: str) -> List[str]:
        """A trajectory controller's joints: ask the controller node for its `joints` parameter."""
        # The topic or action is "<controller node>/<something>", so the node is everything before the last slash.
        owner = topic.rsplit("/", 1)[0] or "/"
        joints = self._param(owner, "joints") if owner != "/" else None
        return list(joints) if isinstance(joints, list) else []

    def _build_model(self, nodes: List[str], n_topics: int) -> RobotModel:
        """Put everything discovered into a RobotModel."""
        p = self.profile
        with self._lock:
            names = list(self._state.joints)        # every joint seen on /joint_states
        # Joints some controller has claimed.
        commandable = {j for c in self._controllers.values() for j in c["joints"]}
        for c in self._controllers.values():     # controllers that did not tell us their joints
            if not c["joints"]:
                # Guess from the names: gripper-like joints go to a gripper controller,
                # the remaining unclaimed joints to an arm controller.
                # (wheels, which turn without limit, are never arm or gripper joints)
                # Joints that merely follow another joint (a gripper's linkage) are not
                # commandable either: the joint they follow is, even if it is never reported.
                def gripper_like(n: str) -> bool:
                    return any(w in n.lower() for w in ("gripper", "finger", "jaw", "hand", "claw"))
                pool = [n for n in names if n not in commandable and n not in self._mimics
                        and self._limits.get(n, (0.0, 0.0, "revolute"))[2] != "continuous"]
                gripperish = [n for n in pool if gripper_like(n)]
                for follower, driver in self._mimics.items():
                    if (gripper_like(follower) or gripper_like(driver)) and driver not in gripperish \
                            and driver not in commandable and driver in self._limits:
                        gripperish.append(driver)
                c["joints"] = gripperish if c["gripper"] else [n for n in pool if n not in gripperish]
                commandable.update(c["joints"])
        arm_joints = [j for c in self._controllers.values() if not c["gripper"] for j in c["joints"]]
        # Joint list with limits from the URDF (or a full turn if the URDF does not know the joint).
        # Only joints a controller can move are listed: a joint that is merely reported (a wheel,
        # say) is not something Eddy can position. A controller's joint is listed even when
        # the robot does not report its position.
        joints = []
        for n in [n for n in names if n in commandable] + sorted(commandable - set(names)):
            lower, upper, kind = self._limits.get(n, (-math.pi, math.pi, "revolute"))
            joints.append(JointSpec(n, lower, upper, kind))
        # Arm geometry for inverse kinematics, using only joints an arm controller can move.
        chain = None
        if self._urdf and arm_joints:
            from ..kinematics import Chain
            try:
                chain = Chain.from_urdf(self._urdf, allowed=set(arm_joints))
            except Exception:
                chain = None
        sensors = [s for s, keys in (("odometry", ("odom", "pose")), ("range", ("scan", "cloud")),
                                     ("joint_state", ("joint_states",))) if any(k in p for k in keys)]
        if p.get("cameras"):
            sensors.append("camera")
        # Notes for `eddy info`: which nodes and topics were found and which ones Eddy chose.
        details = {"nodes": f"{len(nodes)} ({', '.join(nodes[:8])}{'...' if len(nodes) > 8 else ''})",
                   "topics": n_topics}
        for key in ("cmd_vel", "odom", "pose", "scan", "joint_states"):
            if key in p:
                details[key] = p[key][0] if isinstance(p[key], tuple) else p[key]
        if "odom" in p and "pose" in p:
            details["position from"] = p["pose"][0] if self._pose_source == "pose" else p["odom"]
        if "cloud" in p and "scan" not in p:
            details["range from"] = f"{p['cloud']} (3D points: assumes the sensor's x axis points forward; things 15 cm or more above the ground)"
        if self.other_robots:
            details["this robot"] = self.namespace or "/"
            details["other robots here"] = ", ".join(self.other_robots) + "  (choose one with -r ros2:<namespace>)"
        for topic, c in self._controllers.items():
            details[f"controller {topic}"] = ", ".join(c["joints"]) or "joints unknown"
        if p.get("cameras"):
            details["cameras"] = ", ".join(p["cameras"])
        name = ET.fromstring(self._urdf).get("name", "ros2-robot") if self._urdf else "ros2-robot"
        return RobotModel(
            name=name, kind=self.kind,
            # A robot commanded by steering angle is car-like from the start; others are found
            # out by trying (see the brain).
            drive=DriveSpec("car-like" if p.get("cmd_vel", ("", ""))[1] in ACKERMANN else "differential",
                            self.max_linear, self.max_angular) if "cmd_vel" in p else None,
            joints=joints, chain=chain, gripper=any(c["gripper"] and c["joints"] for c in self._controllers.values()),
            sensors=sensors, details=details)

    # ---- incoming data -------------------------------------------------------
    # These run on the ROS spin thread each time a message arrives. Each one just
    # stores the latest value under the lock.

    def _on_pose(self, msg) -> None:
        """A published pose (with or without covariance): treated like odometry."""
        self._on_odom(msg, source="pose")

    def _on_cloud(self, msg, topic: str = "") -> None:
        """3D points: keep only the distance to the nearest thing straight ahead."""
        with self._lock:
            self._cloud_fields.setdefault(topic, len(msg.fields))
            if self._cloud_choice not in (None, topic):
                return
        columns = cloud_xyz(msg)
        r = cloud_front_range(*columns) if columns is not None else None
        with self._lock:
            self._state.range_ahead = r
            self._range_time = time.monotonic()
            self._seen.add("cloud")

    def _on_odom(self, msg, source: str = "odom") -> None:
        """Odometry: keep x, y and the heading."""
        pose = msg.pose.pose if hasattr(msg.pose, "pose") else msg.pose
        with self._lock:
            self._pose_counts[source] += 1
            self._seen.add(source)
            # Before the source is decided, odometry wins; afterwards only the chosen one counts.
            if self._pose_source not in (None, source) or (self._pose_source is None and source == "pose"
                                                           and self._pose_counts["odom"]):
                return
        q = pose.orientation
        # Heading (yaw) from the orientation quaternion.
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        with self._lock:
            self._state.x, self._state.y, self._state.z = pose.position.x, pose.position.y, pose.position.z
            self._state.heading, self._state.has_odometry = yaw, True
            self._odom_time = time.monotonic()

    def _on_scan(self, msg) -> None:
        """Laser scan: keep only the distance to the nearest thing straight ahead."""
        r = front_range(msg.ranges, msg.angle_min, msg.angle_increment, msg.range_min, msg.range_max)
        back = front_range(msg.ranges, msg.angle_min, msg.angle_increment, msg.range_min, msg.range_max, center=math.pi)
        # Narrow beams along the robot's own axis, for checking that a straight drive really
        # closes the distance (the wide corridor above would also pick up a wall alongside).
        axis = [front_range(msg.ranges, msg.angle_min, msg.angle_increment, msg.range_min, msg.range_max,
                            half_width=0.05, center=c) for c in (0.0, math.pi)]
        # A laser that sees all the way round also shows how far the robot has really turned.
        full_circle = abs(len(msg.ranges) * msg.angle_increment - 2 * math.pi) < 0.2
        step = 0.0
        if full_circle and self._scan_prev is not None and len(self._scan_prev) == len(msg.ranges):
            step = scan_rotation(self._scan_prev, msg.ranges, msg.angle_increment)
        self._scan_prev = list(msg.ranges) if full_circle else None
        with self._lock:
            self._state.range_ahead, self._state.range_behind, self._state.range_max = r, back, msg.range_max
            self._state.axis_ahead, self._state.axis_behind = axis
            self._range_time = time.monotonic()
            if full_circle:
                self._state.scan_turn = (self._state.scan_turn or 0.0) + step
            self._seen.add("scan")

    def _on_joints(self, msg) -> None:
        """Joint states: keep each joint's position, and how fast it is changing."""
        now = time.monotonic()
        with self._lock:
            for name, pos in zip(msg.name, msg.position):
                self._state.joints[name] = pos
                old = self._jtrack.get(name)
                if old is None:
                    self._jtrack[name] = {"pos": pos, "t": now, "vel": 0.0, "period": 0.05, "count": 1}
                    continue
                dt = now - old["t"]
                if dt < 1e-3:
                    continue
                old.update(vel=(pos - old["pos"]) / dt, period=min(0.5, 0.7 * old["period"] + 0.3 * dt),
                           pos=pos, t=now, count=old["count"] + 1)

    def _on_description(self, msg) -> None:
        """The robot's URDF."""
        with self._lock:
            self._urdf = msg.data

    # ---- commands ------------------------------------------------------------

    # ---- camera --------------------------------------------------------------

    def camera_frame(self) -> Optional[bytes]:
        """The camera's latest picture as JPEG bytes, or None if none has arrived.

        Nothing is subscribed to until a picture is first asked for, so a camera costs nothing
        unless someone is looking. If the feed tried delivers nothing in 3 s, the next is tried.
        """
        feeds = self.profile.get("camera_feeds") or []
        if not feeds or self.node is None:
            return None
        now = time.monotonic()
        if self._camera_sub is None or (self._jpeg is None and now - self._camera_since > 3.0 and len(feeds) > 1):
            if self._camera_sub is not None:
                self.node.destroy_subscription(self._camera_sub)
                self._camera_try = (self._camera_try + 1) % len(feeds)
            self._subscribe_camera(*feeds[self._camera_try])
            self._camera_since = now
        return self._jpeg

    def _subscribe_camera(self, topic: str, kind: str) -> None:
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CompressedImage, Image
        self.camera_note = f"Waiting for a picture on {topic}."
        if kind == IMAGES[1]:
            self._camera_sub = self.node.create_subscription(
                CompressedImage, topic, lambda msg: self._on_compressed(msg, topic), qos_profile_sensor_data)
        else:
            self._camera_sub = self.node.create_subscription(
                Image, topic, lambda msg: self._on_image(msg, topic), qos_profile_sensor_data)

    def _on_compressed(self, msg, topic: str) -> None:
        """A compressed picture: JPEG ones are passed on as they are."""
        data = bytes(msg.data)
        if data[:2] == b"\xff\xd8":            # every JPEG file starts with these two bytes
            self._jpeg = data
        else:
            self.camera_note = f"{topic} is compressed as '{msg.format}', which the page cannot show (it needs JPEG)."

    def _on_image(self, msg, topic: str) -> None:
        """An uncompressed picture: converted to JPEG with OpenCV, if the robot has it."""
        now = time.monotonic()
        if now - self._camera_encoded < 0.1:    # ten pictures a second is plenty, and converting costs CPU
            return
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.camera_note = (f"{topic} is uncompressed, and turning it into pictures a browser can show needs "
                                "OpenCV on the robot (python3-opencv). A compressed camera topic needs nothing.")
            return
        channels = {"rgb8": 3, "bgr8": 3, "mono8": 1, "rgba8": 4, "bgra8": 4}.get(msg.encoding)
        if channels is None:
            self.camera_note = f"{topic} uses the picture format '{msg.encoding}', which Eddy does not convert."
            return
        picture = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(msg.height, msg.step)[:, :msg.width * channels]
        picture = picture.reshape(msg.height, msg.width, channels)
        if msg.encoding.startswith("rgb"):
            picture = picture[:, :, [2, 1, 0] + ([3] if channels == 4 else [])]    # OpenCV wants blue first
        ok, jpeg = cv2.imencode(".jpg", np.ascontiguousarray(picture[:, :, :3] if channels == 4 else picture),
                                [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            self._jpeg, self._camera_encoded = jpeg.tobytes(), now

    def state(self) -> RobotState:
        """A copy of the latest state, so the caller's copy cannot change under it."""
        with self._lock:
            s = self._state
            age = None if self._odom_time is None else time.monotonic() - self._odom_time
            range_age = None if self._range_time is None else time.monotonic() - self._range_time
            return RobotState(s.x, s.y, s.heading, dict(s.joints), s.range_ahead, s.gripper, s.has_odometry,
                              range_behind=s.range_behind, range_max=s.range_max, odom_age=age,
                              z=s.z, range_age=range_age, scan_turn=s.scan_turn, axis_ahead=s.axis_ahead, axis_behind=s.axis_behind)

    def set_velocity(self, linear: float, angular: float, vertical: float = 0.0, lateral: float = 0.0) -> None:
        """Publish one velocity command, in whichever Twist flavour the robot uses."""
        if "cmd_vel" not in self.profile:
            raise Unsupported("no velocity command topic found on this robot")
        if self._ackermann:
            if lateral:
                raise Unsupported("this robot takes a speed and a steering angle, so it cannot move sideways")
            # Speed and steering angle. The angle that gives the wanted turn rate at this speed
            # (bicycle model), limited to what steering racks usually allow. Standing still, a
            # car cannot turn, so the angle is zero.
            from ackermann_msgs.msg import AckermannDrive, AckermannDriveStamped
            drive = AckermannDrive()
            drive.speed = float(linear)
            if abs(linear) > 1e-6:
                drive.steering_angle = max(-0.4, min(0.4, math.atan(self.wheelbase * angular / linear)))
            if self._ackermann.endswith("Stamped"):
                msg = AckermannDriveStamped()
                msg.header.stamp = self.node.get_clock().now().to_msg()
                msg.header.frame_id = "base_link"
                msg.drive = drive
                self._cmd_pub.publish(msg)
            else:
                self._cmd_pub.publish(drive)
            return
        from geometry_msgs.msg import Twist, TwistStamped
        twist = Twist()
        twist.linear.x, twist.angular.z = float(linear), float(angular)
        twist.linear.z = float(vertical)        # climb rate; ground robots ignore it
        twist.linear.y = float(lateral)         # sideways speed; bases that cannot move sideways ignore it
        if self._twist_stamped:
            msg = TwistStamped()
            msg.header.stamp = self.node.get_clock().now().to_msg()
            msg.header.frame_id = "base_link"
            msg.twist = twist
            self._cmd_pub.publish(msg)
        else:
            self._cmd_pub.publish(twist)

    def _send_trajectory(self, controller: dict, positions: Dict[str, float], duration: float) -> None:
        """Send one controller a single-point trajectory: be at these positions after `duration`."""
        from builtin_interfaces.msg import Duration
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
        current = self.state().joints
        point = JointTrajectoryPoint()
        # Controllers want a value for every one of their joints: keep the others where they are.
        point.positions = [float(positions.get(j, current.get(j, 0.0))) for j in controller["joints"]]
        point.time_from_start = Duration(sec=int(duration), nanosec=int((duration % 1) * 1e9))
        msg = JointTrajectory()
        msg.joint_names = list(controller["joints"])
        msg.points = [point]
        if "grip" in controller:
            # A gripper that takes one position: send the first (only) joint's target.
            self._send_grip(controller, float(point.positions[0]))
        elif "client" in controller:
            # Action-driven controller: send the trajectory as a goal. We do not wait for the
            # result here; the brain watches the joint states to see the move finish.
            from control_msgs.action import FollowJointTrajectory
            if not controller["client"].wait_for_server(timeout_sec=1.0):
                raise Unsupported("the joint controller is not answering")
            sent = controller["client"].send_goal_async(FollowJointTrajectory.Goal(trajectory=msg))
            # Give the controller a moment to say whether it takes the goal, so a refusal
            # (wrong joints, say) is reported instead of passing as a move that happened.
            deadline = time.time() + 1.0
            while not sent.done() and time.time() < deadline:
                time.sleep(0.02)
            if sent.done() and not sent.result().accepted:
                raise RuntimeError("the joint controller rejected the move (joints: "
                                   + ", ".join(controller["joints"]) + ")")
            if sent.done():
                # Keep the goal, so a stop can cancel it while it is still running.
                controller["goal"] = sent.result()
                controller["result"] = controller["goal"].get_result_async()
        else:
            controller["pub"].publish(msg)

    def _send_grip(self, controller: dict, position: float) -> None:
        """Send a gripper-command controller its target opening."""
        from control_msgs.action import GripperCommand
        if not controller["grip"].wait_for_server(timeout_sec=1.0):
            raise Unsupported("the gripper controller is not answering")
        goal = GripperCommand.Goal()
        goal.command.position = position
        goal.command.max_effort = 10.0      # newtons; a gentle default, since the graph does not say
        controller["grip"].send_goal_async(goal)

    def set_joints(self, positions: Dict[str, float], duration: float = 1.0) -> None:
        """Hand each joint target to the controller that owns that joint."""
        if not self._controllers:
            raise Unsupported("no joint trajectory controller found on this robot")
        unclaimed = set(positions)
        for c in self._controllers.values():
            mine = {j: v for j, v in positions.items() if j in c["joints"]}
            if mine:
                self._send_trajectory(c, mine, duration)
                unclaimed -= set(mine)
        if unclaimed:
            raise Unsupported(f"no controller accepts joints: {', '.join(sorted(unclaimed))}")

    def set_gripper(self, opening: float) -> None:
        """Open or close the gripper by mapping 0..1 onto each finger joint's own limits."""
        grippers = [c for c in self._controllers.values() if c["gripper"] and c["joints"]]
        if not grippers:
            raise Unsupported("no gripper controller found on this robot")
        for c in grippers:
            target = {}
            for j in c["joints"]:
                lower, upper, _ = self._limits.get(j, (0.0, 1.0, "revolute"))
                target[j] = lower + min(1.0, max(0.0, opening)) * (upper - lower)
            self._send_trajectory(c, target, 1.0)

    def _gripper_joints(self):
        """The gripper controllers' joints and the joints that follow them, and the controllers' own."""
        own = [j for c in self._controllers.values() if c["gripper"] for j in c["joints"]]
        followers = [f for f, d in self._mimics.items() if d in own]
        return own, followers

    def gripper_feedback(self) -> Optional[Dict[str, float]]:
        """The reported positions of the gripper's joints (its own, or ones that follow it)."""
        own, followers = self._gripper_joints()
        with self._lock:
            seen = {j: float(self._state.joints[j]) for j in own + followers if j in self._state.joints}
        return seen or None

    def gripper_at(self, opening: float) -> Optional[bool]:
        """Whether the gripper's own joints already read the targets this opening would send them."""
        own, _ = self._gripper_joints()
        with self._lock:
            seen = {j: self._state.joints[j] for j in own if j in self._state.joints}
        if not seen:
            return None
        def target(j):
            lower, upper, _ = self._limits.get(j, (0.0, 1.0, "revolute"))
            return lower + min(1.0, max(0.0, opening)) * (upper - lower)
        return all(abs(v - target(j)) < 0.02 * max(1e-3, abs(self._limits.get(j, (0.0, 1.0, ""))[1]
                                                                - self._limits.get(j, (0.0, 1.0, ""))[0])) + 1e-3
                   for j, v in seen.items())

    def hold_joints(self, names) -> None:
        """Stop these joints. First the controller's own stop, where it has one (a ros2_control
        trajectory controller holds where it is on an empty trajectory, or when its goal is
        cancelled); then a hold placed where each joint will be, not where it was last read."""
        from trajectory_msgs.msg import JointTrajectory
        names = set(names)
        native = False
        for c in self._controllers.values():
            if c["gripper"] or not names & set(c["joints"]):
                continue
            if "client" in c:
                if c.get("goal") is not None and not c["result"].done():
                    c["goal"].cancel_goal_async()
                    native = True
                c["goal"] = None
            elif "pub" in c:
                msg = JointTrajectory()
                msg.joint_names = list(c["joints"])
                c["pub"].publish(msg)
                native = True
        if native:
            # Give the controller's stop a moment, then judge the arm's speed from readings taken
            # after it, so an arm the controller already stopped is not pushed on.
            with self._lock:
                counts = {n: self._jtrack[n]["count"] for n in names if n in self._jtrack}
            deadline = time.monotonic() + 0.3
            while time.monotonic() < deadline:
                with self._lock:
                    if all(self._jtrack[n]["count"] >= k + 2 for n, k in counts.items()):
                        break
                time.sleep(0.01)
        now = time.monotonic()
        targets = {}
        with self._lock:
            for n in names:
                t = self._jtrack.get(n)
                lower, upper, _ = self._limits.get(n, (-math.pi, math.pi, "revolute"))
                if t is None:
                    targets[n] = float(self._state.joints.get(n, 0.0))
                else:
                    targets[n] = hold_target(t["pos"], t["vel"], now - t["t"], t["period"], (lower, upper))
        self.set_joints(targets, 0.2)

    def stop(self) -> None:
        """Send zero velocity three times, in case one message is lost."""
        if self.node is not None and "cmd_vel" in self.profile:
            for _ in range(3):
                self.set_velocity(0.0, 0.0)
                time.sleep(0.02)

    def close(self) -> None:
        """Stop the robot and leave the ROS graph."""
        if self.node is None:
            return
        try:
            self.stop()
        finally:
            # Order matters: stop spinning and wait for the thread before tearing ROS down,
            # or the process aborts on exit.
            self._executor.shutdown(timeout_sec=2.0)
            self._spin.join(timeout=2.0)
            for c in self._controllers.values():
                for key in ("client", "grip"):
                    if key in c:
                        c[key].destroy()
            self.node.destroy_node()
            self.node = None
            if self._we_started_rclpy and self._rclpy.ok():
                self._rclpy.shutdown()
