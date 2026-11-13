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

TWIST = "geometry_msgs/msg/Twist"
TWIST_STAMPED = "geometry_msgs/msg/TwistStamped"
ODOM = "nav_msgs/msg/Odometry"
SCAN = "sensor_msgs/msg/LaserScan"
JOINT_STATE = "sensor_msgs/msg/JointState"
TRAJECTORY = "trajectory_msgs/msg/JointTrajectory"
IMAGES = ("sensor_msgs/msg/Image", "sensor_msgs/msg/CompressedImage")
STRING = "std_msgs/msg/String"


def classify_graph(topics: Iterable[Tuple[str, Sequence[str]]],
                   sub_counts: Optional[Dict[str, int]] = None,
                   pub_counts: Optional[Dict[str, int]] = None) -> Dict[str, object]:
    """Decide what each topic on a robot is for.

    topics: (name, [types]) pairs as returned by node.get_topic_names_and_types().
    sub_counts / pub_counts: how many nodes listen to / publish each topic, when known.
    A command topic nobody listens to is ranked last; so is a sensor topic nobody publishes.
    """
    sub_counts, pub_counts = sub_counts or {}, pub_counts or {}
    by_type: Dict[str, List[str]] = {}
    for name, types in topics:
        for t in types:
            by_type.setdefault(t, []).append(name)

    def pick(names: List[str], preferred: str, counts: Dict[str, int], known: bool) -> Optional[str]:
        if not names:
            return None

        def rank(n: str):
            last = n.rsplit("/", 1)[-1]
            return (known and counts.get(n, 0) == 0,   # nobody on the other end
                    last != preferred,                 # not the conventional name
                    preferred not in n,
                    n.count("/"), len(n), n)
        return sorted(names, key=rank)[0]

    profile: Dict[str, object] = {}
    twists = [(n, TWIST) for n in by_type.get(TWIST, [])] + [(n, TWIST_STAMPED) for n in by_type.get(TWIST_STAMPED, [])]
    if twists:
        types = dict(twists)
        # Only topics that look like velocity commands, unless there is exactly one Twist topic.
        named = [n for n in types if "cmd_vel" in n or "cmd" in n.rsplit("/", 1)[-1]]
        pool = named or (list(types) if len(types) == 1 else [])
        best = pick(pool, "cmd_vel", sub_counts, bool(sub_counts))
        if best:
            profile["cmd_vel"] = (best, types[best])
    for key, type_name, preferred, counts in (
            ("odom", ODOM, "odom", pub_counts), ("scan", SCAN, "scan", pub_counts),
            ("joint_states", JOINT_STATE, "joint_states", pub_counts)):
        best = pick(by_type.get(type_name, []), preferred, counts, bool(pub_counts))
        if best:
            profile[key] = best
    trajectories = sorted(n for n in by_type.get(TRAJECTORY, []) if not sub_counts or sub_counts.get(n, 0) > 0)
    grippers = [n for n in trajectories if any(w in n.lower() for w in ("gripper", "hand", "claw"))]
    profile["arm_trajectories"] = [n for n in trajectories if n not in grippers]
    profile["gripper_trajectories"] = grippers
    profile["cameras"] = sorted(n for t in IMAGES for n in by_type.get(t, []))
    descriptions = [n for n in by_type.get(STRING, []) if n.rsplit("/", 1)[-1] == "robot_description"]
    if descriptions:
        profile["description"] = sorted(descriptions, key=len)[0]
    return profile


def urdf_limits(urdf: str) -> Dict[str, Tuple[float, float, str]]:
    out = {}
    for el in ET.fromstring(urdf).findall("joint"):
        kind = el.get("type", "fixed")
        limit = el.find("limit")
        lower = float(limit.get("lower", -math.pi)) if limit is not None else -math.pi
        upper = float(limit.get("upper", math.pi)) if limit is not None else math.pi
        if kind == "continuous":
            lower, upper = -math.pi, math.pi
        out[el.get("name")] = (lower, upper, kind)
    return out


def front_range(ranges: Sequence[float], angle_min: float, angle_increment: float,
                range_min: float, range_max: float, half_width: float = 0.26) -> Optional[float]:
    """Nearest valid laser return within +/- half_width radians of straight ahead."""
    best, seen = None, False
    for i, r in enumerate(ranges):
        a = angle_min + i * angle_increment
        a = math.atan2(math.sin(a), math.cos(a))
        if abs(a) > half_width:
            continue
        seen = True
        if r is None or math.isnan(r) or math.isinf(r) or r < range_min or r > range_max:
            continue
        best = r if best is None else min(best, r)
    if best is None and seen:
        return range_max   # beams point ahead and nothing is within range
    return best


class Ros2Adapter(Adapter):
    kind = "ros2"

    def __init__(self, discover_seconds: float = 3.0, cmd_vel: Optional[str] = None,
                 node_name: str = "eddy", max_linear: float = 0.5, max_angular: float = 1.5):
        self.discover_seconds = discover_seconds
        self.cmd_vel_override = cmd_vel
        self.node_name = node_name
        self.max_linear, self.max_angular = max_linear, max_angular
        self.node = None
        self.profile: Dict[str, object] = {}
        self._state = RobotState()
        self._lock = threading.Lock()
        self._urdf: Optional[str] = None
        self._controllers: Dict[str, dict] = {}   # topic -> {"pub": publisher, "joints": [...]}
        self._limits: Dict[str, Tuple[float, float, str]] = {}
        self._we_started_rclpy = False

    # ---- connection ----------------------------------------------------------

    def connect(self) -> RobotModel:
        try:
            import rclpy
            from rclpy.executors import SingleThreadedExecutor
        except ImportError as exc:
            raise RuntimeError("ROS 2 is not available in this Python. Source your ROS 2 setup "
                               "(e.g. `source /opt/ros/jazzy/setup.bash`) and try again.") from exc
        if not rclpy.ok():
            rclpy.init()
            self._we_started_rclpy = True
        self._rclpy = rclpy
        self.node = rclpy.create_node(self.node_name)
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self.node)
        self._spin = threading.Thread(target=self._executor.spin, daemon=True)
        self._spin.start()

        topics = self._discover()
        me = self.node.get_name()
        subs = {n: self.node.count_subscribers(n) for n, _ in topics}
        pubs = {n: self.node.count_publishers(n) for n, _ in topics}
        self.profile = classify_graph(topics, subs, pubs)
        if self.cmd_vel_override:
            types = dict((n, t[0]) for n, t in topics)
            self.profile["cmd_vel"] = (self.cmd_vel_override, types.get(self.cmd_vel_override, TWIST))
        self._wire()
        nodes = sorted({(ns.rstrip("/") + "/" + n) for n, ns in self.node.get_node_names_and_namespaces() if n != me})
        return self._build_model(nodes, len(topics))

    def _discover(self) -> List[Tuple[str, List[str]]]:
        """Wait until the list of topics stops changing."""
        deadline = time.time() + self.discover_seconds
        last, stable_since = None, time.time()
        while time.time() < deadline:
            topics = sorted(self.node.get_topic_names_and_types())
            if topics != last:
                last, stable_since = topics, time.time()
            elif len(topics) > 2 and time.time() - stable_since > 1.0:
                break
            time.sleep(0.2)
        return last or []

    def _wire(self) -> None:
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
        from geometry_msgs.msg import Twist, TwistStamped
        node, p = self.node, self.profile
        if "cmd_vel" in p:
            topic, type_name = p["cmd_vel"]
            self._twist_stamped = type_name == TWIST_STAMPED
            self._cmd_pub = node.create_publisher(TwistStamped if self._twist_stamped else Twist, topic, 10)
        if "odom" in p:
            from nav_msgs.msg import Odometry
            node.create_subscription(Odometry, p["odom"], self._on_odom, qos_profile_sensor_data)
        if "scan" in p:
            from sensor_msgs.msg import LaserScan
            node.create_subscription(LaserScan, p["scan"], self._on_scan, qos_profile_sensor_data)
        if "joint_states" in p:
            from sensor_msgs.msg import JointState
            node.create_subscription(JointState, p["joint_states"], self._on_joints, qos_profile_sensor_data)
        if "description" in p:
            from std_msgs.msg import String
            latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                                 reliability=ReliabilityPolicy.RELIABLE)
            node.create_subscription(String, p["description"], self._on_description, latched)
        from trajectory_msgs.msg import JointTrajectory
        for topic in list(p["arm_trajectories"]) + list(p["gripper_trajectories"]):
            self._controllers[topic] = {
                "pub": node.create_publisher(JointTrajectory, topic, 10),
                "joints": self._controller_joints(topic),
                "gripper": topic in p["gripper_trajectories"],
            }
        # Give the first messages (description, joint states, odometry) a moment to arrive.
        deadline = time.time() + 2.0
        while time.time() < deadline:
            with self._lock:
                have_desc = "description" not in p or self._urdf is not None
                have_joints = "joint_states" not in p or bool(self._state.joints)
            if have_desc and have_joints:
                break
            time.sleep(0.05)
        if self._urdf is None:
            self._urdf = self._param("/robot_state_publisher", "robot_description")
        if self._urdf:
            try:
                self._limits = urdf_limits(self._urdf)
            except ET.ParseError:
                self._urdf = None

    def _param(self, node_name: str, param: str, timeout: float = 1.5):
        """Read one parameter from another node, or None."""
        from rcl_interfaces.srv import GetParameters
        client = self.node.create_client(GetParameters, f"{node_name}/get_parameters")
        try:
            if not client.wait_for_service(timeout_sec=timeout):
                return None
            future = client.call_async(GetParameters.Request(names=[param]))
            deadline = time.time() + timeout
            while not future.done() and time.time() < deadline:
                time.sleep(0.02)
            if not future.done() or not future.result().values:
                return None
            v = future.result().values[0]
            return {4: v.string_value, 9: list(v.string_array_value)}.get(v.type)
        finally:
            self.node.destroy_client(client)

    def _controller_joints(self, topic: str) -> List[str]:
        """A trajectory controller's joints: ask the controller node for its `joints` parameter."""
        owner = topic.rsplit("/", 1)[0] or "/"
        joints = self._param(owner, "joints") if owner != "/" else None
        return list(joints) if isinstance(joints, list) else []

    def _build_model(self, nodes: List[str], n_topics: int) -> RobotModel:
        p = self.profile
        with self._lock:
            names = list(self._state.joints)
        commandable = {j for c in self._controllers.values() for j in c["joints"]}
        for c in self._controllers.values():     # controllers that did not tell us their joints
            if not c["joints"]:
                pool = [n for n in names if n not in commandable]
                gripperish = [n for n in pool if any(w in n.lower() for w in ("gripper", "finger", "hand", "claw"))]
                c["joints"] = gripperish if c["gripper"] else [n for n in pool if n not in gripperish]
                commandable.update(c["joints"])
        arm_joints = [j for c in self._controllers.values() if not c["gripper"] for j in c["joints"]]
        joints = []
        for n in names:
            lower, upper, kind = self._limits.get(n, (-math.pi, math.pi, "revolute"))
            joints.append(JointSpec(n, lower, upper, kind))
        chain = None
        if self._urdf and arm_joints:
            from ..kinematics import Chain
            try:
                chain = Chain.from_urdf(self._urdf, allowed=set(arm_joints))
            except Exception:
                chain = None
        sensors = [s for s, k in (("odometry", "odom"), ("range", "scan"), ("joint_state", "joint_states")) if k in p]
        if p.get("cameras"):
            sensors.append("camera")
        details = {"nodes": f"{len(nodes)} ({', '.join(nodes[:8])}{'...' if len(nodes) > 8 else ''})",
                   "topics": n_topics}
        for key in ("cmd_vel", "odom", "scan", "joint_states"):
            if key in p:
                details[key] = p[key][0] if isinstance(p[key], tuple) else p[key]
        for topic, c in self._controllers.items():
            details[f"controller {topic}"] = ", ".join(c["joints"]) or "joints unknown"
        if p.get("cameras"):
            details["cameras"] = ", ".join(p["cameras"])
        name = ET.fromstring(self._urdf).get("name", "ros2-robot") if self._urdf else "ros2-robot"
        return RobotModel(
            name=name, kind=self.kind,
            drive=DriveSpec("differential", self.max_linear, self.max_angular) if "cmd_vel" in p else None,
            joints=joints, chain=chain, gripper=any(c["gripper"] and c["joints"] for c in self._controllers.values()),
            sensors=sensors, details=details)

    # ---- incoming data -------------------------------------------------------

    def _on_odom(self, msg) -> None:
        q = msg.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        with self._lock:
            self._state.x, self._state.y = msg.pose.pose.position.x, msg.pose.pose.position.y
            self._state.heading, self._state.has_odometry = yaw, True

    def _on_scan(self, msg) -> None:
        r = front_range(msg.ranges, msg.angle_min, msg.angle_increment, msg.range_min, msg.range_max)
        with self._lock:
            self._state.range_ahead = r

    def _on_joints(self, msg) -> None:
        with self._lock:
            for name, pos in zip(msg.name, msg.position):
                self._state.joints[name] = pos

    def _on_description(self, msg) -> None:
        with self._lock:
            self._urdf = msg.data

    # ---- commands ------------------------------------------------------------

    def state(self) -> RobotState:
        with self._lock:
            s = self._state
            return RobotState(s.x, s.y, s.heading, dict(s.joints), s.range_ahead, s.gripper, s.has_odometry)

    def set_velocity(self, linear: float, angular: float) -> None:
        if "cmd_vel" not in self.profile:
            raise Unsupported("no velocity command topic found on this robot")
        from geometry_msgs.msg import Twist, TwistStamped
        twist = Twist()
        twist.linear.x, twist.angular.z = float(linear), float(angular)
        if self._twist_stamped:
            msg = TwistStamped()
            msg.header.stamp = self.node.get_clock().now().to_msg()
            msg.header.frame_id = "base_link"
            msg.twist = twist
            self._cmd_pub.publish(msg)
        else:
            self._cmd_pub.publish(twist)

    def _send_trajectory(self, controller: dict, positions: Dict[str, float], duration: float) -> None:
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
        controller["pub"].publish(msg)

    def set_joints(self, positions: Dict[str, float], duration: float = 1.0) -> None:
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
        grippers = [c for c in self._controllers.values() if c["gripper"] and c["joints"]]
        if not grippers:
            raise Unsupported("no gripper controller found on this robot")
        for c in grippers:
            target = {}
            for j in c["joints"]:
                lower, upper, _ = self._limits.get(j, (0.0, 1.0, "revolute"))
                target[j] = lower + min(1.0, max(0.0, opening)) * (upper - lower)
            self._send_trajectory(c, target, 1.0)
        with self._lock:
            self._state.gripper = opening

    def stop(self) -> None:
        if self.node is not None and "cmd_vel" in self.profile:
            for _ in range(3):
                self.set_velocity(0.0, 0.0)
                time.sleep(0.02)

    def close(self) -> None:
        if self.node is None:
            return
        try:
            self.stop()
        finally:
            # Order matters: stop spinning and wait for the thread before tearing ROS down,
            # or the process aborts on exit.
            self._executor.shutdown(timeout_sec=2.0)
            self._spin.join(timeout=2.0)
            self.node.destroy_node()
            self.node = None
            if self._we_started_rclpy and self._rclpy.ok():
                self._rclpy.shutdown()
