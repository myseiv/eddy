"""Stand-ins for the two real robots Eddy was tried on, for testing away from the lab.

This is NOT Hiwonder's software. It reproduces, on a real ROS 2 graph, the things observed
on the two real robots, with Eddy's simulator as the physics:

  EDDY_MOCK_KIND=landerpi   mecanum base with a 5-joint arm and a gripper
  EDDY_MOCK_KIND=car        car-steered base, no arm ("rosmentor")

Observed and reproduced:
  - two velocity topics: /cmd_vel goes through an app node that passes turning on about six
    times weaker; /controller/cmd_vel is the direct one the robot's own nodes publish to
  - laser on /scan_raw sweeping 0..2*pi, range_max 25
  - arm and gripper driven by FollowJointTrajectory actions, on nodes that do not answer
    parameter requests, and that reject a goal naming the wrong joints
  - the gripper controller drives one joint, r_joint, which is not in /joint_states; five
    linkage joints that follow it are
  - wheels reported in /joint_states
  - the car ignores a turn command while standing still
Taken from Hiwonder's MentorPi source (driver/controller), which the owner supplied:
  - the car computes a steering angle from linear and angular speed, limited to 45 degrees
    with a 0.145 m wheelbase (tightest turn 0.145 m), and does nothing when linear is zero
  - on the MentorPi the app path (/cmd_vel) limits speed to 0.2 m/s and 0.5 rad/s
  - /odom is a filter over the commanded speed, laser odometry and the IMU, not wheel
    encoders; here it is simply the true pose
  - the driver for the mecanum base reads a sideways speed (linear.y), limited to 0.2 m/s
    (odom_publisher_node.py), and Hiwonder's own programs drive sideways that way. Sliding has
    not been observed on the real LanderPi; here the base slides exactly as commanded
  - the car driver moves the steering servo only when the turn rate is not zero, so a command
    with a turn rate of exactly 0 leaves the steering where the last turn put it (seen on the
    real car on 2026-11-17: "straight" drives after a turn veered 7 to 42 degrees)
Assumed (not observed): the linkage joints are marked <mimic> in the URDF, their names, the
arm's link lengths, and that laser angle 0 is the robot's front.

The true state is written to /tmp/truth.json every tick so a test can compare.
"""
import json
import math
import os
import time

import rclpy
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Image, JointState, LaserScan
from std_msgs.msg import String

from eddy.adapters.sim import SimAdapter

KIND = os.environ.get("EDDY_MOCK_KIND", "landerpi")
ARM = ["joint1", "joint2", "joint3", "joint4", "joint5"]
LINKAGE = ["gripper_l_in_joint", "gripper_l_out_joint", "gripper_r_in_joint", "gripper_r_out_joint", "gripper_l_joint"]
WHEELS = ["wheel_lf_Joint", "wheel_rf_Joint", "wheel_lb_Joint", "wheel_rb_Joint"]


def urdf() -> str:
    """The robot description: wheels for both kinds, and the arm and gripper for the LanderPi."""
    out = [f'<robot name="{"landerpi" if KIND == "landerpi" else "rosmentor"}"><link name="base_link"/>']
    for w in WHEELS:
        out.append(f'<link name="{w}_l"/><joint name="{w}" type="continuous"><parent link="base_link"/>'
                   f'<child link="{w}_l"/><axis xyz="0 1 0"/></joint>')
    if KIND == "landerpi":
        # joint, parent, child, origin, axis, limit
        arm = [("joint1", "base_link", "a1", "0 0 0.10", "0 0 1", 2.09), ("joint2", "a1", "a2", "0 0 0.04", "0 1 0", 2.09),
               ("joint3", "a2", "a3", "0 0 0.13", "0 1 0", 2.09), ("joint4", "a3", "a4", "0 0 0.13", "0 1 0", 2.09),
               ("joint5", "a4", "a5", "0 0 0.05", "0 0 1", 2.09)]
        for name, parent, child, xyz, axis, lim in arm:
            out.append(f'<link name="{child}"/><joint name="{name}" type="revolute"><parent link="{parent}"/>'
                       f'<child link="{child}"/><origin xyz="{xyz}"/><axis xyz="{axis}"/>'
                       f'<limit lower="-{lim}" upper="{lim}" velocity="1.5" effort="1"/></joint>')
        out.append('<link name="tool"/><joint name="tool_joint" type="fixed"><parent link="a5"/><child link="tool"/>'
                   '<origin xyz="0 0 0.08"/></joint>')
        out.append('<link name="r_link"/><joint name="r_joint" type="revolute"><parent link="a5"/><child link="r_link"/>'
                   '<axis xyz="1 0 0"/><limit lower="-1.57" upper="0" velocity="1.5" effort="1"/></joint>')
        for j in LINKAGE:
            out.append(f'<link name="{j}_l"/><joint name="{j}" type="revolute"><parent link="a5"/><child link="{j}_l"/>'
                       f'<axis xyz="1 0 0"/><limit lower="-1.57" upper="1.57" velocity="1.5" effort="1"/>'
                       f'<mimic joint="r_joint"/></joint>')
    return "".join(out) + "</robot>"


class App(Node):
    """The phone-app path: takes /cmd_vel and passes it on with turning made much weaker."""
    def __init__(self):
        super().__init__("app_node")
        self.out = self.create_publisher(Twist, "/controller/cmd_vel", 10)
        self.create_subscription(Twist, "/cmd_vel", self.on_cmd, 10)

    def on_cmd(self, msg: Twist):
        if KIND == "car":       # MentorPi source: the app path limits the speeds
            msg.linear.x = max(-0.2, min(0.2, msg.linear.x))
            msg.angular.z = max(-0.5, min(0.5, msg.angular.z))
        else:                   # LanderPi: turning seen about six times weaker on this path
            msg.angular.z /= 6.0
        self.out.publish(msg)


class Base(Node):
    """The base: listens on /controller/cmd_vel and publishes odometry, the laser and joint states."""
    def __init__(self, sim: SimAdapter, arm: dict):
        super().__init__("odom_publisher")
        self.sim, self.arm, self.last_cmd, self.wheel = sim, arm, 0.0, 0.0
        self.create_subscription(Twist, "/controller/cmd_vel", self.on_cmd, 10)
        self.odom = self.create_publisher(Odometry, "/odom", 10)
        self.scan = self.create_publisher(LaserScan, "/scan_raw", qos_profile_sensor_data)
        self.joints = self.create_publisher(JointState, "/joint_states", 10)
        self.camera = self.create_publisher(Image, "/ascamera/camera_publisher/rgb0/image", 10)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
        self.description = self.create_publisher(String, "/robot_description", latched)
        self.description.publish(String(data=urdf()))
        self.create_timer(0.05, self.tick)

    def on_cmd(self, msg: Twist):
        self.last_cmd = time.monotonic()
        # The mecanum base also takes a sideways speed, limited to 0.2 m/s as in Hiwonder's driver.
        self.sim.set_velocity(msg.linear.x, msg.angular.z, lateral=max(-0.2, min(0.2, msg.linear.y)))

    def ray(self, angle: float) -> float:
        """Distance to the room's wall along one laser beam."""
        s = self.sim
        dx, dy = math.cos(s.heading + angle), math.sin(s.heading + angle)
        hits = [((s.room if d > 0 else -s.room) - pos) / d for pos, d in ((s.x, dx), (s.y, dy)) if abs(d) > 1e-9]
        return max(0.0, min(hits) - s.radius)

    def tick(self):
        s = self.sim
        if time.monotonic() - self.last_cmd > 0.5 and (s.v or s.w or s.vy):
            s.set_velocity(0.0, 0.0)                    # command timeout
        # The arm's joints move towards their targets at 1.5 rad/s.
        for name, target in self.arm["target"].items():
            now = self.arm["now"][name]
            self.arm["now"][name] = now + max(-0.075, min(0.075, target - now))
        stamp = self.get_clock().now().to_msg()
        odom = Odometry()
        odom.header.stamp, odom.header.frame_id, odom.child_frame_id = stamp, "odom", "base_footprint"
        odom.pose.pose.position.x, odom.pose.pose.position.y = s.x, s.y
        odom.pose.pose.orientation.z, odom.pose.pose.orientation.w = math.sin(s.heading / 2), math.cos(s.heading / 2)
        self.odom.publish(odom)
        scan = LaserScan()
        scan.header.stamp, scan.header.frame_id = stamp, "lidar_frame"
        scan.angle_min, scan.angle_max, scan.angle_increment = 0.0, 2 * math.pi, math.radians(1)
        scan.range_min, scan.range_max = 0.05, 25.0
        scan.ranges = [self.ray(i * scan.angle_increment) for i in range(360)]
        self.scan.publish(scan)
        self.wheel += s.v * 0.05 / 0.03
        js = JointState()
        js.header.stamp = stamp
        names, positions = list(WHEELS), [self.wheel] * 4
        if KIND == "landerpi":
            names += ARM + LINKAGE
            positions += [self.arm["now"][j] for j in ARM] + [self.arm["now"]["r_joint"]] * len(LINKAGE)
        js.name, js.position = names, positions
        self.joints.publish(js)
        with open("/tmp/truth.tmp", "w") as f:
            json.dump({"x": s.x, "y": s.y, "heading": s.heading, "bumped": s.bumped, "joints": self.arm["now"]}, f)
        os.replace("/tmp/truth.tmp", "/tmp/truth.json")


class Controller(Node):
    """A joint controller driven only by an action, which does not answer parameter requests."""
    def __init__(self, name: str, joints, arm: dict):
        super().__init__(name, start_parameter_services=False)
        self.joints, self.arm = joints, arm
        self.server = ActionServer(self, FollowJointTrajectory, f"/{name}/follow_joint_trajectory", self.run,
                                   goal_callback=self.on_goal, cancel_callback=lambda _: CancelResponse.ACCEPT)

    def on_goal(self, goal):
        ok = sorted(goal.trajectory.joint_names) == sorted(self.joints) and goal.trajectory.points
        if not ok:
            self.get_logger().error(f"rejected goal for joints {list(goal.trajectory.joint_names)}")
        return GoalResponse.ACCEPT if ok else GoalResponse.REJECT

    def run(self, handle):
        t = handle.request.trajectory
        self.arm["target"].update(dict(zip(t.joint_names, t.points[-1].positions)))
        handle.succeed()
        return FollowJointTrajectory.Result()


def main():
    rclpy.init()
    sim = SimAdapter(name=KIND, arm=False, car=(0.145 if KIND == "car" else False),
                     slides=(KIND == "landerpi"), steering_holds=(KIND == "car")).go_live()
    names = ARM + ["r_joint"] if KIND == "landerpi" else []
    arm = {"now": {n: 0.0 for n in names}, "target": {n: 0.0 for n in names}}
    nodes = [App(), Base(sim, arm)]
    if KIND == "landerpi":
        nodes += [Controller("arm_controller", ARM, arm), Controller("gripper_controller", ["r_joint"], arm)]
    executor = MultiThreadedExecutor()
    for n in nodes:
        executor.add_node(n)
    print(f"mock {KIND} up", flush=True)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"mock stopped ({type(exc).__name__})", flush=True)


if __name__ == "__main__":
    main()
