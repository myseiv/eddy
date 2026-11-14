"""A stand-in ROS 2 robot for testing Eddy's ROS adapter against real rclpy.

It looks, from the ROS graph, like an ordinary mobile manipulator that knows nothing
about Eddy: a base that listens on /cmd_vel and publishes /odom, /scan and
/joint_states, an arm and a gripper behind ros2_control-style trajectory topics,
and a latched /robot_description. Its physics is Eddy's simulator.

Like a real base, it stops if velocity commands stop arriving for half a second.
"""
import math
import os
import time

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import JointState, LaserScan
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory

from eddy.adapters.sim import SimAdapter

# The arm controller's joints, and how far (metres) the gripper finger slides when fully open.
ARM = ["base_yaw", "shoulder", "elbow", "wrist"]
FINGER_TRAVEL = 0.04

# The robot description this mock publishes: two wheels, a 4-joint arm and one sliding finger.
URDF = """<robot name="mockbot">
  <link name="base_link"/><link name="wheel_l"/><link name="wheel_r"/>
  <link name="l1"/><link name="l2"/><link name="l3"/><link name="l4"/><link name="tool"/><link name="finger_link"/>
  <joint name="wheel_left_joint" type="continuous"><parent link="base_link"/><child link="wheel_l"/>
    <origin xyz="0 0.12 0"/><axis xyz="0 1 0"/></joint>
  <joint name="wheel_right_joint" type="continuous"><parent link="base_link"/><child link="wheel_r"/>
    <origin xyz="0 -0.12 0"/><axis xyz="0 1 0"/></joint>
  <joint name="base_yaw" type="revolute"><parent link="base_link"/><child link="l1"/>
    <origin xyz="0 0 0.10"/><axis xyz="0 0 1"/><limit lower="-3.14159" upper="3.14159" velocity="1.5" effort="1"/></joint>
  <joint name="shoulder" type="revolute"><parent link="l1"/><child link="l2"/>
    <origin xyz="0 0 0.05"/><axis xyz="0 1 0"/><limit lower="-1.9" upper="1.9" velocity="1.5" effort="1"/></joint>
  <joint name="elbow" type="revolute"><parent link="l2"/><child link="l3"/>
    <origin xyz="0 0 0.20"/><axis xyz="0 1 0"/><limit lower="-2.4" upper="2.4" velocity="1.5" effort="1"/></joint>
  <joint name="wrist" type="revolute"><parent link="l3"/><child link="l4"/>
    <origin xyz="0 0 0.20"/><axis xyz="0 1 0"/><limit lower="-1.9" upper="1.9" velocity="1.5" effort="1"/></joint>
  <joint name="tool_joint" type="fixed"><parent link="l4"/><child link="tool"/><origin xyz="0 0 0.08"/></joint>
  <joint name="finger" type="prismatic"><parent link="tool"/><child link="finger_link"/>
    <origin xyz="0 0 0.02"/><axis xyz="0 1 0"/><limit lower="0" upper="0.04" velocity="0.1" effort="1"/></joint>
</robot>"""


class Base(Node):
    """The mobile base: takes velocity commands and publishes odometry, a laser scan and joint states."""
    def __init__(self, sim: SimAdapter):
        super().__init__("mock_base")
        self.sim = sim
        self.last_cmd = 0.0     # when the last velocity command arrived
        self.wheel = 0.0        # wheel rotation angle, for the joint states
        self.create_subscription(Twist, "/cmd_vel", self.on_cmd, 10)
        self.decoy = self.create_publisher(Twist, "/teleop/cmd_vel", 10)     # a Twist topic nobody listens to
        self.odom = self.create_publisher(Odometry, "/odom", 10)
        self.scan = self.create_publisher(LaserScan, "/scan", qos_profile_sensor_data)
        self.joints = self.create_publisher(JointState, "/joint_states", 10)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
        self.description = self.create_publisher(String, "/robot_description", latched)
        self.description.publish(String(data=URDF))
        self.create_timer(0.05, self.tick)      # publish sensors 20 times a second

    def on_cmd(self, msg: Twist):
        """A velocity command arrived: pass it to the simulated rover."""
        self.last_cmd = time.monotonic()
        self.sim.set_velocity(msg.linear.x, msg.angular.z)

    def ray(self, angle: float) -> float:
        """Distance to the room's wall along one laser beam, `angle` away from straight ahead."""
        s = self.sim
        dx, dy = math.cos(s.heading + angle), math.sin(s.heading + angle)
        hits = []
        for pos, d in ((s.x, dx), (s.y, dy)):
            if d > 1e-9:
                hits.append((s.room - pos) / d)
            elif d < -1e-9:
                hits.append((-s.room - pos) / d)
        return max(0.0, min(hits) - s.radius)

    def tick(self):
        """Publish one round of odometry, laser scan and joint states."""
        s = self.sim
        if time.monotonic() - self.last_cmd > 0.5 and (s.v or s.w):
            s.set_velocity(0.0, 0.0)                    # command timeout, like real bases have
        now = self.get_clock().now().to_msg()
        odom = Odometry()
        odom.header.stamp, odom.header.frame_id, odom.child_frame_id = now, "odom", "base_link"
        odom.pose.pose.position.x, odom.pose.pose.position.y = s.x, s.y
        odom.pose.pose.orientation.z, odom.pose.pose.orientation.w = math.sin(s.heading / 2), math.cos(s.heading / 2)
        odom.twist.twist.linear.x, odom.twist.twist.angular.z = s.v, s.w
        self.odom.publish(odom)

        # A 360-beam scan, one beam per degree.
        scan = LaserScan()
        scan.header.stamp, scan.header.frame_id = now, "laser"
        scan.angle_min, scan.angle_max, scan.angle_increment = -math.pi, math.pi, math.radians(1)
        scan.range_min, scan.range_max = 0.05, 8.0
        scan.ranges = [self.ray(scan.angle_min + i * scan.angle_increment) for i in range(360)]
        self.scan.publish(scan)

        # Wheel angle: distance moved this tick divided by a 4 cm wheel radius.
        self.wheel += s.v * 0.05 / 0.04
        js = JointState()
        js.header.stamp = now
        js.name = ARM + ["finger", "wheel_left_joint", "wheel_right_joint"]
        js.position = [s.joints[n] for n in ARM] + [s.gripper * FINGER_TRAVEL, self.wheel, self.wheel]
        self.joints.publish(js)


class TrajectoryController(Node):
    """Stands in for a ros2_control joint_trajectory_controller."""

    def __init__(self, name: str, joints, apply):
        super().__init__(name)
        self.joints, self.apply = joints, apply
        # Real controllers expose their joints as a parameter; Eddy reads it to learn them.
        self.declare_parameter("joints", joints)
        if os.environ.get("EDDY_MOCK_ACTIONS"):
            # Action mode: take trajectories as FollowJointTrajectory goals, as many real arms do.
            from control_msgs.action import FollowJointTrajectory
            from rclpy.action import ActionServer
            self._result = FollowJointTrajectory.Result
            self.server = ActionServer(self, FollowJointTrajectory, f"/{name}/follow_joint_trajectory", self.on_goal)
        else:
            self.create_subscription(JointTrajectory, f"/{name}/joint_trajectory", self.on_trajectory, 10)

    def on_goal(self, goal_handle):
        """An action goal arrived: apply its trajectory and report success."""
        self.on_trajectory(goal_handle.request.trajectory)
        goal_handle.succeed()
        return self._result()

    def on_trajectory(self, msg: JointTrajectory):
        """Apply the last point of a trajectory, if it names exactly this controller's joints."""
        if sorted(msg.joint_names) != sorted(self.joints) or not msg.points:
            self.get_logger().error(f"rejected trajectory for joints {list(msg.joint_names)}")   # as the real one does
            return
        self.apply(dict(zip(msg.joint_names, msg.points[-1].positions)))


def main():
    """Start the base and both controllers on one simulated robot and run until interrupted."""
    rclpy.init()
    sim = SimAdapter(name="mockbot").go_live()
    nodes = [
        Base(sim),
        TrajectoryController("arm_controller", ARM, sim.set_joints),
        TrajectoryController("gripper_controller", ["finger"], lambda p: sim.set_gripper(p["finger"] / FINGER_TRAVEL)),
    ]
    executor = SingleThreadedExecutor()
    for n in nodes:
        executor.add_node(n)
    print("mock robot up", flush=True)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    except Exception as exc:            # killed mid-publish by the test script: not a failure
        print(f"mock robot stopped ({type(exc).__name__})", flush=True)


if __name__ == "__main__":
    main()
