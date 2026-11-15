"""Makes Gazebo's stock Ackermann vehicle look like a robot that is commanded by speed and
steering angle (ackermann_msgs), as racing platforms are, instead of by a velocity message.

It listens for AckermannDriveStamped on /drive in the normal ROS domain and passes the
equivalent velocity to Gazebo through a second, private ROS domain, so the velocity topic is
invisible to Eddy. This shim is test scaffolding written for Eddy's tests; the vehicle and
its physics are Gazebo's.
"""
import math

import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Twist
from rclpy.executors import SingleThreadedExecutor

WHEELBASE = 1.0     # of Gazebo's example vehicle, metres

public, private = rclpy.Context(), rclpy.Context()
rclpy.init(context=public, domain_id=0)
rclpy.init(context=private, domain_id=1)
inside = rclpy.create_node("shim_out", context=private)
out = inside.create_publisher(Twist, "/model/vehicle_blue/cmd_vel", 10)
node = rclpy.create_node("vehicle_interface", context=public)


def on_drive(msg):
    t = Twist()
    t.linear.x = msg.drive.speed
    t.angular.z = msg.drive.speed * math.tan(msg.drive.steering_angle) / WHEELBASE
    out.publish(t)


node.create_subscription(AckermannDriveStamped, "/drive", on_drive, 10)
executor = SingleThreadedExecutor(context=public)
executor.add_node(node)
print("ackermann shim up", flush=True)
try:
    executor.spin()
except KeyboardInterrupt:
    pass
