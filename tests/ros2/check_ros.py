"""End-to-end check of Eddy on a live ROS 2 graph. Run by tests/ros2/run.sh inside a ROS container,
with mock_robot.py already running. Eddy is told nothing about the robot: it has to work it out."""
import math
import sys

from eddy import Eddy
from eddy.adapters.ros2 import Ros2Adapter
from eddy.kinematics import demo_arm
from eddy.senses.speak import Speaker

failures = []


def check(label, ok, detail=""):
    """Record one result and print it as a PASS or FAIL line."""
    if not ok:
        failures.append(label)
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  ({detail})" if detail else ""), flush=True)


# Connect with no hints at all: everything below is what Eddy worked out from the ROS graph.
adapter = Ros2Adapter()
eddy = Eddy(adapter, Speaker("silent"))
model = eddy.start()
print(model.summary(), "\n", flush=True)

caps = set(model.capabilities())
check("reads the graph and finds what the robot can do",
      {"drive", "joints", "reach", "gripper", "odometry", "range", "joint_state"} <= caps, ", ".join(sorted(caps)))
check("picks /cmd_vel, not the Twist topic nobody listens to", model.details.get("cmd_vel") == "/cmd_vel")
check("names the robot from its URDF", model.name == "mockbot")
check("learns each controller's joints from the controller itself",
      [c["joints"] for c in adapter._controllers.values() if not c["gripper"]] == [["base_yaw", "shoulder", "elbow", "wrist"]]
      and [c["joints"] for c in adapter._controllers.values() if c["gripper"]] == [["finger"]],
      ", ".join(adapter._controllers))
check("reads joint limits from the URDF", model.joint("shoulder").lower == -1.9 and model.joint("finger").upper == 0.04)
check("builds the arm chain without the gripper finger or wheels",
      model.chain.joint_names == ["base_yaw", "shoulder", "elbow", "wrist"] and abs(model.chain.reach() - 0.63) < 1e-6)

# Driving, measured from the robot's own odometry.
r = eddy.tell("drive forward 50 cm")
s = adapter.state()
check("drive forward 50 cm (measured from /odom)", r.ok and abs(s.x - 0.5) < 0.05 and abs(s.y) < 0.03, f"x={s.x:.3f}")

r = eddy.tell("turn left 90 degrees")
s = adapter.state()
check("turn left 90 degrees", r.ok and abs(math.degrees(s.heading) - 90) < 6, f"heading={math.degrees(s.heading):.1f}")

# The wall is closer than 4 m: Eddy must stop for it using the laser scan.
r = eddy.tell("drive forward 4 meters")
s = adapter.state()
check("stops for the wall seen on /scan", "in the way" in r.message and 0.1 < s.range_ahead < 0.35 and s.y < 1.85,
      f"{r.message} range={s.range_ahead:.2f}")

# Arm: solve inverse kinematics, send it through the trajectory controller, and check where
# the tool ended up (the mock arm has the same geometry as demo_arm).
target = (0.2, 0.1, 0.3)
r = eddy.tell("reach 0.2 0.1 0.3")
joints = adapter.state().joints
tool = demo_arm().position([joints[n] for n in ["base_yaw", "shoulder", "elbow", "wrist"]])
err = math.dist(tool, target)
check("arm reaches a point through the trajectory controller", r.ok and err < 0.01, f"{r.message} {err * 1000:.1f} mm off")

# Gripper: the finger joint should travel from open (0.04) to closed (0).
before = adapter.state().joints["finger"]
r = eddy.tell("close the gripper")
adapter.wait(0.5)
after = adapter.state().joints["finger"]
check("gripper closes", r.ok and before > 0.03 and after < 0.005, f"finger {before:.3f} -> {after:.3f}")

r = eddy.tell("home")
check("arm goes home", r.ok and all(abs(adapter.state().joints[n]) < 0.03 for n in model.chain.joint_names), r.message)

r = eddy.tell("status")
check("reports its state", r.ok and "mockbot" in r.message, r.message)
eddy.close()

print("\n" + ("All ROS 2 checks passed." if not failures else f"FAILED: {'; '.join(failures)}"))
sys.exit(1 if failures else 0)
