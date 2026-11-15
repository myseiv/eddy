"""Eddy on a Franka Panda run by stock ros2_control controllers (see panda_up.sh).
Arm positions are checked against robot_state_publisher's TF, which is computed independently
of Eddy's own kinematics."""
import math
import sys
import time

import rclpy
import tf2_ros
from rclpy.executors import SingleThreadedExecutor
import threading

from eddy import Eddy
from eddy.adapters.ros2 import Ros2Adapter
from eddy.senses.speak import Speaker

failures = []


def check(label, ok, detail=""):
    if not ok:
        failures.append(label)
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  ({detail})" if detail else ""), flush=True)


adapter = Ros2Adapter()
eddy = Eddy(adapter, Speaker("silent"))
model = eddy.start()
print(model.summary(), "\n", flush=True)

# A second node that only listens to TF.
tf_node = rclpy.create_node("panda_tf_check")
buffer = tf2_ros.Buffer()
listener = tf2_ros.TransformListener(buffer, tf_node)
ex = SingleThreadedExecutor(); ex.add_node(tf_node)
threading.Thread(target=ex.spin, daemon=True).start()
time.sleep(1.5)


def tf_nearest(target):
    """The link TF places closest to `target`, and how far away it is."""
    best = (None, 9.9)
    for link in ("panda_link7", "panda_link8", "panda_hand"):
        try:
            t = buffer.lookup_transform("panda_link0", link, rclpy.time.Time()).transform.translation
            d = math.dist((t.x, t.y, t.z), target)
            if d < best[1]:
                best = (link, d)
        except Exception:
            pass
    return best


def tell(text):
    r = eddy.tell(text)
    time.sleep(0.6)
    print(f"> {text}\n    {'OK  ' if r.ok else 'FAIL'} {r.message}", flush=True)
    return r


caps = set(model.capabilities())
arm = [c["joints"] for c in adapter._controllers.values() if not c["gripper"]]
grip = [c["joints"] for c in adapter._controllers.values() if c["gripper"]]
check("finds the arm, its geometry and the gripper", {"joints", "reach", "gripper"} <= caps, ", ".join(sorted(caps)))
check("learns the arm's seven joints from the controller", arm == [[f"panda_joint{i}" for i in range(1, 8)]], str(arm))
check("learns the gripper's joint from the gripper controller", grip == [["panda_finger_joint1"]], str(grip))

r = tell("move joint1 to 20 degrees")
check("moves one joint, named loosely", r.ok and abs(math.degrees(adapter.state().joints["panda_joint1"]) - 20) < 1, r.message)
r = tell("set panda_joint2 to 300 degrees")
check("refuses a target past the joint's limit", not r.ok and "can only go" in r.message)

for target in ((0.4, 0.1, 0.5), (0.3, -0.2, 0.6), (0.5, 0.0, 0.3)):
    r = tell("reach %.2f %.2f %.2f" % target)
    link, off = tf_nearest(target)
    check(f"reach {target}: TF agrees", r.ok and off < 0.01, f"{link} is {off * 1000:.1f} mm from the target")
before = buffer.lookup_transform("panda_link0", link, rclpy.time.Time()).transform.translation
r = tell("move the arm up 5 cm")
time.sleep(0.5)
after = buffer.lookup_transform("panda_link0", link, rclpy.time.Time()).transform.translation
check("relative move: up 5 cm", r.ok and abs((after.z - before.z) - 0.05) < 0.005 and abs(after.x - before.x) < 0.005,
      f"dz {after.z - before.z:+.3f}, dx {after.x - before.x:+.3f}")
r = tell("reach 2 0 0")
check("refuses a point out of reach", not r.ok)

r = tell("close the gripper"); time.sleep(1.0)
closed = adapter.state().joints.get("panda_finger_joint1")
r2 = tell("open the gripper"); time.sleep(1.0)
opened = adapter.state().joints.get("panda_finger_joint1")
check("gripper closes and opens", r.ok and r2.ok and closed is not None and closed < 0.005 and opened > 0.03,
      f"finger {closed} -> {opened}; {r.message}")
r = tell("home")
check("goes home", r.ok, r.message)
r = tell("drive forward 1 meter")
check("refuses to drive (it is an arm)", not r.ok)

eddy.close()
print("\npanda: " + ("all checks passed" if not failures else "FAILED: " + "; ".join(failures)))
sys.exit(1 if failures else 0)
