"""What Eddy does when things go wrong on a live ROS 2 robot: a sensor goes quiet in the
middle of a drive, and "stop" arrives while it is moving. Run by tests/ros2/run.sh with
mock_robot.py running. (The mock can be told to stop publishing a sensor on /mock/fault.)"""
import math
import sys
import threading
import time

import rclpy
from std_msgs.msg import String

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
eddy.start()
fault = adapter.node.create_publisher(String, "/mock/fault", 10)
time.sleep(0.5)


def during(sentence, after, action):
    """Say a sentence, and `after` seconds into it do something. Returns (result, seconds it took)."""
    out = {}
    began = time.time()
    t = threading.Thread(target=lambda: out.update(r=eddy.tell(sentence)))
    t.start()
    time.sleep(after)
    action()
    t.join()
    return out["r"], time.time() - began


def moving():
    a = adapter.state(); time.sleep(0.4); b = adapter.state()
    return math.hypot(b.x - a.x, b.y - a.y) > 0.005 or abs(b.heading - a.heading) > 0.01


def home():
    fault.publish(String(data="all_on")); time.sleep(1.2)


# The earlier checks leave the robot parked facing a wall. Face the open room first.
eddy.tell("turn around")

# 1. The laser goes quiet 1 s into a forward drive.
r, took = during("drive forward 1.5 meters", 1.0, lambda: fault.publish(String(data="scan_off")))
check("laser goes quiet mid-drive: Eddy stops within about 2 s and says why",
      not r.ok and "range sensor" in r.message and took < 3.5, f"{took:.1f} s; {r.message}")
r = eddy.tell("drive forward 20 cm")
check("refuses to drive on while the laser is quiet", not r.ok and "range sensor" in r.message, r.message)
r = eddy.tell("turn left 30 degrees")
check("still turns on the spot (that needs no laser)", r.ok, r.message)
home()
r = eddy.tell("drive forward 10 cm")
check("drives again once the laser is back", r.ok, r.message)

# 2. The odometry goes quiet 1 s into a drive.
r, took = during("drive forward 1 meter", 1.0, lambda: fault.publish(String(data="odom_off")))
check("odometry goes quiet mid-drive: Eddy gives up within a few seconds",
      not r.ok and ("no longer getting anywhere" in r.message or "no effect" in r.message) and took < 8, f"{took:.1f} s; {r.message}")
check("and the robot is not left moving", not moving())
home()

# 3. "stop" while driving: how long until the robot is actually still.
x = {}
def stop_now():
    x["t0"] = time.time(); x["r"] = eddy.tell("stop"); x["t1"] = time.time()
r, took = during("drive forward 1 meter", 1.0, stop_now)
still_after = None
for i in range(20):
    if not moving():
        still_after = time.time() - x["t0"]; break
check("stop during a drive: obeyed, and the drive reports it was interrupted",
      x["r"].ok and "Interrupted" in r.message, r.message)
check("stop takes effect quickly", x["t1"] - x["t0"] < 0.5 and still_after is not None and still_after < 1.5,
      f"stop returned in {x['t1'] - x['t0']:.2f} s; robot still within {still_after if still_after is None else round(still_after, 1)} s")

eddy.close()
print("\nfaults: " + ("all checks passed" if not failures else "FAILED: " + "; ".join(failures)))
sys.exit(1 if failures else 0)
