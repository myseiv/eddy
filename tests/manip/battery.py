"""One set of checks for robots that drive AND have an arm, run unchanged on several of them.

    python3 tests/manip/battery.py sim        Eddy's own simulator (no ROS)
    python3 tests/manip/battery.py mock       the ROS 2 mock mobile manipulator (tests/ros2/mock_robot.py)
    python3 tests/manip/battery.py landerpi   the LanderPi stand-in (tests/ros2/mock_hiwonder.py)
    python3 tests/manip/battery.py gzpanda    Gazebo's stock vehicle and a stock Franka Panda on one
                                              ROS graph (tests/sims), neither written for Eddy

The ROS ones are started by tests/manip/run.sh. Every result is compared with what the robot
really did: the simulator's own state, or the true pose and joints its software reports. Where
the tool's position is checked, it is computed from the robot's true joint angles.
"""
import json
import math
import re
import subprocess
import sys
import threading
import time

WHICH = sys.argv[1] if len(sys.argv) > 1 else "sim"
failures = []


def check(label, ok, detail=""):
    if not ok:
        failures.append(label)
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  ({detail})" if detail else ""), flush=True)


# ---- the robots ---------------------------------------------------------------------

class Target:
    """How to read what a robot really did. Subclasses fill in pose() and joints()."""
    drive_tol, tool_tol, settle = 0.04, 0.01, 0.4

    def finger(self):
        """A number that grows as the gripper opens, or None."""
        return None

    def wait(self, seconds):
        time.sleep(seconds)


class SimTarget(Target):
    def __init__(self):
        from eddy.adapters.sim import SimAdapter
        self.sim = SimAdapter(name="sim-manipulator", room=30.0).go_live()   # real time, so commands can overlap
        self.adapter = self.sim

    def pose(self):
        return self.sim.x, self.sim.y, self.sim.heading

    def joints(self):
        return dict(self.sim.joints)

    def finger(self):
        return self.sim.gripper


class MockTarget(Target):
    """The mock's odometry and joint states are its simulator's true state, by construction."""
    drive_tol = 0.05

    def __init__(self):
        from eddy.adapters.ros2 import Ros2Adapter
        self.adapter = Ros2Adapter()

    def pose(self):
        s = self.adapter.state()
        return s.x, s.y, s.heading

    def joints(self):
        return dict(self.adapter.state().joints)

    def finger(self):
        return self.adapter.state().joints.get("finger")


class LanderPiTarget(Target):
    """The stand-in writes its true state to /tmp/truth.json every tick."""
    drive_tol = 0.04

    def __init__(self):
        from eddy.adapters.ros2 import Ros2Adapter
        self.adapter = Ros2Adapter()

    def _truth(self):
        with open("/tmp/truth.json") as f:
            return json.load(f)

    def pose(self):
        t = self._truth()
        return t["x"], t["y"], t["heading"]

    def joints(self):
        return dict(self._truth()["joints"])

    def finger(self):
        return self._truth()["joints"].get("r_joint")


class GzPandaTarget(Target):
    """Base: Gazebo's true pose of vehicle_blue. Arm: the Panda's joints from its mock hardware."""
    drive_tol, settle = 0.12, 1.0

    def __init__(self):
        from eddy.adapters.ros2 import Ros2Adapter
        self.adapter = Ros2Adapter()

    def pose(self):
        for _ in range(5):
            out = subprocess.run(["ign", "topic", "-e", "-t", "/world/diff_drive/dynamic_pose/info", "-n", "1"],
                                 capture_output=True, text=True, timeout=20).stdout
            m = re.search(r'name: "vehicle_blue"\s+id: \d+\s+position \{([^}]*)\}\s+orientation \{([^}]*)\}', out)
            if m:
                p = dict(re.findall(r"(\w): (\S+)", m.group(1)))
                q = dict(re.findall(r"(\w): (\S+)", m.group(2)))
                return float(p.get("x", 0)), float(p.get("y", 0)), 2 * math.atan2(float(q.get("z", 0)), float(q.get("w", 1)))
        raise RuntimeError("no pose from Gazebo")

    def joints(self):
        return dict(self.adapter.state().joints)

    def finger(self):
        return self.adapter.state().joints.get("panda_finger_joint1")


T = {"sim": SimTarget, "mock": MockTarget, "landerpi": LanderPiTarget, "gzpanda": GzPandaTarget}[WHICH]()

from eddy.brain import Eddy                     # noqa: E402  (after the target, which may start ROS)
from eddy.senses.speak import Speaker           # noqa: E402

eddy = Eddy(T.adapter, Speaker("silent"))
model = eddy.start()
chain = model.chain
print(model.summary(), "\n", flush=True)


def tell(text):
    """Say a sentence; return the result and the robot's true state before and after."""
    a, ja = T.pose(), T.joints()
    t0 = time.time()
    r = eddy.tell(text)
    took = time.time() - t0
    T.wait(T.settle)
    b, jb = T.pose(), T.joints()
    moved = math.hypot(b[0] - a[0], b[1] - a[1])
    print(f"> {text}\n    {'OK  ' if r.ok else 'FAIL'} {r.message}\n    truth: base moved {moved:.3f} m, "
          f"arm joints moved up to {arm_change(ja, jb):.3f} rad ({took:.1f} s)", flush=True)
    return r, moved, ja, jb


def arm_change(ja, jb):
    return max((abs(jb.get(n, 0.0) - ja.get(n, 0.0)) for n in chain.joint_names), default=0.0)


def tool(joints=None):
    """Where the tool really is (arm base frame), from the true joint angles."""
    j = joints or T.joints()
    return tuple(float(v) for v in chain.position([j.get(n, 0.0) for n in chain.joint_names]))


def in_background(text):
    out = []
    t = threading.Thread(target=lambda: out.append(eddy.tell(text)))
    t.start()
    return t, out


# Two arm poses this arm can certainly reach: 40 and 65 percent of the way across each joint's range.
lo_hi = [(j.lower, j.upper) for j in chain.active]
A = tool(dict(zip(chain.joint_names, [lo + 0.40 * (hi - lo) for lo, hi in lo_hi])))
B = tool(dict(zip(chain.joint_names, [lo + 0.65 * (hi - lo) for lo, hi in lo_hi])))
reach = lambda p: "reach %.3f %.3f %.3f" % p

# ---- the checks ---------------------------------------------------------------------

caps = set(model.capabilities())
check("finds both the base and the arm", {"drive", "reach", "gripper"} <= caps, ", ".join(sorted(caps)))
check("the arm chain holds no wheels and no gripper fingers",
      not any(w in n.lower() for n in chain.joint_names for w in ("wheel", "finger", "jaw", "gripper")), ", ".join(chain.joint_names))

eddy.tell("home")
r, moved, ja, jb = tell("drive forward 50 cm")
check("drives 50 cm with the arm at home", r.ok and abs(moved - 0.5) < T.drive_tol, f"{moved:.3f} m")

r, moved, ja, jb = tell(reach(A))
err = math.dist(tool(jb), A)
check("reaches a point after driving (inverse kinematics on the real joints)", r.ok and err < T.tool_tol,
      f"{err * 1000:.1f} mm off; base moved {moved:.3f} m")

r, moved, ja, jb = tell("drive forward 30 cm")
check("drives with the arm held out, and the arm stays put", r.ok and abs(moved - 0.3) < T.drive_tol and arm_change(ja, jb) < 0.02,
      f"base {moved:.3f} m, arm moved {arm_change(ja, jb):.3f} rad")

# A relative move from a pose where it is possible: Eddy's own solver is asked first, so this checks
# that the arm really goes where Eddy says, not how big the arm's workspace is.
start, step = None, None
for frac in (0.5, 0.45, 0.55, 0.35, 0.6):
    q = dict(zip(chain.joint_names, [lo + frac * (hi - lo) for lo, hi in lo_hi]))
    for dz, word in ((0.05, "up"), (-0.05, "down")):
        p = tool(q)
        if chain.ik((p[0], p[1], p[2] + dz), q0=[q[n] for n in chain.joint_names]).success:
            start, step = p, (dz, word)
            break
    if start:
        break
if start is None:
    check("moves the tool 5 cm up or down, relative to where it is", False, "no pose found where Eddy's solver can do it")
else:
    eddy.tell(reach(start))
    before = tool()
    r, moved, ja, jb = tell(f"move the arm {step[1]} 5 cm")
    after = tool(jb)
    check(f"moves the tool 5 cm {step[1]}, relative to where it is", r.ok and abs(after[2] - before[2] - step[0]) < 0.01
          and math.hypot(after[0] - before[0], after[1] - before[1]) < 0.01 and moved < 0.01,
          f"dz {after[2] - before[2]:+.3f}, sideways {math.hypot(after[0] - before[0], after[1] - before[1]):.3f}; {r.message}")

eddy.tell("go back 80 cm")

# Stop in the middle of an arm move.
eddy.tell(reach(A))
worker, out = in_background(reach(B))
time.sleep(0.35)
eddy.tell("stop")
worker.join(timeout=15)
q1 = T.joints()
T.wait(1.0)
q2 = T.joints()
short = math.dist(tool(q2), B)
check("stop interrupts an arm move, and the arm then holds still",
      out and "interrupted" in out[0].message.lower() and arm_change(q1, q2) < 0.02 and short > 0.02,
      f"{out[0].message if out else 'no answer'}; drifted {arm_change(q1, q2):.3f} rad after; {short * 100:.1f} cm short of the target")

# A drive arriving in the middle of an arm move takes over.
eddy.tell(reach(A))
pose0 = T.pose()
worker, out = in_background(reach(B))
time.sleep(0.35)
r = eddy.tell("drive forward 20 cm")
worker.join(timeout=15)
T.wait(T.settle)
q1, p1 = T.joints(), T.pose()
T.wait(1.0)
q2 = T.joints()
moved = math.hypot(p1[0] - pose0[0], p1[1] - pose0[1])
check("a drive takes over from an arm move: the arm holds, the base drives",
      out and "interrupted" in out[0].message.lower() and r.ok and abs(moved - 0.2) < T.drive_tol and arm_change(q1, q2) < 0.02,
      f"arm: {out[0].message if out else 'no answer'}; base {moved:.3f} m; {r.message}")

# Stop in the middle of a drive, with the arm out.
ja, p0 = T.joints(), T.pose()
worker, out = in_background("drive forward 1 meter")
time.sleep(1.2)
eddy.tell("stop")
worker.join(timeout=15)
T.wait(1.0)
jb, p1 = T.joints(), T.pose()
moved = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
check("stop ends a drive part way, and the arm is not disturbed",
      out and out[0].message.startswith("Interrupted") and 0.05 < moved < 0.7 and arm_change(ja, jb) < 0.02,
      f"base {moved:.3f} of 1 m; arm moved {arm_change(ja, jb):.3f} rad")
eddy.tell("go back %d cm" % round(moved * 100))

# Arm, then base, in one sentence.
eddy.tell("home")
r, moved, ja, jb = tell(reach(B) + " and drive forward 20 cm")
check("one sentence, arm then base, done in order", r.ok and "Then:" in r.message and math.dist(tool(jb), B) < T.tool_tol
      and abs(moved - 0.2) < T.drive_tol, f"tool {math.dist(tool(jb), B) * 1000:.1f} mm off, base {moved:.3f} m")

# Gripper, checked against the gripper's real joint.
f0 = T.finger()
r1 = eddy.tell("close the gripper"); T.wait(1.0); f1 = T.finger()
r2 = eddy.tell("open the gripper"); T.wait(1.0); f2 = T.finger()
check("the gripper really closes and opens", r1.ok and r2.ok and f1 is not None and f2 is not None and f2 - f1 > 0.01,
      f"gripper joint {f0} -> closed {f1} -> open {f2}")

# Refusals leave everything where it was.
ja, p0 = T.joints(), T.pose()
r_far = eddy.tell("reach 3 0 0.2")
first = chain.joint_names[0]
r_lim = eddy.tell(f"set joint {first} to 500 degrees" if not first[-1].isdigit() else f"set {first} to 500 degrees")
T.wait(T.settle)
jb, p1 = T.joints(), T.pose()
check("refuses a point out of reach and a joint past its limit, moving nothing",
      not r_far.ok and not r_lim.ok and arm_change(ja, jb) < 0.01 and math.hypot(p1[0] - p0[0], p1[1] - p0[1]) < 0.01,
      f"{r_far.message} / {r_lim.message}")

r, moved, ja, jb = tell("home")
off = max(abs(jb.get(j.name, 0.0) - max(j.lower, min(j.upper, 0.0))) for j in chain.active)
check("goes home (every arm joint to zero, or to its nearest limit)", r.ok and off < 0.03, f"furthest joint {off:.3f} rad off")

eddy.close()
print(f"\n{WHICH}: " + ("all checks passed" if not failures else "FAILED: " + "; ".join(failures)), flush=True)
sys.exit(1 if failures else 0)
