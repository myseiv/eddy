"""Eddy flying Gazebo's stock X3 quadcopter, with every result compared against the
simulator's true pose. Sentences are the ones a person would type."""
import math
import re
import subprocess
import sys
import time

from eddy import Eddy
from eddy.adapters import open_robot
from eddy.senses.speak import Speaker

failures, notes = [], []


def truth():
    """The drone's true x, y, z and heading, straight from the simulator."""
    for _ in range(5):
        out = subprocess.run(["ign", "topic", "-e", "-t", "/world/multicopter/dynamic_pose/info", "-n", "1"],
                             capture_output=True, text=True, timeout=20).stdout
        m = re.search(r'name: "[xX]3"\s+id: \d+\s+position \{([^}]*)\}(?:\s+orientation \{([^}]*)\})?', out)
        if m:
            p = dict(re.findall(r"(\w): (\S+)", m.group(1)))
            q = dict(re.findall(r"(\w): (\S+)", m.group(2) or ""))
            return (float(p.get("x", 0)), float(p.get("y", 0)), float(p.get("z", 0)),
                    2 * math.atan2(float(q.get("z", 0)), float(q.get("w", 1))))
    raise RuntimeError("no pose from the simulator")


def check(label, ok, detail=""):
    if not ok:
        failures.append(label)
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  ({detail})" if detail else ""), flush=True)


def tell(text):
    a, t0 = truth(), time.time()
    r = eddy.tell(text)
    took = time.time() - t0
    time.sleep(1.5)
    b = truth()
    flat = math.hypot(b[0] - a[0], b[1] - a[1])
    turned = math.degrees(math.atan2(math.sin(b[3] - a[3]), math.cos(b[3] - a[3])))
    print(f"> {text}\n    {'OK  ' if r.ok else 'FAIL'} {r.message}\n    truth: height {a[2]:.2f} -> {b[2]:.2f} m, moved {flat:.2f} m "
          f"across the ground, turned {turned:+.0f} deg  ({took:.1f} s)", flush=True)
    return r, b[2] - a[2], flat, turned, b


adapter = open_robot("ros2")
eddy = Eddy(adapter, Speaker("silent"))
model = eddy.start()
print(model.summary(), "\n", flush=True)
caps = set(model.capabilities())
check("finds the velocity command and odometry with no configuration", {"drive", "odometry"} <= caps,
      f"{', '.join(sorted(caps))}; command on {model.details.get('cmd_vel')}")

r, dz, flat, turned, b = tell("status")
ground = b[2]
r, dz, flat, turned, b = tell("take off")
check("takes off to 1 m (true height gain within 0.15 m)", r.ok and abs(dz - 1.0) < 0.15, f"{dz:+.2f} m")
check("now knows it is a flying robot", model.drive.kind == "flying", model.drive.kind)
notes.append(f"take off (1 m asked): truly {dz:+.2f} m")
time.sleep(4)
drift = abs(truth()[2] - b[2])
check("hovers: height holds for 4 s (within 0.1 m)", drift < 0.1, f"drifted {drift:.2f} m")
r, dz, flat, turned, b = tell("go up 50 cm")
check("climbs 50 cm (within 0.1 m)", r.ok and abs(dz - 0.5) < 0.1, f"{dz:+.2f} m")
r, dz, flat, turned, b = tell("fly forward 2 meters")
check("flies forward 2 m (within 0.3 m, height held within 0.15 m)", r.ok and abs(flat - 2.0) < 0.3 and abs(dz) < 0.15,
      f"{flat:.2f} m, height changed {dz:+.2f}")
notes.append(f"fly forward 2 m: truly {flat:.2f} m")
r, dz, flat, turned, b = tell("turn left 90 degrees")
check("turns left 90 degrees in the air (within 15)", r.ok and abs(turned - 90) < 15, f"{turned:+.0f}")
notes.append(f"turn left 90: truly {turned:+.0f}")
r, dz, flat, turned, b = tell("fly forward 1 meter and turn right 90 degrees")
check("two commands in one sentence", r.ok and abs(flat - 1.0) < 0.3 and abs(turned + 90) < 15, f"{flat:.2f} m, {turned:+.0f}")
r, dz, flat, turned, b = tell("go back 1 meter")
check("flies backwards 1 m (within 0.3 m)", r.ok and abs(flat - 1.0) < 0.3, f"{flat:.2f} m")
r, dz, flat, turned, b = tell("status")
check("status gives its height", "height" in r.message, r.message)
r, dz, flat, turned, b = tell("descend 50 cm")
check("descends 50 cm (within 0.1 m)", r.ok and abs(dz + 0.5) < 0.1, f"{dz:+.2f} m")
r, dz, flat, turned, b = tell("land")
check("lands (true height back within 0.1 m of the ground)", r.ok and abs(b[2] - ground) < 0.1, f"height {b[2]:.2f} m, ground {ground:.2f}")
r, dz, flat, turned, b = tell("land")
check("landing again says it is already down", r.ok and "already" in r.message, r.message)
r, dz, flat, turned, b = tell("take off to 2 meters")
check("takes off to 2 m (within 0.2 m)", r.ok and abs(dz - 2.0) < 0.2, f"{dz:+.2f} m")
r, dz, flat, turned, b = tell("land")
check("lands again", r.ok and abs(b[2] - ground) < 0.1, f"height {b[2]:.2f} m")
r, *_ = tell("open the gripper")
check("refuses what it does not have", not r.ok)

eddy.close()
print("\nmeasured:\n  " + "\n  ".join(notes))
print("\ndrone: " + ("all checks passed" if not failures else "FAILED: " + "; ".join(failures)))
sys.exit(1 if failures else 0)
