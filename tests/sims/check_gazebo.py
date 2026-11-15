"""Eddy driving one of Gazebo's stock vehicles, with every result compared against the
simulator's true pose (not the robot's own odometry).
Usage: check_gazebo.py WORLD MODEL [ROBOT_SPEC]      e.g.  check_gazebo.py diff_drive vehicle_blue ros2"""
import math
import re
import subprocess
import sys
import time

from eddy import Eddy
from eddy.adapters import open_robot
from eddy.senses.speak import Speaker

WORLD, MODEL = sys.argv[1], sys.argv[2]
SPEC = sys.argv[3] if len(sys.argv) > 3 else "ros2"
FRONT_X = float(sys.argv[4]) if len(sys.argv) > 4 else 10.0     # x of whatever is in this vehicle's lane ahead
OTHER = sys.argv[5] if len(sys.argv) > 5 else None              # another vehicle that must not move
FRONT, REAR = 0.85, 1.15            # how far the body reaches ahead of and behind the model origin (m)
WALL_FRONT, WALL_BACK = 10.0, -4.0  # x of the wall faces (see make_world.py)
failures, notes = [], []


def truth(model=None):
    """The vehicle's true x, y and heading, straight from the simulator."""
    MODEL = model or globals()["MODEL"]
    for _ in range(5):
        out = subprocess.run(["ign", "topic", "-e", "-t", f"/world/{WORLD}/dynamic_pose/info", "-n", "1"],
                             capture_output=True, text=True, timeout=20).stdout
        m = re.search(r'name: "%s"\s+id: \d+\s+position \{([^}]*)\}\s+orientation \{([^}]*)\}' % MODEL, out)
        if m:
            p = dict(re.findall(r"(\w): (\S+)", m.group(1)))
            q = dict(re.findall(r"(\w): (\S+)", m.group(2)))
            z, w = float(q.get("z", 0)), float(q.get("w", 1))
            return float(p.get("x", 0)), float(p.get("y", 0)), 2 * math.atan2(z, w)
    raise RuntimeError("no pose from the simulator")


def check(label, ok, detail=""):
    if not ok:
        failures.append(label)
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  ({detail})" if detail else ""), flush=True)


def tell(text):
    a, t0 = truth(), time.time()
    r = eddy.tell(text)
    took = time.time() - t0
    time.sleep(1.0)
    b = truth()
    moved = math.hypot(b[0] - a[0], b[1] - a[1])
    turned = math.degrees(math.atan2(math.sin(b[2] - a[2]), math.cos(b[2] - a[2])))
    print(f"> {text}\n    {'OK  ' if r.ok else 'FAIL'} {r.message}\n    truth: moved {moved:.2f} m, turned {turned:+.0f} deg, "
          f"now at ({b[0]:+.2f}, {b[1]:+.2f}) facing {math.degrees(b[2]):+.0f}  ({took:.1f} s)", flush=True)
    return r, moved, turned, b, took


other_start = truth(OTHER) if OTHER else None
adapter = open_robot(SPEC)
eddy = Eddy(adapter, Speaker("silent"))
model = eddy.start()
print(model.summary(), "\n", flush=True)
caps = set(model.capabilities())
check("finds drive, odometry and the laser with no configuration", {"drive", "odometry", "range"} <= caps, ", ".join(sorted(caps)))

r, moved, turned, b, took = tell("status")
check("status gives a position and a range", "nearest thing ahead" in r.message and "I am at" in r.message)

# Backing into the wall behind. There is no rear sensor, so the only protection is noticing
# that the robot has stopped getting anywhere.
x0 = b[0]
gap = (x0 - REAR) - WALL_BACK
r, moved, turned, b, took = tell("go back 5 meters")
pushing = took - gap / 0.2
check("backing into a wall: notices and gives up within a few seconds",
      not r.ok and ("no longer getting anywhere" in r.message or "slipping" in r.message) and pushing < 6, f"{r.message[:90]}; about {max(0, pushing):.0f} s after contact")
notes.append(f"backing into the wall: reply came about {max(0, pushing):.0f} s after contact")

r, moved, turned, b, took = tell("drive forward 2 meters")
check("drives 2 m (true distance within 0.25 m)", r.ok and abs(moved - 2.0) < 0.25, f"{moved:.2f} m")
notes.append(f"asked 2.00 m forward: truly {moved:.2f} m")
r, moved, turned, b, took = tell("turn left 90 degrees")
car = model.drive.kind == "car-like"
# Gazebo's car-steered vehicle reports about 20% more turn than it makes in a tight circle,
# and Eddy can only go by that odometry, so the car is allowed 25 degrees instead of 15.
TOL = 25 if car else 15
check(f"turns left 90 degrees (true angle within {TOL})", r.ok and abs(turned - 90) < TOL, f"{turned:+.0f}")
if car:
    check("says it had to roll forward to turn", "cannot turn on the spot" in r.message, r.message[:120])
notes.append(f"asked 90 left: truly {turned:+.0f}")
r, moved, turned, b, took = tell("turn right 90 degrees")
check(f"turns right 90 degrees (true angle within {TOL})", r.ok and abs(turned + 90) < TOL, f"{turned:+.0f}")
notes.append(f"asked 90 right: truly {turned:+.0f}")
r, moved, turned, b, took = tell("go forward slowly 50 cm")
check("drives 50 cm slowly (within 0.15 m)", r.ok and abs(moved - 0.5) < 0.15, f"{moved:.2f} m")
r, moved, turned, b, took = tell("drive forward 10 meters")
front_gap = FRONT_X - (b[0] + FRONT * math.cos(b[2]))
check("stops for what is ahead without touching it", "in the way" in r.message and front_gap > 0.02, f"front is {front_gap:.2f} m from it")
notes.append(f"wall stop: front {front_gap:.2f} m from the wall")
r, moved, turned, b, took = tell("drive forward 1 meter")
check("refuses to drive on into the wall", not r.ok and moved < 0.05, r.message)
# A car needs room in front to turn, so it backs away further.
back = 4 if car else 1
r, moved, turned, b, took = tell(f"go back {back} meters")
check(f"backs away {back} m", r.ok and abs(moved - back) < 0.2, f"{moved:.2f} m")
r, moved, turned, b, took = tell("turn left 90 degrees")
check(f"turns left 90 degrees again, in the open (within {TOL})", r.ok and abs(turned - 90) < TOL, f"{turned:+.0f}")
r, moved, turned, b, took = tell("wander for 40 seconds")
check("wanders for 40 s", r.ok, r.message)
if not car:
    # Turning on the spot right up against something can jam the body (Eddy does not know the
    # robot's outline). Whatever happens, Eddy must not report a turn that did not happen.
    tell("drive forward 10 meters")
    r, moved, turned, b, took = tell("turn around")
    honest = (not r.ok) or abs(abs(turned) - 180) < 25
    check("turn around right next to an obstacle: completes, or says it is jammed", honest, f"{r.message[:110]}; truly {turned:+.0f}")
    notes.append(f"turn around next to an obstacle: '{r.message[:70]}', truly {turned:+.0f}")
r, *_ = tell("stop")
check("stop", r.ok)

if OTHER:
    o = truth(OTHER)
    drift = math.hypot(o[0] - other_start[0], o[1] - other_start[1])
    check(f"the other robot ({OTHER}) was left alone", drift < 0.02, f"it moved {drift:.3f} m")
    check("Eddy says which robot it is using and that another is there",
          "other robots here" in model.details and OTHER in str(model.details.get("other robots here")), str(model.details.get("this robot")))
s = adapter.state()
t = truth()
notes.append(f"end of run: odometry says heading {math.degrees(s.heading):+.0f}, truth {math.degrees(t[2]):+.0f}")
eddy.close()
print("\nmeasured:\n  " + "\n  ".join(notes))
print(f"\n{MODEL}: " + ("all checks passed" if not failures else "FAILED: " + "; ".join(failures)))
sys.exit(1 if failures else 0)
