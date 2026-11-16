"""Measure how well the laser's reading of a turn agrees with the truth (Gazebo, diff_drive world).

Not a pass/fail test: it prints, for turns in the open and turns hard up against an obstacle,
what the odometry said, what the laser said and what the simulator says really happened. Used
to choose how far the laser may fall short of the odometry before a turn is called jammed.

    ./tests/sims/run.sh measure
"""
import math
import re
import subprocess
import sys
import time

sys.path.insert(0, "/eddy/src")
from eddy.adapters import open_robot
from eddy.brain import Eddy
from eddy.senses.speak import Speaker

MODEL = "vehicle_blue"


def truth():
    for _ in range(5):
        out = subprocess.run(["ign", "topic", "-e", "-n", "1", "-t", "/world/diff_drive/pose/info"],
                             capture_output=True, text=True, timeout=20).stdout
        m = re.search(r'name: "%s"\s+id: \d+\s+position \{([^}]*)\}\s+orientation \{([^}]*)\}' % MODEL, out)
        if m:
            q = dict(re.findall(r"(\w): (-?[\d.e-]+)", m.group(2)))
            z, w = float(q.get("z", 0)), float(q.get("w", 1))
            return 2 * math.atan2(z, w)
    raise RuntimeError("no pose")


eddy = Eddy(open_robot("ros2"), Speaker("silent"))
eddy.start()


def turn(sentence, where):
    eddy.turn_check = None
    a = truth()
    r = eddy.tell(sentence)
    time.sleep(1.0)
    b = truth()
    true = abs(math.degrees(math.atan2(math.sin(b - a), math.cos(b - a))))
    c = eddy.turn_check
    if c is None:
        print(f"TURN {where:<8} {sentence:<24} no laser reading; truly {true:.0f}; {r.message[:60]}", flush=True)
        return
    odo, laser = math.degrees(c["odometry"]), math.degrees(c["laser"])
    # For "turn around" the true turn of 180 or so wraps; use the laser/odometry sign-free values.
    if "around" in sentence and true < 90 and odo > 150:
        true = 360 - true if true > 180 else true
    print(f"TURN {where:<8} {sentence:<24} odometry {odo:6.1f}  laser {laser:6.1f}  truth {true:6.1f}  "
          f"laser/odometry {laser / max(odo, 1e-6):.2f}  truth/odometry {true / max(odo, 1e-6):.2f}  "
          f"{'ok' if r.ok else 'REFUSED'}: {r.message[:50]}", flush=True)


for sentence in ("turn left 90 degrees", "turn right 90 degrees", "turn left 45 degrees", "turn right 45 degrees",
                 "turn around", "turn right 180 degrees", "turn left 30 degrees", "turn right 30 degrees",
                 "turn left 120 degrees", "turn right 120 degrees"):
    turn(sentence, "open")
# Hard up against the obstacle, several times, approaching from slightly different headings.
for nudge in ("", "turn left 10 degrees", "turn right 20 degrees", "turn left 10 degrees"):
    if nudge:
        eddy.tell(nudge)
    eddy.tell("drive forward 10 meters")
    turn("turn around", "obstacle")
    eddy.tell("drive forward 1 meter")          # away from it again, facing the open side
    turn("turn around", "obstacle")             # back to face it, from a metre off
eddy.tell("stop")
eddy.close()
