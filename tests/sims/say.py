"""Say sentences to Eddy the way a person would at `eddy shell`, and print each reply with
what the robot's own topics then report. Usage: say.py [--robot SPEC] "sentence" ..."""
import math
import sys
import time

from eddy import Eddy
from eddy.adapters import open_robot
from eddy.senses.speak import Speaker

args = sys.argv[1:]
spec = "ros2"
if args and args[0] == "--robot":
    spec, args = args[1], args[2:]
adapter = open_robot(spec)
eddy = Eddy(adapter, Speaker("silent"))
print(eddy.start().summary(), "\n", flush=True)
for sentence in args:
    a, t0 = adapter.state(), time.time()
    r = eddy.tell(sentence)
    time.sleep(0.5)
    b = adapter.state()
    bits = []
    if b.has_odometry:
        turned = math.degrees(math.atan2(math.sin(b.heading - a.heading), math.cos(b.heading - a.heading)))
        bits.append(f"moved {math.hypot(b.x - a.x, b.y - a.y):.2f} m, turned {turned:+.0f} deg")
    changed = {n: v for n, v in b.joints.items() if abs(v - a.joints.get(n, v)) > 0.005}
    if changed:
        bits.append("joints now " + ", ".join(f"{n}={v:+.3f}" for n, v in changed.items()))
    if b.range_ahead is not None:
        bits.append(f"range {b.range_ahead:.2f} m")
    print(f"> {sentence}\n    {'OK  ' if r.ok else 'FAIL'} {r.message}\n    robot: {'; '.join(bits) or 'no change'}  ({time.time() - t0:.1f} s)", flush=True)
eddy.close()
