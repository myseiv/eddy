"""Run a battery of sentences through Eddy against mock_hiwonder.py and compare each reply
with the mock's true state. Run by tests/ros2/run_hiwonder.sh; EDDY_MOCK_KIND says which robot."""
import json
import math
import os
import sys
import time

from eddy import Eddy
from eddy.adapters.ros2 import Ros2Adapter
from eddy.senses.speak import Speaker

KIND = os.environ.get("EDDY_MOCK_KIND", "landerpi")
failures = []


def truth():
    with open("/tmp/truth.json") as f:
        return json.load(f)


def check(label, ok, detail=""):
    if not ok:
        failures.append(label)
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  ({detail})" if detail else ""), flush=True)


def tell(text):
    """Say one sentence to Eddy; return the result plus what truly changed."""
    a, t0 = truth(), time.time()
    r = eddy.tell(text)
    time.sleep(0.3)
    b = truth()
    moved = math.hypot(b["x"] - a["x"], b["y"] - a["y"])
    turned = math.degrees(math.atan2(math.sin(b["heading"] - a["heading"]), math.cos(b["heading"] - a["heading"])))
    print(f"> {text}\n    {'OK  ' if r.ok else 'FAIL'} {r.message}\n    truth: moved {moved:.2f} m, turned {turned:+.0f} deg, "
          f"{time.time() - t0:.1f} s, bumped={b['bumped']}", flush=True)
    return r, moved, turned, b


adapter = Ros2Adapter()
eddy = Eddy(adapter, Speaker("silent"))
model = eddy.start()
print(model.summary(), "\n", flush=True)
caps = set(model.capabilities())

check("picks the direct velocity topic", model.details.get("cmd_vel") == "/controller/cmd_vel", str(model.details.get("cmd_vel")))
check("finds odometry and the laser", {"odometry", "range"} <= caps, ", ".join(sorted(caps)))
check("does not list wheels as joints", not any("wheel" in j.name for j in model.joints))

r, moved, turned, _ = tell("status")
check("status reports a range ahead", "nearest thing ahead" in r.message)
r, moved, turned, _ = tell("drive forward 50 cm")
check("drives 50 cm", r.ok and abs(moved - 0.5) < 0.03, f"{moved:.2f}")
r, moved, turned, _ = tell("go back 30 cm")
check("reverses 30 cm", r.ok and abs(moved - 0.3) < 0.03, f"{moved:.2f}")
r, moved, turned, _ = tell("turn left 90 degrees")
check("turns left 90 degrees", r.ok and abs(turned - 90) < 4, f"{turned:.0f}")
if KIND == "car":
    check("says it had to roll forward to turn", "cannot turn on the spot" in r.message)
    check("is now known to be car-like", model.drive.kind == "car-like", model.drive.kind)
r, moved, turned, _ = tell("turn right 45 degrees")
check("turns right 45 degrees", r.ok and abs(turned + 45) < 4, f"{turned:.0f}")
r, moved, turned, _ = tell("turn left for 2 seconds")
check("timed turn turns", r.ok and turned > 20, f"{turned:.0f}")
r, moved, turned, _ = tell("turn right 90 degrees and go forward 20 cm")
check("two commands in one sentence", r.ok and "Then:" in r.message and abs(turned + 90) < 5, r.message)
r, moved, turned, b = tell("drive forward 6 meters")
check("stops for the wall on the laser, without touching it", "in the way" in r.message and not b["bumped"])
r, moved, turned, b = tell("go back 6 meters")
check("stops for the wall behind it without touching it (the laser sees all round)",
      "in the way" in r.message and not b["bumped"], r.message)
r, moved, turned, b = tell("drive forward 1 meter")
# Sideways: the mecanum base slides; the car is found not to, and is not moved.
a = truth()
r, moved, turned, b = tell("slide left 30 cm")
# How far it truly went to its own left, and forwards, from where it was.
left = (b["y"] - a["y"]) * math.cos(a["heading"]) - (b["x"] - a["x"]) * math.sin(a["heading"])
ahead = (b["x"] - a["x"]) * math.cos(a["heading"]) + (b["y"] - a["y"]) * math.sin(a["heading"])
if KIND == "landerpi":
    check("slides 30 cm to its left without turning or rolling forward",
          r.ok and abs(left - 0.3) < 0.03 and abs(ahead) < 0.02 and abs(turned) < 2, f"left {left:.2f}, ahead {ahead:.2f}; {r.message}")
    check("has learned that it can move sideways", model.drive.slides is True)
    r, moved, turned, b2 = tell("slide right 30 cm")
    check("slides back to where it was", r.ok and math.hypot(b2["x"] - a["x"], b2["y"] - a["y"]) < 0.03, r.message)
    r, moved, *_ = tell("slide left 3 meters")
    check("refuses a slide longer than it will do blind", not r.ok and moved < 0.01, r.message)
else:
    check("a car asked to slide is told it cannot, and does not move",
          not r.ok and "sideways" in r.message and moved < 0.01 and abs(turned) < 1, r.message)
    r, moved, *_ = tell("slide right 20 cm")
    check("and is refused at once the second time", not r.ok and moved < 0.01, r.message)

# Straight after a turn: on the real car the steering stayed turned and "straight" drives veered.
tell("turn left 45 degrees")
r, moved, turned, b = tell("drive forward 50 cm")
check("drives straight after a turn (the steering is re-centred)", r.ok and abs(turned) < 4, f"veered {turned:+.1f} deg; {r.message}")
r, moved, turned, b = tell("go back 50 cm")
check("reverses straight after a turn", r.ok and abs(turned) < 4, f"veered {turned:+.1f} deg; {r.message}")

r, moved, turned, b = tell("wander for 25 seconds")
check("wanders and turns away from walls", r.ok and "0 times" not in r.message, r.message)
r, *_ = tell("dance")
check("refuses nonsense", not r.ok)

if KIND == "landerpi":
    check("finds the arm, its geometry and the gripper", {"joints", "reach", "gripper"} <= caps, ", ".join(sorted(caps)))
    arm = [c["joints"] for c in adapter._controllers.values() if not c["gripper"]]
    grip = [c["joints"] for c in adapter._controllers.values() if c["gripper"]]
    check("works out the arm controller's joints", arm == [["joint1", "joint2", "joint3", "joint4", "joint5"]], str(arm))
    check("works out the gripper controller's joint", grip == [["r_joint"]], str(grip))
    r, _, _, b = tell("move joint1 to 20 degrees")
    check("moves one joint", r.ok and abs(math.degrees(b["joints"]["joint1"]) - 20) < 1.5, f"{math.degrees(b['joints']['joint1']):.1f}")
    r, _, _, b = tell("set joint 2 to -15 degrees")
    check("joint named loosely", r.ok and abs(math.degrees(b["joints"]["joint2"]) + 15) < 1.5, r.message)
    r, _, _, b = tell("set joint3 to 400 degrees")
    check("refuses a joint target past its limit", not r.ok and "can only go" in r.message, r.message)
    r, _, _, b = tell("reach 0.15 0.05 0.25")
    tool = model.chain.position([b["joints"][n] for n in model.chain.joint_names])
    err = math.dist(tool, (0.15, 0.05, 0.25))
    check("reaches a point with the real joints", r.ok and err < 0.01, f"{err * 1000:.1f} mm off; {r.message}")
    r, *_ = tell("reach 1 0 0")
    check("refuses a point out of reach", not r.ok)
    before = truth()["joints"]["r_joint"]
    r, _, _, b = tell("close the gripper")
    closed = b["joints"]["r_joint"]
    r2, _, _, b = tell("open the gripper")
    opened = b["joints"]["r_joint"]
    check("gripper closes and opens", r.ok and r2.ok and abs(closed - opened) > 1.0, f"{before:.2f} -> {closed:.2f} -> {opened:.2f}; {r.message}")
    r, _, _, b = tell("home")
    check("arm goes home", r.ok and all(abs(b["joints"][n]) < 0.03 for n in model.chain.joint_names), r.message)
else:
    check("has no arm abilities", not ({"joints", "reach", "gripper"} & caps), ", ".join(sorted(caps)))
    r, *_ = tell("open the gripper")
    check("refuses gripper commands", not r.ok)

eddy.close()
print(f"\n{KIND}: {'all checks passed' if not failures else 'FAILED: ' + '; '.join(failures)}")
sys.exit(1 if failures else 0)
