"""`eddy selftest`: drive the whole stack end to end on the simulator.

Runs anywhere Eddy is installed (laptop, Raspberry Pi, container) with no hardware.
Each line is one behaviour checked from the outside: a sentence or hand sign goes
in, and the simulated robot's actual position is measured afterwards.
"""
from __future__ import annotations

import math
import threading
import time

from . import gestures
from .adapters.line import LineAdapter, LineServer
from .adapters.sim import SimAdapter
from .brain import Eddy
from .senses.listen import strip_wake_word
from .senses.speak import Speaker


def run(verbose: bool = True) -> int:
    """Run every check. Returns 0 if all passed, 1 otherwise (used as the exit code)."""
    failures = []

    def check(label: str, ok: bool, detail: str = "") -> None:
        """Record one result and print it as a PASS or FAIL line."""
        if not ok:
            failures.append(label)
        if verbose:
            print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  ({detail})" if detail else ""), flush=True)

    def fresh():
        """A new simulated robot in the middle of its room, with a silent Eddy attached."""
        sim = SimAdapter()
        eddy = Eddy(sim, Speaker("silent"))
        eddy.start()
        return sim, eddy

    if verbose:
        print("Eddy self-test (simulated rover + arm)\n")

    # -- plug and play: discover abilities
    sim, eddy = fresh()
    caps = eddy.model.capabilities()
    check("discovers the robot's abilities", {"drive", "joints", "reach", "gripper", "range"} <= set(caps), ", ".join(caps))

    # -- driving from sentences
    r = eddy.tell("drive forward 1 meter")
    check("drive forward 1 meter", r.ok and abs(sim.x - 1.0) < 0.03 and abs(sim.y) < 0.01, f"x={sim.x:.3f}")
    r = eddy.tell("turn left 90 degrees")
    check("turn left 90 degrees", r.ok and abs(math.degrees(sim.heading) - 90) < 3, f"heading={math.degrees(sim.heading):.1f}")
    # The rover now faces +y, so backing up moves it to negative y.
    r = eddy.tell("go back 50 cm")
    check("go back 50 cm", r.ok and abs(sim.y + 0.5) < 0.03, f"y={sim.y:.3f}")

    # -- safety: obstacle ahead
    # The wall is 2 m away; asked for 5 m, Eddy must stop short of it by itself.
    r = eddy.tell("drive forward 5 meters")
    gap = sim.state().range_ahead
    check("stops for an obstacle instead of hitting it", (not sim.bumped) and 0.15 < gap < 0.30 and "in the way" in r.message,
          f"stopped {gap:.2f} m from the wall")
    r = eddy.tell("drive forward 1 meter")
    check("refuses to drive into an obstacle", not r.ok and not sim.bumped, r.message)

    # -- inverse kinematics
    sim, eddy = fresh()
    target = (0.20, 0.10, 0.30)
    r = eddy.tell("reach 0.2 0.1 0.3")
    # Measure where the tool really ended up, from the simulated joint angles.
    tool = sim.chain.position([sim.joints[n] for n in sim.chain.joint_names])
    err = math.dist(tool, target)
    check("arm reaches a point (inverse kinematics)", r.ok and err < 0.005, f"tool is {err * 1000:.1f} mm from target")
    r = eddy.tell("reach 2 0 0.3")
    check("refuses a point beyond its reach", not r.ok, r.message)
    r = eddy.tell("home")
    check("arm goes home", r.ok and all(abs(v) < 0.03 for v in sim.joints.values()))
    eddy.tell("close gripper")
    closed = sim.gripper
    eddy.tell("open the gripper")
    check("gripper closes and opens", closed == 0.0 and sim.gripper == 1.0)

    # -- hand signs
    # Each call to sign() is one camera frame. A sign acts on the fifth frame in a row.
    sim, eddy = fresh()
    acted = [eddy.sign(gestures.fake_hand(["index"])) for _ in range(5)]
    check("hand sign 'point' drives forward", acted[-1] is not None and acted[-1].ok and sim.x > 0.08,
          f"x={sim.x:.2f}")
    check("a sign must be held, not glimpsed", all(a is None for a in acted[:4]))
    acted = [eddy.sign(gestures.fake_hand(["index", "middle", "ring", "pinky"], "left")) for _ in range(5)]
    check("hand sign 'open palm' stops", acted[-1] is not None and "Stopped" in acted[-1].message)
    before = sim.heading
    for _ in range(5):
        eddy.sign(gestures.fake_hand([], "right"))
    check("hand sign 'thumb right' turns right", sim.heading < before - 0.2)

    # -- stop wins over a running command
    # Real-time simulator, with the drive running in another thread, so "stop" arrives mid-drive.
    sim = SimAdapter(realtime=True)
    eddy = Eddy(sim, Speaker("silent"))
    eddy.start()
    worker = threading.Thread(target=lambda: eddy.tell("drive forward 1.5 meters"))
    worker.start()
    time.sleep(0.4)
    eddy.tell("stop")
    worker.join(timeout=3)
    check("'stop' interrupts a drive in progress", not worker.is_alive() and 0.01 < sim.x < 0.4 and sim.v == 0,
          f"stopped at x={sim.x:.2f} of 1.5")

    # -- autonomy
    sim, eddy = fresh()
    r = eddy.tell("wander for 60 seconds")
    travelled = abs(sim.x) + abs(sim.y)
    check("wanders on its own without hitting walls", r.ok and not sim.bumped and "turned away" in r.message and travelled > 0.2,
          r.message)

    # -- speaking and listening plumbing
    eddy.tell("say hello there")
    check("speaks", eddy.speaker.history[-1:] == ["hello there"])
    r = eddy.tell("status")
    check("reports its state", r.ok and "I can" in r.message)
    check("wake word filters speech", strip_wake_word("eddy turn left") == "turn left" and strip_wake_word("turn left") is None)
    r = eddy.tell("make me a sandwich")
    check("says so when it does not understand", not r.ok)

    # -- a robot with no arm and a robot with no wheels: same commands, honest answers
    eddy = Eddy(SimAdapter(arm=False), Speaker("silent"))
    eddy.start()
    check("arm command on a robot without an arm is declined", not eddy.tell("reach 0.2 0 0.3").ok)
    eddy = Eddy(SimAdapter(drive=False), Speaker("silent"))
    eddy.start()
    check("drive command on a robot without wheels is declined", not eddy.tell("drive forward 1 meter").ok)

    # -- a robot reached over the network with the line protocol
    # One simulated robot is served on a local port; a second Eddy drives it through that port.
    remote = SimAdapter(name="remote-rover").go_live()
    server = LineServer(remote, host="127.0.0.1").start()
    try:
        eddy = Eddy(LineAdapter(f"tcp:127.0.0.1:{server.port}"), Speaker("silent"))
        model = eddy.start()
        check("line protocol: discovers a networked robot", model.name == "remote-rover" and model.chain is not None,
              ", ".join(model.capabilities()))
        r = eddy.tell("drive forward 20 cm")
        check("line protocol: drives it", r.ok and abs(remote.x - 0.2) < 0.04, f"x={remote.x:.3f}")
        r = eddy.tell("reach 0.15 0 0.35")
        tool = remote.chain.position([remote.joints[n] for n in remote.chain.joint_names])
        err = math.dist(tool, (0.15, 0, 0.35))
        check("line protocol: inverse kinematics on it", r.ok and err < 0.02, f"{err * 1000:.1f} mm off")
        eddy.close()
    finally:
        server.stop()

    # -- two robots talking to each other
    # (The checks themselves live in link.py, next to the code they test.)
    try:
        from .link import selftest as link_selftest
    except ImportError:
        link_selftest = None
    if link_selftest is not None:
        link_selftest(check)

    if verbose:
        print("\n" + ("All checks passed." if not failures else f"{len(failures)} check(s) FAILED: " + "; ".join(failures)))
    return 1 if failures else 0
