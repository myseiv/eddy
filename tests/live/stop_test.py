"""A guided live test that a moving robot can be stopped by a hand sign, by voice, or by both.

Eddy drives a simulated robot in real time (no hardware) while the camera and the microphone are
both on. Each round starts a slow 8 second drive and tells you how to stop it. Some rounds must
NOT stop it (a thumbs up, talking to someone else, doing nothing), because a stop that fires
when it should not is also a fault.

    .venv/bin/python tests/live/stop_test.py            (results also saved to stop_results.json)
    .venv/bin/python tests/live/stop_test.py dry        (no camera or mic: the stop is typed by the script,
                                                         to check the test itself)

Each line ends up as one of:
    correct        the robot stopped when it should, or kept going when it should
    NOT STOPPED    the robot ran its whole drive although a stop was given
    STOPPED BY THE WRONG SENSE   it stopped, but not by the voice or hand sign the round asked for
    WRONG STOP     the robot stopped when nothing should have stopped it
    observed       a case where the right behaviour is a design choice: it just records what happened
"""
import json
import sys
import threading
import time

from eddy import gestures
from eddy.adapters.sim import SimAdapter
from eddy.brain import Eddy
from eddy.model import Drive
from eddy.senses.speak import Speaker

CAMERA = 0      # the camera index; on the owner's MacBook the built-in camera was 1 (0 was the iPhone)
DRIVE_M, SPEED = 1.6, 0.2       # an 8 second drive: long enough to react
WAIT = 4.0          # seconds to read the instruction before the drive starts
DRY = "dry" in sys.argv[1:]

# (what to do, what must happen: "stop", "go" (must keep going) or "info", and which sense must do it)
ROUNDS = [
    ("OPEN PALM: hands down now, then show an open palm to the camera after NOW and hold it", "stop", "sign"),
    ("FIST: hands down now, then show a fist after NOW and hold it", "stop", "sign"),
    ('VOICE: hands down. After NOW say "Eddy, stop"', "stop", "voice"),
    ('VOICE: hands down. After NOW say "Eddy, halt"', "stop", "voice"),
    ('VOICE, NO NAME: hands down. After NOW just say "stop" (without Eddy). Records what happens', "info", "voice"),
    ('BOTH AT ONCE: hands down. After NOW show an open palm AND say "Eddy, stop" together', "stop", "either"),
    ("THUMBS UP: hold a thumbs up to the camera for the whole drive. It must NOT stop", "go", None),
    ('TALKING: say "what time is it" to the room, twice, during the drive. It must NOT stop', "go", None),
    ("NOTHING: hands down, silent. The drive must finish by itself", "go", None),
    ("OPEN PALM again with your OTHER hand: hands down, then after NOW", "stop", "sign"),
]


def main():
    sim = SimAdapter(realtime=True, room=30.0)       # a big room, so the robot never reaches a wall
    eddy = Eddy(sim, Speaker("silent"))
    eddy.start()
    events = []                                 # (seconds, what), from every sense
    clock = time.time

    def note(what):
        events.append((clock(), what))
    eddy.listeners.append(lambda result: note(f"result: {result}"))
    status = {"text": "starting", "done": False}

    if not DRY:
        from eddy.senses.listen import Listener, heard_command
        from eddy.senses.vision import Camera, HandTracker
        listener = Listener()

        def hear():
            try:
                for phrase in listener.phrases():
                    sentence, unclear = heard_command(phrase, "eddy")
                    note(f"VOICE {phrase!r} -> {sentence!r}{' (unclear)' if unclear else ''}")
                    print(f"         (heard: {phrase!r})", flush=True)
                    if sentence and not unclear:
                        eddy.tell(sentence)
            except Exception as exc:
                print(f"\n!!! the microphone thread stopped: {exc}\n", flush=True)
        threading.Thread(target=hear, daemon=True).start()
        camera, tracker = Camera(CAMERA), HandTracker()
        cv2 = camera._cv2

    rows = []

    def script():
        print("\nEddy stop test. Each round: read the instruction, the drive starts, do what it says.", flush=True)
        print("The robot is simulated, nothing real moves.\n", flush=True)
        for number, (do, must, via) in enumerate(ROUNDS, 1):
            print(f"[{number}/{len(ROUNDS)}]  {do}", flush=True)
            status["text"] = f"[{number}/{len(ROUNDS)}] {do}"
            time.sleep(WAIT)
            x0, started = sim.x, clock()
            mark = len(events)
            finished = {}
            drive = threading.Thread(target=lambda: finished.update(result=eddy.execute(Drive(linear=SPEED, distance=DRIVE_M)), at=clock()))
            drive.start()
            print("         NOW", flush=True)
            status["text"] = "NOW  " + do
            if DRY and must == "stop":
                def fake():
                    note(f"{'VOICE' if via == 'voice' else 'SIGN'} fake -> 'stop' Stopped")
                    eddy.tell("stop")
                threading.Timer(1.5, fake).start()
            drive.join(timeout=DRIVE_M / SPEED + 3)
            ran = (finished.get("at", clock()) - started)
            moved = sim.x - x0
            time.sleep(1.0)
            creep = abs(sim.x - x0 - moved)
            mine = events[mark:]
            signs = [e for e in mine if e[1].startswith("SIGN") and "Stopped" in e[1]]
            voices = [e for e in mine if e[1].startswith("VOICE") and "'stop'" in e[1].split("->")[-1] and "(unclear)" not in e[1]]
            got = sorted(signs + voices)
            stopped = ran < DRIVE_M / SPEED - 1.0 and sim.v == 0
            first = round(got[0][0] - started, 2) if got else None
            source = ("hand sign" if got[0] in signs else "voice") if got else ("something else" if stopped else None)
            if must == "info":
                verdict = "observed"
            elif must == "stop":
                right = via == "either" or (via == "sign" and bool(signs)) or (via == "voice" and bool(voices))
                verdict = ("correct" if right else "STOPPED BY THE WRONG SENSE") if stopped and creep < 0.02 else "NOT STOPPED"
            else:
                verdict = "WRONG STOP" if stopped else "correct"
            print(f"         {'stopped' if stopped else 'drove the whole way'} after {ran:.1f} s, {moved:.2f} m moved"
                  f", {creep:.3f} m creep afterwards" + (f", stopped by {source}" + (f" {first} s after NOW" if first is not None else "") if source else "")
                  + f"   ->  {verdict}\n", flush=True)
            rows.append({"do": do, "must": must, "stopped": stopped, "seconds_driven": round(ran, 2), "metres": round(moved, 2),
                         "creep_m": round(creep, 3), "first_stop_s": first, "stop_source": source,
                         "events": [(round(t - started, 2), w) for t, w in events[mark:]][:30], "verdict": verdict})
        status["done"] = True

    worker = threading.Thread(target=script, daemon=True)
    worker.start()
    if DRY:
        worker.join()
    else:
        for frame in camera.frames():
            if status["done"]:
                break
            landmarks = tracker.landmarks(frame)
            result = eddy.sign(landmarks)
            if result is not None:
                note(f"SIGN {result.message}")
            sign = gestures.classify(landmarks) if landmarks else None
            picture = cv2.flip(frame, 1)
            for i, text in enumerate((status["text"][:70], f"Eddy sees: {sign or ('hand, no sign' if landmarks else 'no hand')}")):
                cv2.putText(picture, text, (10, 28 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
                cv2.putText(picture, text, (10, 28 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow("Eddy stop test", picture)
            cv2.waitKey(1)
        camera.close()
        cv2.destroyAllWindows()
    tally = {v: sum(r["verdict"] == v for r in rows) for v in ("correct", "NOT STOPPED", "WRONG STOP", "STOPPED BY THE WRONG SENSE", "observed")}
    print("Done. " + ", ".join(f"{k} {v}" for k, v in tally.items()) + f" of {len(rows)}.")
    if not DRY:
        with open("stop_results.json", "w") as f:
            json.dump({"tally": tally, "rows": rows}, f, indent=1)
        print("Saved to stop_results.json")


if __name__ == "__main__":
    sys.exit(main())
