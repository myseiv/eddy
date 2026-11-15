"""A guided test of Eddy's hand signs with a real camera and a real hand.

It names a sign, you hold it up for a few seconds, and it records what the camera saw and
what Eddy would have done. Nothing is driven and no pictures are saved, only counts.

    .venv/bin/python tests/vision/sign_test.py            (results also saved to sign_results.json)

Each line ends up as one of:
    correct        Eddy would have done the right thing (including correctly doing nothing)
    WRONG ACTION   Eddy would have acted when it should not, or done a different thing
    missed         a real sign that Eddy would not have acted on
"""
import json
import math
import sys
import time
from collections import Counter

from eddy import gestures
from eddy.senses.vision import Camera, HandTracker

MEANS = {"open_palm": "stop", "fist": "stop", "point": "forward a little", "victory": "back a little",
         "thumb_left": "turn left", "thumb_right": "turn right"}

# (what to do, the signs Eddy may act on; [] means "must do nothing")
SCRIPT = [
    ("OPEN PALM facing the camera, about half a metre away", ["open_palm"]),
    ("FIST, about half a metre away", ["fist"]),
    ("POINT: index finger straight up, other fingers closed", ["point"]),
    ("VICTORY: index and middle fingers up", ["victory"]),
    ("THUMB pointing to YOUR LEFT, other fingers closed (the camera sees it on the image's right: turn right)", ["thumb_right"]),
    ("THUMB pointing to YOUR RIGHT, other fingers closed (turn left)", ["thumb_left"]),
    ("THUMB UP (this sign has no action)", []),
    ("OPEN PALM again, but lean back as far from the camera as you can reach", ["open_palm"]),
    ("OPEN PALM with your OTHER hand", ["open_palm"]),
    ("POINT with your OTHER hand", ["point"]),
    ("NO HAND: keep your hands out of view and sit still", []),
    ("WAVE: pass your open hand quickly through the picture once or twice", []),
    ("NOT A SIGN: scratch your head or rest your chin on your hand", []),
    ("OPEN PALM while moving your hand slowly side to side", ["open_palm"]),
]
HOLD = 5.0      # seconds each one is watched
READY = 3.0     # seconds to get the hand into place first
CAMERA = 1      # the MacBook camera is 1 on this Mac (0 is the iPhone); set back to 0 after these tests


def show(cv2, frame, landmarks, sign, lines):
    """Show what the camera sees, mirrored like a mirror, with the hand points and the sign Eddy reads."""
    h, w = frame.shape[:2]
    picture = frame.copy()
    for x, y, _ in landmarks or []:
        cv2.circle(picture, (int(x * h), int(y * h)), 4, (0, 255, 0), -1)
    picture = cv2.flip(picture, 1)       # only the display is mirrored; Eddy reads the unmirrored frame
    for i, (text, colour) in enumerate(lines + [(f"Eddy reads: {sign or ('hand, no sign' if landmarks else 'no hand')}", (255, 255, 0))]):
        cv2.putText(picture, text, (10, 28 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(picture, text, (10, 28 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
    cv2.imshow("Eddy sign test", picture)
    cv2.waitKey(1)


def main():
    camera, tracker = Camera(CAMERA), HandTracker()
    cv2 = camera._cv2
    frames = camera.frames()
    rows = []
    print("\nEddy hand-sign test. Hold each sign up when it says GO, until it says done.")
    print("Nothing will move, and no pictures are saved.\n", flush=True)
    chosen = {int(a) for a in sys.argv[1:]} or None       # e.g. `sign_test.py 2 5 13` runs just those steps
    for number, (do, allowed) in enumerate(SCRIPT, 1):
        if chosen and number not in chosen:
            continue
        print(f"[{number}/{len(SCRIPT)}]  {do}", flush=True)
        end = time.time() + READY
        for frame in frames:                    # keep the camera running while you get ready
            left = end - time.time()
            if left <= 0:
                break
            landmarks = tracker.landmarks(frame)
            show(cv2, frame, landmarks, gestures.classify(landmarks) if landmarks else None,
                 [(f"[{number}/{len(SCRIPT)}] {do}", (255, 255, 255)), (f"get ready... {left:.0f}", (0, 200, 255))])
        print("         GO", flush=True)
        hold = gestures.GestureFilter()
        seen, fired, hands, total, first = Counter(), [], 0, 0, None
        runs, run, last, angles = {}, 0, None, []
        began = time.time()
        for frame in frames:
            if time.time() - began >= HOLD:
                break
            total += 1
            landmarks = tracker.landmarks(frame)
            sign = gestures.classify(landmarks) if landmarks else None
            hands += landmarks is not None
            seen[sign or ("hand, no sign" if landmarks else "no hand")] += 1
            run = run + 1 if sign == last else 1
            last = sign
            if sign:
                runs[sign] = max(runs.get(sign, 0), run)
            if landmarks:      # which way the thumb points: 0 = image right, 90 = up, 180 = image left
                angles.append(math.degrees(math.atan2(-(landmarks[4][1] - landmarks[2][1]), landmarks[4][0] - landmarks[2][0])))
            show(cv2, frame, landmarks, sign,
                 [(f"[{number}/{len(SCRIPT)}] {do}", (255, 255, 255)), (f"GO  {HOLD - (time.time() - began):.0f}", (0, 255, 0))])
            acted = hold.update(sign)
            if acted and gestures.to_command(acted) is not None:
                fired.append(acted)
                first = first if first is not None else round(time.time() - began, 2)
        acted_on = sorted(set(fired))
        what = lambda a: MEANS[a]            # open palm and fist both mean "stop", so judge by the meaning
        if not allowed:
            verdict = "correct" if not acted_on else "WRONG ACTION"
        elif not acted_on:
            verdict = "missed"
        else:
            verdict = "correct" if all(what(a) in {what(x) for x in allowed} for a in acted_on) else "WRONG ACTION"
        would = ", ".join(f"{a} ({MEANS[a]})" for a in acted_on) or "nothing"
        print(f"         done. camera saw: {dict(seen.most_common())}\n         Eddy would do: {would}"
              + (f", first after {first} s" if first is not None else "")
              + f"\n         longest unbroken run (frames): {runs}   ->  {verdict}\n", flush=True)
        rows.append({"do": do, "allowed": allowed, "frames": total, "frames_with_hand": hands, "seen": dict(seen),
                     "longest_run_frames": runs, "median_thumb_angle": round(sorted(angles)[len(angles) // 2]) if angles else None,
                     "acted_on": acted_on, "times_acted": len(fired), "first_after_s": first, "verdict": verdict})
    camera.close()
    cv2.destroyAllWindows()
    tally = {v: sum(r["verdict"] == v for r in rows) for v in ("correct", "WRONG ACTION", "missed")}
    print(f"Done. correct {tally['correct']}, WRONG ACTION {tally['WRONG ACTION']}, missed {tally['missed']} of {len(rows)}.")
    with open("sign_results.json", "w") as f:
        json.dump({"tally": tally, "rows": rows}, f, indent=1)
    print("Saved to sign_results.json")


if __name__ == "__main__":
    sys.exit(main())
