"""A guided test of Eddy's hearing with a real voice and a real microphone.

It shows a sentence, you say it, and it records what the recogniser heard and what Eddy
would have done. Nothing is driven: this only tests listening and understanding.

    .venv/bin/python tests/voice/voice_test.py            (results also saved to voice_results.json)
    .venv/bin/python tests/voice/voice_test.py wake       (only the wake word: saved to wake_results.json)

Each line ends up as one of:
    correct        Eddy would have done the right thing (including correctly doing nothing)
    WRONG ACTION   Eddy would have acted when it should not, or done a different thing
    missed         a real command that Eddy would not have carried out
"""
import json
import queue
import sys
import time

import sounddevice

from eddy import intents
from eddy.senses.listen import Listener, heard_command

# (what to say, the commands Eddy should end up with; [] means "must do nothing")
SCRIPT = [
    ("Eddy, drive forward one meter", ["Drive"]),
    ("Eddy, turn left ninety degrees", ["Drive"]),
    ("Eddy, go back fifty centimeters", ["Drive"]),
    ("Eddy, turn right", ["Drive"]),
    ("Eddy, stop", ["Stop"]),
    ("Eddy, open the gripper", ["Gripper"]),
    ("Eddy, close the gripper", ["Gripper"]),
    ("Eddy, home", ["Home"]),
    ("Eddy, status", ["Report"]),
    ("Eddy, turn around", ["Drive"]),
    ("Eddy, drive forward slowly two meters", ["Drive"]),
    ("Eddy, turn left and go forward one meter", ["Drive", "Drive"]),
    ("Eddy, move the arm up five centimeters", ["MoveBy"]),
    ("Eddy, wander for ten seconds", ["Wander"]),
    ("Eddy, take off", ["TakeOff"]),
    ("Eddy, land", ["Land"]),
    ("Eddy, don't drive forward", []),
    ("Eddy, do not turn left", []),
    ("Eddy, never open the gripper", []),
    ("Eddy, go forward or turn left", []),
    ("Eddy, please don't move", ["Stop"]),
    ("Drive forward one meter      (without saying Eddy)", []),
    ("What time is it?             (just talking, not to Eddy)", []),
    ("Eddy, drive to the kitchen", []),
]


# The wake-word test: Eddy must ignore everything that does not start with its name.
WAKE = [
    ("Drive forward one meter       (do NOT say Eddy)", []),
    ("Turn left ninety degrees      (do NOT say Eddy)", []),
    ("Go back fifty centimeters     (do NOT say Eddy)", []),
    ("Open the gripper              (do NOT say Eddy)", []),
    ("Turn right                    (do NOT say Eddy)", []),
    ("Take off                      (do NOT say Eddy)", []),
    ("Any day now, drive forward    (do NOT say Eddy)", []),
    ("Ready, turn left              (do NOT say Eddy)", []),
    ("Eddy, drive forward one meter", ["Drive"]),
    ("Eddy, turn right", ["Drive"]),
    ("Eddy, open the gripper", ["Gripper"]),
    ("Eddy, stop", ["Stop"]),
]


def main():
    global SCRIPT
    wake_only = len(sys.argv) > 1 and sys.argv[1] == "wake"
    if wake_only:
        SCRIPT = WAKE
    listener = Listener()
    recognizer = listener._recognizer(listener.samplerate)
    recognizer.SetWords(True)       # also report each word's confidence and length
    audio: "queue.Queue[bytes]" = queue.Queue()
    rows = []
    print("\nEddy voice test. Say each sentence in your normal voice when it appears.")
    print("You have 10 seconds for each one. Nothing will move.\n", flush=True)
    with sounddevice.RawInputStream(samplerate=listener.samplerate, blocksize=4000, dtype="int16", channels=1,
                                    callback=lambda data, *_: audio.put(bytes(data))):
        for number, (say, expected) in enumerate(SCRIPT, 1):
            time.sleep(1.0)
            while not audio.empty():            # forget anything said before the prompt
                audio.get()
            recognizer.Reset()
            print(f"[{number}/{len(SCRIPT)}]  SAY:  {say}", flush=True)
            heard, words, deadline = "", [], time.time() + 10
            while time.time() < deadline:
                try:
                    chunk = audio.get(timeout=0.5)
                except queue.Empty:
                    continue
                if recognizer.AcceptWaveform(chunk):
                    result = json.loads(recognizer.Result())
                    text = result.get("text", "").strip()
                    if text.replace("[unk]", "").strip():
                        heard = text
                        words = [[w["word"], round(w["conf"], 2), round(w["end"] - w["start"], 2)] for w in result.get("result", [])]
                        break
            sentence, unclear = heard_command(heard) if heard else (None, False)
            commands = [] if (sentence is None or unclear) else [type(c).__name__ for c in intents.parse_all(sentence)]
            if commands == expected:
                verdict = "correct"
            elif commands:
                verdict = "WRONG ACTION"
            else:
                verdict = "missed"
            would = ", ".join(commands) or ("nothing (only partly heard)" if unclear else "nothing")
            print(f"         heard: {heard or '(nothing)'}\n         Eddy would do: {would}   ->  {verdict}\n", flush=True)
            rows.append({"say": say, "expected": expected, "heard": heard, "would_do": commands, "unclear": unclear, "verdict": verdict,
                         "words": words})
    tally = {v: sum(r["verdict"] == v for r in rows) for v in ("correct", "WRONG ACTION", "missed")}
    print(f"Done. correct {tally['correct']}, WRONG ACTION {tally['WRONG ACTION']}, missed {tally['missed']} of {len(rows)}.")
    name = "wake_results.json" if wake_only else "voice_results.json"
    with open(name, "w") as f:
        json.dump({"tally": tally, "rows": rows}, f, indent=1)
    print(f"Saved to {name}")


if __name__ == "__main__":
    sys.exit(main())
