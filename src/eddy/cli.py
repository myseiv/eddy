"""The `eddy` command.

Each sub-command (`eddy scan`, `eddy shell`, ...) is one cmd_* function below. main() at the
bottom defines the options and calls the right one. A cmd_* function returns the exit
code: 0 for success, 1 if something the user asked for did not work.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import platform
import shutil
import sys
import threading
import time

from . import __version__
from .adapters import open_robot, ros2_available, scan
from .brain import Eddy, Safety
from .senses.speak import Speaker


def _eddy(args, speaker: str = "print", log: bool = False) -> Eddy:
    """Build an Eddy for the robot and safety options given on the command line.
    Not every sub-command has --speed and --min-range, hence the getattr defaults."""
    adapter = open_robot(args.robot, key=getattr(args, "key", None))
    safety = Safety(speed_scale=getattr(args, "speed", 1.0), min_range=getattr(args, "min_range", 0.25))
    # Every run is logged unless --no-log is given. The file is named after the time and the robot.
    keep, log = log, None
    if keep and not getattr(args, "no_log", True):
        from .runlog import RunLog
        log = RunLog.open_for(args.robot if args.robot != "auto" else "robot")
    return Eddy(adapter, Speaker(speaker), safety, log=log)


def _say_where_the_log_is(eddy: Eddy) -> None:
    if eddy.log is not None:
        print(f"Keeping a log of this run in {eddy.log.path}  (see it with: eddy log)", flush=True)


def _start_mark(text):
    """Turn "--start x,y,degrees" into three numbers."""
    try:
        x, y, heading = (float(v) for v in text.replace(",", " ").split())
        return x, y, heading
    except ValueError:
        raise ValueError(f"--start needs three numbers: x,y,heading in degrees (got '{text}')") from None


def cmd_scan(args) -> int:
    """`eddy scan`: list the robots reachable from this machine."""
    print("Robots Eddy can attach to from here:\n")
    for spec, note in scan():
        print(f"  {spec:<28} {note}")
    print("\nUse one with:  eddy info -r <name>   or   eddy shell -r <name>")
    return 0


def cmd_info(args) -> int:
    """`eddy info`: connect to a robot and print what it can do."""
    with _eddy(args) as eddy:
        print(eddy.model.summary())
    return 0


def cmd_do(args) -> int:
    """`eddy do "<sentence>" ...`: carry out each sentence in order, then exit."""
    ok = True
    with _eddy(args, "auto" if args.speak else "print", log=True) as eddy:
        for sentence in args.sentence:
            result = eddy.tell(sentence)
            print(f"> {sentence}\n  {result}")
            ok = ok and result.ok
    return 0 if ok else 1


def cmd_shell(args) -> int:
    """`eddy shell`: an interactive prompt; each line typed is a command for the robot."""
    with _eddy(args, "auto" if args.speak else "print", log=True) as eddy:
        print(eddy.model.summary())
        _say_where_the_log_is(eddy)
        print('\nType what you want the robot to do ("drive forward 1 meter", "turn left", "reach 0.2 0 0.3",\n'
              '"status", "stop"). Ctrl-D or "quit" to leave.\n')
        while True:
            try:
                line = input("eddy> ").strip()
            except (EOFError, KeyboardInterrupt):      # Ctrl-D or Ctrl-C leaves the shell
                print()
                break
            if line in ("quit", "exit"):
                break
            if line:
                print(eddy.tell(line))
    return 0


def cmd_serve(args) -> int:
    """`eddy serve`: put this machine's robot on the network so another machine can drive it."""
    from .adapters.line import LineServer
    adapter = open_robot(args.robot, key=args.key)
    # The simulator normally runs on its own clock; served over the network it must move in real time.
    if hasattr(adapter, "go_live"):
        adapter.go_live()
    server = LineServer(adapter, port=args.port, key=args.key, watchdog=args.watchdog).start()
    print(f"Serving {server.model.name} on port {server.port}. Connect with:  eddy shell -r tcp:<this-host>:{server.port}"
          + (" --key <the same key>" if args.key else ""), flush=True)
    if not args.key:
        print("WARNING: no --key given, so anyone who can reach this port can drive the robot. "
              "Use  --key <secret>  on any network you do not fully control.", flush=True)
    print(f"The robot stops by itself {args.watchdog:g} s after drive commands stop arriving.", flush=True)
    # The server works in a background thread; this thread just waits for Ctrl-C.
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        server.stop()
    return 0


def cmd_run(args) -> int:
    """Everything on at once: the robot, plus whichever senses were asked for."""
    eddy = _eddy(args, "auto" if args.speak else "print", log=True)
    eddy.start()
    print(eddy.model.summary(), flush=True)
    _say_where_the_log_is(eddy)
    # Print the outcome of every command, whichever sense it came from.
    eddy.listeners.append(lambda result: print(f"  {result}", flush=True))
    threads = []

    def background(name, target):
        """Run a sense in its own thread. If it fails, say so and let the rest keep running."""
        def guarded():
            try:
                target()
            except Exception as exc:
                print(f"[{name}] stopped: {exc}", flush=True)
        t = threading.Thread(target=guarded, daemon=True, name=name)
        t.start()
        threads.append(t)

    # --voice: listen on the microphone and act on sentences addressed to Eddy.
    if args.voice:
        from .senses.listen import Listener, heard_command
        listener = Listener(args.voice_model)
        wake = None if args.no_wake_word else "eddy"

        def hear():
            """Thread: every phrase heard, minus the wake word, becomes a command."""
            for phrase in listener.phrases():
                sentence, unclear = heard_command(phrase, wake)
                if sentence and unclear:
                    # Part of it was not caught. Acting on the rest could do the opposite of
                    # what was said, so say so and do nothing.
                    print(f"[heard only part] {sentence}  -> not acted on", flush=True)
                    eddy.speaker.say("I did not catch all of that.")
                elif sentence:
                    print(f"[heard] {sentence}", flush=True)
                    eddy.tell(sentence, source="voice")
        background("voice", hear)
        print("Listening." + ("" if args.no_wake_word else ' Start each command with "Eddy".'), flush=True)

    # --gestures: watch the camera and act on hand signs.
    if args.gestures:
        from .senses.vision import Camera, HandTracker
        camera, tracker = Camera(args.camera), HandTracker()

        def watch():
            """Thread: feed every camera frame's hand landmarks to Eddy."""
            for frame in camera.frames():
                eddy.sign(tracker.landmarks(frame))
        background("gestures", watch)
        print("Watching for hand signs: palm = stop, point = forward, V = back, thumb sideways = turn.", flush=True)

    # --peers: join the robot-to-robot link.
    node = None
    if args.peers or args.peer or args.start:
        from .link import PEER_PORT, PeerNode, attach, roll_call
        name = args.name or eddy.model.name
        try:
            node = PeerNode(name, key=args.key, port=args.peer_port)
        except OSError:
            # The usual port is taken (a second robot on this machine): take any free one.
            node = PeerNode(name, key=args.key, port=0)
        node.start()
        attach(node, eddy, obey=args.obey, start=_start_mark(args.start) if args.start else None)
        print(f"Peer link up as '{node.name}' on port {node.port}"
              f"{' (will obey signed commands from peers)' if args.obey else ''}.", flush=True)
        if not args.key:
            print("No --key given: peers can see this robot but none can command or stop it, "
                  "and it cannot command them.", flush=True)
        # Peers given by address, for networks where robots cannot find each other by themselves.
        for address in args.peer:
            host, _, port = address.partition(":")
            node.add_address(host, int(port or PEER_PORT))
            print(f"Looking for a peer at {host}:{port or PEER_PORT}.", flush=True)

    # --web: a page with the camera, a command box and a stop button.
    page = None
    if args.web:
        from .web import WebPage
        # The simulator normally jumps its clock forward; watched on a page it must move in real time.
        if hasattr(eddy.adapter, "go_live"):
            eddy.adapter.go_live()
        page = WebPage(eddy, port=args.web_port).start()
        print("Web page up. Open this on a laptop or phone on the same network:", flush=True)
        for link in page.links():
            print(f"  {link}", flush=True)

    if args.say:
        eddy.speaker.say(args.say)
    # With a terminal attached, also take typed commands; as a background service, just keep running.
    try:
        if sys.stdin.isatty():
            print('Type commands too. "quit" to leave.', flush=True)
            while True:
                line = input("eddy> ").strip()
                if line in ("quit", "exit"):
                    break
                if line.startswith("@") and node is not None:      # "@other-robot turn left", "@all stop"
                    peer, _, sentence = line[1:].partition(" ")
                    if peer != "all":
                        print(node.request(peer, sentence))
                    elif sentence.strip() in ("stop", "halt", "freeze"):
                        for name, answer in node.broadcast("stop").items():
                            print(f"{name}: {answer if isinstance(answer, Exception) else answer['message']}")
                    else:
                        for name in sorted(node.peers):
                            print(node.request(name, sentence))
                elif line == "peers" and node is not None:
                    print("\n".join(roll_call(node)) or "no peers seen yet")
                elif line:
                    eddy.tell(line)
        else:
            while True:
                time.sleep(3600)
    except (EOFError, KeyboardInterrupt):
        print()
    finally:
        # Always leave the network and stop the robot on the way out.
        if node is not None:
            node.stop()
        if page is not None:
            page.stop()
        eddy.close()
    return 0


def cmd_hear(args) -> int:
    """`eddy hear file.wav`: recognise the speech in a recording and carry it out
    (tests listening without a microphone)."""
    from .senses.listen import Listener, heard_command
    text, unclear = heard_command(Listener(args.voice_model).transcribe(args.wav), None)
    print(f"[heard] {text or '(nothing)'}")
    if not text:
        return 1
    if unclear:
        print("  part of that was not caught, so it was not acted on")
        return 1
    with _eddy(args, log=True) as eddy:
        result = eddy.tell(text, source="voice")
        print(f"  {result}")
    return 0 if result.ok else 1


def cmd_see(args) -> int:
    """Name the hand sign in each image file (tests hand signs without a live camera)."""
    import cv2
    from . import gestures
    from .senses.vision import HandTracker
    tracker = HandTracker()
    ok = True
    for path in args.image:
        frame = cv2.imread(path)
        if frame is None:
            print(f"{path}: not an image")
            ok = False
            continue
        landmarks = tracker.landmarks(frame)
        sign = gestures.classify(landmarks) if landmarks else None
        command = gestures.to_command(sign)
        print(f"{path}: {'no hand found' if landmarks is None else (sign or 'a hand, but no sign Eddy knows')}"
              + (f"  ->  {type(command).__name__}" if command else ""))
    tracker.close()
    return 0 if ok else 1


def cmd_doctor(args) -> int:
    """Say what works on this machine and what is missing."""
    def have(module):
        """True if this Python package is installed."""
        return importlib.util.find_spec(module) is not None

    def row(ok, label, note=""):
        """Print one line of the report."""
        print(f"  [{'ok' if ok else '--'}] {label:<26} {note}")

    print(f"Eddy {__version__}\n")
    machine = platform.machine()
    board = ""
    # On a Raspberry Pi (and other ARM boards) this file holds the board's name.
    try:
        with open("/proc/device-tree/model") as f:
            board = f.read().strip("\x00\n ")
    except OSError:
        pass
    row(True, "machine", f"{platform.system()} {machine} {board}".strip())
    row(sys.version_info >= (3, 9), "python", platform.python_version())
    row(have("numpy"), "numpy (kinematics)", "" if have("numpy") else "pip install numpy")
    print("\nRobots:")
    row(True, "simulator", "always available")
    row(ros2_available(), "ROS 2", os.environ.get("ROS_DISTRO") or "not sourced (source /opt/ros/<distro>/setup.bash)")
    row(have("serial"), "serial robots", "" if have("serial") else "pip install 'eddy-ai[serial]'")
    row(True, "network robots (tcp:)", "always available")
    print("\nSenses:")
    voice = next((p for p in ("espeak-ng", "espeak", "say", "spd-say") if shutil.which(p)), None)
    row(bool(voice), "speak", voice or "install espeak-ng (sudo apt install espeak-ng)")
    from .senses.listen import default_model_path
    listen_ok = have("vosk") and have("sounddevice")
    row(listen_ok and bool(default_model_path()), "listen",
        (default_model_path() or "no speech model: see `eddy run --help`") if listen_ok
        else "pip install 'eddy-ai[voice]'")
    row(have("cv2"), "see (camera)", "" if have("cv2") else "pip install 'eddy-ai[vision]'")
    row(have("mediapipe"), "hand signs", "" if have("mediapipe") else "pip install 'eddy-ai[vision]'")
    # On Linux, cameras show up as /dev/video0, /dev/video1, ...
    cams = [d for d in (os.listdir("/dev") if os.path.isdir("/dev") else []) if d.startswith("video")]
    if platform.system() == "Linux":
        row(bool(cams), "camera device", ", ".join(sorted(cams)[:4]) or "none found under /dev/video*")
    print("\nRun `eddy selftest` to check the whole stack on the simulator.")
    return 0


def cmd_log(args) -> int:
    """`eddy log`: show what happened in the latest run (or the run log file given)."""
    from . import runlog
    path = args.file or runlog.latest()
    if not path:
        print(f"No run logs yet in {runlog.default_dir()}. Runs of `eddy do`, `shell` and `run` are logged there.")
        return 1
    entries = runlog.read(path)
    print(path)
    for e in entries:
        if e.get("event") == "start":
            print(f"{e['at']}  attached to {e.get('robot')} ({', '.join(e.get('capabilities', []))})")
        elif e.get("event") == "command":
            where = ""
            after = e.get("after") or {}
            if after.get("has_odometry"):
                where = f"  [x {after['x']:.2f}, y {after['y']:.2f}]"
            said = e.get("said") or e.get("command")
            print(f"{e['at']}  {e.get('source', '?'):<8} {'ok  ' if e.get('ok') else 'FAIL'} {said}  ->  {e.get('message')}{where}")
    commands = [e for e in entries if e.get("event") == "command"]
    print(f"\n{len(commands)} command(s), {sum(1 for e in commands if not e.get('ok'))} not carried out. "
          "The file has the full state before and after each one.")
    return 0


def cmd_selftest(args) -> int:
    """`eddy selftest`: run the end-to-end checks on the simulator."""
    from .selftest import run
    return run(verbose=not args.quiet)


def main(argv=None) -> int:
    """Parse the command line and run the chosen sub-command. Returns the exit code."""
    parser = argparse.ArgumentParser(prog="eddy", description="Eddy: one controller for any robot.")
    parser.add_argument("--version", action="version", version=f"eddy {__version__}")
    sub = parser.add_subparsers(dest="command")

    def robot_arg(p):
        """Add the options shared by every sub-command that talks to a robot."""
        p.add_argument("-r", "--robot", default=os.environ.get("EDDY_ROBOT", "auto"),
                       help="sim, ros2, serial:<port>[@baud], tcp:<host>:<port>, or auto (default)")
        p.add_argument("--speed", type=float, default=1.0, help="scale every speed, e.g. 0.5 for half")
        p.add_argument("--min-range", type=float, default=0.25, help="stop this far (m) from obstacles ahead")
        p.add_argument("--key", default=os.environ.get("EDDY_KEY"),
                       help="shared secret: for a tcp: robot served with one, for `serve`, and for --peers")
        p.add_argument("--no-log", action="store_true", help="do not keep a log of this run")

    # One parser per sub-command; set_defaults(fn=...) records which function runs it.
    sub.add_parser("doctor", help="check what works on this machine").set_defaults(fn=cmd_doctor)
    sub.add_parser("scan", help="list robots Eddy can attach to").set_defaults(fn=cmd_scan)
    p = sub.add_parser("info", help="show what a robot can do")
    robot_arg(p)
    p.set_defaults(fn=cmd_info)
    p = sub.add_parser("do", help='carry out sentences, e.g. eddy do "drive forward 1 meter" "turn left"')
    robot_arg(p)
    p.add_argument("sentence", nargs="+")
    p.add_argument("--speak", action="store_true", help="speak out loud")
    p.set_defaults(fn=cmd_do)
    p = sub.add_parser("shell", help="type commands to the robot")
    robot_arg(p)
    p.add_argument("--speak", action="store_true")
    p.set_defaults(fn=cmd_shell)
    def run_args(p):
        """The options of `eddy run` (also used by `eddy web`, which is `run` with the page on)."""
        robot_arg(p)
        p.add_argument("--speak", action="store_true", help="speak out loud")
        p.add_argument("--say", help="say this once at start-up")
        p.add_argument("--voice", action="store_true", help="listen for spoken commands")
        p.add_argument("--voice-model", help="folder of a Vosk speech model")
        p.add_argument("--no-wake-word", action="store_true", help='act on every sentence, not only ones starting "Eddy"')
        p.add_argument("--gestures", action="store_true", help="watch the camera for hand signs")
        p.add_argument("--camera", type=int, default=0)
        p.add_argument("--peers", action="store_true", help="find and talk to other Eddy robots on the network")
        p.add_argument("--name", help="this robot's name on the peer link")
        p.add_argument("--peer", action="append", default=[], metavar="HOST[:PORT]",
                       help="another Eddy robot's address, where robots do not find each other by themselves (repeatable)")
        p.add_argument("--start", metavar="X,Y,DEGREES",
                       help="where this robot is now, measured from a floor mark all the robots share, "
                            "so they can tell each other where things are (e.g. --start 2,0,180)")
        p.add_argument("--peer-port", type=int, default=42425, help="port for messages from peers")
        p.add_argument("--obey", action="store_true", help="carry out commands sent by peers that know the key")
        p.add_argument("--web", action="store_true", help="serve a web page: camera, commands and a stop button")
        p.add_argument("--web-port", type=int, default=8080)

    p = sub.add_parser("run", help="run Eddy with senses (voice, hand signs, peers, web page)")
    run_args(p)
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("web", help="control the robot from a web page (camera, commands, stop button)")
    run_args(p)
    p.set_defaults(fn=cmd_run, web=True)
    p = sub.add_parser("serve", help="put a robot on the network (Eddy line protocol over TCP)")
    robot_arg(p)
    p.add_argument("--port", type=int, default=9000)
    p.add_argument("--watchdog", type=float, default=0.5,
                   help="stop the robot this many seconds after drive commands stop arriving")
    p.set_defaults(fn=cmd_serve)
    p = sub.add_parser("hear", help="recognise speech in a WAV file and carry it out (tests listening without a mic)")
    robot_arg(p)
    p.add_argument("wav")
    p.add_argument("--voice-model")
    p.set_defaults(fn=cmd_hear)
    p = sub.add_parser("see", help="name the hand sign in image files (tests hand signs without a camera)")
    p.add_argument("image", nargs="+")
    p.set_defaults(fn=cmd_see)
    p = sub.add_parser("log", help="show what happened in the latest run")
    p.add_argument("file", nargs="?", help="a run log file (default: the newest)")
    p.set_defaults(fn=cmd_log)
    p = sub.add_parser("selftest", help="check the whole stack end to end on the simulator")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(fn=cmd_selftest)

    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 0
    try:
        return args.fn(args)
    except KeyboardInterrupt:
        return 130      # the conventional exit code for "interrupted with Ctrl-C"
    except (RuntimeError, ValueError, OSError, TimeoutError) as exc:
        # Expected problems (robot not found, missing package, ...): one clear line, no traceback.
        print(f"eddy: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
