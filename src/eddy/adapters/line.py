"""The Eddy line protocol: plain text over a serial port or a TCP socket.

This is how Eddy reaches robots whose motors hang off a microcontroller (Arduino,
ESP32, Pico) or any program that would rather open a socket than use ROS.

    host sends      device answers
    EDDY?           {"eddy":1,"name":"rover","drive":{"max_linear":0.4,"max_angular":2.0},
                     "joints":[{"name":"pan","lower":-1.5,"upper":1.5}],"gripper":false,
                     "sensors":["odometry","range"],"watchdog":0.5}
    V 0.2 0.0       OK        (drive: metres/second, radians/second)
    J pan=0.5 T1.0  OK        (joint targets in radians, T = seconds to get there)
    G 0.5           OK        (gripper, 0 closed .. 1 open)
    S               OK        (stop everything)
    ?               {"x":0,"y":0,"th":0,"joints":{"pan":0.5},"range":1.2,"gripper":0.5}

"watchdog" is how many seconds after the last V command the robot stops by itself. Firmware
must do this: if the cable or radio drops, nothing on the host side can stop the motors.

LineServer is the other end of the same protocol: it puts any adapter on the
network. That is used for tests, and to drive a robot from another machine.
"""
from __future__ import annotations

import json
import socket
import socketserver
import threading
import time
from typing import Dict, Optional

from ..model import DriveSpec, JointSpec, RobotModel, RobotState, Unsupported
from .base import Adapter


class _SocketTransport:
    """Makes a TCP socket look like a serial port (write / readline / close), so LineAdapter
    can use either without caring which it has."""

    def __init__(self, host: str, port: int, timeout: float):
        self.timeout = timeout
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.file = self.sock.makefile("rwb")

    def write(self, data: bytes) -> None:
        self.file.write(data)
        self.file.flush()       # send now rather than waiting for the buffer to fill

    def readline(self) -> bytes:
        try:
            return self.file.readline()
        except (socket.timeout, TimeoutError):
            return b""          # like a serial port: nothing arrived in time

    def reset_input_buffer(self) -> None:
        """Throw away anything that has arrived but has not been read (late answers)."""
        self.sock.setblocking(False)
        try:
            while self.file.read(4096):
                pass
        except (BlockingIOError, OSError, ValueError):
            pass
        finally:
            self.sock.settimeout(self.timeout)

    def close(self) -> None:
        try:
            self.file.close()
        finally:
            self.sock.close()


class LineAdapter(Adapter):
    """Talks the Eddy line protocol. target: 'serial:/dev/ttyUSB0@115200' or 'tcp:host:port'."""
    kind = "line"

    def __init__(self, target: str, timeout: float = 2.0, transport=None, reply_timeout: float = 0.4):
        """timeout: how long to wait while connecting (a board may still be starting up).
        reply_timeout: how long to wait for an answer once connected. It is short on purpose: a
        control loop that sat waiting two seconds for one lost answer would let the robot's own
        watchdog stop it mid-move.
        transport: an already open connection to use instead of opening one (for tests)."""
        self.target = target
        self.timeout = timeout
        self.reply_timeout = reply_timeout
        self.io = transport
        self.lock = threading.Lock()        # one request and its reply at a time
        self.model: Optional[RobotModel] = None

    def _open(self):
        """Open the connection named by the target string."""
        scheme, _, rest = self.target.partition(":")
        if scheme == "tcp":
            host, _, port = rest.rpartition(":")
            return _SocketTransport(host or "127.0.0.1", int(port), self.timeout)
        if scheme == "serial":
            try:
                import serial  # pyserial
            except ImportError as exc:
                raise RuntimeError("serial robots need pyserial: pip install 'eddy-ai[serial]'") from exc
            # "port@baud"; 115200 if no baud rate is given.
            port, _, baud = rest.partition("@")
            dev = serial.Serial(port, int(baud or 115200), timeout=self.timeout)
            dev.reset_input_buffer()        # throw away boot messages the board may have printed
            return dev
        raise ValueError(f"unknown line target '{self.target}'")

    def _ask(self, line: str) -> str:
        """Send one line and return the device's one-line answer.

        Every request here can safely be sent twice, so one lost answer is retried once before
        giving up. Anything left over from an earlier exchange is thrown away first, so a late
        answer can never be mistaken for the answer to a newer question.
        """
        skipped = ""
        with self.lock:
            for attempt in range(2):
                if hasattr(self.io, "reset_input_buffer"):
                    try:
                        self.io.reset_input_buffer()
                    except Exception:
                        pass
                self.io.write((line + "\n").encode())
                # Boards also print things nobody asked for (a boot banner, debug messages). An
                # answer in this protocol is "OK", "ERR ..." or a JSON object; anything else is
                # skipped, a few lines at most.
                reply = ""
                for _ in range(8):
                    reply = self.io.readline().decode(errors="replace").strip()
                    if not reply or reply == "OK" or reply.startswith(("ERR", "{")):
                        break
                    skipped = reply
                if reply:
                    break
        if not reply:
            if skipped:
                raise RuntimeError(f"{self.target} answered '{line.split()[0]}' with something that is not the "
                                   f"Eddy protocol: {skipped[:60]!r}")
            raise TimeoutError(f"no answer from {self.target} to '{line.split()[0]}'")
        # "ERR unsupported ..." means the robot lacks the ability; any other ERR is a failure.
        if reply.startswith("ERR"):
            if "unsupported" in reply:
                raise Unsupported(reply[3:].strip())
            raise RuntimeError(f"{self.target} refused '{line}': {reply}")
        return reply

    def _handshake(self, patience: float = 6.0) -> dict:
        """Ask the device what it is, allowing for a board that is still starting up.

        Opening a serial port resets many boards (Arduino, ESP32): they are deaf for a second
        or two and then print a banner. So the question is repeated until a proper answer
        comes back or `patience` seconds have gone by.
        """
        deadline, last = time.monotonic() + patience, None
        while True:
            try:
                hello = json.loads(self._ask("EDDY?"))
                if isinstance(hello, dict) and "eddy" in hello:
                    return hello
                last = RuntimeError(f"{self.target} did not answer the Eddy handshake")
            except Unsupported:
                raise
            except (TimeoutError, OSError, ValueError, RuntimeError) as exc:
                last = exc if isinstance(exc, TimeoutError) else \
                    RuntimeError(f"{self.target} did not answer the Eddy handshake ({exc})")
            if time.monotonic() >= deadline:
                raise last
            time.sleep(0.2)

    def connect(self) -> RobotModel:
        """Do the handshake: ask the device what it is and build its model from the answer."""
        if self.io is None:
            self.io = self._open()
        hello = self._handshake()
        # Connected: from here on answers are expected promptly.
        try:
            if hasattr(self.io, "sock"):
                self.io.timeout = self.reply_timeout
                self.io.sock.settimeout(self.reply_timeout)
            elif hasattr(self.io, "timeout"):
                self.io.timeout = self.reply_timeout
        except Exception:
            pass
        drive = hello.get("drive")
        self.model = RobotModel(
            name=hello.get("name", "robot"), kind=self.kind,
            drive=DriveSpec(drive.get("kind", "differential"), drive.get("max_linear", 0.5),
                            drive.get("max_angular", 1.5)) if drive else None,
            joints=[JointSpec(j["name"], j.get("lower", -3.14159), j.get("upper", 3.14159),
                              j.get("kind", "revolute")) for j in hello.get("joints", [])],
            gripper=bool(hello.get("gripper")), sensors=list(hello.get("sensors", [])),
            details={"link": self.target},
        )
        # A drive that keeps going when the link drops is the one thing Eddy cannot make safe
        # from its side: only the robot's own firmware can stop the motors when commands stop
        # arriving. The handshake can say it does ("watchdog": seconds); if it does not, say so.
        if drive:
            watchdog = hello.get("watchdog")
            self.model.details["link loss"] = (
                f"the robot stops by itself {watchdog} s after commands stop" if watchdog else
                "WARNING: this robot does not say it stops by itself if the link drops. If the cable or "
                "radio fails while it is driving, nothing will stop it. Its firmware needs a motor watchdog.")
        # A device may include its arm's URDF, which gives Eddy the geometry it needs to reach points.
        if hello.get("urdf"):
            from ..kinematics import Chain
            self.model.chain = Chain.from_urdf(hello["urdf"])
        return self.model

    def close(self) -> None:
        """Stop the robot (best effort) and close the connection."""
        if self.io is not None:
            try:
                self.stop()
            except Exception:
                pass
            self.io.close()
            self.io = None

    # Each command below is one line of the protocol shown at the top of this file.

    def set_velocity(self, linear: float, angular: float, vertical: float = 0.0) -> None:
        if vertical:
            raise Unsupported("robots on the line protocol cannot be flown")
        self._ask(f"V {linear:.4f} {angular:.4f}")

    def set_joints(self, positions: Dict[str, float], duration: float = 1.0) -> None:
        pairs = " ".join(f"{n}={v:.5f}" for n, v in positions.items())
        self._ask(f"J {pairs} T{duration:.2f}")

    def set_gripper(self, opening: float) -> None:
        self._ask(f"G {opening:.3f}")

    def stop(self) -> None:
        self._ask("S")

    def state(self) -> RobotState:
        """Ask the device where it is. Fields the device leaves out fall back to zero or None."""
        s = json.loads(self._ask("?"))
        return RobotState(s.get("x", 0.0), s.get("y", 0.0), s.get("th", 0.0), s.get("joints", {}),
                          s.get("range"), s.get("gripper"),
                          has_odometry="odometry" in (self.model.sensors if self.model else []))


class LineServer:
    """Serve an adapter over TCP with the line protocol. One robot, many clients."""

    def __init__(self, adapter: Adapter, host: str = "0.0.0.0", port: int = 0, urdf: Optional[str] = None):
        """Connect the adapter and prepare the server. port=0 lets the system pick a free port."""
        self.adapter = adapter
        self.model = adapter.connect()
        # Pass the arm geometry on to clients, so they can do inverse kinematics too.
        self.urdf = urdf or (self.model.chain.to_urdf() if self.model.chain is not None else None)
        self.lock = threading.Lock()        # one command reaches the robot at a time
        outer = self

        class Handler(socketserver.StreamRequestHandler):
            """One client connection: answer every line it sends until it disconnects."""

            def handle(self):
                for raw in self.rfile:
                    reply = outer.handle(raw.decode(errors="replace").strip())
                    self.wfile.write((reply + "\n").encode())
                    self.wfile.flush()

        class Server(socketserver.ThreadingTCPServer):
            """TCP server with a thread per client."""
            allow_reuse_address = True
            daemon_threads = True

        self.server = Server((host, port), Handler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def start(self) -> "LineServer":
        """Begin accepting clients in a background thread."""
        self.thread.start()
        return self

    def stop(self) -> None:
        """Stop serving and release the robot."""
        self.server.shutdown()
        self.server.server_close()
        self.adapter.close()

    def handle(self, line: str) -> str:
        """Carry out one protocol line on the wrapped adapter and return the answer line."""
        if not line:
            return "ERR empty"
        op, _, rest = line.partition(" ")       # the command letter, then its arguments
        try:
            with self.lock:
                if op == "EDDY?":
                    # Handshake: describe the robot as JSON.
                    m = self.model
                    hello = {"eddy": 1, "name": m.name, "gripper": m.gripper, "sensors": m.sensors,
                             "joints": [{"name": j.name, "lower": j.lower, "upper": j.upper, "kind": j.kind}
                                        for j in m.joints]}
                    if m.drive:
                        hello["drive"] = {"kind": m.drive.kind, "max_linear": m.drive.max_linear,
                                          "max_angular": m.drive.max_angular}
                    if self.urdf:
                        hello["urdf"] = self.urdf
                    return json.dumps(hello)
                if op == "V":
                    lin, ang = (float(v) for v in rest.split())
                    self.adapter.set_velocity(lin, ang)
                elif op == "J":
                    # "name=value" tokens are joint targets; a "T<seconds>" token is the duration.
                    positions, duration = {}, 1.0
                    for token in rest.split():
                        if token.startswith("T") and "=" not in token:
                            duration = float(token[1:])
                        else:
                            name, _, value = token.partition("=")
                            positions[name] = float(value)
                    self.adapter.set_joints(positions, duration)
                elif op == "G":
                    self.adapter.set_gripper(float(rest))
                elif op == "S":
                    self.adapter.stop()
                elif op == "?":
                    s = self.adapter.state()
                    return json.dumps({"x": s.x, "y": s.y, "th": s.heading, "joints": s.joints,
                                       "range": s.range_ahead, "gripper": s.gripper})
                else:
                    return f"ERR unknown command {op}"
            return "OK"
        except Unsupported as exc:
            return f"ERR unsupported {exc}"
        except Exception as exc:  # malformed input must never take the robot's server down
            return f"ERR {type(exc).__name__} {exc}"
