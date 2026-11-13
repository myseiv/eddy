"""The Eddy line protocol: plain text over a serial port or a TCP socket.

This is how Eddy reaches robots whose motors hang off a microcontroller (Arduino,
ESP32, Pico) or any program that would rather open a socket than use ROS.

    host sends      device answers
    EDDY?           {"eddy":1,"name":"rover","drive":{"max_linear":0.4,"max_angular":2.0},
                     "joints":[{"name":"pan","lower":-1.5,"upper":1.5}],"gripper":false,
                     "sensors":["odometry","range"]}
    V 0.2 0.0       OK        (drive: metres/second, radians/second)
    J pan=0.5 T1.0  OK        (joint targets in radians, T = seconds to get there)
    G 0.5           OK        (gripper, 0 closed .. 1 open)
    S               OK        (stop everything)
    ?               {"x":0,"y":0,"th":0,"joints":{"pan":0.5},"range":1.2,"gripper":0.5}

LineServer is the other end of the same protocol: it puts any adapter on the
network. That is used for tests, and to drive a robot from another machine.
"""
from __future__ import annotations

import json
import socket
import socketserver
import threading
from typing import Dict, Optional

from ..model import DriveSpec, JointSpec, RobotModel, RobotState, Unsupported
from .base import Adapter


class _SocketTransport:
    def __init__(self, host: str, port: int, timeout: float):
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.file = self.sock.makefile("rwb")

    def write(self, data: bytes) -> None:
        self.file.write(data)
        self.file.flush()

    def readline(self) -> bytes:
        return self.file.readline()

    def close(self) -> None:
        try:
            self.file.close()
        finally:
            self.sock.close()


class LineAdapter(Adapter):
    """Talks the Eddy line protocol. target: 'serial:/dev/ttyUSB0@115200' or 'tcp:host:port'."""
    kind = "line"

    def __init__(self, target: str, timeout: float = 2.0, transport=None):
        self.target = target
        self.timeout = timeout
        self.io = transport
        self.lock = threading.Lock()
        self.model: Optional[RobotModel] = None

    def _open(self):
        scheme, _, rest = self.target.partition(":")
        if scheme == "tcp":
            host, _, port = rest.rpartition(":")
            return _SocketTransport(host or "127.0.0.1", int(port), self.timeout)
        if scheme == "serial":
            try:
                import serial  # pyserial
            except ImportError as exc:
                raise RuntimeError("serial robots need pyserial: pip install 'eddy-ai[serial]'") from exc
            port, _, baud = rest.partition("@")
            dev = serial.Serial(port, int(baud or 115200), timeout=self.timeout)
            dev.reset_input_buffer()
            return dev
        raise ValueError(f"unknown line target '{self.target}'")

    def _ask(self, line: str) -> str:
        with self.lock:
            self.io.write((line + "\n").encode())
            reply = self.io.readline().decode(errors="replace").strip()
        if not reply:
            raise TimeoutError(f"no answer from {self.target} to '{line.split()[0]}'")
        if reply.startswith("ERR"):
            if "unsupported" in reply:
                raise Unsupported(reply[3:].strip())
            raise RuntimeError(f"{self.target} refused '{line}': {reply}")
        return reply

    def connect(self) -> RobotModel:
        if self.io is None:
            self.io = self._open()
        hello = json.loads(self._ask("EDDY?"))
        if "eddy" not in hello:
            raise RuntimeError(f"{self.target} did not answer the Eddy handshake")
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
        if hello.get("urdf"):
            from ..kinematics import Chain
            self.model.chain = Chain.from_urdf(hello["urdf"])
        return self.model

    def close(self) -> None:
        if self.io is not None:
            try:
                self.stop()
            except Exception:
                pass
            self.io.close()
            self.io = None

    def set_velocity(self, linear: float, angular: float) -> None:
        self._ask(f"V {linear:.4f} {angular:.4f}")

    def set_joints(self, positions: Dict[str, float], duration: float = 1.0) -> None:
        pairs = " ".join(f"{n}={v:.5f}" for n, v in positions.items())
        self._ask(f"J {pairs} T{duration:.2f}")

    def set_gripper(self, opening: float) -> None:
        self._ask(f"G {opening:.3f}")

    def stop(self) -> None:
        self._ask("S")

    def state(self) -> RobotState:
        s = json.loads(self._ask("?"))
        return RobotState(s.get("x", 0.0), s.get("y", 0.0), s.get("th", 0.0), s.get("joints", {}),
                          s.get("range"), s.get("gripper"),
                          has_odometry="odometry" in (self.model.sensors if self.model else []))


class LineServer:
    """Serve an adapter over TCP with the line protocol. One robot, many clients."""

    def __init__(self, adapter: Adapter, host: str = "0.0.0.0", port: int = 0, urdf: Optional[str] = None):
        self.adapter = adapter
        self.model = adapter.connect()
        self.urdf = urdf or (self.model.chain.to_urdf() if self.model.chain is not None else None)
        self.lock = threading.Lock()
        outer = self

        class Handler(socketserver.StreamRequestHandler):
            def handle(self):
                for raw in self.rfile:
                    reply = outer.handle(raw.decode(errors="replace").strip())
                    self.wfile.write((reply + "\n").encode())
                    self.wfile.flush()

        class Server(socketserver.ThreadingTCPServer):
            allow_reuse_address = True
            daemon_threads = True

        self.server = Server((host, port), Handler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def start(self) -> "LineServer":
        self.thread.start()
        return self

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.adapter.close()

    def handle(self, line: str) -> str:
        if not line:
            return "ERR empty"
        op, _, rest = line.partition(" ")
        try:
            with self.lock:
                if op == "EDDY?":
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
