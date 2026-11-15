"""A stand-in for a microcontroller robot (Arduino, ESP32, Pico) that speaks Eddy's line
protocol, for testing the `serial:` and `tcp:` paths without hardware.

It is written separately from Eddy's own LineServer, the way real firmware would be: a small
loop that reads bytes, keeps motor state, and answers. It behaves like a real board in the
ways that tend to break host software:
  - it prints a boot banner (with junk bytes) when the port is opened, and is deaf while booting
  - it ends lines with \\r\\n
  - it has a motor watchdog: no drive command for `watchdog` seconds and it stops by itself
and it can be told to misbehave (see the `faults` argument).

Serial is done with a pseudo-terminal, so Eddy opens a real device path with pyserial.
"""
import json
import math
import os
import socket
import threading
import time

BANNER = b"\x00\xfe\xff ets Jul 29 2019 12:21:46\r\nrst:0x1 (POWERON_RESET),boot:0x13\r\nrover firmware 0.3 ready\r\n"


class FakeMCU:
    def __init__(self, link="serial", watchdog=0.5, boot=0.3, odometry=True, range_sensor=True,
                 joints=("pan", "tilt"), gripper=True, wall=1.5, **faults):
        """faults: latency=s (delay every answer), drop_every=n (never answer every nth request),
        die_at=s (go silent after s seconds of commands, as if the cable were pulled),
        reset_at=s (reboot once at s: banner again, state zeroed), freeze_odometry=True (report
        a position that never changes), bad_state=True (answer '?' with something that is not JSON)."""
        self.watchdog, self.boot, self.faults = watchdog, boot, faults
        self.has_odometry, self.has_range, self.gripper_fitted, self.wall = odometry, range_sensor, gripper, wall
        self.joint_names = list(joints)
        self._zero()
        self.requests, self.first_command, self.did_reset, self.alive = 0, None, False, True
        self.log = []                           # every line received, for the tests to inspect
        if link == "serial":
            self.master, slave = os.openpty()
            import tty
            tty.setraw(slave)                   # a real serial line does not echo or rewrite bytes
            self.port = f"serial:{os.ttyname(slave)}@115200"
            self._slave = slave                 # kept open so the device node stays valid
            self._read, self._write = (lambda: os.read(self.master, 256)), (lambda b: os.write(self.master, b))
            threading.Thread(target=self._boot_then_serve, daemon=True).start()
        else:
            self.listener = socket.socket()
            self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.listener.bind(("127.0.0.1", 0))
            self.listener.listen(1)
            self.port = f"tcp:127.0.0.1:{self.listener.getsockname()[1]}"
            threading.Thread(target=self._accept, daemon=True).start()
        threading.Thread(target=self._physics, daemon=True).start()

    def _zero(self):
        self.x = self.y = self.th = self.v = self.w = 0.0
        self.joints = {n: 0.0 for n in getattr(self, "joint_names", [])}
        self.grip, self.last_drive = 1.0, 0.0

    # ---- the "board" -----------------------------------------------------------------

    def _accept(self):
        conn, _ = self.listener.accept()
        self._read, self._write = (lambda: conn.recv(256)), conn.sendall
        self._boot_then_serve()

    def _boot_then_serve(self):
        self._write(BANNER)
        time.sleep(self.boot)
        buffer = b""
        while self.alive:
            try:
                chunk = self._read()
            except OSError:
                return
            if not chunk:
                return
            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                self._handle(line.decode(errors="replace").strip())

    def _physics(self):
        last = time.monotonic()
        while self.alive:
            time.sleep(0.005)
            now = time.monotonic()
            dt, last = now - last, now
            if self.watchdog and now - self.last_drive > self.watchdog:
                self.v = self.w = 0.0           # nobody is driving: stop the motors
            self.th += self.w * dt
            self.x += self.v * math.cos(self.th) * dt
            self.y += self.v * math.sin(self.th) * dt

    def _handle(self, line):
        if not line:
            return
        self.log.append(line)
        self.requests += 1
        now = time.monotonic()
        if line != "EDDY?" and self.first_command is None:
            self.first_command = now
        since = now - self.first_command if self.first_command else 0.0
        f = self.faults
        if f.get("die_at") is not None and since >= f["die_at"]:
            return                               # cable pulled: nothing comes back, ever
        if f.get("reset_at") is not None and since >= f["reset_at"] and not self.did_reset:
            self.did_reset = True                # brown-out: the board reboots and forgets everything
            self._zero()
            self._write(BANNER)
            return
        if f.get("drop_every") and self.requests % f["drop_every"] == 0:
            return
        if f.get("latency"):
            time.sleep(f["latency"])
        self._write((self._answer(line) + "\r\n").encode())

    def _answer(self, line):
        op, _, rest = line.partition(" ")
        try:
            if op == "EDDY?":
                sensors = (["odometry"] if self.has_odometry else []) + (["range"] if self.has_range else [])
                return json.dumps({"eddy": 1, "name": "fake-mcu-rover", "drive": {"max_linear": 0.4, "max_angular": 2.0},
                                   "joints": [{"name": n, "lower": -1.5, "upper": 1.5} for n in self.joint_names],
                                   "gripper": self.gripper_fitted, "sensors": sensors,
                                   **({"watchdog": self.watchdog} if self.watchdog else {})})
            if op == "V":
                v, w = (float(t) for t in rest.split())
                self.v, self.w, self.last_drive = v, w, time.monotonic()
                return "OK"
            if op == "J":
                for token in rest.split():
                    if "=" in token:
                        name, value = token.split("=")
                        if name not in self.joints:
                            return f"ERR no joint {name}"
                        self.joints[name] = float(value)
                return "OK"
            if op == "G":
                if not self.gripper_fitted:
                    return "ERR unsupported no gripper fitted"
                self.grip = float(rest)
                return "OK"
            if op == "S":
                self.v = self.w = 0.0
                return "OK"
            if op == "?":
                if self.faults.get("bad_state"):
                    return "x=0.1 y=0.0 th=0.0"
                out = {"joints": self.joints, "gripper": self.grip}
                if self.has_odometry:
                    frozen = self.faults.get("freeze_odometry")
                    out.update(x=0.0 if frozen else round(self.x, 4), y=0.0 if frozen else round(self.y, 4),
                               th=0.0 if frozen else round(self.th, 4))
                if self.has_range:
                    out["range"] = round(max(0.0, self.wall - self.x), 3)
                return json.dumps(out)
            return "ERR unknown command"
        except Exception as exc:
            return f"ERR {type(exc).__name__}"

    def close(self):
        self.alive = False
