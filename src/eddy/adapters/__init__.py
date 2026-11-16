"""Finding robots and opening them.

A robot is named by a short spec string:
    sim                          the built-in simulated rover + arm
    ros2                         whatever ROS 2 robot is on this machine's ROS graph
    serial:/dev/ttyUSB0@115200   a microcontroller speaking the Eddy line protocol
    tcp:192.168.1.20:9000        the same protocol over the network
    auto                         the first real robot found, else the simulator
"""
from __future__ import annotations

import importlib.util
import os
from typing import List, Optional, Tuple

from .base import Adapter


def open_robot(spec: str = "auto", key: Optional[str] = None) -> Adapter:
    """Turn a spec string into the right adapter. The adapter is not connected yet.
    key: the shared secret for a network robot that was served with one.

    Each adapter module is imported only when its kind is asked for, so a machine
    without ROS or pyserial can still use the others.
    """
    spec = (spec or "auto").strip()
    # "auto": take the first real robot the scan finds; fall back to the simulator.
    if spec == "auto":
        found = [s for s, _ in scan() if s != "sim"]
        spec = found[0] if found else "sim"
    # Whatever follows the colon is the adapter's own option:
    # "sim:<name>" names the simulated robot; "ros2:<topic>" forces the velocity topic, and
    # "ros2:<namespace>" picks one robot when several share the ROS graph.
    if spec == "sim" or spec.startswith("sim:"):
        from .sim import SimAdapter
        return SimAdapter(name=spec.partition(":")[2] or "sim-rover")
    if spec == "ros2" or spec.startswith("ros2:"):
        from .ros2 import Ros2Adapter
        return Ros2Adapter(cmd_vel=spec.partition(":")[2] or None)
    if spec.startswith(("serial:", "tcp:")):
        from .line import LineAdapter
        return LineAdapter(spec, key=key)
    raise ValueError(f"don't know how to open robot '{spec}' (try: sim, ros2, serial:<port>, tcp:<host>:<port>)")


def ros2_available() -> bool:
    """True if this shell has a ROS 2 environment sourced and its Python library can be imported."""
    return bool(os.environ.get("ROS_DISTRO")) and importlib.util.find_spec("rclpy") is not None


def scan(probe_serial: bool = True) -> List[Tuple[str, str]]:
    """List robots Eddy could attach to from this machine: (spec, description)."""
    found: List[Tuple[str, str]] = []
    if ros2_available():
        found.append(("ros2", f"ROS 2 {os.environ.get('ROS_DISTRO')} graph on this machine"))
    # Serial ports can only be listed if pyserial is installed.
    if importlib.util.find_spec("serial") is not None:
        from serial.tools import list_ports
        from .line import LineAdapter
        for port in list_ports.comports():
            if "bluetooth" in port.device.lower():
                continue        # macOS lists Bluetooth pseudo-ports; they are never robots
            spec, note = f"serial:{port.device}", port.description or "serial device"
            if probe_serial:
                # Try the Eddy handshake to learn whether the device really is an Eddy robot.
                adapter = LineAdapter(spec, timeout=2.5)
                try:
                    note = f"{adapter.connect().name} (answers Eddy protocol)"
                except Exception:
                    note += " (did not answer the Eddy handshake)"
                finally:
                    try:
                        adapter.close()
                    except Exception:
                        pass
            found.append((spec, note))
    # The simulator is always available, and always listed last.
    found.append(("sim", "built-in simulated rover with a 4-joint arm"))
    return found
