"""The run log: one line per command, with what was asked, what happened and where the robot was.

Each run writes its own file of JSON lines (one JSON object per line), so a session with a
robot can be read back afterwards, plotted, or attached to a bug report. The first line
describes the robot; every later line is one command.

    {"event": "start", "robot": "rosmentor", "capabilities": [...], "details": {...}}
    {"event": "command", "source": "typed", "said": "drive forward 1 meter", "command": "Drive",
     "ok": true, "message": "Done: moved 1.00 m.", "seconds": 5.3, "before": {...}, "after": {...}}

Logging must never get in the way of the robot: if the file cannot be written, the log
switches itself off and says so once.
"""
from __future__ import annotations

import dataclasses
import json
import os
import re
import sys
import threading
import time
from typing import Optional


def default_dir() -> str:
    """Where run logs go: $EDDY_LOG_DIR if set, otherwise ~/.eddy/logs."""
    return os.environ.get("EDDY_LOG_DIR") or os.path.join(os.path.expanduser("~"), ".eddy", "logs")


def snapshot(state) -> dict:
    """The parts of a RobotState worth keeping, as plain JSON values."""
    keep = ("x", "y", "heading", "z", "range_ahead", "range_behind", "gripper", "has_odometry",
            "odom_age", "range_age", "joints")
    out = {}
    for name in keep:
        value = getattr(state, name, None)
        if isinstance(value, float):
            value = round(value, 4)
        elif isinstance(value, dict):
            value = {k: round(float(v), 4) for k, v in value.items()}
        out[name] = value
    return out


class RunLog:
    """Appends entries to one file. Safe to call from several threads."""

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        self._file = open(path, "a", encoding="utf-8")

    @classmethod
    def open_for(cls, robot_name: str, directory: Optional[str] = None) -> Optional["RunLog"]:
        """Start a new log file named after the time and the robot. None if it cannot be created."""
        directory = directory or default_dir()
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", robot_name or "robot").strip("-") or "robot"
        path = os.path.join(directory, f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_{safe}.jsonl")
        try:
            os.makedirs(directory, exist_ok=True)
            return cls(path)
        except OSError as exc:
            print(f"[log] not keeping a run log: {exc}", file=sys.stderr, flush=True)
            return None

    def write(self, event: str, **fields) -> None:
        """Append one entry. Never raises: a full disk must not stop a robot."""
        entry = {"t": round(time.time(), 3), "at": time.strftime("%Y-%m-%d %H:%M:%S"), "event": event}
        entry.update(fields)
        with self._lock:
            if self._file is None:
                return
            try:
                self._file.write(json.dumps(entry, default=_plain) + "\n")
                self._file.flush()
            except (OSError, ValueError) as exc:
                print(f"[log] stopped keeping the run log: {exc}", file=sys.stderr, flush=True)
                self._file = None

    def close(self) -> None:
        with self._lock:
            if self._file is not None:
                try:
                    self._file.close()
                except OSError:
                    pass
                self._file = None


def _plain(value):
    """Turn things JSON does not know (commands, numpy numbers) into plain values."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def latest(directory: Optional[str] = None) -> Optional[str]:
    """The newest run log file, or None if there is none."""
    directory = directory or default_dir()
    try:
        files = sorted(f for f in os.listdir(directory) if f.endswith(".jsonl"))
    except OSError:
        return None
    return os.path.join(directory, files[-1]) if files else None


def read(path: str) -> list:
    """Every entry of a run log, skipping any line that is not complete JSON (a run cut short)."""
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                entries.append(json.loads(line))
            except ValueError:
                continue
    return entries
