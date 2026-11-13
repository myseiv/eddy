"""Speaking. Uses whatever speech program the machine already has; otherwise prints."""
from __future__ import annotations

import shutil
import subprocess
from typing import List, Optional

# program -> arguments placed before the text
_ENGINES = (("espeak-ng", []), ("espeak", []), ("say", []), ("spd-say", ["--wait"]))


class Speaker:
    def __init__(self, backend: str = "auto"):
        """backend: auto, print, silent, or the name of a speech program."""
        self.backend = backend
        self.history: List[str] = []
        self.command: Optional[List[str]] = None
        if backend not in ("print", "silent"):
            for program, args in _ENGINES:
                if backend in ("auto", program) and shutil.which(program):
                    self.command = [program] + args
                    self.backend = program
                    break
            else:
                self.backend = "print"

    def say(self, text: str) -> None:
        self.history.append(text)
        if self.backend == "silent":
            return
        print(f"[eddy] {text}", flush=True)
        if self.command:
            try:
                subprocess.run(self.command + [text], timeout=30, check=False,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except (OSError, subprocess.TimeoutExpired):
                pass   # losing the voice must never stop the robot
