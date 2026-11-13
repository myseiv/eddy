"""Eddy AI: a plug-and-play controller that wraps any robot's existing code."""
__version__ = "0.1.0"

# The names most programs need, importable straight from `eddy`:
#   from eddy import Eddy, open_robot
from .brain import Eddy, Result, Safety  # noqa: E402,F401
from .adapters import open_robot, scan  # noqa: E402,F401
