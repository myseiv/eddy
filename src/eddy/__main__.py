"""Lets `python -m eddy ...` do the same as the `eddy` command."""
import sys

from .cli import main

sys.exit(main())
