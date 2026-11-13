"""End-to-end tests: the whole stack through its public entry points (self-test and CLI),
on the simulator."""
import subprocess
import sys

from eddy import selftest
from eddy.cli import main


def test_selftest_passes(capsys):
    """The built-in self-test passes and prints no FAIL line."""
    assert selftest.run(verbose=True) == 0
    assert "FAIL" not in capsys.readouterr().out


def test_cli_do_on_simulator(capsys):
    """`eddy do` carries out sentences in order and reports what happened."""
    assert main(["do", "-r", "sim", "drive forward 1 meter", "turn left", "status"]) == 0
    out = capsys.readouterr().out
    assert "moved 1.00 m" in out and "turned 90 degrees" in out and "x 1.00" in out


def test_cli_do_fails_cleanly_on_nonsense(capsys):
    """A sentence Eddy does not understand gives exit code 1, not a crash."""
    assert main(["do", "-r", "sim", "fly to the moon"]) == 1


def test_cli_info_scan_doctor(capsys):
    """The three inspection commands run and print what they should."""
    assert main(["info", "-r", "sim"]) == 0
    assert "base_yaw" in capsys.readouterr().out
    assert main(["scan"]) == 0
    assert "sim" in capsys.readouterr().out
    assert main(["doctor"]) == 0
    assert "simulator" in capsys.readouterr().out


def test_cli_bad_robot_is_a_clean_error(capsys):
    """An unknown or unreachable robot gives exit code 2 and no traceback."""
    assert main(["info", "-r", "warpdrive:9"]) == 2
    assert main(["info", "-r", "tcp:127.0.0.1:1"]) == 2     # nothing listening


def test_module_entry_point():
    """`python -m eddy` works as well as the `eddy` command."""
    out = subprocess.run([sys.executable, "-m", "eddy", "--version"], capture_output=True, text=True)
    assert out.returncode == 0 and "eddy 0.1.0" in out.stdout
