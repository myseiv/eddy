"""Eddy against a microcontroller-style robot over a real serial device (a pseudo-terminal)
and over TCP, including the ways such links go wrong. These run in real time."""
import math
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent / "firmware"))
from fake_mcu import FakeMCU  # noqa: E402

from eddy.adapters import open_robot  # noqa: E402
from eddy.brain import Eddy  # noqa: E402
from eddy.senses.speak import Speaker  # noqa: E402

pytest.importorskip("serial")


def attach(**kw):
    mcu = FakeMCU(**kw)
    eddy = Eddy(open_robot(mcu.port), Speaker("silent"))
    eddy.start()
    return mcu, eddy


def test_serial_board_that_boots_slowly_and_prints_a_banner():
    mcu, eddy = attach(boot=1.0)        # deaf for a second after the port opens, then a banner
    try:
        assert eddy.model.name == "fake-mcu-rover"
        assert {"drive", "joints", "gripper", "odometry", "range"} <= set(eddy.model.capabilities())
        assert "stops by itself" in eddy.model.details["link loss"]
        r = eddy.tell("drive forward 30 cm")
        assert r.ok and abs(mcu.x - 0.30) < 0.03, (r.message, mcu.x)
        r = eddy.tell("turn left 90 degrees")
        assert r.ok and abs(math.degrees(mcu.th) - 90) < 6, (r.message, mcu.th)
        assert eddy.tell("set joint pan to 30 degrees").ok and abs(mcu.joints["pan"] - math.radians(30)) < 1e-3
        assert eddy.tell("close the gripper").ok and mcu.grip == 0.0
        assert not eddy.tell("reach 0.2 0 0.3").ok          # the board sent no arm geometry
    finally:
        eddy.close(); mcu.close()


def test_tcp_board():
    mcu, eddy = attach(link="tcp")
    try:
        r = eddy.tell("drive forward 30 cm and turn right 45 degrees")
        assert r.ok and abs(mcu.x - 0.30) < 0.03 and abs(math.degrees(mcu.th) + 45) < 6, r.message
    finally:
        eddy.close(); mcu.close()


def test_board_with_no_position_sensor_is_driven_by_time():
    mcu, eddy = attach(odometry=False, range_sensor=False)
    try:
        r = eddy.tell("drive forward for 2 seconds")
        assert r.ok and "no position" in r.message and abs(mcu.x - 0.40) < 0.04, (r.message, mcu.x)
        assert not eddy.tell("wander for 5 seconds").ok     # no range sensor: refused
    finally:
        eddy.close(); mcu.close()


def test_slow_link_does_not_stretch_a_timed_drive():
    mcu, eddy = attach(latency=0.08)
    try:
        began = time.monotonic()
        r = eddy.tell("drive forward for 2 seconds")
        assert r.ok and time.monotonic() - began < 3.0 and abs(mcu.x - 0.40) < 0.06, (r.message, mcu.x)
    finally:
        eddy.close(); mcu.close()


def test_lost_answers_are_retried():
    mcu, eddy = attach(drop_every=15)
    try:
        r = eddy.tell("drive forward 40 cm")
        assert r.ok and abs(mcu.x - 0.40) < 0.04, (r.message, mcu.x)
    finally:
        eddy.close(); mcu.close()


def test_cable_pulled_mid_drive():
    mcu, eddy = attach(die_at=1.0)
    try:
        began = time.monotonic()
        r = eddy.tell("drive forward 1 meter")
        assert not r.ok and "lost contact" in r.message and "cannot confirm" in r.message, r.message
        assert time.monotonic() - began < 6.0
        time.sleep(0.7)
        assert mcu.v == 0.0                                 # the board's own watchdog stopped it
    finally:
        eddy.close(); mcu.close()


def test_board_without_a_watchdog_is_flagged():
    mcu, eddy = attach(watchdog=0)
    try:
        assert "WARNING" in eddy.model.details["link loss"] and "watchdog" in eddy.model.details["link loss"]
    finally:
        eddy.close(); mcu.close()


def test_answers_that_are_not_the_protocol_are_named():
    mcu, eddy = attach(bad_state=True)
    try:
        r = eddy.tell("status")
        assert not r.ok and "not the Eddy protocol" in r.message, r.message
    finally:
        eddy.close(); mcu.close()


def test_frozen_position_reading_is_noticed():
    mcu, eddy = attach(freeze_odometry=True)
    try:
        r = eddy.tell("drive forward 50 cm")
        assert not r.ok and "no effect" in r.message and mcu.v == 0.0, r.message
    finally:
        eddy.close(); mcu.close()
