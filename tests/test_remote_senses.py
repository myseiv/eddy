"""Hand signs and voice run on the controlling computer; the robot is reached over TCP.

The robot (a simulator, served on a local port) runs only Eddy's core and has no camera, no microphone
and no extra packages. A second Eddy, standing in for the laptop, takes hand signs and spoken
sentences and drives the robot through the line protocol. So the camera and voice never have to be
installed on the robot.
"""
import threading
import time

import pytest

from eddy import gestures
from eddy.adapters.line import LineAdapter, LineServer
from eddy.adapters.sim import SimAdapter
from eddy.brain import Eddy
from eddy.senses.listen import heard_command
from eddy.senses.speak import Speaker

OPEN = ["index", "middle", "ring", "pinky"]


@pytest.fixture
def pair():
    """A served robot and the 'laptop' Eddy connected to it."""
    robot = SimAdapter(name="far-rover", room=30.0).go_live()      # moves on its own clock, like a real robot
    server = LineServer(robot, host="127.0.0.1").start()
    laptop = Eddy(LineAdapter(f"tcp:127.0.0.1:{server.port}"), Speaker("silent"))
    laptop.start()
    yield robot, laptop
    laptop.close()
    server.stop()


def show(laptop, hand, frames):
    """Feed the same hand to Eddy for this many camera frames; returns the results that acted."""
    return [r for r in (laptop.sign(hand) for _ in range(frames)) if r is not None]


def drive_in_background(laptop):
    """Start a long drive in another thread, as a spoken or typed command would."""
    t = threading.Thread(target=lambda: laptop.tell("drive forward 1.5 meters"), daemon=True)
    t.start()
    time.sleep(0.6)             # let it get going
    return t


def test_a_pointing_hand_at_the_laptop_moves_a_robot_it_has_no_camera_on(pair):
    robot, laptop = pair
    acted = show(laptop, gestures.fake_hand(["index"]), 10)
    assert acted and acted[0].ok
    assert robot.x > 0.08


def test_an_open_palm_stops_a_moving_remote_robot_quickly(pair):
    robot, laptop = pair
    t = drive_in_background(laptop)
    moving_at = robot.x
    assert robot.v > 0 and moving_at > 0.02
    began = time.time()
    acted = show(laptop, gestures.fake_hand(OPEN, "left"), 5)
    t.join(timeout=3)
    assert acted and "Stopped" in acted[0].message
    assert not t.is_alive() and robot.v == 0
    assert time.time() - began < 1.0
    stopped_at = robot.x
    time.sleep(0.5)
    assert abs(robot.x - stopped_at) < 0.01       # no creep once stopped
    assert stopped_at < 0.8                       # far short of the 1.5 m asked for


def test_a_fist_stops_it_too(pair):
    robot, laptop = pair
    t = drive_in_background(laptop)
    assert robot.v > 0                            # really moving before the stop
    show(laptop, gestures.fake_hand([]), 5)
    t.join(timeout=3)
    assert not t.is_alive() and robot.v == 0


def test_a_spoken_stop_reaches_the_remote_robot(pair):
    robot, laptop = pair
    t = drive_in_background(laptop)
    assert robot.v > 0
    sentence, unclear = heard_command("eddie stop", "eddy")
    assert sentence == "stop" and not unclear
    laptop.tell(sentence)
    t.join(timeout=3)
    assert not t.is_alive() and robot.v == 0


def test_a_thumb_sideways_turns_the_remote_robot(pair):
    robot, laptop = pair
    before = robot.heading
    show(laptop, gestures.fake_hand([], "right"), 10)
    assert robot.heading < before - 0.2          # turned right


def test_a_glimpsed_sign_does_nothing_remotely(pair):
    robot, laptop = pair
    assert show(laptop, gestures.fake_hand(["index"]), 9) == []
    assert robot.x == 0 and robot.v == 0
