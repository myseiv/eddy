import math

import numpy as np
import pytest

from eddy import gestures, intents
from eddy.adapters.line import LineAdapter, LineServer
from eddy.adapters.ros2 import TWIST, TWIST_STAMPED, classify_graph, front_range, urdf_limits
from eddy.adapters.sim import SimAdapter
from eddy.kinematics import Chain, demo_arm
from eddy.model import Drive, Gripper, Home, MoveTo, Report, Say, Stop, Unsupported, Wander

URDF = """
<robot name="testbot">
  <link name="base_link"/><link name="wheel_l"/><link name="wheel_r"/>
  <link name="l1"/><link name="l2"/><link name="l3"/><link name="tool"/>
  <joint name="wheel_left" type="continuous"><parent link="base_link"/><child link="wheel_l"/>
    <origin xyz="0 0.1 0"/><axis xyz="0 1 0"/></joint>
  <joint name="wheel_right" type="continuous"><parent link="base_link"/><child link="wheel_r"/>
    <origin xyz="0 -0.1 0"/><axis xyz="0 1 0"/></joint>
  <joint name="pan" type="revolute"><parent link="base_link"/><child link="l1"/>
    <origin xyz="0 0 0.1"/><axis xyz="0 0 1"/><limit lower="-3.0" upper="3.0" velocity="1" effort="1"/></joint>
  <joint name="lift" type="revolute"><parent link="l1"/><child link="l2"/>
    <origin xyz="0 0 0.05" rpy="0 0 0"/><axis xyz="0 1 0"/><limit lower="-1.5" upper="1.5" velocity="1" effort="1"/></joint>
  <joint name="elbow" type="revolute"><parent link="l2"/><child link="l3"/>
    <origin xyz="0 0 0.2"/><axis xyz="0 1 0"/><limit lower="-2.2" upper="2.2" velocity="1" effort="1"/></joint>
  <joint name="tool_fixed" type="fixed"><parent link="l3"/><child link="tool"/><origin xyz="0 0 0.2"/></joint>
</robot>
"""


# ---- kinematics ------------------------------------------------------------------

def test_fk_of_straight_arm():
    arm = demo_arm()
    assert np.allclose(arm.position([0, 0, 0, 0]), [0, 0, 0.63])
    # shoulder bent 90 degrees forward: everything above the shoulder lies along +x
    assert np.allclose(arm.position([0, math.pi / 2, 0, 0]), [0.48, 0, 0.15], atol=1e-9)


def test_jacobian_matches_finite_differences():
    arm = demo_arm()
    q = np.array([0.3, -0.5, 0.9, 0.2])
    J = arm.jacobian(q)
    for i in range(4):
        dq = np.zeros(4)
        dq[i] = 1e-6
        numeric = (arm.position(q + dq) - arm.position(q - dq)) / 2e-6
        assert np.allclose(J[:, i], numeric, atol=1e-6)


def test_ik_reaches_many_random_points_within_limits():
    arm = demo_arm()
    rng = np.random.default_rng(1)
    lo = np.array([j.lower for j in arm.active])
    hi = np.array([j.upper for j in arm.active])
    for _ in range(40):
        target = arm.position(rng.uniform(lo, hi))     # reachable by construction
        result = arm.ik(target)
        assert result.success, f"failed for {target}: {result.error}"
        assert np.linalg.norm(arm.position(result.q) - target) < 1e-3
        assert np.all(np.array(result.q) >= lo - 1e-9) and np.all(np.array(result.q) <= hi + 1e-9)


def test_ik_reports_failure_for_unreachable_point():
    result = demo_arm().ik([1.5, 0, 0])
    assert not result.success and result.error > 0.5


def test_urdf_picks_the_arm_not_a_wheel():
    chain = Chain.from_urdf(URDF)
    assert chain.joint_names == ["pan", "lift", "elbow"]
    assert np.allclose(chain.position([0, 0, 0]), [0, 0, 0.55])
    assert chain.active[1].lower == -1.5
    with pytest.raises(ValueError):
        Chain.from_urdf(URDF, tip="nope")


def test_urdf_round_trip_keeps_geometry():
    arm = demo_arm()
    again = Chain.from_urdf(arm.to_urdf())
    q = [0.4, -0.7, 1.1, 0.3]
    assert again.joint_names == arm.joint_names
    assert np.allclose(again.position(q), arm.position(q), atol=1e-6)


def test_urdf_with_rotated_origin():
    urdf = """<robot name="r"><link name="a"/><link name="b"/><link name="c"/>
      <joint name="j1" type="revolute"><parent link="a"/><child link="b"/>
        <origin xyz="0 0 0" rpy="0 1.5707963 0"/><axis xyz="0 0 1"/><limit lower="-1" upper="1"/></joint>
      <joint name="j2" type="fixed"><parent link="b"/><child link="c"/><origin xyz="0 0 1"/></joint></robot>"""
    chain = Chain.from_urdf(urdf)
    assert np.allclose(chain.position([0.5]), [1, 0, 0], atol=1e-6)   # pitched 90 deg: z becomes x


# ---- sentences -------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("drive forward 1 meter", Drive(linear=0.2, distance=1.0)),
    ("Eddy, go forward two metres please", Drive(linear=0.2, distance=2.0)),
    ("move back 30 cm", Drive(linear=-0.2, distance=0.3)),
    ("reverse for 2 seconds", Drive(linear=-0.2, duration=2.0)),
    ("go forward", Drive(linear=0.2, distance=0.5)),
    ("drive forward slowly 1 m", Drive(linear=0.1, distance=1.0)),
    ("turn left", Drive(angular=0.8, angle=math.pi / 2)),
    ("turn right 45 degrees", Drive(angular=-0.8, angle=math.radians(45))),
    ("rotate clockwise ninety degrees", Drive(angular=-0.8, angle=math.pi / 2)),
    ("turn around", Drive(angular=0.8, angle=math.pi)),
    ("stop", Stop()), ("HALT!", Stop()), ("freeze", Stop()),
    ("open the gripper", Gripper(1.0)), ("grab it", Gripper(0.0)),
    ("go home", Home()),
    ("reach 0.2 0 0.3", MoveTo(0.2, 0.0, 0.3)),
    ("move the arm to 20 -5 30 cm", MoveTo(0.2, -0.05, 0.3)),
    ("say Hello World", Say("Hello World")),
    ("status", Report()),
    ("explore for 30 seconds", Wander(30.0)),
    ("go back to meters", Drive(linear=-0.2, distance=2.0)),          # speech-to-text homophones
    ("drive forward for meters", Drive(linear=0.2, distance=4.0)),
    ("reverse for 2 seconds", Drive(linear=-0.2, duration=2.0)),
    ("drive for six seconds", Drive(linear=0.2, duration=6.0)),
])
def test_sentences(text, expected):
    got = intents.parse(text)
    assert type(got) is type(expected)
    for key, value in vars(expected).items():
        assert getattr(got, key) == pytest.approx(value), f"{text}: {key}"


@pytest.mark.parametrize("text", ["", "make me a sandwich", "what is the weather", "arm", "go"])
def test_unknown_sentences_do_nothing(text):
    assert intents.parse(text) is None


def test_stop_wins_inside_a_longer_sentence():
    assert isinstance(intents.parse("no no stop going forward"), Stop)


# ---- hand signs ------------------------------------------------------------------

@pytest.mark.parametrize("extended,thumb,expected", [
    (["index", "middle", "ring", "pinky"], "left", "open_palm"),
    (["index", "middle", "ring", "pinky"], None, "open_palm"),
    ([], None, "fist"),
    (["index"], None, "point"),
    (["index", "middle"], None, "victory"),
    ([], "up", "thumb_up"), ([], "down", "thumb_down"),
    ([], "left", "thumb_left"), ([], "right", "thumb_right"),
    (["index", "pinky"], None, None),
])
def test_hand_signs(extended, thumb, expected):
    for scale in (0.05, 0.1, 0.3):                      # near and far hands
        assert gestures.classify(gestures.fake_hand(extended, thumb, scale=scale)) == expected


def test_hand_signs_on_real_photos():
    """Landmarks MediaPipe found in real photographs of hands (see tests/data/real_hands.json)."""
    import json
    import pathlib
    data = json.loads((pathlib.Path(__file__).parent / "data" / "real_hands.json").read_text())
    for hand in data["hands"]:
        assert gestures.classify(hand["landmarks"]) == hand["expect"], hand["photo"]


def test_gesture_filter_needs_a_steady_sign_and_repeats_a_held_one():
    f = gestures.GestureFilter(hold_frames=3, repeat_frames=4)
    seen = [f.update(g) for g in ["point", "fist", "point", "point", "point", "point", "point", "point", "point", None]]
    assert seen == [None, None, None, None, "point", None, None, None, "point", None]


def test_bad_landmarks_are_ignored():
    assert gestures.classify(None) is None
    assert gestures.classify([[0, 0]] * 5) is None


# ---- ROS 2 graph reading (no ROS needed) ---------------------------------------------

def test_classify_a_typical_mobile_manipulator():
    topics = [
        ("/cmd_vel", [TWIST]), ("/cmd_vel_nav", [TWIST]), ("/teleop/cmd_vel", [TWIST]),
        ("/odom", ["nav_msgs/msg/Odometry"]), ("/odometry/filtered", ["nav_msgs/msg/Odometry"]),
        ("/scan", ["sensor_msgs/msg/LaserScan"]), ("/joint_states", ["sensor_msgs/msg/JointState"]),
        ("/arm_controller/joint_trajectory", ["trajectory_msgs/msg/JointTrajectory"]),
        ("/gripper_controller/joint_trajectory", ["trajectory_msgs/msg/JointTrajectory"]),
        ("/camera/image_raw", ["sensor_msgs/msg/Image"]), ("/robot_description", ["std_msgs/msg/String"]),
        ("/rosout", ["rcl_interfaces/msg/Log"]),
    ]
    p = classify_graph(topics)
    assert p["cmd_vel"] == ("/cmd_vel", TWIST)
    assert p["odom"] == "/odom" and p["scan"] == "/scan" and p["joint_states"] == "/joint_states"
    assert p["arm_trajectories"] == ["/arm_controller/joint_trajectory"]
    assert p["gripper_trajectories"] == ["/gripper_controller/joint_trajectory"]
    assert p["cameras"] == ["/camera/image_raw"] and p["description"] == "/robot_description"


def test_classify_prefers_the_topic_someone_listens_to():
    topics = [("/cmd_vel", [TWIST]), ("/diff_drive_controller/cmd_vel", [TWIST_STAMPED])]
    p = classify_graph(topics, sub_counts={"/cmd_vel": 0, "/diff_drive_controller/cmd_vel": 1})
    assert p["cmd_vel"] == ("/diff_drive_controller/cmd_vel", TWIST_STAMPED)


def test_classify_does_not_guess_among_unrelated_twists():
    p = classify_graph([("/wind", [TWIST]), ("/current", [TWIST])])
    assert "cmd_vel" not in p
    assert classify_graph([("/turtle1/cmd_vel", [TWIST])])["cmd_vel"][0] == "/turtle1/cmd_vel"
    assert classify_graph([])["arm_trajectories"] == []


def test_front_range():
    inc = math.radians(1)
    ranges = [5.0] * 360
    ranges[0], ranges[5], ranges[90], ranges[355] = 2.0, 1.2, 0.3, float("inf")
    assert front_range(ranges, 0.0, inc, 0.1, 10.0) == 1.2          # the 0.3 is off to the side
    assert front_range([float("inf")] * 360, 0.0, inc, 0.1, 10.0) == 10.0
    assert front_range([1.0] * 10, math.radians(90), inc, 0.1, 10.0) is None   # no beams point ahead


def test_urdf_limits():
    limits = urdf_limits(URDF)
    assert limits["lift"] == (-1.5, 1.5, "revolute")
    assert limits["wheel_left"][2] == "continuous"


# ---- line protocol -----------------------------------------------------------------

@pytest.fixture
def served():
    sim = SimAdapter(name="wire-bot").go_live()
    server = LineServer(sim, host="127.0.0.1").start()
    yield sim, server
    server.stop()


def test_line_protocol_round_trip(served):
    sim, server = served
    robot = LineAdapter(f"tcp:127.0.0.1:{server.port}")
    model = robot.connect()
    assert model.name == "wire-bot" and model.drive.max_linear == 0.5 and len(model.joints) == 4
    robot.set_velocity(0.2, 0.0)
    assert sim.v == 0.2
    robot.set_joints({"elbow": 0.5})
    assert sim.targets["elbow"] == 0.5
    robot.set_gripper(0.25)
    assert robot.state().gripper == 0.25
    robot.stop()
    assert sim.v == 0.0
    robot.close()


def test_line_server_survives_garbage(served):
    sim, server = served
    for bad in ["", "V", "V fast slow", "J nope=1", "J elbow", "G wide", "BOGUS 1 2", "\x00\xff"]:
        assert server.handle(bad).startswith("ERR"), bad
    assert server.handle("V 0.1 0") == "OK"            # still alive afterwards


def test_line_adapter_reports_unsupported(served):
    server = LineServer(SimAdapter(arm=False), host="127.0.0.1").start()
    try:
        robot = LineAdapter(f"tcp:127.0.0.1:{server.port}")
        robot.connect()
        with pytest.raises(Unsupported):
            robot.set_gripper(1.0)
        robot.close()
    finally:
        server.stop()


def test_line_adapter_rejects_a_non_eddy_device():
    import socket
    import threading
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)

    def not_a_robot():
        conn, _ = listener.accept()
        conn.recv(100)
        conn.sendall(b"hello I am a printer\n")
        conn.close()
    threading.Thread(target=not_a_robot, daemon=True).start()
    with pytest.raises(Exception):
        LineAdapter(f"tcp:127.0.0.1:{listener.getsockname()[1]}").connect()
    listener.close()
