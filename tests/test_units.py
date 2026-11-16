"""Unit tests: each part of Eddy on its own, with no robot and no hardware.

Run with:  .venv/bin/python -m pytest -q
"""
import math

import numpy as np
import pytest

from eddy import gestures, intents
from eddy.adapters.line import LineAdapter, LineServer
from eddy.adapters.ros2 import TWIST, TWIST_STAMPED, classify_graph, front_range, urdf_limits, urdf_mimics, cloud_front_range, robot_namespaces, scan_rotation
from eddy.adapters.sim import SimAdapter
from eddy.kinematics import Chain, demo_arm
from eddy.brain import Eddy
from eddy.senses.speak import Speaker
from eddy.model import Drive, Gripper, Home, MoveTo, Report, Say, SetJoints, Stop, Unsupported, Wander

# A small robot description used by several tests: two wheels (continuous joints) and a
# three-joint arm (pan, lift, elbow) ending in a fixed tool link.
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
    """Forward kinematics gives the tool position we can work out by hand."""
    arm = demo_arm()
    assert np.allclose(arm.position([0, 0, 0, 0]), [0, 0, 0.63])
    # shoulder bent 90 degrees forward: everything above the shoulder lies along +x
    assert np.allclose(arm.position([0, math.pi / 2, 0, 0]), [0.48, 0, 0.15], atol=1e-9)


def test_jacobian_matches_finite_differences():
    """The Jacobian agrees with nudging each joint a tiny amount and measuring the tool."""
    arm = demo_arm()
    q = np.array([0.3, -0.5, 0.9, 0.2])
    J = arm.jacobian(q)
    for i in range(4):
        dq = np.zeros(4)
        dq[i] = 1e-6
        numeric = (arm.position(q + dq) - arm.position(q - dq)) / 2e-6
        assert np.allclose(J[:, i], numeric, atol=1e-6)


def test_ik_reaches_many_random_points_within_limits():
    """Inverse kinematics solves 40 random reachable targets to under 1 mm, inside the joint limits."""
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
    """A point far outside the arm's reach is reported as a failure, not a wrong answer."""
    result = demo_arm().ik([1.5, 0, 0])
    assert not result.success and result.error > 0.5


def test_urdf_picks_the_arm_not_a_wheel():
    """Reading a URDF finds the arm chain, ignores the wheels, and keeps the joint limits."""
    chain = Chain.from_urdf(URDF)
    assert chain.joint_names == ["pan", "lift", "elbow"]
    assert np.allclose(chain.position([0, 0, 0]), [0, 0, 0.55])
    assert chain.active[1].lower == -1.5
    with pytest.raises(ValueError):
        Chain.from_urdf(URDF, tip="nope")


def test_urdf_round_trip_keeps_geometry():
    """Writing a chain out as URDF and reading it back gives the same arm."""
    arm = demo_arm()
    again = Chain.from_urdf(arm.to_urdf())
    q = [0.4, -0.7, 1.1, 0.3]
    assert again.joint_names == arm.joint_names
    assert np.allclose(again.position(q), arm.position(q), atol=1e-6)


def test_urdf_with_rotated_origin():
    """A joint whose frame is rotated (rpy) is handled correctly."""
    urdf = """<robot name="r"><link name="a"/><link name="b"/><link name="c"/>
      <joint name="j1" type="revolute"><parent link="a"/><child link="b"/>
        <origin xyz="0 0 0" rpy="0 1.5707963 0"/><axis xyz="0 0 1"/><limit lower="-1" upper="1"/></joint>
      <joint name="j2" type="fixed"><parent link="b"/><child link="c"/><origin xyz="0 0 1"/></joint></robot>"""
    chain = Chain.from_urdf(urdf)
    assert np.allclose(chain.position([0.5]), [1, 0, 0], atol=1e-6)   # pitched 90 deg: z becomes x


# ---- sentences -------------------------------------------------------------------

# Each pair is a sentence and the exact command it must become.
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
    ("set joint1 to 0.3", SetJoints({"joint1": 0.3})),
    ("move joint elbow to 90 degrees", SetJoints({"elbow": math.pi / 2})),
    ("set gripper_base_joint to -20 deg", SetJoints({"gripper_base_joint": math.radians(-20)})),
    ("say Hello World", Say("Hello World")),
    ("status", Report()),
    ("explore for 30 seconds", Wander(30.0)),
    ("go back to meters", Drive(linear=-0.2, distance=2.0)),          # speech-to-text homophones
    ("drive forward for meters", Drive(linear=0.2, distance=4.0)),
    ("reverse for 2 seconds", Drive(linear=-0.2, duration=2.0)),
    ("drive for six seconds", Drive(linear=0.2, duration=6.0)),
])
def test_sentences(text, expected):
    """The parser turns the sentence into the right command with the right numbers."""
    got = intents.parse(text)
    assert type(got) is type(expected)
    for key, value in vars(expected).items():
        assert getattr(got, key) == pytest.approx(value), f"{text}: {key}"


@pytest.mark.parametrize("text", ["", "make me a sandwich", "what is the weather", "arm", "go"])
def test_unknown_sentences_do_nothing(text):
    """Anything Eddy does not understand becomes no command at all."""
    assert intents.parse(text) is None


def test_joints_can_be_named_loosely():
    """A joint can be named exactly, without capitals, by its number, or by a unique word in
    its name; an ambiguous word matches nothing."""
    from eddy.model import JointSpec, RobotModel
    model = RobotModel("r", joints=[JointSpec("joint1"), JointSpec("joint2"), JointSpec("Elbow_Pitch"),
                                    JointSpec("gripper_left"), JointSpec("gripper_right")])
    assert model.find_joint("joint1").name == "joint1"
    assert model.find_joint("1").name == "joint1"
    assert model.find_joint("elbow_pitch").name == "Elbow_Pitch"
    assert model.find_joint("elbow").name == "Elbow_Pitch"
    assert model.find_joint("gripper") is None and model.find_joint("wrist") is None


def test_stop_wins_inside_a_longer_sentence():
    """"stop" anywhere in a sentence means stop, even next to driving words."""
    assert isinstance(intents.parse("no no stop going forward"), Stop)


# ---- hand signs ------------------------------------------------------------------

# Each row: which fingers are extended, where the thumb points, and the sign that must be named.
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
    """Made-up hands are named correctly whether the hand is small (far) or large (near) in the image."""
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
    """A moving sign counts after twice the hold (6 frames here), repeats every 4 while held, and a change resets the count."""
    f = gestures.GestureFilter(hold_frames=3, repeat_frames=4)
    seen = [f.update(g) for g in ["point", "fist", "point"] + ["point"] * 9 + [None]]
    assert seen == [None] * 7 + ["point"] + [None] * 3 + ["point"] + [None]


def test_stop_signs_act_after_the_short_hold():
    """Stopping must stay quick: a fist or open palm counts after the short hold, a pointing hand needs longer."""
    for sign in ("fist", "open_palm"):
        f = gestures.GestureFilter(hold_frames=3)
        assert [f.update(sign) for _ in range(3)] == [None, None, sign]
    f = gestures.GestureFilter(hold_frames=3)
    assert "point" not in [f.update("point") for _ in range(5)]


def test_a_thumb_halfway_between_sideways_and_up_is_no_sign():
    """The wavering thumb that flipped between left, right and up in the live test now names nothing."""
    lm = gestures.fake_hand([], "right")
    lm[3] = [lm[2][0] + 0.05, lm[2][1] - 0.05]
    lm[4] = [lm[2][0] + 0.1, lm[2][1] - 0.1]        # sticking out, exactly diagonal
    assert gestures.classify(lm) is None


def test_bad_landmarks_are_ignored():
    """No hand, or the wrong number of points, is simply not a sign."""
    assert gestures.classify(None) is None
    assert gestures.classify([[0, 0]] * 5) is None


# ---- ROS 2 graph reading (no ROS needed) ---------------------------------------------

def test_classify_a_typical_mobile_manipulator():
    """From a realistic list of topics, each role is given to the conventional topic."""
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
    """A velocity topic with a subscriber beats a better-named one nobody listens to."""
    topics = [("/cmd_vel", [TWIST]), ("/diff_drive_controller/cmd_vel", [TWIST_STAMPED])]
    p = classify_graph(topics, sub_counts={"/cmd_vel": 0, "/diff_drive_controller/cmd_vel": 1})
    assert p["cmd_vel"] == ("/diff_drive_controller/cmd_vel", TWIST_STAMPED)


def test_classify_prefers_the_topic_the_robots_own_software_uses():
    """Seen on a Hiwonder LanderPi: two velocity topics both reach the motors, but the robot's
    own programs all publish to one of them. That one wins over the better-named one."""
    topics = [("/cmd_vel", [TWIST]), ("/controller/cmd_vel", [TWIST])]
    p = classify_graph(topics, sub_counts={"/cmd_vel": 1, "/controller/cmd_vel": 1},
                       pub_counts={"/cmd_vel": 0, "/controller/cmd_vel": 5})
    assert p["cmd_vel"][0] == "/controller/cmd_vel"
    # With nothing to tell them apart, the conventional name still wins.
    p = classify_graph(topics, sub_counts={"/cmd_vel": 1, "/controller/cmd_vel": 1},
                       pub_counts={"/cmd_vel": 0, "/controller/cmd_vel": 0})
    assert p["cmd_vel"][0] == "/cmd_vel"


def test_classify_does_not_guess_among_unrelated_twists():
    """Several Twist topics that do not look like velocity commands are left alone; a single cmd_vel is used."""
    p = classify_graph([("/wind", [TWIST]), ("/current", [TWIST])])
    assert "cmd_vel" not in p
    assert classify_graph([("/turtle1/cmd_vel", [TWIST])])["cmd_vel"][0] == "/turtle1/cmd_vel"
    assert classify_graph([])["arm_trajectories"] == []


def test_classify_finds_action_driven_controllers():
    """Controllers offered as FollowJointTrajectory actions are found; one that also has a
    trajectory topic is used through the topic and not listed twice."""
    follow = ["control_msgs/action/FollowJointTrajectory"]
    actions = [("/arm_controller/follow_joint_trajectory", follow),
               ("/gripper_controller/follow_joint_trajectory", follow),
               ("/navigate_to_pose", ["nav2_msgs/action/NavigateToPose"])]
    p = classify_graph([("/cmd_vel", [TWIST])], actions=actions)
    assert p["arm_actions"] == ["/arm_controller/follow_joint_trajectory"]
    assert p["gripper_actions"] == ["/gripper_controller/follow_joint_trajectory"]
    both = classify_graph([("/arm_controller/joint_trajectory", ["trajectory_msgs/msg/JointTrajectory"])], actions=actions)
    assert both["arm_actions"] == [] and both["arm_trajectories"] == ["/arm_controller/joint_trajectory"]
    assert classify_graph([])["arm_actions"] == []


def test_front_range():
    """Only laser beams pointing ahead count; invalid readings are skipped."""
    inc = math.radians(1)
    ranges = [5.0] * 360
    ranges[0], ranges[5], ranges[90], ranges[355] = 2.0, 1.2, 0.3, float("inf")
    assert front_range(ranges, 0.0, inc, 0.1, 10.0) == 1.2          # the 0.3 is off to the side
    assert front_range([float("inf")] * 360, 0.0, inc, 0.1, 10.0) == 10.0
    assert front_range([1.0] * 10, math.radians(90), inc, 0.1, 10.0) is None   # no beams point ahead


def test_urdf_limits():
    """Joint limits and types are read from the URDF."""
    limits = urdf_limits(URDF)
    assert limits["lift"] == (-1.5, 1.5, "revolute")
    assert limits["wheel_left"][2] == "continuous"


# ---- line protocol -----------------------------------------------------------------

@pytest.fixture
def served():
    """A live simulated robot served over the line protocol on a local port; stopped after the test."""
    sim = SimAdapter(name="wire-bot").go_live()
    server = LineServer(sim, host="127.0.0.1").start()
    yield sim, server
    server.stop()


def test_line_protocol_round_trip(served):
    """Every command sent through the protocol arrives at the robot on the other end."""
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
    """Malformed lines get an ERR answer and never take the server down."""
    sim, server = served
    for bad in ["", "V", "V fast slow", "J nope=1", "J elbow", "G wide", "BOGUS 1 2", "\x00\xff"]:
        assert server.handle(bad).startswith("ERR"), bad
    assert server.handle("V 0.1 0") == "OK"            # still alive afterwards


def test_line_adapter_reports_unsupported(served):
    """A robot without a gripper answers "unsupported", which surfaces as the Unsupported exception."""
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
    """Connecting to something that is not an Eddy robot fails cleanly."""
    import socket
    import threading
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)

    def not_a_robot():
        """Accept one connection and answer with something that is not the handshake."""
        conn, _ = listener.accept()
        conn.recv(100)
        conn.sendall(b"hello I am a printer\n")
        conn.close()
    threading.Thread(target=not_a_robot, daemon=True).start()
    with pytest.raises(Exception):
        LineAdapter(f"tcp:127.0.0.1:{listener.getsockname()[1]}").connect()
    listener.close()


def test_a_robot_that_cannot_turn_on_the_spot_is_turned_while_rolling():
    sim = SimAdapter(car=True, arm=False)
    with Eddy(sim, Speaker("silent")) as eddy:
        first = eddy.execute(Drive(angular=0.8, angle=math.pi / 2))
        assert first.ok and "cannot turn on the spot" in first.message, first.message
        assert abs(abs(sim.heading) - math.pi / 2) < 0.1
        assert eddy.model.drive.kind == "car-like"
        # The second turn does not waste time trying on the spot again.
        before = sim.clock
        assert eddy.execute(Drive(angular=-0.8, angle=math.pi / 2)).ok
        assert sim.clock - before < 6 and abs(sim.heading) < 0.1


def test_a_drive_with_no_effect_is_given_up_quickly():
    class Dead(SimAdapter):
        def set_velocity(self, linear, angular):
            pass
    sim = Dead(arm=False)
    with Eddy(sim, Speaker("silent")) as eddy:
        r = eddy.execute(Drive(linear=0.2, distance=1.0))
        assert not r.ok and "no effect" in r.message and sim.clock < 3
        r = eddy.execute(Drive(angular=0.8, angle=1.0))
        assert not r.ok and "heading did not change" in r.message


def test_a_drive_that_stops_getting_anywhere_is_given_up():
    """Reversing into a wall (no rear sensor): Eddy stops soon after progress stops."""
    sim = SimAdapter(arm=False)
    with Eddy(sim, Speaker("silent")) as eddy:
        r = eddy.tell("go back 5 meters")
        assert not r.ok and "no longer getting anywhere" in r.message, r.message
        assert sim.clock < 15       # 1.85 m at 0.2 m/s is about 9 s, plus the 2 s stall window


def test_wander_works_on_a_car_like_robot():
    sim = SimAdapter(car=True, arm=False)
    with Eddy(sim, Speaker("silent")) as eddy:
        r = eddy.tell("wander for 60 seconds")
        assert r.ok and "0 times" not in r.message, r.message
        assert eddy.model.drive.kind == "car-like" and math.hypot(sim.x, sim.y) < 1.9


def test_car_like_turn_with_no_room_says_why():
    sim = SimAdapter(car=True, arm=False)
    with Eddy(sim, Speaker("silent")) as eddy:
        sim.x = 1.65
        r = eddy.tell("turn left 90 degrees")
        assert not r.ok and "roll forward to turn" in r.message, r.message
        sim.x = 0.0
        r = eddy.tell("spin")
        assert r.ok and "rolling forward 0.00" not in r.message, r.message


def test_a_joint_target_past_its_limit_is_refused():
    sim = SimAdapter()
    with Eddy(sim, Speaker("silent")) as eddy:
        r = eddy.tell("set joint shoulder to 500 degrees")
        assert not r.ok and "can only go from" in r.message and abs(sim.joints["shoulder"]) < 1e-6
        assert eddy.tell("set joint shoulder to 30 degrees").ok


def test_several_commands_in_one_sentence():
    assert [type(c).__name__ for c in intents.parse_all("turn left and move 50cm")] == ["Drive", "Drive"]
    assert [type(c).__name__ for c in intents.parse_all("stop and go forward")] == ["Stop"]
    assert [type(c).__name__ for c in intents.parse_all("say hello and goodbye")] == ["Say"]
    assert [type(c).__name__ for c in intents.parse_all("reach 0.2, 0.1, 0.3")] == ["MoveTo"]
    assert intents.parse_all("dance and sing") == []
    sim = SimAdapter(arm=False)
    with Eddy(sim, Speaker("silent")) as eddy:
        r = eddy.tell("turn left 90 degrees, then go forward 50 cm and turn right")
        assert r.ok and abs(sim.y - 0.5) < 0.02 and abs(sim.heading) < 0.05, r.message
        r = eddy.tell("go forward 5 meters and turn left")       # the wall stops the first step
        assert "did not do the remaining 1" in r.message and abs(sim.heading) < 0.05, r.message


def test_urdf_mimic_joints_are_found():
    urdf = ('<robot name="r"><joint name="a" type="revolute"/>'
            '<joint name="b" type="revolute"><mimic joint="a"/></joint></robot>')
    assert urdf_mimics(urdf) == {"b": "a"}


def test_classify_finds_point_clouds_and_published_poses():
    topics = [("/lidar/points", ["sensor_msgs/msg/PointCloud2"]),
              ("/slam/pose", ["geometry_msgs/msg/PoseWithCovarianceStamped"]),
              ("/goal_pose", ["geometry_msgs/msg/PoseStamped"]),
              ("/initialpose", ["geometry_msgs/msg/PoseWithCovarianceStamped"])]
    pubs = {"/lidar/points": 1, "/slam/pose": 1, "/goal_pose": 1, "/initialpose": 0}
    p = classify_graph(topics, {}, pubs)
    assert p["cloud"] == "/lidar/points"
    assert p["pose"] == ("/slam/pose", "geometry_msgs/msg/PoseWithCovarianceStamped")
    assert "cmd_vel" not in p and "scan" not in p


def test_cloud_front_range_looks_only_ahead_and_above_the_ground():
    #        ahead     to the side   ground      behind     the robot itself
    x = [3.0, 5.0,     2.0,          1.0,        -1.0,      0.1]
    y = [0.1, -0.2,    1.5,          0.0,        0.0,       0.0]
    z = [0.0, 0.3,     0.0,          -0.6,       0.0,       0.0]
    assert abs(cloud_front_range(x, y, z) - math.hypot(3.0, 0.1)) < 1e-9
    assert cloud_front_range([2.0], [1.5], [0.0]) is None
    assert cloud_front_range([float("nan")], [0.0], [0.0]) is None


def test_cloud_range_ignores_sloping_ground_and_sees_what_stands_on_it():
    """A sensor 0.6 m up and tilted down sees the floor as a slope; a box on it is the obstacle."""
    rng = np.random.default_rng(0)
    gx, gy = rng.uniform(0.6, 12, 4000), rng.uniform(-4, 4, 4000)
    gz = -0.6 - 0.08 * gx + rng.normal(0, 0.01, 4000)          # the ground, in the sensor's frame
    assert cloud_front_range(gx, gy, gz) is None                # ground alone is not an obstacle
    bx, by = np.full(40, 4.0), rng.uniform(-0.2, 0.2, 40)
    bz = (-0.6 - 0.08 * 4.0) + rng.uniform(0.2, 0.5, 40)       # a box 4 m ahead, 20 to 50 cm tall
    r = cloud_front_range(np.concatenate((gx, bx)), np.concatenate((gy, by)), np.concatenate((gz, bz)))
    assert r is not None and abs(r - 4.0) < 0.1


def test_relative_arm_moves():
    from eddy.model import MoveBy
    assert intents.parse("move the arm up 5 cm") == MoveBy(0, 0, 0.05)
    assert intents.parse("raise the arm") == MoveBy(0, 0, 0.05)
    assert intents.parse("arm left 10 centimeters") == MoveBy(0, 0.1, 0)
    assert intents.parse("lower the arm 2 cm") == MoveBy(0, 0, -0.02)
    assert isinstance(intents.parse("open the hand"), Gripper) and isinstance(intents.parse("turn left"), Drive)
    sim = SimAdapter()
    with Eddy(sim, Speaker("silent")) as eddy:
        assert eddy.tell("reach 0.2 0.1 0.3").ok
        r = eddy.tell("move the arm up 5 cm")
        tool = sim.chain.position([sim.joints[n] for n in sim.chain.joint_names])
        assert r.ok and np.allclose(tool, (0.2, 0.1, 0.35), atol=0.003), (r.message, tool)
    with Eddy(SimAdapter(arm=False), Speaker("silent")) as eddy:
        assert not eddy.tell("raise the arm").ok


def test_classify_finds_gripper_command_actions():
    p = classify_graph([], {}, {}, [("/hand_controller/gripper_cmd", ["control_msgs/action/GripperCommand"]),
                                    ("/arm_controller/follow_joint_trajectory", ["control_msgs/action/FollowJointTrajectory"])])
    assert p["gripper_commands"] == ["/hand_controller/gripper_cmd"]
    assert p["arm_actions"] == ["/arm_controller/follow_joint_trajectory"] and p["gripper_actions"] == []


def test_robot_namespaces():
    T, O, S = "geometry_msgs/msg/Twist", "nav_msgs/msg/Odometry", "sensor_msgs/msg/LaserScan"
    two = [("/tb1/cmd_vel", [T]), ("/tb1/odom", [O]), ("/tb2/cmd_vel", [T]), ("/tb2/odom", [O]), ("/tb2/scan", [S])]
    assert robot_namespaces(two) == ["/tb1", "/tb2"]
    # A second command topic on one robot is not a second robot.
    one = [("/cmd_vel", [T]), ("/controller/cmd_vel", [T]), ("/odom", [O]), ("/scan_raw", [S])]
    assert robot_namespaces(one) == [""]
    deep = [("/model/blue/cmd_vel", [T]), ("/model/blue/odometry", [O]), ("/model/green/cmd_vel", [T]), ("/model/green/odometry", [O])]
    assert robot_namespaces(deep) == ["/model/blue", "/model/green"]


def test_classify_finds_a_steering_angle_command():
    A = "ackermann_msgs/msg/AckermannDriveStamped"
    p = classify_graph([("/drive", [A]), ("/odom", ["nav_msgs/msg/Odometry"])], {"/drive": 1}, {"/odom": 1})
    assert p["cmd_vel"] == ("/drive", A)
    # A velocity topic, where there is one, is still preferred.
    p = classify_graph([("/drive", [A]), ("/cmd_vel", [TWIST])], {"/drive": 1, "/cmd_vel": 1}, {})
    assert p["cmd_vel"] == ("/cmd_vel", TWIST)


def test_scan_rotation_measures_a_turn_from_two_scans():
    def room_scan(x, y, heading, half=3.0):
        out = []
        for i in range(360):
            a = heading + math.radians(i) - math.pi
            dx, dy = math.cos(a), math.sin(a)
            hits = [((half if d > 0 else -half) - p) / d for p, d in ((x, dx), (y, dy)) if abs(d) > 1e-9]
            out.append(min(hits))
        return out
    inc = math.radians(1)
    for turn in (0.0, 0.05, -0.08, 0.2):
        got = scan_rotation(room_scan(0.5, -0.3, 0.3), room_scan(0.5, -0.3, 0.3 + turn), inc)
        assert abs(got - turn) < math.radians(1.5), (turn, got)
    assert scan_rotation([1.0] * 10, [1.0] * 12, inc) == 0.0


def test_flying_sentences_and_a_simulated_drone():
    from eddy.model import Climb, Land, TakeOff
    assert intents.parse("take off") == TakeOff(1.0)
    assert intents.parse("take off to 2 meters") == TakeOff(2.0)
    assert intents.parse("land") == Land()
    assert intents.parse("go up 50 cm") == Climb(0.5)
    assert intents.parse("descend 1 meter") == Climb(-1.0)
    assert intents.parse("fly forward 2 meters") == Drive(linear=0.2, distance=2.0)
    assert isinstance(intents.parse("move the arm up 5 cm"), type(intents.parse("raise the arm")))
    sim = SimAdapter(arm=False, flies=True)
    with Eddy(sim, Speaker("silent")) as eddy:
        assert not eddy.tell("drive forward 1 meter").ok              # on the ground: nothing happens
        r = eddy.tell("take off")
        assert r.ok and abs(sim.z - 1.0) < 0.03 and eddy.model.drive.kind == "flying", r.message
        assert eddy.tell("go up 50 cm").ok and abs(sim.z - 1.5) < 0.03
        assert eddy.tell("fly forward 1 meter").ok and abs(sim.x - 1.0) < 0.02
        assert "height" in eddy.tell("status").message
        r = eddy.tell("land")
        assert r.ok and sim.z == 0.0 and "Landed" in r.message, r.message
        assert "already on the ground" in eddy.tell("land").message


def test_a_ground_robot_told_to_take_off_says_it_does_not_fly():
    sim = SimAdapter(arm=False)
    with Eddy(sim, Speaker("silent")) as eddy:
        r = eddy.tell("take off")
        assert not r.ok and "does not seem to be a flying robot" in r.message and sim.clock < 4
        assert eddy.model.drive.kind != "flying"


@pytest.mark.parametrize("sentence", [
    "don't drive forward", "do not go forward 1 meter", "never turn left", "don't open the gripper",
    "go forward 1 meter or turn left", "not forward, backward 1 meter", "drive forward unless something is there",
    "go forward 5", "drive forward -5 meters", "turn left -90 degrees", "go up -1 meter", "go up 3",
])
def test_negations_choices_and_unclear_amounts_are_not_acted_on(sentence):
    assert intents.parse_all(sentence) == [], sentence


def test_stop_always_wins_and_bare_turn_numbers_are_degrees():
    for s in ("don't move", "do not move", "no, stop", "cancel", "don't stop", "go forward and stop"):
        assert intents.parse_all(s) == [Stop()], s
    assert intents.parse("turn left 45") == Drive(angular=0.8, angle=math.radians(45))


def test_moves_too_long_to_finish_are_refused_before_starting():
    sim = SimAdapter(arm=False)
    with Eddy(sim, Speaker("silent")) as eddy:
        for s in ("drive forward 1000 meters", "turn left 100000 degrees", "turn left for 100 minutes"):
            r = eddy.tell(s)
            assert not r.ok and "at a time" in r.message and sim.clock < 0.5, (s, r.message)
        assert abs(sim.x) < 1e-6 and abs(sim.heading) < 1e-6


def test_a_partly_heard_command_is_not_acted_on():
    from eddy.senses.listen import heard_command
    assert heard_command("eddy drive forward one meter") == ("drive forward one meter", False)
    assert heard_command("eddy [unk] drive forward") == ("drive forward", True)        # "don't" was not caught
    assert heard_command("[unk] eddy turn left") == ("turn left", False)               # chatter before the name is fine
    assert heard_command("drive forward one meter") == (None, False)                   # not addressed to Eddy
    assert heard_command("eddy [unk] stop [unk]") == ("stop", False)                   # stop is always safe
    assert heard_command("eddy") == (None, False) and heard_command("eddy [unk]") == (None, False)
    assert heard_command("go forward", None) == ("go forward", False)


def test_spoken_negations_number_words_and_run_together_commands():
    for s in ("don't drive forward", "go forward or turn left", "go forward five", "turn left go forward one meter"):
        assert intents.parse_all(s) == [], s
    for w in ("don't", "not", "never", "or", "and", "cancel"):
        assert w in intents.VOCABULARY.split()


# ---- link safety -------------------------------------------------------------------

def test_served_robot_stops_when_commands_stop_arriving(served):
    """The laptop goes quiet mid-drive (crash, Wi-Fi gone) with the connection still open:
    the served robot stops by itself after the watchdog time."""
    import time
    sim, server = served
    robot = LineAdapter(f"tcp:127.0.0.1:{server.port}")
    model = robot.connect()
    assert "stops by itself" in model.details["link loss"]
    robot.set_velocity(0.2, 0.0)
    assert sim.v == 0.2
    time.sleep(server.watchdog + 0.4)
    assert sim.v == 0.0
    robot.close()


def test_served_robot_keeps_driving_while_commands_keep_arriving(served):
    """The watchdog must not cut a normal drive short: velocity re-sent every tick keeps it going."""
    import time
    sim, server = served
    robot = LineAdapter(f"tcp:127.0.0.1:{server.port}")
    robot.connect()
    end = time.monotonic() + 3 * server.watchdog
    while time.monotonic() < end:
        robot.set_velocity(0.2, 0.0)
        time.sleep(0.05)
    assert sim.v == 0.2 and sim.x > 0.2
    robot.close()


def test_served_robot_stops_when_the_client_disconnects():
    """The connection itself drops mid-drive: the robot is stopped at once, not after the watchdog."""
    import time
    sim = SimAdapter(name="wire-bot").go_live()
    server = LineServer(sim, host="127.0.0.1", watchdog=30.0).start()   # so only the disconnect can stop it
    try:
        robot = LineAdapter(f"tcp:127.0.0.1:{server.port}")
        robot.connect()
        robot.set_velocity(0.2, 0.0)
        robot.io.close()            # the cable is pulled: no stop command is sent
        robot.io = None
        time.sleep(0.3)
        assert sim.v == 0.0
    finally:
        server.stop()


@pytest.fixture
def locked():
    """A served robot that only takes commands from clients holding the key."""
    sim = SimAdapter(name="locked-bot").go_live()
    server = LineServer(sim, host="127.0.0.1", key="s3cret").start()
    yield sim, server
    server.stop()


def test_served_robot_with_a_key_takes_the_right_key(locked):
    sim, server = locked
    robot = LineAdapter(f"tcp:127.0.0.1:{server.port}", key="s3cret")
    assert robot.connect().name == "locked-bot"
    robot.set_velocity(0.2, 0.0)
    assert sim.v == 0.2
    robot.close()


def test_served_robot_with_a_key_refuses_no_key_and_the_wrong_key(locked):
    sim, server = locked
    for key in (None, "wrong"):
        robot = LineAdapter(f"tcp:127.0.0.1:{server.port}", key=key)
        with pytest.raises(RuntimeError, match="key"):
            robot.connect()
        robot.close()
    assert sim.v == 0.0


def test_served_robot_with_a_key_ignores_commands_sent_without_logging_in(locked):
    """A program that skips the handshake and just sends commands gets nowhere, and learns nothing."""
    import socket
    sim, server = locked
    with socket.create_connection(("127.0.0.1", server.port), timeout=2) as sock:
        f = sock.makefile("rwb")
        for line in ("V 0.2 0", "J elbow=0.5", "G 1", "?", "AUTH 00", "V 0.2 0"):
            f.write((line + "\n").encode())
            f.flush()
            assert f.readline().decode().startswith("ERR locked"), line
        f.write(b"EDDY?\n")
        f.flush()
        hello = f.readline().decode()
        assert "auth" in hello and "locked-bot" not in hello and "joints" not in hello
    assert sim.v == 0.0 and sim.targets["elbow"] == 0.0


def test_a_login_answer_cannot_be_replayed_on_another_connection(locked):
    """The server asks a new question on every connection, so a recorded answer is useless later."""
    import hashlib, hmac, json, socket
    sim, server = locked

    def session():
        sock = socket.create_connection(("127.0.0.1", server.port), timeout=2)
        f = sock.makefile("rwb")
        f.write(b"EDDY?\n")
        f.flush()
        return sock, f, json.loads(f.readline())["auth"]

    s1, f1, nonce1 = session()
    answer = hmac.new(b"s3cret", nonce1.encode(), hashlib.sha256).hexdigest()
    s2, f2, nonce2 = session()
    assert nonce1 != nonce2
    f2.write(f"AUTH {answer}\n".encode())
    f2.flush()
    assert f2.readline().decode().startswith("ERR locked")
    f1.write(f"AUTH {answer}\n".encode())
    f1.flush()
    assert f1.readline().decode().strip() == "OK"
    s1.close()
    s2.close()


def test_a_peer_command_meant_for_one_robot_is_refused_by_another():
    """A signed command recorded on its way to one robot must not work on a second robot
    that shares the key."""
    import json
    from eddy.link import PeerNode
    a = PeerNode("alpha", key="k", host="127.0.0.1")
    b = PeerNode("bravo", key="k", host="127.0.0.1")
    c = PeerNode("charlie", key="k", host="127.0.0.1")
    try:
        for node in (b, c):
            node.on("tell", lambda sender, body: "moved")
        line = (json.dumps(a._sign({"type": "tell", "body": "drive forward 1 meter"}, to="bravo")) + "\n").encode()
        refused = c._receive(line)
        assert not refused["ok"] and "bravo" in refused["error"]
        assert b._receive(line) == {"ok": True, "body": "moved"}
        # A signed command that names nobody is refused too.
        nameless = (json.dumps(a._sign({"type": "tell", "body": "x"})) + "\n").encode()
        assert not b._receive(nameless)["ok"]
    finally:
        for node in (a, b, c):
            node._server.server_close()


# ---- robots working together -------------------------------------------------------

def _peer_robot(name, made, start=None, obey=True, realtime=False, hold=10.0, room=2.0):
    """A simulated robot with its own Eddy and a peer node on this machine (no discovery)."""
    from eddy.brain import Safety
    from eddy.link import PeerNode, attach
    sim = SimAdapter(name=name, realtime=realtime, room=room)
    eddy = Eddy(sim, Speaker("silent"), Safety(peer_hold=hold))
    eddy.start()
    node = PeerNode(name, key="k", host="127.0.0.1").start(discovery=False)
    attach(node, eddy, obey=obey, start=start)
    made.append(node)
    return sim, eddy, node


@pytest.fixture
def made():
    nodes = []
    yield nodes
    for node in nodes:
        node.stop()


def test_a_peer_can_be_added_by_address_alone(made):
    """Where discovery is blocked, an address is enough: the robot is asked its name."""
    _, _, a = _peer_robot("alpha", made)
    sim_b, _, b = _peer_robot("bravo", made)
    a.add_address("127.0.0.1", b.port)
    assert a.wait_for("bravo", 5)
    assert "Done" in a.request("bravo", "drive forward 20 cm") and abs(sim_b.x - 0.2) < 0.03


def test_a_peer_added_by_address_is_found_once_it_starts(made):
    """The other robot is switched on later: the address is tried again until it answers."""
    import socket
    _, _, a = _peer_robot("alpha", made)
    with socket.socket() as s:              # find a free port for the robot that is not up yet
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    a.add_address("127.0.0.1", port)
    assert not a.wait_for("late", 1)
    from eddy.link import PeerNode
    late = PeerNode("late", key="k", host="127.0.0.1", port=port).start(discovery=False)
    made.append(late)
    assert a.wait_for("late", 6)


def test_a_peer_cannot_take_over_a_move_its_operator_asked_for(made):
    """The robot's own operator comes first: while their command runs, a peer is turned away."""
    import threading, time
    _, _, a = _peer_robot("alpha", made)
    sim_b, eddy_b, b = _peer_robot("bravo", made, realtime=True)
    a.add_peer("bravo", "127.0.0.1", b.port)
    own = threading.Thread(target=lambda: eddy_b.tell("drive forward 60 cm"))
    own.start()
    time.sleep(0.5)
    answer = a.request("bravo", "turn left 90 degrees")
    own.join(timeout=10)
    assert "own operator" in answer
    assert abs(sim_b.x - 0.6) < 0.04 and abs(sim_b.heading) < 0.05      # the operator's drive finished, untouched
    assert "Done" in a.request("bravo", "drive forward 10 cm")          # and afterwards the peer is welcome


def test_a_peer_can_still_stop_a_move_the_operator_asked_for(made):
    """Stopping is always allowed, whoever started the move."""
    import threading, time
    _, _, a = _peer_robot("alpha", made)
    sim_b, eddy_b, b = _peer_robot("bravo", made, realtime=True)
    a.add_peer("bravo", "127.0.0.1", b.port)
    own = threading.Thread(target=lambda: eddy_b.tell("drive forward 1.5 meters"))
    own.start()
    time.sleep(0.5)
    assert "Stopped" in a.send("bravo", "stop")["message"]
    own.join(timeout=5)
    assert sim_b.x < 0.5 and sim_b.v == 0


def test_a_peer_cannot_restart_a_robot_its_operator_just_stopped(made):
    import time
    _, _, a = _peer_robot("alpha", made)
    sim_b, eddy_b, b = _peer_robot("bravo", made, hold=1.0)
    a.add_peer("bravo", "127.0.0.1", b.port)
    eddy_b.tell("stop")
    assert "stopped it" in a.request("bravo", "drive forward 20 cm") and sim_b.x == 0.0
    time.sleep(1.1)
    assert "Done" in a.request("bravo", "drive forward 20 cm")
    # A stop that came from a peer does not lock the peers out.
    a.send("bravo", "stop")
    assert "Done" in a.request("bravo", "drive forward 10 cm")


def test_robots_started_from_a_shared_mark_report_positions_in_one_frame(made):
    """Alpha starts on the mark facing along it; bravo starts 2 m further on, facing back.
    After each drives, both positions are given from the mark, so the gap between them is right."""
    from eddy.link import roll_call
    sim_a, eddy_a, a = _peer_robot("alpha", made, start=(0.0, 0.0, 0.0))
    sim_b, eddy_b, b = _peer_robot("bravo", made, start=(2.0, 0.0, 180.0))
    a.add_peer("bravo", "127.0.0.1", b.port)
    eddy_a.tell("drive forward 50 cm")
    eddy_b.tell("drive forward 50 cm")
    there = a.send("bravo", "state")
    assert there["frame"] == "shared" and abs(there["x"] - 1.5) < 0.03 and abs(there["y"]) < 0.03
    assert abs(abs(there["heading"]) - math.pi) < 0.05
    line = roll_call(a)[0]              # "bravo: x 1.50, y 0.00 from the shared mark, 1.00 m from me"
    assert line.endswith("m from me") and abs(float(line.split(", ")[-1].split()[0]) - 1.0) < 0.03
    # A sideways move too: bravo turns left (now facing -y in the shared frame) and drives.
    eddy_b.tell("turn left 90 degrees")
    eddy_b.tell("drive forward 40 cm")
    there = a.send("bravo", "state")
    assert abs(there["x"] - 1.5) < 0.05 and abs(there["y"] + 0.4) < 0.05


def test_without_a_shared_mark_positions_are_marked_as_the_robots_own(made):
    from eddy.link import roll_call
    _, _, a = _peer_robot("alpha", made)
    _, _, b = _peer_robot("bravo", made)
    a.add_peer("bravo", "127.0.0.1", b.port)
    assert a.send("bravo", "state")["frame"] == "own"
    assert "own frame" in roll_call(a)[0]


def test_an_obstacle_is_reported_where_it_is_in_the_shared_frame(made):
    """Bravo starts 5 m along and 5 m across from the mark and drives into its wall, which is
    2 m ahead of it: the others are told about something near (7, 5), not near (2, 0)."""
    import time
    _, _, a = _peer_robot("alpha", made, start=(0.0, 0.0, 0.0))
    sim_b, eddy_b, b = _peer_robot("bravo", made, start=(5.0, 5.0, 0.0))
    b.add_peer("alpha", "127.0.0.1", a.port)
    eddy_b.tell("drive forward 5 meters")
    deadline = time.time() + 3
    while time.time() < deadline and not a.inbox:
        time.sleep(0.05)
    sender, kind, body = a.inbox[0]
    assert (sender, kind, body["what"], body["frame"]) == ("bravo", "event", "obstacle", "shared")
    assert abs(body["x"] - 7.0) < 0.3 and abs(body["y"] - 5.0) < 0.05


# ---- run log -----------------------------------------------------------------------

def test_run_log_records_every_command_with_the_state_before_and_after(tmp_path):
    from eddy import runlog
    log = runlog.RunLog.open_for("log bot/1", str(tmp_path))
    eddy = Eddy(SimAdapter(name="log-bot"), Speaker("silent"), log=log)
    eddy.start()
    eddy.tell("drive forward 50 cm")
    eddy.tell("make me a sandwich", source="voice")
    eddy.tell("turn left 90 degrees and reach 9 9 9")
    eddy.stop()
    eddy.close()
    assert runlog.latest(str(tmp_path)) == log.path and "/" not in log.path[len(str(tmp_path)) + 1:]
    entries = runlog.read(log.path)
    assert entries[0]["event"] == "start" and entries[0]["robot"] == "log-bot" and "drive" in entries[0]["capabilities"]
    drive, unknown, turn, reach = entries[1:5]
    assert drive["said"] == "drive forward 50 cm" and drive["command"] == "Drive" and drive["ok"]
    assert drive["before"]["x"] == 0.0 and abs(drive["after"]["x"] - 0.5) < 0.03 and drive["args"]["distance"] == 0.5
    assert unknown["source"] == "voice" and unknown["command"] is None and not unknown["ok"]
    assert turn["command"] == "Drive" and turn["ok"] and reach["command"] == "MoveTo" and not reach["ok"]
    assert "beyond my reach" in reach["message"]
    assert [e["command"] for e in entries[5:]] == ["Stop", "Stop"] and entries[5]["before"] is None


def test_a_run_log_that_cannot_be_written_does_not_stop_the_robot(tmp_path):
    from eddy import runlog
    assert runlog.RunLog.open_for("x", str(tmp_path / "file" / "sub")) is not None
    blocker = tmp_path / "blocked"
    blocker.write_text("a file where the folder should be")
    assert runlog.RunLog.open_for("x", str(blocker)) is None        # says so, returns nothing
    log = runlog.RunLog.open_for("x", str(tmp_path))
    log._file.close()                                               # the disk "fails" under it
    sim = SimAdapter()
    eddy = Eddy(sim, Speaker("silent"), log=log)
    assert eddy.tell("drive forward 20 cm").ok and abs(sim.x - 0.2) < 0.03


# ---- web page ----------------------------------------------------------------------

JPEG = b"\xff\xd8\xff\xe0" + b"pretend picture" + b"\xff\xd9"


@pytest.fixture
def page():
    """A live simulated robot with a (pretend) camera, and its web page on a local port."""
    from eddy.web import WebPage
    sim = SimAdapter(name="web-bot").go_live()
    sim.camera_frame = lambda: JPEG
    eddy = Eddy(sim, Speaker("silent"))
    eddy.start()
    web = WebPage(eddy, host="127.0.0.1", port=0, code="c0de").start()
    yield sim, eddy, web
    web.stop()
    sim.close()


def _http(web, path, body=None, code="c0de"):
    """GET (or POST when a body is given) a path on the page; returns (status, bytes)."""
    import urllib.error, urllib.request
    url = f"http://127.0.0.1:{web.port}{path}" + (f"?t={code}" if code is not None else "")
    try:
        with urllib.request.urlopen(urllib.request.Request(url, data=body), timeout=5) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def test_web_page_shows_the_robot_and_drives_it(page):
    import json, time
    sim, eddy, web = page
    status, html = _http(web, "/", code=None)
    assert status == 200 and b"STOP" in html and b"c0de" not in html      # the page itself gives nothing away
    state = json.loads(_http(web, "/state")[1])
    assert state["name"] == "web-bot" and "drive" in state["capabilities"] and state["camera"] and state["x"] == 0.0
    assert _http(web, "/tell", b"drive forward 20 cm")[0] == 202          # answered at once, carried out behind
    deadline = time.time() + 6
    while time.time() < deadline and not json.loads(_http(web, "/state")[1])["results"]:
        time.sleep(0.1)
    state = json.loads(_http(web, "/state")[1])
    assert state["results"][-1]["ok"] and "moved 0.20 m" in state["results"][-1]["message"]
    assert abs(sim.x - 0.2) < 0.03 and abs(state["x"] - 0.2) < 0.03


def test_web_stop_button_stops_a_drive_in_progress(page):
    import json, time
    sim, eddy, web = page
    _http(web, "/tell", b"drive forward 1.5 meters")
    time.sleep(0.6)
    assert sim.v > 0 and json.loads(_http(web, "/state")[1])["busy"]
    status, body = _http(web, "/stop", b"")
    assert status == 200 and "Stopped" in json.loads(body)["message"]
    time.sleep(0.3)
    assert sim.v == 0 and sim.x < 0.5


def test_web_page_does_nothing_without_the_code(page):
    sim, eddy, web = page
    for code in (None, "", "wrong", "c0d"):
        assert _http(web, "/state", code=code)[0] == 401
        assert _http(web, "/camera.mjpg", code=code)[0] == 401
        assert _http(web, "/tell", b"drive forward 1 meter", code=code)[0] == 401
        assert _http(web, "/stop", b"", code=code)[0] == 401
    assert sim.v == 0 and sim.x == 0
    assert _http(web, "/tell", b"x" * 501)[0] == 413 and _http(web, "/tell", b"")[0] == 400
    assert _http(web, "/nowhere")[0] == 404


def test_web_page_streams_the_camera(page):
    import urllib.request
    sim, eddy, web = page
    with urllib.request.urlopen(f"http://127.0.0.1:{web.port}/camera.mjpg?t=c0de", timeout=5) as r:
        assert "multipart/x-mixed-replace" in r.headers["Content-Type"]
        chunk = r.read(len(JPEG) + 80)
    assert b"Content-Type: image/jpeg" in chunk and JPEG in chunk


def test_web_page_says_why_there_is_no_picture():
    import json
    from eddy.web import WebPage
    eddy = Eddy(SimAdapter(), Speaker("silent"))
    eddy.start()
    web = WebPage(eddy, host="127.0.0.1", port=0).start()
    try:
        assert len(web.code) >= 6 and web.links()[0].endswith(f":{web.port}/?t={web.code}")
        state = json.loads(_http(web, "/state", code=web.code)[1])
        assert state["camera"] is False and "no camera" in state["camera_note"]
    finally:
        web.stop()


def test_camera_feeds_are_tried_in_a_sensible_order():
    """Compressed colour first (already JPEG), raw colour next, depth last."""
    img, comp = "sensor_msgs/msg/Image", "sensor_msgs/msg/CompressedImage"
    profile = classify_graph([("/cam/depth/image_raw", [img]), ("/cam/rgb/image_raw", [img]),
                              ("/cam/rgb/image_raw/compressed", [comp]), ("/cam/depth/image_raw/compressedDepth", [comp])])
    assert [n for n, _ in profile["camera_feeds"]] == ["/cam/rgb/image_raw/compressed", "/cam/rgb/image_raw",
                                                        "/cam/depth/image_raw/compressedDepth", "/cam/depth/image_raw"]


def test_an_uncompressed_camera_picture_is_turned_into_jpeg():
    """A raw picture (as most camera drivers publish) becomes a JPEG a browser can show,
    with its colours the right way round."""
    cv2 = pytest.importorskip("cv2")
    from types import SimpleNamespace
    from eddy.adapters.ros2 import Ros2Adapter
    adapter = Ros2Adapter()
    red = bytes([255, 0, 0]) * 16 * 16       # 16 x 16, every pixel red, in rgb8
    adapter._on_image(SimpleNamespace(height=16, width=16, step=48, encoding="rgb8", data=red), "/cam")
    assert adapter._jpeg[:2] == b"\xff\xd8"
    blue, green, red_ = cv2.imdecode(np.frombuffer(adapter._jpeg, np.uint8), cv2.IMREAD_COLOR)[8, 8]
    assert red_ > 200 and blue < 60 and green < 60
    # A format Eddy does not convert is explained, not shown wrongly.
    other = Ros2Adapter()
    other._on_image(SimpleNamespace(height=2, width=2, step=4, encoding="16UC1", data=bytes(8)), "/depth")
    assert other._jpeg is None and "16UC1" in other.camera_note
    # A compressed picture that is not JPEG is explained too.
    other._on_compressed(SimpleNamespace(format="png", data=b"\x89PNG...."), "/cam/compressed")
    assert other._jpeg is None and "png" in other.camera_note


# ---- sideways driving --------------------------------------------------------------

def test_sideways_sentences():
    from eddy.model import MoveBy, Slide
    assert intents.parse("slide left 20 cm") == Slide(0.2)
    assert intents.parse("strafe right 30 cm") == Slide(-0.3)
    assert intents.parse("Eddy, sidestep left") == Slide(0.2)
    assert intents.parse("move sideways to the right half a meter") == Slide(-0.5)
    assert intents.parse("slide left slowly 10 cm") == Slide(0.1, 0.075)
    for unclear in ("slide 20 cm", "slide left 20", "don't slide left", "slide left -20 cm", "slide left or right"):
        assert intents.parse(unclear) is None, unclear
    assert isinstance(intents.parse("slide the arm left 5 cm"), MoveBy)         # the arm, not the base
    assert [type(c).__name__ for c in intents.parse_all("slide left 20 cm and drive forward 30 cm")] == ["Slide", "Drive"]


def _eddy_on(sim):
    eddy = Eddy(sim, Speaker("silent"))
    eddy.start()
    return eddy


def test_a_base_that_can_move_sideways_slides_the_distance_asked():
    sim = SimAdapter(slides=True)
    eddy = _eddy_on(sim)
    assert eddy.model.drive.slides is None                                      # not known until tried
    r = eddy.tell("slide left 30 cm")
    assert r.ok and "slid 0.30 m to the left" in r.message
    assert abs(sim.y - 0.3) < 0.01 and abs(sim.x) < 0.005 and sim.heading == 0.0
    assert eddy.model.drive.slides is True
    r = eddy.tell("slide right 50 cm")
    assert r.ok and abs(sim.y + 0.2) < 0.01 and abs(sim.x) < 0.005
    # Sideways is relative to the way the robot faces: after a left turn, "left" is back along -x.
    eddy.tell("turn left 90 degrees")
    r = eddy.tell("slide left 20 cm")
    assert r.ok and abs(sim.x + 0.2) < 0.02 and abs(sim.y + 0.2) < 0.02


def test_a_base_that_cannot_move_sideways_is_found_out_by_trying():
    sim = SimAdapter()                                                          # ordinary two-wheel base
    eddy = _eddy_on(sim)
    r = eddy.tell("slide left 20 cm")
    assert not r.ok and "does not seem to be able to move sideways" in r.message
    assert (sim.x, sim.y, sim.v, sim.w) == (0.0, 0.0, 0.0, 0.0) and eddy.model.drive.slides is False
    before = sim.clock
    r = eddy.tell("slide right 20 cm")                                          # now known: refused at once
    assert not r.ok and "tried earlier" in r.message and sim.clock == before
    assert eddy.tell("drive forward 20 cm").ok                                  # ordinary driving is unaffected


def test_sliding_is_refused_where_it_cannot_be_done_safely():
    eddy = _eddy_on(SimAdapter(slides=True))
    r = eddy.tell("slide left 2 meters")
    assert not r.ok and "at most 1 m" in r.message
    car = SimAdapter(car=True)
    eddy = _eddy_on(car)
    eddy.tell("turn left 45 degrees")                                           # finds out it steers like a car
    r = eddy.tell("slide left 20 cm")
    assert not r.ok and "steers like a car" in r.message
    assert not _eddy_on(SimAdapter(drive=False)).tell("slide left 20 cm").ok


def test_sliding_into_a_wall_stops_and_says_so():
    sim = SimAdapter(slides=True, room=0.5)                                     # walls 0.35 m to each side of the robot
    eddy = _eddy_on(sim)
    r = eddy.tell("slide left 80 cm")
    assert not r.ok and "blocking it on that side" in r.message and sim.vy == 0.0
    assert eddy.model.drive.slides is True                                      # it can slide; it was blocked
    r = eddy.tell("slide left 20 cm")
    assert not r.ok and "blocking" in r.message
    assert eddy.tell("slide right 20 cm").ok


def test_stop_interrupts_a_slide():
    import threading, time
    sim = SimAdapter(slides=True, realtime=True)
    eddy = _eddy_on(sim)
    out = []
    worker = threading.Thread(target=lambda: out.append(eddy.tell("slide left 90 cm")))
    worker.start()
    time.sleep(0.6)
    eddy.tell("stop")
    worker.join(timeout=3)
    assert not worker.is_alive() and "Interrupted" in out[0].message and 0.01 < sim.y < 0.4 and sim.vy == 0.0


def test_a_robot_reached_over_the_line_protocol_cannot_slide(served):
    sim, server = served
    eddy = Eddy(LineAdapter(f"tcp:127.0.0.1:{server.port}"), Speaker("silent"))
    eddy.start()
    r = eddy.tell("slide left 20 cm")
    assert not r.ok and "no sideways command" in r.message and sim.y == 0.0
    eddy.close()


def test_half_a_meter_is_half_a_meter():
    """ "half a meter" used to be read as "a meter": twice the distance asked for."""
    assert intents.parse("drive forward half a meter").distance == 0.5
    assert intents.parse("go back a quarter of a meter").distance == 0.25
    assert intents.parse("turn left for half a second").duration == 0.5
    assert intents.parse("drive forward a meter").distance == 1.0
    # "one and a half meters" used to be read as half a metre.
    assert [c.distance for c in intents.parse_all("drive forward one and a half meters")] == [1.5]
    assert [c.distance for c in intents.parse_all("go back two and a half meters")] == [2.5]
    assert intents.parse("drive forward half meter").distance == 0.5
