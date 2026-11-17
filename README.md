# Eddy AI

A plug-and-play controller that sits on top of a robot's existing code and controls it.

Eddy does not replace what already runs on a robot. It attaches to it, works out what the
robot can do, and then accepts the same commands on every robot: typed, spoken, shown as a
hand sign, sent by a program, or sent by another robot.

```
  typed text   speech   hand signs   another robot   your program
        \         |          |            |              /
         +--------+----------+------------+-------------+
         |                    Eddy                      |   sentences -> commands
         |   safety limits · inverse kinematics · stop  |   commands  -> motion
         +----------------------+-----------------------+
                                |  one adapter interface
        +-----------+-----------+-----------+------------+
        |  ros2     |  serial:  |   tcp:    |    sim     |
        | existing  | Arduino / | networked | built-in   |
        | ROS nodes | ESP32     | robot     | simulator  |
        +-----------+-----------+-----------+------------+
```

## Try it (no robot needed)

```bash
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
```

```bash
.venv/bin/eddy selftest
```

```bash
.venv/bin/eddy shell -r sim
```

Then type things like `drive forward 1 meter`, `turn left 90 degrees`, `slide left 20 cm` (bases that can move sideways; try it with `-r sim:mecanum`), `reach 0.2 0.1 0.3`,
`close the gripper`, `wander for 30 seconds`, `status`, `stop`.

## Put it on a Raspberry Pi

From this folder on your computer (the Pi only needs SSH enabled):

```bash
./scripts/deploy.sh pi@raspberrypi.local --all
```

Or, on the Pi itself, once the code is on GitHub:

```bash
git clone <your-repo-url> eddy_ai && cd eddy_ai && ./scripts/install.sh --all
```

`--all` adds serial robots, voice and vision; leave it off for the core only. The installer
ends by running `eddy doctor` and `eddy selftest` on the Pi. To start Eddy at boot:

```bash
./scripts/install.sh --service "run -r ros2 --speak --peers --key my-secret"
```

## Attach it to a robot

| Robot | Command | What Eddy needs from it |
|---|---|---|
| Anything running ROS 2 | `eddy info -r ros2` | Nothing new. Eddy reads the graph (all topics and nodes) and finds the velocity topic, odometry, laser, joint states, trajectory controllers and URDF by itself. |
| Arduino / ESP32 / Pico on USB | `eddy info -r serial:/dev/ttyUSB0` | Firmware that answers the Eddy line protocol (7 one-line commands, documented in `src/eddy/adapters/line.py`). |
| A robot on the network | `eddy info -r tcp:192.168.1.20:9000` | The same protocol on a TCP port. `eddy serve` provides it for any robot Eddy can already reach. Serve with `--key <secret>` and connect with the same `--key`, so only you can drive it. A served robot stops by itself if the link drops. |
| No robot | `eddy info -r sim` | Nothing. |

`eddy scan` lists what it can find. `eddy doctor` says what is installed and what is missing.

## Control it from a web page

```bash
eddy web -r ros2
```

Eddy prints a link. Open it on a laptop or phone on the same network: the robot's camera, where
it is, a box to type commands in, and a big stop button. Nothing extra to install. The link
carries a short code that changes every start, so only someone who saw the robot's terminal can
drive it. A camera that publishes compressed pictures needs nothing; an uncompressed one needs
OpenCV on the robot (the page says so if that is the case).

## See what happened

Every `eddy do`, `shell`, `run` and `web` session is logged, one line per command, with the
robot's state before and after:

```bash
eddy log
```

The files are in `~/.eddy/logs` (JSON lines). `--no-log` switches it off.

## Everything on

```bash
eddy run -r ros2 --speak --voice --gestures --peers --key my-secret --obey
```

- `--voice` listens offline; say "Eddy, drive forward one meter".
- `--gestures` watches the camera: open palm or fist = stop, point = forward, V = back, thumb sideways = turn that way.
- `--peers` finds other Eddy robots on the network. In the prompt, `peers` lists them and where they are, `@name turn left` sends a command to one, and `@all stop` stops them all.
- `--peer 192.168.1.21` adds a robot by address, for networks where robots do not find each other (many Wi-Fi networks).
- `--start 2,0,180` says where this robot is now (x, y in metres and heading in degrees) from a mark on the floor that all the robots use. With it, robots tell each other positions that mean the same thing to all of them.
- The robot's own operator comes first: a peer cannot take over a move the operator asked for, or restart the robot for 10 s after the operator stopped it. A peer can always stop it.
- `--obey` lets peers that hold the same `--key` command this robot. Without it, the robot answers questions and honours `stop`, but will not move for anyone else.

## Safety built in

- Speeds are clamped to the robot's limits (`--speed 0.5` halves everything).
- Eddy will not start, and will stop, a forward drive when its distance sensor sees something closer than `--min-range` (0.25 m).
- `stop` (typed, spoken, signed, or from a peer) interrupts whatever is running.
- Arm targets are checked with inverse kinematics first; unreachable points are refused, joint limits are respected.
- Commands from other robots must be signed with the shared key, are time-limited and cannot be replayed.

Eddy only knows about obstacles straight ahead that the robot's own range sensor reports. It
is not a certified safety system: keep a physical emergency stop on any robot that can hurt someone.

## Tests

```bash
.venv/bin/python -m pytest -q
```

```bash
./tests/ros2/run.sh
```

```bash
./tests/peers/run.sh
```

```bash
./tests/ros2/run_hiwonder.sh
```

```bash
./tests/sims/run.sh
```

```bash
./tests/manip/run.sh
```

All but the first need Docker; run them one at a time (containers see each other's ROS topics).
`tests/manip/run.sh` runs 13 checks for robots that drive and have an arm on three of them,
including Gazebo's stock vehicle carrying a stock Franka Panda. `tests/ros2/run.sh` runs Eddy against a mock robot on real
ROS 2 Jazzy, and `tests/peers/run.sh` runs two robots on separate machines talking to each
other. `run_hiwonder.sh` uses stand-ins for two real robots Eddy has been tried on.
`tests/sims/run.sh` uses public simulators that were not written for Eddy (a Franka Panda arm
on stock ros2_control controllers, Gazebo's stock differential-drive and car-steered
vehicles, and its stock quadcopter) and compares every result with the simulator's true pose; it builds a 4 GB image
the first time and takes about ten minutes.

See [docs/PLAN.md](docs/PLAN.md) for the design, what has been verified, and what comes next.
