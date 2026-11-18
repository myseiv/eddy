# Setting Eddy up

Step-by-step guides for attaching Eddy to a robot. Pick the one that matches how your robot is
controlled:

| Your robot | Guide | Eddy connects with |
|---|---|---|
| Runs ROS 2 (most research robots) | [ros2.md](ros2.md) | `-r ros2` |
| Is reachable over the network, or you want to drive a robot from another computer | [tcp.md](tcp.md) | `-r tcp:<host>:<port>` |
| Has its motors on an Arduino, ESP32 or Pico over USB | [microcontroller.md](microcontroller.md) | `-r serial:<port>` |

Every guide starts from the same install, below.

## 1. Install Eddy

On the computer that will run Eddy (a Raspberry Pi on the robot, or a laptop).

Python 3.9 or newer is needed:

```bash
python3 --version
```

Get the code:

```bash
git clone https://github.com/myseiv/eddy.git
```

```bash
cd eddy
```

Install the core (simulator, ROS 2, network robots):

```bash
./scripts/install.sh
```

Or install everything (adds serial robots, offline voice, camera and hand signs):

```bash
./scripts/install.sh --all
```

The installer puts Eddy in `~/.eddy` and links the `eddy` command into `~/.local/bin`. If
`eddy` is not found afterwards, open a new terminal, or add that folder to your PATH:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

### Installing from a laptop instead

If the robot's computer has SSH enabled, you can copy and install from your laptop in one step:

```bash
./scripts/deploy.sh pi@raspberrypi.local --all
```

## 2. Check the install

```bash
eddy doctor
```

```bash
eddy selftest
```

`doctor` says what is installed and what is missing. `selftest` runs the end-to-end checks on
the built-in simulator; every line should say PASS.

## 3. Try it with no robot

```bash
eddy shell -r sim
```

Type `drive forward 1 meter`, `turn left 90 degrees`, `status`, then `stop`. Type `quit` or
press Ctrl+D to leave.

Then follow the guide for your robot.

## Safety, before any real robot

- Put the robot on blocks (wheels off the ground) for the first commands.
- Keep a physical emergency stop within reach. Eddy is not a certified safety system.
- Start slow: add `--speed 0.3` to any command to scale every speed down.
