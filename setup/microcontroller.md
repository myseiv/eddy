# Setting Eddy up on a microcontroller robot (Arduino, ESP32, Pico)

For robots whose motors hang off a small board connected by USB. The board runs a short
firmware that answers Eddy's line protocol, seven one-line commands, and Eddy does the rest:
understanding commands, distances and turns, safety checks, voice and talking to other robots.

> There is no tested reference firmware yet. `firmware/eddy_rover/eddy_rover.ino` is a
> starting point for a two-wheel robot. It follows the protocol below but has **not been run
> on a real board**. Check it with the wheels off the ground.

## 1. Install Eddy with serial support

On the computer the board plugs into (a Raspberry Pi or a laptop):

```bash
./scripts/install.sh --serial
```

## 2. Put firmware on the board

1. Open `setup/firmware/eddy_rover/eddy_rover.ino` in the Arduino IDE.
2. Set the motor pins and the robot's numbers (wheel spacing, top speed) at the top of the file
   to match your robot.
3. Select your board and port, then press Upload.

Writing your own firmware instead? Follow "The Eddy line protocol" below.

## 3. Find the board's port

Linux or Raspberry Pi:

```bash
ls /dev/ttyUSB* /dev/ttyACM*
```

macOS:

```bash
ls /dev/cu.usb*
```

Or let Eddy look:

```bash
eddy scan
```

On Linux, if the port is refused, add yourself to the `dialout` group, then log out and back in:

```bash
sudo usermod -aG dialout $USER
```

## 4. Check the board answers (optional)

pyserial's terminal comes with the serial install. Close the Arduino serial monitor first:

```bash
~/.eddy/venv/bin/python -m serial.tools.miniterm /dev/ttyUSB0 115200 --eol LF
```

Type `EDDY?` and press Enter: the board should answer with one line of JSON. Type `?` for its
state, then press Ctrl+] to leave.

## 5. Let Eddy look at the robot

```bash
eddy info -r serial:/dev/ttyUSB0
```

At a baud rate other than 115200:

```bash
eddy info -r serial:/dev/ttyUSB0@57600
```

## 6. First motion: wheels off the ground

```bash
eddy do -r serial:/dev/ttyUSB0 --speed 0.3 "drive forward 10 cm" "status"
```

Then pull the USB cable while the wheels turn: the motors must stop within the watchdog time
(0.5 s). If they do not, fix the firmware before going further. Nothing on the computer can
stop the motors once the cable is out.

## 7. Drive it

```bash
eddy shell -r serial:/dev/ttyUSB0
```

To drive it from another computer, serve it on the network ([tcp.md](tcp.md)):

```bash
eddy serve -r serial:/dev/ttyUSB0 --port 9000 --key <your secret>
```

## The Eddy line protocol

Plain text, one line per message, ending in `\n` (`\r\n` is fine). 115200 baud by default.
Eddy sends a line and waits for a one-line answer: `OK`, a line starting `ERR`, or a JSON
object. Anything else the board prints (a boot banner, debug text) is skipped.

| Eddy sends | Board answers | Meaning |
|---|---|---|
| `EDDY?` | JSON description (below) | What are you? |
| `V 0.2 0.0` | `OK` | Drive: forward speed in m/s, turn rate in rad/s (left is positive). |
| `J pan=0.5 tilt=-0.2 T1.0` | `OK` | Move joints to these angles (radians), taking T seconds. |
| `G 0.5` | `OK` | Gripper: 0 closed, 1 open. |
| `S` | `OK` | Stop everything now. |
| `?` | JSON state (below) | Where are you, what do you see? |

Description, answered to `EDDY?` (one line):

```json
{"eddy":1,"name":"rover","drive":{"max_linear":0.4,"max_angular":2.0},"joints":[{"name":"pan","lower":-1.5,"upper":1.5}],"gripper":false,"sensors":["odometry","range"],"watchdog":0.5}
```

- Leave out `drive` if the robot cannot drive. Use `"joints":[]` if it has none.
- `sensors`: include `"odometry"` only if the board tracks its position (wheel encoders), and
  `"range"` only if it has a distance sensor pointing forward.
- `watchdog`: seconds after the last `V` after which the board stops the motors by itself.
  **The firmware must do this.**

State, answered to `?` (one line):

```json
{"x":0,"y":0,"th":0,"joints":{"pan":0.5},"range":1.2,"gripper":0.5}
```

`x` and `y` are metres and `th` is the heading in radians, from where the board started. `range`
is metres to the nearest thing ahead. Leave out what the board does not have.

For a command the robot cannot do, answer `ERR unsupported <reason>` (for example `J` on a
robot with no joints). Eddy then tells the user the robot lacks that ability. Any other `ERR`
counts as a failure.

Without odometry, Eddy still drives: it works out how long to drive at a known speed, and says
it could not measure the result. Distances are rougher.

## If something goes wrong

| What you see | What to try |
|---|---|
| "no answer ... to 'EDDY?'" | Wrong port or baud rate; close the Arduino serial monitor; check step 4. |
| "not the Eddy protocol" | The board answers, but not with OK, ERR or JSON. Check the firmware's replies. |
| Permission denied on the port | The `dialout` step in 3. |
| Drives too far or too short | Set the wheel numbers in the firmware; add encoders and report odometry. |
