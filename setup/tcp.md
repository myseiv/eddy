# Setting Eddy up over the network (TCP)

Two uses:

- **Drive a robot from another computer.** Eddy runs on the robot and puts it on the network
  with `eddy serve`; you drive it from your laptop with `-r tcp:<robot>:<port>`.
- **A robot that speaks the Eddy line protocol on a TCP port itself**, for example an ESP32
  on Wi-Fi or a program you wrote. The protocol is in
  [microcontroller.md](microcontroller.md#the-eddy-line-protocol).

Both computers need Eddy installed ([README.md](README.md) step 1) and must be on the same
network.

## Drive a robot from another computer

### 1. On the robot: find its address

```bash
hostname -I
```

Note the first address, for example `192.168.1.20`.

### 2. On the robot: serve it with a key

Pick a secret key, so that only someone who knows it can drive the robot. For a ROS 2 robot
(source ROS 2 first, as in [ros2.md](ros2.md)):

```bash
eddy serve -r ros2 --port 9000 --key <your secret>
```

For a microcontroller robot on USB:

```bash
eddy serve -r serial:/dev/ttyUSB0 --port 9000 --key <your secret>
```

To try it without a robot:

```bash
eddy serve -r sim --port 9000 --key <your secret>
```

Leave this running. The served robot stops by itself 0.5 s after drive commands stop arriving,
and at once if your connection drops (change the 0.5 s with `--watchdog`).

### 3. On your laptop: check you can reach it

```bash
eddy info -r tcp:192.168.1.20:9000 --key <your secret>
```

It should print what the robot can do. With a wrong or missing key it is refused.

### 4. On your laptop: drive it

```bash
eddy shell -r tcp:192.168.1.20:9000 --key <your secret>
```

Or one command at a time:

```bash
eddy do -r tcp:192.168.1.20:9000 --key <your secret> "drive forward 50 cm" "status"
```

## Several robots talking to each other

Each robot runs `eddy run` with `--peers` and the same key. On robot A:

```bash
eddy run -r ros2 --peers --name alpha --key <shared secret> --obey
```

On robot B:

```bash
eddy run -r ros2 --peers --name bravo --key <shared secret> --obey
```

Then, in either prompt: `peers` lists the others, `@bravo turn left` sends a command to one,
`@all stop` stops them all. Without `--obey` a robot answers questions and honours stop, but
will not move for another robot.

If the robots do not find each other (many Wi-Fi networks block the multicast Eddy uses), give
the address directly:

```bash
eddy run -r ros2 --peers --name alpha --peer 192.168.1.21 --key <shared secret> --obey
```

To share positions, start every robot at a measured spot from one floor mark (x and y in
metres, heading in degrees):

```bash
eddy run -r ros2 --peers --name alpha --start 0,0,0 --key <shared secret> --obey
```

## If something goes wrong

| What you see | What to try |
|---|---|
| Connection refused or times out | Check the address with `hostname -I`, that `eddy serve` is still running, and that a firewall is not blocking the port. |
| "only takes commands from someone holding its key" | Pass the same `--key` the robot was served with. |
| The robot stops in the middle of a drive | The link is slower than the watchdog; serve with `--watchdog 1.0`. |
| Peers not found | Use `--peer <address>` as above. |

## Limits

The key is checked once per connection. The lines after that are not encrypted: it keeps
strangers on the network out, not someone who can intercept the traffic. Use it on a network
you trust.
