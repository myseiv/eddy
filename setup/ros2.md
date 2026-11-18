# Setting Eddy up on a ROS 2 robot

Eddy joins the robot's ROS 2 graph as one extra node. It reads the topics, nodes, controllers
and URDF that are already there and works out how to drive the robot. Nothing on the robot is
changed and no configuration file is needed.

Tested with ROS 2 Humble and Jazzy.

## 1. Install Eddy on a machine that can see the robot's ROS 2 graph

Either the robot's own computer, or a computer on the same network with the same ROS 2
distribution. Follow [README.md](README.md) step 1.

If the robot's ROS 2 runs inside a Docker container (Hiwonder robots do), run Eddy inside that
container. See "Robot in a Docker container" below.

## 2. Load ROS 2 in the terminal

Eddy needs ROS 2's Python library, `rclpy`, so source ROS 2 first. For Humble:

```bash
source /opt/ros/humble/setup.bash
```

For Jazzy:

```bash
source /opt/ros/jazzy/setup.bash
```

If the robot uses a domain ID, use the same one:

```bash
export ROS_DOMAIN_ID=<the robot's domain id>
```

Check that you can see the robot's topics:

```bash
ros2 topic list
```

You should see the robot's velocity topic (often `/cmd_vel`) and, ideally, `/odom` and `/scan`.

## 3. Let Eddy look at the robot (read-only, nothing moves)

```bash
eddy scan
```

```bash
eddy info -r ros2
```

`info` prints what Eddy found: the velocity topic it will use, odometry, laser, joint states,
cameras, arm controllers and their joints, the gripper, and joint limits from the URDF. If
something is missing here, Eddy will refuse the commands that need it.

## 4. First motion: wheels off the ground

Lift the robot so the wheels spin freely, then:

```bash
eddy do -r ros2 --speed 0.3 "drive forward 10 cm" "status"
```

The wheels should turn briefly and stop by themselves.

## 5. Drive it

Put the robot on the floor with space around it:

```bash
eddy shell -r ros2 --speed 0.5
```

Try `drive forward 50 cm`, `turn left 90 degrees`, `back 20 cm`, `status`, `stop`. For a robot
with an arm: `reach 0.2 0.0 0.2`, `open the gripper`, `close the gripper`, `home`.

## 6. Optional extras

Control it from a phone or laptop browser (camera, commands, stop button):

```bash
eddy web -r ros2
```

Voice, hand signs and talking to other robots, all at once:

```bash
eddy run -r ros2 --speak --voice --gestures --peers --key <a shared secret>
```

Start Eddy at boot (sources ROS 2 automatically):

```bash
./scripts/install.sh --service "run -r ros2 --speak"
```

## Robot in a Docker container

Find the container's name:

```bash
docker ps
```

Copy the code into it:

```bash
docker cp ~/eddy_ai <container>:/root/eddy_ai
```

Open a shell inside it:

```bash
docker exec -it <container> bash
```

Inside the container, source ROS 2 and run Eddy from the copied folder:

```bash
source /opt/ros/humble/setup.bash
```

```bash
cd /root/eddy_ai && PYTHONPATH=src python3 -m eddy info -r ros2
```

Use `PYTHONPATH=src python3 -m eddy` wherever the guides say `eddy`.

## If something goes wrong

| What you see | What to try |
|---|---|
| `rclpy` not found | Source ROS 2 (step 2) in the same terminal. |
| `eddy info` finds no velocity topic | Check `ros2 topic list` shows the robot's topics; check `ROS_DOMAIN_ID`. |
| It picked the wrong velocity topic | Eddy prefers a topic something subscribes to. Check with `ros2 topic info -v <topic>`. |
| Turns overshoot | The robot's odometry may lag; try `--speed 0.3`. |
| Drives are refused near walls | Something is closer than 0.25 m ahead on the laser; change with `--min-range`. |
