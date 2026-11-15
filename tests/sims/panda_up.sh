#!/usr/bin/env bash
# Inside the eddy-sims image: start a Franka Panda arm on ros2_control's mock hardware, with the
# stock controllers from moveit_resources (a joint trajectory controller for the arm and a
# gripper action controller for the hand). Nothing here is written for Eddy.
source /opt/ros/humble/setup.bash
cd /opt/ros/humble/share/moveit_resources_panda_moveit_config/config
python3 - <<'PY'
import subprocess, yaml
urdf = subprocess.check_output(["xacro", "panda.urdf.xacro"]).decode()
yaml.safe_dump({"/**": {"ros__parameters": {"robot_description": urdf}}}, open("/tmp/description.yaml", "w"))
PY
ros2 run robot_state_publisher robot_state_publisher --ros-args --params-file /tmp/description.yaml >/tmp/rsp.log 2>&1 &
ros2 run controller_manager ros2_control_node --ros-args --params-file /tmp/description.yaml --params-file ros2_controllers.yaml >/tmp/control.log 2>&1 &
sleep 3
ros2 run controller_manager spawner joint_state_broadcaster panda_arm_controller panda_hand_controller >/tmp/spawn.log 2>&1
echo "panda up: $(ros2 control list_controllers 2>/dev/null | tr '\n' ';')"
