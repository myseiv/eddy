# Public simulators and controllers that Eddy is tested against (none of them written for Eddy).
# Built by tests/sims/run.sh as the image "eddy-sims".
FROM ros:humble-ros-base
RUN apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
      ros-humble-ros-gz ros-humble-ros-gz-sim-demos \
      ros-humble-controller-manager ros-humble-ros2-controllers ros-humble-gripper-controllers \
      ros-humble-moveit-resources-panda-moveit-config ros-humble-xacro ros-humble-robot-state-publisher \
      ros-humble-control-msgs ros-humble-ackermann-msgs \
      libegl1 libgl1-mesa-dri libglx-mesa0 python3-numpy \
    && rm -rf /var/lib/apt/lists/*
