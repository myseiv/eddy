"""Turn one of Gazebo's stock example worlds into a test world for Eddy.

The vehicles, their drive plugins and the physics are Gazebo's own, unchanged. This script
only adds what a test needs around them: walls, a box to stop for, and a laser on each vehicle.
Usage: make_world.py diff_drive|ackermann_steering  >  world.sdf
"""
import re
import sys

kind = sys.argv[1]
sdf = open(f"/usr/share/ignition/ignition-gazebo6/worlds/{kind}.sdf").read()

SENSORS = """
    <plugin filename="ignition-gazebo-sensors-system" name="ignition::gazebo::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>"""


def laser(model: str) -> str:
    """A flat 360-degree laser, mounted high on the front of the chassis so it sees over the wheels."""
    return f"""
        <sensor name='laser' type='gpu_lidar'>
          <pose>0.9 0 0.45 0 0 0</pose>
          <topic>model/{model}/scan</topic>
          <ignition_frame_id>{model}/chassis</ignition_frame_id>
          <update_rate>10</update_rate>
          <lidar>
            <scan><horizontal><samples>360</samples><resolution>1</resolution>
              <min_angle>-3.14159</min_angle><max_angle>3.12414</max_angle></horizontal>
              <vertical><samples>1</samples><resolution>1</resolution><min_angle>0</min_angle><max_angle>0</max_angle></vertical></scan>
            <range><min>0.1</min><max>30.0</max><resolution>0.01</resolution></range>
          </lidar>
          <always_on>1</always_on>
        </sensor>"""


def box(name: str, x: float, y: float, sx: float, sy: float, sz: float = 2.0) -> str:
    return f"""
    <model name='{name}'><static>true</static><pose>{x} {y} {sz / 2} 0 0 0</pose>
      <link name='link'>
        <collision name='c'><geometry><box><size>{sx} {sy} {sz}</size></box></geometry></collision>
        <visual name='v'><geometry><box><size>{sx} {sy} {sz}</size></box></geometry></visual>
      </link></model>"""


# The sensors system goes after the world's first plugin.
first_plugin_end = sdf.index("</plugin>") + len("</plugin>")
sdf = sdf[:first_plugin_end] + SENSORS + sdf[first_plugin_end:]
# A laser inside each vehicle's chassis link (the first <link> after each vehicle model starts).
for model in re.findall(r"<model name='(vehicle_\w+)'>", sdf):
    start = sdf.index(f"<model name='{model}'>")
    link = sdf.index("<link name='chassis'>", start) + len("<link name='chassis'>")
    sdf = sdf[:link] + laser(model) + sdf[link:]
# A room from x=-4 to x=10 and y=-8 to y=8, and a box in the green vehicle's lane.
room = (box("wall_front", 10.25, 0, 0.5, 17) + box("wall_back", -4.25, 0, 0.5, 17)
        + box("wall_left", 3, 8.25, 15, 0.5) + box("wall_right", 3, -8.25, 15, 0.5)
        + box("crate", 5.0, -2.0, 0.6, 0.6, 1.5))
sdf = sdf.replace("</world>", room + "\n  </world>")
sys.stdout.write(sdf)
