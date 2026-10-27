# UR5e and RG2 live visualization

`simulation_node` mirrors ROS feedback into MuJoCo using forward kinematics.
The UR5e uses the installed `ur_description` kinematics and STL collision meshes
(plain shaded appearance). The RG2 has a **schematic body and fingers** with
measured opening-dependent axial reach, not its exact CAD or linkage.
This is a kinematic display; it does not
simulate grasp forces or predict collisions.

## Build and run (ROS 2 Humble)

The workspace's existing Python environment has MuJoCo installed. Use the same
Python environment to build and launch so the executable can import MuJoCo:

```bash
cd ~/risl_assembly_ws
source /opt/ros/humble/setup.bash
source .venv/bin/activate
python -m colcon build --packages-select risl_ur risl_camera_localizer risl_simulation risl_onrobot_ros --symlink-install
source install/setup.bash
ros2 launch risl_simulation simulation.launch.py
```

Required ROS packages: `ur_description`, `xacro`, `tf2_ros`, `tf2_geometry_msgs`,
plus standard ROS Python/message/launch packages listed in `package.xml`.
If missing, install dependencies with `rosdep install --from-paths
src/risl_simulation --ignore-src -r -y`, and install Python dependencies into your
environment with `python -m pip install 'mujoco>=3.0' numpy`.
A desktop display is required for the viewer. For operation without a window:

```bash
ros2 run risl_simulation simulation_node --ros-args -p show_viewer:=false
```

Start your existing UR driver to supply `/joint_states` (`sensor_msgs/JointState`).
Only position values are used; message order and partial joint updates are supported.
The node reports missing arm joints every five seconds. With no feedback, joints
stay at their last value (zero at startup).

Start the RG2 driver with periodic measured feedback to supply
`/gripper_queried_width` (`std_msgs/Float64`, inner opening in **millimeters**,
0–110):

```bash
ros2 run risl_onrobot_ros gripper_control --ros-args -p feedback_rate:=5.0
```

This opt-in rate uses the driver's existing Modbus connection and single-threaded
executor. Do not run a second gripper driver at the same time. The default rate
is zero, preserving query-only operation; you can also refresh once with:

```bash
ros2 topic pub --once /gripper_command std_msgs/msg/String "{data: 'query'}"
```

The display never sends robot or gripper commands. Opening stays at its last
measured value, or the configured initial width before feedback arrives. A slow
or unavailable Modbus connection can delay polling and command handling.

## Match your physical setup

Copy `config/simulation.yaml`, edit it, and launch with
`ros2 launch risl_simulation simulation.launch.py params_file:=/absolute/path/to/simulation.yaml`.

- Set `joint_prefix` and topic names to match the UR driver if namespaced.
- Set `kinematics_params` to the same calibrated YAML used by the UR driver for
  agreement with the real arm. Otherwise nominal UR5e kinematics are used.
- Set `gripper_mount_xyz` (meters) and `gripper_mount_rpy` (radians) for the
  adjustment from `tool0` to the calibrated gripper frame. Leave these at zero
  for the setup measured in `GRIPPER_TABLE`: those measured distances already
  start at the flange (zero TCP offset) and include its mounting hardware.
  Only add a transform for changes relative to that calibrated setup, to avoid
  counting an adapter twice. These are startup parameters.
- The world defaults to UR's `base`, matching this workspace's camera localizer.
  `world_frame: base_link` is also supported. A configured joint prefix is applied
  to the frame name. Object poses in other frames use TF at their message
  timestamp and are skipped if TF is unavailable.
  An empty frame ID is assumed to mean the configured world frame.

For accurate RG2 geometry, export your SolidWorks parts as separate binary STL
files for the fixed assembly and each moving finger, in meters, retaining their
shared assembly coordinates. Include the mounting origin, finger pivot/axis
locations, and the assembly pose used for export. STEP exports can also serve as
source files but require mesh conversion before MuJoCo can load them. A URDF
with meshes and joint definitions is another suitable input. The current
approximation follows the measured fingertip depth but does not reproduce the
RG2's internal linkage or swept geometry.

## Measured fingertip reach

`risl_ur.custom_libraries.gripper_geometry` is the shared source of `GRIPPER_TABLE`
and `GRIPPER_FINGER_OFFSET`; `gripper_utils` re-exports both for existing callers.
The controller and simulation use the same piecewise linear interpolation.
Table keys are opening in **0.1 mm**, while ROS feedback is **mm**; table values
are axial distance in **meters** from zero-offset `tool0` to the distal finger
edge. Reach varies from 0.153 m closed to 0.114 m at 100 mm opening. Openings
100–110 mm hold the last measured reach, matching the existing controller's
clamping behavior; there are no measurements beyond 100 mm.

Each width update moves the fingers symmetrically and shifts both along local
tool Z. The 0.010 m half-length perpendicular to the opening also comes from
the controller calibration. Orange sites mark the inner fingertip centers; red
sites mark their leading edges. These positions follow the arm's full 3D pose.
The controller's `vertical_offset` represents its virtual table level and is
not added to the physical gripper geometry.

## Recognized objects

Run the camera localizer alongside the simulation (select your camera ID):

```bash
ros2 run risl_camera_localizer localize --camera-id 4
```

By default, the simulation subscribes to these `geometry_msgs/PoseStamped` topics:

| Object | Topic | Geometry |
| --- | --- | --- |
| Wrench | `/vision/object_poses/wrench` | Body-aligned `Wrench.STL` |
| Allen key | `/vision/object_poses/allen_key` | Body-aligned `Allen Key.STL` |
| Jenga marker 6 | `/vision/object_poses/jenga_6` | 72 × 24 × 14 mm box |

`risl_camera_localizer.object_models` now provides the same meshes and body-frame
alignment to both contour processing and MuJoCo, including millimeter-to-meter
conversion in the display. The Jenga origin lies on the top surface; its solid
extends 14 mm along negative local Z. Tool meshes retain their original body
origins rather than being centered on their bounding boxes.

Objects stay hidden until their first valid pose, and disappear after
`object_timeout` seconds without a valid update (default 2.0; zero holds the last
pose indefinitely). They reappear on fresh feedback. This timeout measures the
pose stream: the localizer may continue publishing its own predicted/held poses
during an occlusion, which will continue to display. No camera is opened by the
simulation and no vision recognition is duplicated.

Configure `object_names` to choose objects, including multiple `jenga_<id>`
instances, and restart the simulation. The localizer currently detects only
marker 6; additional IDs also require changing its detector's ID filter.
The bridge supports publishing additional instances on the same `/vision` topic
prefix. Change `object_pose_topic_prefix` if replaying or remapping other topics.
The old single `target_object_name` parameter and `/object_poses` topic default
have been replaced with this object list and `/vision/object_poses` prefix.
Recognized objects are pose-driven visual bodies; no contact forces are simulated.
