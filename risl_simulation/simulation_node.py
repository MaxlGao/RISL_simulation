import mujoco
import mujoco.viewer
import numpy as np
from contextlib import nullcontext

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
from tf2_geometry_msgs import do_transform_pose
from tf2_ros import Buffer, TransformException, TransformListener

from risl_simulation.robot_model import build_scene, JointStateMirror


class SimulationNode(Node):
    def __init__(self):
        super().__init__("simulation_node")

        defaults = {
            'joint_states_topic': '/joint_states',
            'joint_prefix': '',
            'gripper_width_topic': '/gripper_queried_width',
            'target_object_name': 'jenga_6',
            'world_frame': 'base',
            'kinematics_params': '',
            'gripper_mount_xyz': [0., 0., 0.],
            'gripper_mount_rpy': [0., 0., 0.],
            'initial_gripper_width_mm': 110.,
            'show_viewer': True,
            'render_rate': 30.,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        params = {name: self.get_parameter(name).value for name in defaults}
        if not np.isfinite(params['render_rate']) or params['render_rate'] <= 0:
            raise ValueError('render_rate must be finite and positive')
        self.world_frame = params['joint_prefix'] + params['world_frame']
        self.model = build_scene(
            prefix=params['joint_prefix'],
            kinematics_params=params['kinematics_params'],
            world_frame=params['world_frame'],
            mount_xyz=params['gripper_mount_xyz'],
            mount_rpy=params['gripper_mount_rpy'])
        self.data = mujoco.MjData(self.model)
        self.mirror = JointStateMirror(self.model, params['joint_prefix'])
        self.mirror.set_width(self.data, params['initial_gripper_width_mm'])
        self.target_object_name = params['target_object_name']
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        body_id = self.model.body("tracked_block").id
        self.mocap_id = self.model.body_mocapid[body_id]

        mujoco.mj_forward(self.model, self.data)

        self.viewer = (mujoco.viewer.launch_passive(self.model, self.data)
                       if params['show_viewer'] else None)

        if self.viewer:
            with self.viewer.lock():
                self.viewer.cam.lookat[:] = [0, 0, 0.35]
                self.viewer.cam.distance = 1.8

        self.joint_subscription = self.create_subscription(
            JointState, params['joint_states_topic'], self.joint_callback,
            qos_profile_sensor_data)
        self.width_subscription = self.create_subscription(
            Float64, params['gripper_width_topic'], self.width_callback,
            qos_profile_sensor_data)

        self.subscription = self.create_subscription(
            PoseStamped,
            f"/object_poses/{self.target_object_name}",
            self.pose_callback,
            qos_profile_sensor_data,
        )

        self.render_timer = self.create_timer(
            1.0 / params['render_rate'], self.refresh_viewer
        )
        self.status_timer = self.create_timer(5.0, self.report_missing_joints)
        self.get_logger().info(
            'UR5e mirror ready; RG2 uses schematic geometry, calibrated '
            'fingertip reach, and measured width feedback. Awaiting joint states.')

    def scene_lock(self):
        return self.viewer.lock() if self.viewer else nullcontext()

    def joint_callback(self, msg):
        try:
            with self.scene_lock():
                self.mirror.update(self.data, msg.name, msg.position)
        except ValueError as error:
            self.get_logger().warning(str(error), throttle_duration_sec=5.0)

    def width_callback(self, msg):
        try:
            with self.scene_lock():
                self.mirror.set_width(self.data, msg.data)
        except ValueError as error:
            self.get_logger().warning(str(error), throttle_duration_sec=5.0)

    def report_missing_joints(self):
        missing = self.mirror.addresses.keys() - self.mirror.seen
        if missing:
            self.get_logger().warning(
                'Waiting for joint states: ' + ', '.join(sorted(missing)))

    def pose_callback(self, msg):
        pose = msg.pose
        if msg.header.frame_id and msg.header.frame_id != self.world_frame:
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.world_frame, msg.header.frame_id,
                    rclpy.time.Time.from_msg(msg.header.stamp))
                pose = do_transform_pose(pose, transform)
            except TransformException as error:
                self.get_logger().warning(
                    f'Cannot transform object pose: {error}',
                    throttle_duration_sec=5.0)
                return
        p = pose.position
        q = pose.orientation

        position = np.array([p.x, p.y, p.z])
        # MuJoCo quaternion order is w, x, y, z.
        quaternion = np.array([q.w, q.x, q.y, q.z])

        norm = np.linalg.norm(quaternion)
        if (
            not np.all(np.isfinite(position))
            or not np.all(np.isfinite(quaternion))
            or norm < 1e-8
        ):
            self.get_logger().warning("Ignoring invalid pose")
            return

        with self.scene_lock():
            self.data.mocap_pos[self.mocap_id] = position
            self.data.mocap_quat[self.mocap_id] = quaternion / norm

    def refresh_viewer(self):
        if self.viewer and not self.viewer.is_running():
            rclpy.try_shutdown()
            return

        with self.scene_lock():
            mujoco.mj_forward(self.model, self.data)

        if self.viewer:
            self.viewer.sync()

    def destroy_node(self):
        if self.viewer:
            self.viewer.close()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = SimulationNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
