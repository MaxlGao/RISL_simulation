import mujoco
import mujoco.viewer
import numpy as np
from contextlib import nullcontext
from functools import partial

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
from tf2_geometry_msgs import do_transform_pose
from tf2_ros import Buffer, TransformException, TransformListener

from risl_simulation.robot_model import build_scene, JointStateMirror
from risl_simulation.recognized_objects import RecognizedObjectMirror, DEFAULT_SCENE_OBJECT_NAMES


class SimulationNode(Node):
    def __init__(self):
        super().__init__("simulation_node")

        defaults = {
            'joint_states_topic': '/joint_states',
            'joint_prefix': '',
            'gripper_width_topic': '/gripper_queried_width',
            'object_names': list(DEFAULT_SCENE_OBJECT_NAMES),
            'object_pose_topic_prefix': '/vision/object_poses',
            'object_timeout': 2.,
            'world_frame': 'base',
            'kinematics_params': '',
            'gripper_mount_xyz': [0., 0., 0.],
            'gripper_mount_rpy': [0., 0., 0.],
            'initial_gripper_width_mm': 110.,
            'show_viewer': True,
            'render_rate': 30.,
            'headlight_ambient': [0.1, 0.1, 0.1],
            'headlight_diffuse': [0.8, 0.8, 0.8],
            'headlight_specular': [0.5, 0.5, 0.5],
            'light_diffuse': [0.7, 0.7, 0.7],
            'light_castshadow': True,
            'camera_view_depth': 0.35,
            'camera_hfov': 69.4,
            'camera_vfov': 42.5,
            'camera_lookat': [0., 0., 0.35],
            'camera_distance': 1.8,
            'camera_azimuth': 90.,
            'camera_elevation': -45.,
            'show_right_ui': False,
            'show_left_ui': False,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        params = {name: self.get_parameter(name).value for name in defaults}
        if not np.isfinite(params['render_rate']) or params['render_rate'] <= 0:
            raise ValueError('render_rate must be finite and positive')
        lookat = params['camera_lookat']
        if len(lookat) != 3 or not np.all(np.isfinite(lookat)):
            raise ValueError('camera_lookat must contain three finite values')
        if (not np.isfinite(params['camera_distance'])
                or params['camera_distance'] <= 0):
            raise ValueError('camera_distance must be finite and positive')
        for name in ('camera_azimuth', 'camera_elevation'):
            if not np.isfinite(params[name]):
                raise ValueError(f'{name} must be finite')
        if not -90 <= params['camera_elevation'] <= 90:
            raise ValueError('camera_elevation must be within -90..90 degrees')
        self.world_frame = params['joint_prefix'] + params['world_frame']
        self.model = build_scene(
            prefix=params['joint_prefix'],
            kinematics_params=params['kinematics_params'],
            world_frame=params['world_frame'],
            mount_xyz=params['gripper_mount_xyz'],
            mount_rpy=params['gripper_mount_rpy'],
            object_names=params['object_names'],
            headlight_ambient=params['headlight_ambient'],
            headlight_diffuse=params['headlight_diffuse'],
            headlight_specular=params['headlight_specular'],
            light_diffuse=params['light_diffuse'],
            light_castshadow=params['light_castshadow'],
            camera_view_depth=params['camera_view_depth'],
            camera_hfov=params['camera_hfov'],
            camera_vfov=params['camera_vfov'])
        self.data = mujoco.MjData(self.model)
        self.mirror = JointStateMirror(self.model, params['joint_prefix'])
        self.mirror.set_width(self.data, params['initial_gripper_width_mm'])
        self.objects = RecognizedObjectMirror(
            self.model, params['object_names'], params['object_timeout'])
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        mujoco.mj_forward(self.model, self.data)

        self.viewer = (mujoco.viewer.launch_passive(self.model, self.data)
                       if params['show_viewer'] else None)

        if self.viewer:
            with self.viewer.lock():
                self.viewer.cam.lookat[:] = params['camera_lookat']
                self.viewer.cam.distance = params['camera_distance']
                self.viewer.cam.azimuth = params['camera_azimuth']
                self.viewer.cam.elevation = params['camera_elevation']

        self.joint_subscription = self.create_subscription(
            JointState, params['joint_states_topic'], self.joint_callback,
            qos_profile_sensor_data)
        self.width_subscription = self.create_subscription(
            Float64, params['gripper_width_topic'], self.width_callback,
            qos_profile_sensor_data)

        topic_prefix = params['object_pose_topic_prefix'].rstrip('/')
        self.object_subscriptions = {
            name: self.create_subscription(
                PoseStamped, f'{topic_prefix}/{name}',
                partial(self.pose_callback, object_name=name), qos_profile_sensor_data)
            for name in params['object_names']}

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

    def pose_callback(self, msg, object_name):
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

        try:
            with self.scene_lock():
                self.objects.update(
                    self.data, object_name, [p.x, p.y, p.z], [q.w, q.x, q.y, q.z],
                    self.get_clock().now().nanoseconds * 1e-9)
        except ValueError as error:
            self.get_logger().warning(
                f'{object_name}: {error}', throttle_duration_sec=5.0)

    def refresh_viewer(self):
        if self.viewer and not self.viewer.is_running():
            rclpy.try_shutdown()
            return

        with self.scene_lock():
            self.objects.expire(self.data, self.get_clock().now().nanoseconds * 1e-9)
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
