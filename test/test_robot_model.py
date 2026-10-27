"""Verify feedback mapping and UR5e kinematics without opening a viewer."""

import mujoco
import numpy as np
import pytest

from risl_simulation.robot_model import ARM_JOINTS, build_scene, JointStateMirror


@pytest.fixture(scope='module')
def model():
    return build_scene()


@pytest.mark.parametrize('angles', [
    np.zeros(6),
    [0.3, -1.2, 0.8, -0.4, 0.7, -0.9],
])
def test_forward_kinematics_matches_ur_dh(model, angles):
    """Compare imported ROS geometry with independent nominal DH kinematics."""
    transform = np.eye(4)
    for theta, a, d, alpha in zip(
            angles, [0, -.425, -.3922, 0, 0, 0],
            [.1625, 0, 0, .1333, .0997, .0996],
            [np.pi / 2, 0, 0, np.pi / 2, -np.pi / 2, 0]):
        c, s = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)
        transform = transform @ np.array([
            [c, -s * ca, s * sa, a * c],
            [s, c * ca, -c * sa, a * s],
            [0, sa, ca, d], [0, 0, 0, 1]])
    data = mujoco.MjData(model)
    mirror = JointStateMirror(model)
    mirror.update(data, list(reversed(ARM_JOINTS)), list(reversed(angles)))
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.body('tool0').xpos,
                               transform[:3, 3], atol=1e-8)
    np.testing.assert_allclose(data.body('tool0').xmat.reshape(3, 3),
                               transform[:3, :3], atol=1e-8)


def test_partial_updates_and_invalid_packets(model):
    data = mujoco.MjData(model)
    mirror = JointStateMirror(model)
    mirror.update(data, ['unrelated', ARM_JOINTS[2]], [99., .3])
    before = data.qpos.copy()
    for names, positions in [
            ([ARM_JOINTS[0]], []),
            ([ARM_JOINTS[0], ARM_JOINTS[0]], [1., 2.]),
            ([ARM_JOINTS[0], ARM_JOINTS[1]], [1., np.nan])]:
        with pytest.raises(ValueError):
            mirror.update(data, names, positions)
        np.testing.assert_array_equal(data.qpos, before)
    assert mirror.seen == {ARM_JOINTS[2]}


def test_measured_gripper_width(model):
    data = mujoco.MjData(model)
    mirror = JointStateMirror(model)
    for width in [0., 52., 110.]:
        mirror.set_width(data, width)
        mujoco.mj_forward(model, data)
        left = data.body('rg2_left_finger').xpos
        right = data.body('rg2_right_finger').xpos
        # Center separation includes both 4 mm half-thicknesses.
        assert np.linalg.norm(left - right) - .008 == pytest.approx(width / 1000)
    for invalid in [np.nan, np.inf, -1., 111.]:
        with pytest.raises(ValueError):
            mirror.set_width(data, invalid)


@pytest.mark.parametrize('width_mm, reach_m', [
    (0., .153), (10., .1515), (20., .150), (30., .1485),
    (40., .147), (50., .144), (60., .141), (70., .1365),
    (80., .132), (90., .123), (100., .114), (110., .114),
])
def test_fingertips_follow_measured_reach(model, width_mm, reach_m):
    """Check inner tip edges in the flange frame at a rotated arm pose."""
    from risl_ur.custom_libraries.gripper_utils import GripperHandler

    data = mujoco.MjData(model)
    mirror = JointStateMirror(model)
    mirror.update(data, ARM_JOINTS, [.3, -1.2, .8, -.4, .7, -.9])
    mirror.set_width(data, width_mm)
    mujoco.mj_forward(model, data)
    tool = data.body('tool0')
    world_to_tool = tool.xmat.reshape(3, 3).T
    for side, direction in [('left', 1), ('right', -1)]:
        # Controller's leading edge is toward -Y for its [0, 180, yaw] tool pose.
        for feature, y in [('tip', 0.), ('edge', -.010)]:
            local = world_to_tool @ (
                data.site(f'rg2_{side}_{feature}').xpos - tool.xpos)
            np.testing.assert_allclose(
                local, [direction * width_mm / 2000., y, reach_m], atol=1e-8)
    # Existing controller API retains its 0.1 mm units and the same calibration.
    assert GripperHandler.gripper_width_to_height(None, width_mm * 10) == (
        pytest.approx(reach_m))


def test_closing_extends_fingers_without_changing_arm_pose(model):
    data = mujoco.MjData(model)
    mirror = JointStateMirror(model)
    mirror.set_width(data, 100.)
    mujoco.mj_forward(model, data)
    tool_before = data.body('tool0').xpos.copy()
    tip_before = data.site('rg2_left_tip').xpos.copy()
    mirror.set_width(data, 0.)
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.body('tool0').xpos, tool_before)
    displacement = data.body('tool0').xmat.reshape(3, 3).T @ (
        data.site('rg2_left_tip').xpos - tip_before)
    np.testing.assert_allclose(displacement, [-.050, 0., .039], atol=1e-8)


def test_prefix_mount_and_base_link_frame():
    model = build_scene(prefix='ur_', world_frame='base_link',
                        mount_xyz=[0., 0., .04], mount_rpy=[0., 0., .5])
    data = mujoco.MjData(model)
    mirror = JointStateMirror(model, 'ur_')
    mirror.update(data, ['ur_elbow_joint'], [.5])
    mujoco.mj_forward(model, data)
    tool = data.body('ur_tool0')
    mount = data.body('rg2_mount')
    expected = tool.xpos + tool.xmat.reshape(3, 3) @ [0., 0., .04]
    np.testing.assert_allclose(mount.xpos, expected, atol=1e-8)
    np.testing.assert_allclose(data.body('ur_base').xmat.reshape(3, 3),
                               np.diag([-1., -1., 1.]), atol=1e-8)
