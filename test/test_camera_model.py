"""Verify camera mounting and optical pyramid geometry without a viewer."""

import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from risl_simulation.camera_model import add_camera
from risl_simulation.robot_model import ARM_JOINTS, build_scene, JointStateMirror


@pytest.mark.parametrize('prefix, world_frame, angles', [
    ('', 'base', [0.] * 6),
    ('ur_', 'base_link', [.3, -1.2, .8, -.4, .7, -.9]),
])
def test_camera_tracks_ee_and_optical_pyramid(prefix, world_frame, angles):
    model = build_scene(
        prefix=prefix, world_frame=world_frame, object_names=['jenga_6'],
        mount_xyz=[.01, .02, .03], mount_rpy=[.1, .2, .3])
    data = mujoco.MjData(model)
    mirror = JointStateMirror(model, prefix)
    mirror.update(data, [prefix + name for name in ARM_JOINTS], angles)
    mujoco.mj_forward(model, data)
    tool = data.body(prefix + 'tool0')
    camera = data.body('wrist_camera')
    rotation = tool.xmat.reshape(3, 3)
    # Independent localizer_bridge transform, unaffected by RG2 mount adjustments.
    np.testing.assert_allclose(
        camera.xpos, tool.xpos + rotation @ [-.012, -.038, -.01], atol=1e-8)
    np.testing.assert_allclose(camera.xmat, tool.xmat, atol=1e-8)

    depth = .35
    width = depth * np.tan(np.deg2rad(69.4 / 2))
    height = depth * np.tan(np.deg2rad(42.5 / 2))
    corners = np.array([[-width, -height, depth], [width, -height, depth],
                        [width, height, depth], [-width, height, depth]])
    for index, corner in enumerate(corners):
        for edge, start, end in (
                ('ray', np.zeros(3), corner),
                ('rim', corner, corners[(index + 1) % 4])):
            name = f'camera_view_{edge}_{index}'
            site = data.site(name)
            axis = site.xmat.reshape(3, 3)[:, 2]
            half_length = model.site(name).size[1]
            endpoints = np.array([site.xpos - half_length * axis,
                                  site.xpos + half_length * axis])
            expected = camera.xpos + np.array([start, end]) @ rotation.T
            # Capsule axes may point in either direction.
            if np.linalg.norm(endpoints[0] - expected[0]) > 1e-8:
                endpoints = endpoints[::-1]
            np.testing.assert_allclose(endpoints, expected, atol=1e-8)


def test_custom_view_dimensions():
    tool = ET.Element('body')
    add_camera(tool, view_depth=.5, hfov=90., vfov=60.)
    ray = tool.find(".//site[@name='camera_view_ray_0']")
    np.testing.assert_allclose(
        list(map(float, ray.get('fromto').split())),
        [0., 0., 0., -.5, -.5 * np.tan(np.pi / 6), .5])


@pytest.mark.parametrize('keyword, invalid', [
    ('view_depth', 0.), ('view_depth', -.1), ('view_depth', np.inf),
    ('hfov', 0.), ('hfov', 180.), ('hfov', np.nan),
    ('vfov', -1.), ('vfov', 180.), ('vfov', np.inf),
])
def test_invalid_view_dimensions(keyword, invalid):
    with pytest.raises(ValueError, match='camera_'):
        add_camera(ET.Element('body'), **{keyword: invalid})
