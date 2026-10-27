"""Check recognized-object geometry, pose mapping, and detection lifecycle."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
import mujoco
import numpy as np
import pytest
import trimesh

from risl_camera_localizer.object_models import load_object_meshes
from risl_simulation.recognized_objects import RecognizedObjectMirror
from risl_simulation.robot_model import build_scene


NAMES = ('wrench', 'allen_key', 'jenga_6', 'jenga_7')


@pytest.fixture(scope='module')
def model():
    return build_scene(object_names=NAMES)


@pytest.mark.parametrize('name, filename, origin', [
    ('allen_key', 'Allen Key.STL', [28.33, 0, 37]),
    ('wrench', 'Wrench.STL', [49, 0, 26]),
])
def test_shared_meshes_preserve_localizer_alignment(name, filename, origin):
    """Compare with the original process_stl translations and rotations."""
    directory = Path(get_package_share_directory('risl_camera_localizer')) / 'STL'
    raw = trimesh.load_mesh(directory / filename)
    vertices = np.asarray(raw.vertices) - origin
    rx90 = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    vertices = vertices @ rx90.T
    if name == 'allen_key':
        vertices += [-50, -10, 0]
    else:
        rz90 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        vertices = vertices @ rz90.T
    np.testing.assert_allclose(load_object_meshes()[name].vertices, vertices, atol=1e-10)


@pytest.mark.parametrize('name', ['wrench', 'allen_key'])
def test_mujoco_mesh_has_correct_scale_origin_and_orientation(model, name):
    data = mujoco.MjData(model)
    mirror = RecognizedObjectMirror(model, NAMES)
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    position = np.array([.2, -.5, .05])
    mirror.update(data, name, position, [1, 0, 0, 1], 1.)
    mujoco.mj_forward(model, data)
    geom = model.geom(f'object_geom_{name}').id
    mesh = model.geom_dataid[geom]
    start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
    vertices = model.mesh_vert[start:start + count]
    actual = vertices @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
    expected = load_object_meshes()[name].vertices * .001 @ rotation.T + position
    np.testing.assert_allclose(actual.min(axis=0), expected.min(axis=0), atol=1e-7)
    np.testing.assert_allclose(actual.max(axis=0), expected.max(axis=0), atol=1e-7)


def test_jenga_pose_is_top_surface_not_center(model):
    data = mujoco.MjData(model)
    mirror = RecognizedObjectMirror(model, NAMES)
    mirror.update(data, 'jenga_6', [.3, -.4, .014], [1, 0, 0, 0], 1.)
    mujoco.mj_forward(model, data)
    geom = model.geom('object_geom_jenga_6').id
    np.testing.assert_allclose(data.geom_xpos[geom], [.3, -.4, .007])
    np.testing.assert_allclose(model.geom_size[geom], [.036, .012, .007])
    assert data.geom_xpos[geom, 2] - model.geom_size[geom, 2] == pytest.approx(0.)


def test_visibility_expiry_reacquisition_and_independent_instances(model):
    data = mujoco.MjData(model)
    mirror = RecognizedObjectMirror(model, NAMES, timeout=2.)
    assert np.all(data.mocap_pos[:, 2] == -10.)
    mirror.update(data, 'jenga_6', [.1, .2, .3], [2, 0, 0, 0], 1.)
    mirror.update(data, 'jenga_7', [.4, .5, .6], [1, 0, 0, 0], 2.)
    mirror.expire(data, 3.1)
    assert data.mocap_pos[mirror.mocap_ids['jenga_6'], 2] == -10.
    np.testing.assert_allclose(data.mocap_pos[mirror.mocap_ids['jenga_7']], [.4, .5, .6])
    mirror.update(data, 'jenga_6', [.7, .8, .9], [2, 0, 0, 0], 3.2)
    np.testing.assert_allclose(data.mocap_quat[mirror.mocap_ids['jenga_6']], [1, 0, 0, 0])
    mirror.expire(data, 0.)  # Clock reset invalidates old timestamps.
    assert not mirror.last_seen


def test_invalid_pose_does_not_refresh_timeout(model):
    data = mujoco.MjData(model)
    mirror = RecognizedObjectMirror(model, NAMES)
    mirror.update(data, 'wrench', [0, 0, 0], [1, 0, 0, 0], 1.)
    for position, quat in [([np.nan, 0, 0], [1, 0, 0, 0]),
                           ([0, 0, 0], [0, 0, 0, 0]),
                           ([0, 0, 0], [np.inf, 0, 0, 0])]:
        with pytest.raises(ValueError):
            mirror.update(data, 'wrench', position, quat, 2.9)
    mirror.expire(data, 3.1)
    assert data.mocap_pos[mirror.mocap_ids['wrench'], 2] == -10.
    assert 'wrench' not in mirror.last_seen


def test_timeout_can_be_disabled(model):
    data = mujoco.MjData(model)
    mirror = RecognizedObjectMirror(model, NAMES, timeout=0.)
    mirror.update(data, 'wrench', [0, 0, .1], [1, 0, 0, 0], 0.)
    mirror.expire(data, 1000.)
    assert data.mocap_pos[mirror.mocap_ids['wrench'], 2] == .1


@pytest.mark.parametrize('names', [[], ['wrench', 'wrench'], ['unknown']])
def test_reject_unsupported_or_ambiguous_geometry(names):
    with pytest.raises(ValueError):
        build_scene(object_names=names)
