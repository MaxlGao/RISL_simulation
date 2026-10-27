"""Render recognized objects in their localizer body frames."""

import xml.etree.ElementTree as ET

import numpy as np
from risl_camera_localizer.object_models import load_object_meshes, object_kind


# Both configured localizer dictionaries (4x4_250 and 5x5_250) use IDs 0..249.
# Preallocate bodies so tags first observed after startup can appear immediately.
DEFAULT_SCENE_OBJECT_NAMES = ('wrench', 'allen_key') + tuple(
    f'jenga_{marker_id}' for marker_id in range(250))


def add_objects(scene, names):
    """Add named mocap bodies using shared localizer geometry in meters."""
    if not names or len(names) != len(set(names)):
        raise ValueError('object_names must be nonempty and unique')
    kinds = {name: object_kind(name) for name in names}
    meshes = load_object_meshes()
    assets = scene.find('asset')
    if assets is None:
        assets = ET.SubElement(scene, 'asset')
    for kind in sorted(set(kinds.values()) - {'jenga'}):
        mesh = meshes[kind]
        ET.SubElement(
            assets, 'mesh', name=f'object_mesh_{kind}',
            vertex=' '.join(map(str, (mesh.vertices * .001).ravel())),
            face=' '.join(map(str, mesh.faces.ravel())))
    colors = {'jenga': '0.7 0.4 0.2 1', 'wrench': '0.65 0.68 0.72 1',
              'allen_key': '0.25 0.28 0.32 1'}
    for name, kind in kinds.items():
        body = ET.SubElement(scene.find('worldbody'), 'body',
                             name=f'object_{name}', mocap='true', pos='0 0 -10')
        attributes = ({'type': 'box', 'size': '0.036 0.012 0.007', 'pos': '0 0 -0.007'}
                      if kind == 'jenga' else
                      {'type': 'mesh', 'mesh': f'object_mesh_{kind}'})
        ET.SubElement(body, 'geom', name=f'object_geom_{name}',
                      rgba=colors[kind], contype='0', conaffinity='0', **attributes)


class RecognizedObjectMirror:
    """Keep unseen or stale bodies hidden; update valid poses independently."""

    def __init__(self, model, names, timeout=2.):
        if not np.isfinite(timeout) or timeout < 0:
            raise ValueError('object_timeout must be finite and non-negative')
        self.timeout = timeout
        self.mocap_ids = {
            name: model.body_mocapid[model.body(f'object_{name}').id]
            for name in names}
        self.last_seen = {}

    def update(self, data, name, position, quaternion, now):
        """Accept finite XYZ and normalize a WXYZ quaternion before displaying."""
        position = np.asarray(position, dtype=float)
        quaternion = np.asarray(quaternion, dtype=float)
        if (position.shape != (3,) or quaternion.shape != (4,)
                or not np.all(np.isfinite(position))
                or not np.all(np.isfinite(quaternion))):
            raise ValueError('Ignoring invalid object pose')
        norm = np.linalg.norm(quaternion)
        if not np.isfinite(norm) or norm < 1e-8:
            raise ValueError('Ignoring invalid object quaternion')
        mocap_id = self.mocap_ids[name]
        data.mocap_pos[mocap_id] = position
        data.mocap_quat[mocap_id] = quaternion / norm
        self.last_seen[name] = now

    def expire(self, data, now):
        """Hide expired objects until fresh feedback arrives; zero disables expiry."""
        if not self.timeout:
            return
        for name, timestamp in list(self.last_seen.items()):
            if now - timestamp > self.timeout or now < timestamp:
                data.mocap_pos[self.mocap_ids[name]] = [0, 0, -10]
                del self.last_seen[name]
