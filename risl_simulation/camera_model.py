"""Schematic wrist camera and optical field of view for the kinematic scene."""

import xml.etree.ElementTree as ET

import numpy as np


def add_camera(tool, view_depth=0.35, hfov=69.4, vfov=42.5):
    """Attach localizer_bridge's camera frame; optical forward is local +Z."""
    if not np.isfinite(view_depth) or view_depth <= 0:
        raise ValueError('camera_view_depth must be finite and positive')
    for name, value in (('camera_hfov', hfov), ('camera_vfov', vfov)):
        if not np.isfinite(value) or not 0 < value < 180:
            raise ValueError(f'{name} must be finite and within 0..180 degrees')

    # Same EE-frame offset and identity rotation as LocalizerBridge.
    # The optical origin is the front of the lens; housing dimensions are schematic.
    camera = ET.SubElement(tool, 'body', name='wrist_camera',
                           pos='-0.012 -0.038 -0.01')
    ET.SubElement(camera, 'geom', name='camera_housing', type='box',
                  pos='0 0 -0.014', size='0.025 0.012 0.008',
                  rgba='0.12 0.14 0.17 1', contype='0', conaffinity='0')
    ET.SubElement(camera, 'geom', name='camera_lens', type='cylinder',
                  pos='0 0 -0.003', size='0.006 0.003',
                  rgba='0.1 0.35 0.55 1', contype='0', conaffinity='0')

    half_width, half_height = view_depth * np.tan(np.deg2rad([hfov, vfov]) / 2)
    corners = [(-half_width, -half_height, view_depth),
               (half_width, -half_height, view_depth),
               (half_width, half_height, view_depth),
               (-half_width, half_height, view_depth)]
    # Sites are visual only: no mass, collision, or filled faces obscuring objects.
    for index, corner in enumerate(corners):
        for edge, start, end in (
                ('ray', (0., 0., 0.), corner),
                ('rim', corner, corners[(index + 1) % 4])):
            ET.SubElement(
                camera, 'site', name=f'camera_view_{edge}_{index}',
                type='capsule', size='0.0007',
                fromto=' '.join(map(str, (*start, *end))),
                rgba='0.1 0.75 1 0.7')
