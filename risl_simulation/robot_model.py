"""Build a kinematic UR5e scene using the installed ROS description."""

from pathlib import Path
from tempfile import TemporaryDirectory
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_share_directory
import mujoco
import numpy as np
import xacro
from risl_simulation.recognized_objects import add_objects, DEFAULT_SCENE_OBJECT_NAMES

from risl_ur.custom_libraries.gripper_geometry import (
    GRIPPER_FINGER_OFFSET, GRIPPER_TABLE, gripper_width_to_reach,
)

ARM_JOINTS = (
    'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
    'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
)
FINGER_JOINTS = ('rg2_left_slide', 'rg2_right_slide')
MIN_FINGER_REACH = min(GRIPPER_TABLE.values())
MAX_FINGER_REACH = max(GRIPPER_TABLE.values())
FINGER_LENGTH = 0.070


def build_scene(prefix='', kinematics_params='', world_frame='base',
                mount_xyz=(0., 0., 0.), mount_rpy=(0., 0., 0.),
                object_names=DEFAULT_SCENE_OBJECT_NAMES,
                headlight_ambient=(0.1, 0.1, 0.1),
                headlight_diffuse=(0.8, 0.8, 0.8),
                headlight_specular=(0.5, 0.5, 0.5),
                light_diffuse=(0.7, 0.7, 0.7), light_castshadow=True):
    """
    Compile ROS URDF to MJCF, retaining fixed tool frames and STL meshes.

    RG2 geometry is schematic; measured width-dependent fingertip reach comes
    from the shared controller calibration. No dynamics are stepped.
    """
    if world_frame not in ('base', 'base_link'):
        raise ValueError('world_frame must be base or base_link (without prefix)')
    if any(len(v) != 3 or not np.all(np.isfinite(v))
           for v in (mount_xyz, mount_rpy)):
        raise ValueError('Mount xyz and rpy must each contain three finite values')
    for name, value in (
            ('headlight_ambient', headlight_ambient),
            ('headlight_diffuse', headlight_diffuse),
            ('headlight_specular', headlight_specular),
            ('light_diffuse', light_diffuse)):
        if (len(value) != 3 or not np.all(np.isfinite(value))
                or np.any(np.asarray(value) < 0)
                or np.any(np.asarray(value) > 1)):
            raise ValueError(f'{name} must contain three finite RGB values in 0..1')
    share = Path(get_package_share_directory('ur_description'))
    mappings = {'name': 'ur5e', 'ur_type': 'ur5e', 'tf_prefix': prefix}
    if kinematics_params:
        mappings['kinematics_params'] = str(Path(kinematics_params).resolve())
    robot = ET.fromstring(xacro.process_file(
        str(share / 'urdf' / 'ur.urdf.xacro'), mappings=mappings).toxml())
    # MuJoCo supports STL collision meshes; the visual DAE files are omitted.
    for link in robot.findall('link'):
        for visual in link.findall('visual'):
            link.remove(visual)
    for mesh in robot.iter('mesh'):
        filename = mesh.get('filename')
        if filename.startswith('package://'):
            package, relative = filename[len('package://'):].split('/', 1)
            mesh.set('filename', str(
                Path(get_package_share_directory(package)) / relative))
    extension = ET.SubElement(robot, 'mujoco')
    ET.SubElement(extension, 'compiler', fusestatic='false', strippath='false')
    arm = mujoco.MjModel.from_xml_string(ET.tostring(robot, encoding='unicode'))
    with TemporaryDirectory() as directory:
        converted = Path(directory) / 'arm.xml'
        mujoco.mj_saveLastXML(str(converted), arm)
        scene = ET.parse(converted).getroot()
    scene.find('compiler').set('angle', 'radian')
    visual = scene.find('visual')
    if visual is None:
        visual = ET.SubElement(scene, 'visual')
    headlight = visual.find('headlight')
    if headlight is None:
        headlight = ET.SubElement(visual, 'headlight')
    for name, value in (('ambient', headlight_ambient),
                        ('diffuse', headlight_diffuse),
                        ('specular', headlight_specular)):
        headlight.set(name, ' '.join(map(str, value)))
    world = scene.find('worldbody')
    # UR base is rotated pi about Z relative to ROS base_link.
    root = ET.Element('body', name='robot_root',
                      euler='0 0 3.141592653589793'
                      if world_frame == 'base' else '0 0 0')
    for child in list(world):
        world.remove(child)
        root.append(child)
    world.append(root)
    for geom in root.iter('geom'):
        geom.set('rgba', '0.65 0.72 0.78 1')
        geom.set('contype', '0')
        geom.set('conaffinity', '0')
    tool = root.find(f".//body[@name='{prefix}tool0']")
    if tool is None:
        raise ValueError('UR description did not preserve tool0')
    gripper = ET.SubElement(
        tool, 'body', name='rg2_mount',
        pos=' '.join(map(str, mount_xyz)), euler=' '.join(map(str, mount_rpy)))
    ET.SubElement(gripper, 'geom', type='cylinder', size='0.032 0.015',
                  pos='0 0 0.015', rgba='0.2 0.2 0.2 1')
    ET.SubElement(gripper, 'geom', type='box', size='0.035 0.025 0.025',
                  pos='0 0 0.055', rgba='0.15 0.15 0.16 1')
    carrier = ET.SubElement(gripper, 'body', name='rg2_finger_carrier')
    # MuJoCo requires inertia on this synthetic moving frame; not used for dynamics.
    ET.SubElement(carrier, 'inertial', pos='0 0 0', mass='0.001',
                  diaginertia='0.000001 0.000001 0.000001')
    ET.SubElement(carrier, 'joint', name='rg2_extension', type='slide',
                  axis='0 0 1', range=f'0 {MAX_FINGER_REACH - MIN_FINGER_REACH}')
    # This extra DOF represents measured reach, not the RG2's internal linkage.
    for side, direction in (('left', 1), ('right', -1)):
        finger = ET.SubElement(carrier, 'body', name=f'rg2_{side}_finger',
                               pos=f'{direction * 0.004} 0 '
                                   f'{MIN_FINGER_REACH - FINGER_LENGTH}')
        ET.SubElement(finger, 'joint', name=f'rg2_{side}_slide', type='slide',
                      axis=f'{direction} 0 0', range='0 0.055')
        ET.SubElement(finger, 'geom', type='box',
                      size=f'0.004 {GRIPPER_FINGER_OFFSET} {FINGER_LENGTH / 2}',
                      pos='0 0 0.035', rgba='0.3 0.3 0.32 1')
        ET.SubElement(finger, 'site', name=f'rg2_{side}_tip',
                      pos=f'{-direction * 0.004} 0 {FINGER_LENGTH}',
                      size='0.002', rgba='1 0.5 0 1')
        ET.SubElement(finger, 'site', name=f'rg2_{side}_edge',
                      pos=f'{-direction * 0.004} '
                          f'{-GRIPPER_FINGER_OFFSET} {FINGER_LENGTH}',
                      size='0.002', rgba='1 0.2 0 1')
    ET.SubElement(world, 'light', name='overhead', pos='0 0 3',
                  diffuse=' '.join(map(str, light_diffuse)),
                  castshadow='true' if light_castshadow else 'false')
    ET.SubElement(world, 'geom', type='plane', size='2 2 0.1',
                  rgba='0.8 0.8 0.8 1')
    add_objects(scene, object_names)
    return mujoco.MjModel.from_xml_string(ET.tostring(scene, encoding='unicode'))


class JointStateMirror:
    """Map ROS positions by name, independent of message ordering or subsets."""

    def __init__(self, model, prefix=''):
        self.addresses = {
            prefix + name: model.jnt_qposadr[model.joint(prefix + name).id]
            for name in ARM_JOINTS
        }
        self.fingers = [model.jnt_qposadr[model.joint(name).id]
                        for name in FINGER_JOINTS]
        self.extension = model.jnt_qposadr[model.joint('rg2_extension').id]
        self.seen = set()

    def update(self, data, names, positions):
        """Reject malformed packets atomically; ignore unrelated joints."""
        if len(names) != len(positions) or len(set(names)) != len(names):
            raise ValueError('JointState names/positions must match and be unique')
        updates = [(name, self.addresses[name], value)
                   for name, value in zip(names, positions)
                   if name in self.addresses]
        if any(not np.isfinite(value) for _, _, value in updates):
            raise ValueError('JointState contains non-finite arm positions')
        for name, address, value in updates:
            data.qpos[address] = value
            self.seen.add(name)

    def set_width(self, data, width_mm):
        """Apply measured opening in mm and interpolated axial fingertip reach."""
        if not np.isfinite(width_mm) or not 0 <= width_mm <= 110:
            raise ValueError('RG2 width must be finite and within 0..110 mm')
        data.qpos[self.fingers] = width_mm / 2000.
        data.qpos[self.extension] = (
            gripper_width_to_reach(width_mm * 10.) - MIN_FINGER_REACH)
