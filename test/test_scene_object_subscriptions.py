"""Check default scene subscriptions and tags first seen after startup."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from geometry_msgs.msg import PoseStamped
import numpy as np
from rclpy.node import Node
import yaml

from risl_simulation import simulation_node as simulation
from risl_simulation.recognized_objects import DEFAULT_SCENE_OBJECT_NAMES


def test_launch_config_keeps_all_jenga_defaults():
    config = Path(__file__).parents[1] / 'config' / 'simulation.yaml'
    params = yaml.safe_load(config.read_text())['simulation_node']['ros__parameters']
    assert tuple(params.get('object_names', DEFAULT_SCENE_OBJECT_NAMES)) == (
        'wrench', 'allen_key', *(f'jenga_{i}' for i in range(250)))


def test_default_node_subscribes_before_tags_appear(monkeypatch):
    # Keep real scene compilation, MuJoCo data, and callbacks; mock ROS transport.
    monkeypatch.setattr(Node, '__init__', lambda self, *args, **kwargs: None)
    params = {}

    def declare_parameter(self, name, default):
        params[name] = False if name == 'show_viewer' else default

    monkeypatch.setattr(simulation.SimulationNode, 'declare_parameter', declare_parameter)
    monkeypatch.setattr(simulation.SimulationNode, 'get_parameter',
                        lambda self, name: SimpleNamespace(value=params[name]))
    callbacks = {}

    def subscribe(self, message_type, topic, callback, qos):
        callbacks[topic] = callback
        return Mock()

    monkeypatch.setattr(simulation.SimulationNode, 'create_subscription', subscribe)
    monkeypatch.setattr(simulation.SimulationNode, 'create_timer', Mock())
    monkeypatch.setattr(simulation.SimulationNode, 'get_logger', Mock())
    clock = Mock()
    clock.now.return_value.nanoseconds = 1_000_000_000
    monkeypatch.setattr(simulation.SimulationNode, 'get_clock', lambda self: clock)
    monkeypatch.setattr(simulation, 'Buffer', Mock())
    monkeypatch.setattr(simulation, 'TransformListener', Mock())

    node = simulation.SimulationNode()
    assert set(node.object_subscriptions) == set(DEFAULT_SCENE_OBJECT_NAMES)
    assert len(node.objects.mocap_ids) == 252
    assert not node.objects.last_seen
    assert np.all(node.data.mocap_pos[:, 2] == -10.)
    for name in DEFAULT_SCENE_OBJECT_NAMES:
        assert f'/vision/object_poses/{name}' in callbacks

    # Deliver independent poses through subscriptions created before detection.
    for marker_id in (0, 6, 7, 249):
        msg = PoseStamped()
        msg.header.frame_id = 'base'
        msg.pose.position.x = marker_id * .001
        msg.pose.position.y = -.4
        msg.pose.position.z = .014
        msg.pose.orientation.w = 1.
        callbacks[f'/vision/object_poses/jenga_{marker_id}'](msg)
    node.refresh_viewer()
    for marker_id in (0, 6, 7, 249):
        geom = node.model.geom(f'object_geom_jenga_{marker_id}').id
        np.testing.assert_allclose(
            node.data.geom_xpos[geom], [marker_id * .001, -.4, .007])
    assert node.data.mocap_pos[node.objects.mocap_ids['jenga_8'], 2] == -10.

    clock.now.return_value.nanoseconds = 4_000_000_000
    node.refresh_viewer()
    assert not node.objects.last_seen
    assert np.all(node.data.mocap_pos[:, 2] == -10.)
