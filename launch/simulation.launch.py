from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from pathlib import Path


def generate_launch_description():
    package_share = Path(
        get_package_share_directory('risl_simulation')
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'params_file', default_value=str(
                package_share / 'config' / 'simulation.yaml')),
        Node(
            package='risl_simulation',
            executable='simulation_node',
            name='simulation_node',
            output='screen',
            parameters=[
                LaunchConfiguration('params_file')
            ],
        )
    ])
