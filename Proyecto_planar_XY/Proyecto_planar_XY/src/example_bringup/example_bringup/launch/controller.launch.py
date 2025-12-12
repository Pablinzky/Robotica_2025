#!/usr/bin/env python3
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    desc_share = get_package_share_directory('prueba_descriptios')
    urdf_path = os.path.join(desc_share, 'urdf', 'URDF_PLANO_XY.urdf')
    rviz_path = os.path.join(desc_share, 'rviz', 'rviz_config.rviz')

    return LaunchDescription([
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            parameters=[{'robot_description': open(urdf_path).read()}],
            output='screen',
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_path],
            output='screen',
        ),
        Node(
            package='example_control',
            executable='controller_manager',
            name='controller_manager',
            output='screen',
        ),
        Node(
            package='example_control',
            executable='hardware_interface',
            name='hardware_interface',
            output='screen',
        ),
        Node(
            package='example_control',
            executable='manipulator_controller',
            name='manipulator_controller',
            output='screen',
        ),
    ])
