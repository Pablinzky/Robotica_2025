from launch import LaunchDescription
from launch.substitutions import Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python import get_package_share_directory
import os

def generate_launch_description():
  #Rutas de paquete "example_description" y archivos
  description_path = get_package_share_directory("example_description")
  model_path = os.path.join(description_path, "urdf", "rrr_robot.urdf")
  rviz_conf_path = os.path.join(description_path, "rviz", "rviz_config.rviz")
  #Modelo URDF como parámetro (con espacio después de 'xacro ')
  robot_description = {
    "robot_description": ParameterValue(
      Command(["xacro ", model_path]),
      value_type=str
    )
  }

  controller_manager_node = Node(
    package='example_control',
    executable="controller_manager"
  )

  manipulator_controller_node = Node(
    package='example_control',
    executable="manipulator_controller"
  )

  hardware_interface_node = Node(
    package='example_control',
    executable="hardware_interface"
  )

  rviz_node = Node(
    package='rviz2',
    executable="rviz2",
    arguments=["-d", rviz_conf_path]
  )
  jsp_node = Node(
    package='joint_state_publisher_gui',
    executable="joint_state_publisher_gui"
  )
  rsp_node = Node(
    package='robot_state_publisher',
    executable="robot_state_publisher",
    parameters=[robot_description]
  )
  return LaunchDescription([
    controller_manager_node,
    manipulator_controller_node,
    hardware_interface_node,
    rviz_node, 
    rsp_node
    ])
