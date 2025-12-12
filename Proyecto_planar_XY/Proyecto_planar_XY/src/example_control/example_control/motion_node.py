#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist, PointStamped

from .geom_model import RobotKinematics
from .actuator_model import RobotDynamics

class ManipulatorController(Node):
  def __init__(self):
    super().__init__("manipulator_controller")

    self.robot_kinematics = RobotKinematics()
    self.robot_kinematics.redirect_print(self.get_logger().info)
    self.robot_kinematics.direct_kinematics()

    self.robot_dynamics = RobotDynamics()
    self.robot_dynamics.define_kinematics(self.robot_kinematics)
    self.robot_dynamics.define_dynamics()

    self.moving = False

    reliable_qos = QoSProfile(
      reliability=ReliabilityPolicy.RELIABLE,
      history=HistoryPolicy.KEEP_LAST,
      depth=10
    )

    self.end_effector_goal_subscriber = self.create_subscription(
      Twist, "/end_effector_goal", self.end_effector_callback, reliable_qos
    )
    self.clicked_point_subscriber = self.create_subscription(
      PointStamped, "/clicked_point", self.clicked_point_callback, reliable_qos
    )
    self.joint_states_subscriber = self.create_subscription(
      JointState, "/joint_states", self.joint_states_callback, reliable_qos
    )

    self.joint_goals_publisher = self.create_publisher(
      JointState, "/joint_goals", reliable_qos
    )

    self.current_joint_states = JointState()
    self.current_joint_states.name = ["shoulder_joint", "arm_joint", "forearm_joint"]
    self.current_joint_states.position = [0.0, 0.0, 0.0]

    self.get_logger().info("Controlador inicializado ...")

  def end_effector_callback(self, msg:Twist):
    if self.moving:
      self.get_logger().warning("Trayectoria en progreso ...")
      return
    self.moving = True
    self.get_logger().info("Punto recibido")

    x_goal  = float(msg.linear.x)
    y_goal  = float(msg.linear.y)
    th_goal = float(msg.angular.z)

    self.robot_kinematics.trajectory_generator(
      self.current_joint_states.position, [x_goal, y_goal, th_goal], 3
    )
    self.robot_kinematics.inverse_kinematics()

    self.robot_dynamics.lagrange_effort_generator()

    # Publicación periódica ANTES de abrir ventanas (para no bloquear)
    self.count = 0
    self.joint_goals = JointState()
    self.joint_goals.name = ["shoulder_joint", "arm_joint", "forearm_joint"]
    self.get_logger().info("Publicando trayectoria de las juntas")
    self.position_publisher_timer = self.create_timer(
      float(self.robot_kinematics.dt), self.trayectory_publisher_callback
    )

    # Ventanas UNA POR UNA (cada función llama plt.show())
    self.robot_kinematics.ws_graph()
    self.robot_kinematics.q_graph()
    self.robot_dynamics.effort_graph()

  def clicked_point_callback(self, msg:PointStamped):
    if self.moving:
      self.get_logger().warning("Trayectoria en progreso...")
      return
    self.moving = True
    self.get_logger().info("Punto objetivo ")

    x_goal = float(msg.point.x)
    y_goal = float(msg.point.y)
    th_goal = 0.0

    self.robot_kinematics.trajectory_generator(
      self.current_joint_states.position, [x_goal, y_goal, th_goal], 3
    )
    self.robot_kinematics.inverse_kinematics()

    self.robot_dynamics.lagrange_effort_generator()

    self.count = 0
    self.joint_goals = JointState()
    self.joint_goals.name = ["shoulder_joint", "arm_joint", "forearm_joint"]
    self.get_logger().info("Publicando trayectoria ")
    self.position_publisher_timer = self.create_timer(
      float(self.robot_kinematics.dt), self.trayectory_publisher_callback
    )

    # Ventanas UNA POR UNA
    self.robot_kinematics.ws_graph()
    self.robot_kinematics.q_graph()
    self.robot_dynamics.effort_graph()

  def trayectory_publisher_callback(self):
    self.joint_goals.header.stamp = self.get_clock().now().to_msg()

    th1 = float(self.robot_kinematics.q_m[0, self.count])
    th2 = float(self.robot_kinematics.q_m[1, self.count])
    th3 = float(self.robot_kinematics.q_m[2, self.count])

    self.joint_goals.position = [th1, th2, th3]
    self.joint_goals_publisher.publish(self.joint_goals)

    self.count += 1
    if self.count >= len(self.robot_kinematics.q_m[0, :]):
      self.count = 0
      self.position_publisher_timer.cancel()
      self.get_logger().info("Trayectoria lista ")
      self.moving = False

  def joint_states_callback(self, msg:JointState):
    self.current_joint_states = msg

def main(args=None):
  try:
    rclpy.init(args=args)
    node = ManipulatorController()
    node.get_logger().info("✅ Listo: publica un punto en /clicked_point para mover el robot y graficar.")
    rclpy.spin(node)
  except KeyboardInterrupt:
    print("Node stopped")
  finally:
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
  main()
