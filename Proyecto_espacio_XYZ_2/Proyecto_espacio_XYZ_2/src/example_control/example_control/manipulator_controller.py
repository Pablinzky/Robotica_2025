#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist, PointStamped
import numpy as np

from .kinematics import RobotKinematics
from .dynamics import RobotDynamics

GROUND_Z = 0.0    
TOUCH_EPS = 0.0    
DO_ADAPTIVE_Z = True  

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

    self.joint_states_warmup_pub = self.create_publisher(
      JointState, "/joint_states", reliable_qos
    )


    self.current_joint_states = JointState()
    self.current_joint_states.name = ["shoulder_joint", "arm_joint", "forearm_joint"]
    self.current_joint_states.position = [0.0, 1.1, -2.1]


    self._warmup_ticks = 0
    def _warmup_pub():
      self.current_joint_states.header.stamp = self.get_clock().now().to_msg()
      self.joint_states_warmup_pub.publish(self.current_joint_states)
      self.joint_goals_publisher.publish(self.current_joint_states)
      self._warmup_ticks += 1
      if self._warmup_ticks > 15:
        self._warmup_timer.cancel()
    self._warmup_timer = self.create_timer(0.033, _warmup_pub)

    self.get_logger().info("Controlador inicializado.")

  def _plan_and_prepare_publish(self, q_start, xyz_goal, duration=3.0, adaptive_z=True, peg_to_ground=False):
    x_goal, y_goal, z_goal = xyz_goal

    if peg_to_ground:
      z_goal = GROUND_Z - TOUCH_EPS

    self.robot_kinematics.trajectory_generator(q_start, [x_goal, y_goal, z_goal], duration)
    self.robot_kinematics.inverse_kinematics()

    if adaptive_z:
      q_end = [
        float(self.robot_kinematics.q_m[0, -1]),
        float(self.robot_kinematics.q_m[1, -1]),
        float(self.robot_kinematics.q_m[2, -1]),
      ]
      xyz_est = np.array(self.robot_kinematics.fk_pos(q_end[0], q_end[1], q_end[2]), dtype=float).reshape(3,)
      z_ref = (GROUND_Z - TOUCH_EPS) if peg_to_ground else z_goal
      z_err = float(xyz_est[2] - z_ref)

      if abs(z_err) > 1e-3:
        z_goal_2 = z_goal - z_err
        self.get_logger().info(f"Corrección adaptativa de Z: err={z_err:+.4f} -> nuevo z_goal={z_goal_2:.4f}")
        self.robot_kinematics.trajectory_generator(q_start, [x_goal, y_goal, z_goal_2], duration)
        self.robot_kinematics.inverse_kinematics()

    self.robot_dynamics.lagrange_effort_generator()

    self.count = 0
    self.joint_goals = JointState()
    self.joint_goals.name = ["shoulder_joint", "arm_joint", "forearm_joint"]
    self.position_publisher_timer = self.create_timer(
      float(self.robot_kinematics.dt), self.trayectory_publisher_callback
    )

    # Gráficas
    self.robot_kinematics.ws_graph()
    self.robot_kinematics.q_graph()
    self.robot_dynamics.effort_graph()

  def end_effector_callback(self, msg:Twist):
    if self.moving:
      self.get_logger().warning("Trayectoria en progreso. Mensaje rechazado")
      return
    self.moving = True
    self.get_logger().info("Punto objetivo recibido (Twist, 3D)")

    xyz_goal = (float(msg.linear.x), float(msg.linear.y), float(msg.linear.z))
    self._plan_and_prepare_publish(self.current_joint_states.position, xyz_goal,
                                   duration=3.0, adaptive_z=DO_ADAPTIVE_Z, peg_to_ground=False)
    self.get_logger().info("Publicando trayectoria de las juntas")

  def clicked_point_callback(self, msg:PointStamped):
    if self.moving:
      self.get_logger().warning("Trayectoria en progreso. Mensaje rechazado")
      return
    self.moving = True
    self.get_logger().info(f"Punto objetivo clickeado (RViz): frame='{msg.header.frame_id}'")

    xyz_goal = (float(msg.point.x), float(msg.point.y), float(msg.point.z))
    self._plan_and_prepare_publish(self.current_joint_states.position, xyz_goal,
                                   duration=3.0, adaptive_z=DO_ADAPTIVE_Z, peg_to_ground=True)
    self.get_logger().info("Publicando trayectoria de las juntas")

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
      self.get_logger().info("Trayectoria finalizada")
      self.moving = False

  def joint_states_callback(self, msg:JointState):
    self.current_joint_states = msg

def main(args=None):
  try:
    rclpy.init(args=args)
    node = ManipulatorController()
    rclpy.spin(node)
  except KeyboardInterrupt:
    print("Node stopped")
  finally:
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
  main()
