#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

class HardwareInterface(Node):
    def __init__(self):
        super().__init__("hardware_interface")

        # Suscribe a objetivos de junta
        self.joint_obj_sub = self.create_subscription(
            JointState, "/joint_hardware_objectives", self.hardware_obj_callback, 10
        )

        # Publica estados de junta
        self.joint_states_pub = self.create_publisher(
            JointState, "/joint_states", 10
        )

        # Orden “canónico” en todo el sistema
        self.joint_order = ["shoulder_joint", "arm_joint", "forearm_joint"]

        # Estado actual (inicial razonable para evitar NaN)
        self.current_joint_state = JointState()
        self.current_joint_state.name = self.joint_order[:]
        self.current_joint_state.position = [0.0, 0.0, 0.0]

        # Frecuencia de publicación 
        self.create_timer(0.1, self.joint_states_timer_callback)

    def hardware_obj_callback(self, msg: JointState):
        # Reordenar según self.joint_order respetando msg.name
        name_to_pos = {}
        if msg.name and len(msg.name) == len(msg.position):
            for n, p in zip(msg.name, msg.position):
                name_to_pos[n] = float(p)
        else:
            # Si no llegaron nombres, asumimos el mismo orden canónico
            for i, n in enumerate(self.joint_order):
                if i < len(msg.position):
                    name_to_pos[n] = float(msg.position[i])

        pos_out = []
        for n in self.joint_order:
            pos_out.append(name_to_pos.get(n, 0.0))

        self.current_joint_state = JointState()
        self.current_joint_state.header.stamp = self.get_clock().now().to_msg()
        self.current_joint_state.name = self.joint_order[:]
        self.current_joint_state.position = pos_out

    def joint_states_timer_callback(self):
        # Publicar el último estado 
        self.current_joint_state.header.stamp = self.get_clock().now().to_msg()
        self.joint_states_pub.publish(self.current_joint_state)

def main(args=None):
    try:
        rclpy.init(args=args)
        node = HardwareInterface()
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Node stopped")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
