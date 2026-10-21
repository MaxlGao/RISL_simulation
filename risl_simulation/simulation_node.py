import rclpy
from rclpy.node import Node


class SimulationNode(Node):
    def __init__(self):
        super().__init__('simulation_node')
        self.declare_parameter('backend', 'mujoco')
        self.declare_parameter('model_path', '')

        backend = self.get_parameter('backend').value
        self.get_logger().info(
            f'Simulation scaffold ready; selected backend: {backend}'
        )


def main(args=None):
    rclpy.init(args=args)
    node = SimulationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()