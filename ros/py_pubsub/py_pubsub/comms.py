import rclpy
from rclpy.node import Node
from rclpy.executors import Executor, MultiThreadedExecutor

from std_msgs.msg import String
from std_msgs.msg import Float64MultiArray

class MinimalSubscriber(Node):

    def __init__(self, topic):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            Float64MultiArray,
            topic,
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info('I heard: ' +str(msg.data))


def main(args=None):
    rclpy.init(args=args)

    fromcontrol = MinimalSubscriber('control_comms')
    fromsensors = MinimalSubscriber('sensors_comms')
    frommotor = MinimalSubscriber('motors_comms')
    fromplanner = MinimalSubscriber('planner_comms')

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(fromcontrol)
    executor.add_node(frommotor)
    executor.add_node(fromsensors)
    executor.add_node(fromplanner)
    
    executor.spin()
    
    rclpy.shutdown()


if __name__ == '__main__':
    main()
