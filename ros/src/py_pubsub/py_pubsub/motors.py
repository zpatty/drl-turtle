import rclpy
from rclpy.node import Node
from rclpy.executors import Executor, MultiThreadedExecutor

from std_msgs.msg import String
from std_msgs.msg import Float64MultiArray


class MinimalSubscriber(Node):

    def __init__(self, topic):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            String,
            topic,
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info('I heard:' + msg.data)

class DynamicSubscriber(Node):

    def __init__(self, topic):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            String,
            topic,
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        if msg.data == 'd1':
            log = 'Running Swimming Trajectory'
        elif msg.data == 'd2':
            log = 'Running Walking Trajectory'
        elif msg.data == 'stop':
            log = 'stopping'
        else: 
            log = 'I heard:' + msg.data
        self.get_logger().info(log)

def main(args=None):
    rclpy.init(args=args)

    from_cv = MinimalSubscriber('cv_motors')
    from_master = DynamicSubscriber('master_motors')

    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(from_cv)
    executor.add_node(from_master)

    executor.spin()
    
    rclpy.shutdown()


if __name__ == '__main__':
    main()
