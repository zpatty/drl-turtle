import rclpy
from rclpy.node import Node
from rclpy.executors import Executor, MultiThreadedExecutor

from std_msgs.msg import String
from std_msgs.msg import Float64MultiArray


class MinimalPublisher(Node):

    def __init__(self, topic):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, topic, 10)
        timer_period = 1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0.

    def timer_callback(self):
        msg = String()
        msg.data = "test " + str(self.i)
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: ' +msg.data)
        self.i += 1

def main(args=None):
    rclpy.init(args=args)

    tomotors = MinimalPublisher('master_motors')
    tocv = MinimalPublisher('master_cv')

    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(tomotors)
    executor.add_node(tocv)
    
    executor.spin()
    
    rclpy.shutdown()


if __name__ == '__main__':
    main()
