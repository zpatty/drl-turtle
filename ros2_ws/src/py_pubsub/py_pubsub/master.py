import rclpy
from rclpy.node import Node
from rclpy.executors import Executor, MultiThreadedExecutor

from std_msgs.msg import String
from std_msgs.msg import Float64MultiArray

"""
to send one-time messages to motors: 'ros2 topic pub -1 master_motors std_msgs/msg/String "data: Hello world"'
replace 'Hello world' with 'd1', 'd2', or 'stop' for demo1, demo2, or stop
"""

class MinimalPublisher(Node):

    def __init__(self, topic):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, topic, 10)
        timer_period = 5  # seconds
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

    #tomotors = DynamicPublisher('master_motors')
    tocv = MinimalPublisher('master_cv')

    # executor = MultiThreadedExecutor(num_threads=2)
    # executor.add_node(tomotors)
    # executor.add_node(tocv)
    
    # executor.spin() 
    rclpy.spin(tocv)

    rclpy.shutdown()


if __name__ == '__main__':
    main()
