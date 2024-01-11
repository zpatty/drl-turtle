import rclpy
from rclpy.node import Node
from rclpy.executors import Executor, MultiThreadedExecutor

from std_msgs.msg import String
from std_msgs.msg import Float64MultiArray


class MinimalPublisher(Node):

    def __init__(self, topic):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, topic, 10)
        timer_period = 7  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0.

    def timer_callback(self):
        msg = String()
        msg.data = "test " + str(self.i)
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: ' +msg.data)
        self.i += 1

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
        self.get_logger().info('I heard: ' +msg.data)


def main(args=None):
    rclpy.init(args=args)

    fromcv = MinimalSubscriber('cv_planner')
    fromsonar = MinimalSubscriber('sonar_planner')
    tomotors = MinimalPublisher('planner_motors')

    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(fromcv)
    executor.add_node(fromsonar)
    executor.add_node(tomotors)
    
    executor.spin()
    
    rclpy.shutdown()


if __name__ == '__main__':
    main()
