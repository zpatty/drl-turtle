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
        self.get_logger().info('I heard:' +str(msg.data))

class MinimalPublisher(Node):

    def __init__(self, topic):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(Float64MultiArray, topic, 10)
        timer_period = 1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0.

    def timer_callback(self):
        msg = Float64MultiArray()
        msg.data = [self.i for j in range(16)]
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: ' +str(msg.data))
        self.i += 1

def main(args=None):
    rclpy.init(args=args)

    fromsensor = MinimalSubscriber('sensors_planner')
    tocontrol = MinimalPublisher('planner_control')
    fromcamera = MinimalSubscriber('camera_planner')
    tocomms = MinimalPublisher('planner_comms')

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(tocontrol)
    executor.add_node(fromsensor)
    executor.add_node(fromcamera)
    executor.add_node(tocomms)

    executor.spin()
    
    rclpy.shutdown()


if __name__ == '__main__':
    main()
