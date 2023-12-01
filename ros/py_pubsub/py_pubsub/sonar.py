import serial
import rclpy
from rclpy.node import Node
from rclpy.executors import Executor, MultiThreadedExecutor

from std_msgs.msg import String
from std_msgs.msg import Float64MultiArray
# from serial.tools import list_ports
# port = list(list_ports.comports())
# for p in port:
#     print(p.device)

xiao = serial.Serial('/dev/ttyACM0', 115200, timeout=3) # instantiates the serial object
# the below will happen in a (while) loop

class MinimalPublisher(Node):


    def __init__(self, topic):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, topic, 10)
        timer_period = 1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        string_to_parse = xiao.readline() # reads from the port
        msg = String()
        msg.data = string_to_parse
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: ' +msg.data)
        self.i += 1

def main(args=None):
    rclpy.init(args=args)

    toplan = MinimalPublisher('sonar_plan')

    rclpy.spin(toplan)

    rclpy.shutdown()


if __name__ == '__main__':
    main()
