import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from turtle_interfaces.msg import TurtleTraj, TurtleSensors
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from continuous_primitive import *
import time

class TrajPub(Node):

# class TurtleRobot(Node, gym.Env):
    """
    This node is responsible for continously reading sensor data and receiving commands from the keyboard node
    to execute specific trajectoreies or handle emergency stops. It also is responsible for sending motor pos commands to the RL node
    for training and deployment purposes.
    TODO: migrate dynamixel motors into this class
    TLDR; this is the node that handles all turtle hardware things
    """

    def __init__(self):
        super().__init__('traj_pub_node')
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=2
        )
        buff_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.traj_pub = self.create_publisher(
            TurtleTraj,
            'turtle_traj',
            qos_profile
        )

        self.motor_state_sub = self.create_subscription(
            TurtleSensors,
            'turtle_sensors',
            self.turtle_state_callback,
            qos_profile
        )

        self.key_subscription = self.create_subscription(
            String,
            'turtle_mode_cmd',
            self.keyboard_callback,
            qos_profile,
            )
        self.flag = 'go'
        self.q = [0]*10
        
    def turtle_state_callback(self, msg):
        # self.q = convert_motors_to_q(np.array(msg.q), np.array(msg.q))
        print(msg)

    def keyboard_callback(self, msg):
        """
        Callback function that updates the mode the turtle should be in.
        This method is what enables us to set "emergency stops" mid-trajectory. 
        """
        # global mode
        if msg.data == 'stop':
            self.flag = 'stop'


def main(args=None):
    rclpy.init(args=args)
    print("yo")
    traj = TurtleTraj()
    trajectory_publisher = TrajPub()
    t_0 = time.time()
    while(trajectory_publisher.flag != 'stop'):
        rclpy.spin_once(trajectory_publisher)
        t = time.time()
        ud = task_space_control_fn_learned(t - t_0, trajectory_publisher.q[:6])
        traj.u = ud
        traj.t = t - t_0
        trajectory_publisher.traj_pub(traj)
    trajectory_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
  main()