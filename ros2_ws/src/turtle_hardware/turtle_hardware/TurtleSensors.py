import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import rclpy.parameter

from rclpy.parameter_event_handler import ParameterEventHandler

import os, sys
submodule = os.path.expanduser("~") + "/drl-turtle/ros2_ws/install/turtle_hardware/lib/python3.12/site-packages/turtle_hardware/"
sys.path.append(submodule)
import transforms3d.quaternions as quat
from turtle_interfaces.msg import TurtleTraj, TurtleSensors, TurtleCtrl, TurtleMode, TurtleState
from std_msgs.msg import String, Float32MultiArray
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
import numpy as np
import serial
import time
import json



class TurtleSensorsNode(Node):

# class TurtleRobot(Node, gym.Env):
    """
    This node is responsible for continously reading sensor data and receiving commands from the keyboard node
    to execute specific trajectoreies or handle emergency stops. It also is responsible for sending motor pos commands to the RL node
    for training and deployment purposes.
    TODO: migrate dynamixel motors into this class
    TLDR; this is the node that handles all turtle hardware things
    """

    def __init__(self, params=None):
        super().__init__('turtle_sensors_node')
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
        timer_cb_group = None
        self.call_timer = self.create_timer(0.005, self._timer_cb, callback_group=timer_cb_group)


        # continously reads from all the sensors and publishes data at end of trajectory
        self.sensors_pub = self.create_publisher(
            TurtleSensors,
            'turtle_imu',
            qos_profile
        )

        self.mode_sub = self.create_subscription(
            TurtleMode,
            'turtle_mode',
            self.turtle_mode_callback,
            qos_profile)

        self.create_rate(1000)
        self.voltage = 12.0
        self.depth = 0.0

        self.acc = np.zeros((3,1))
        self.gyr = np.zeros((3,1))
        self.quat_vec = np.zeros((4,1))
        self.off_flag = False

        self.xiao = serial.Serial('/dev/ttyACM0', 115200, timeout=3)   
        self.xiao.reset_input_buffer()        

        self.print_sensors = True

        # t = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        # self.folder_name =  "data/" + t
        # os.makedirs(self.folder_name)
        


    def _timer_cb(self):
        # print(self.mode)
        # t_meas = time.time()
        self.read_sensors()
        # print(self.mode)
        # print(f"[DEBUG] dt_read: {time.time() - t_meas}\n")

        self.publish_turtle_sensors()


    def np2msg(self, mat):
        """
        flattens nd numpy array into a lst for ros pkg messaging
        """
        nq = 10
        squeezed = np.reshape(mat, (nq * mat.shape[1]))
        return squeezed.tolist()
    
    def publish_turtle_sensors(self):
        """
        Send data out
        """
        turtle_msg = TurtleSensors()
        # turtle_msg.q = self.q
        # turtle_msg.dq = self.dq
        turtle_msg.imu.quat = self.quat_vec
        # # angular velocity
        turtle_msg.imu.gyr = self.gyr
        # print("acc msg")
        # # linear acceleration
        turtle_msg.imu.acc = self.acc
        turtle_msg.voltage = self.voltage
        turtle_msg.depth = self.depth
        # publish msg 
        self.sensors_pub.publish(turtle_msg)
    
    def turtle_mode_callback(self, msg):
        if msg.mode == "kill":
            raise KeyboardInterrupt
        
    
    def read_sensors(self):
        """
        Appends current sensor reading to the turtle node's sensor data structs
        """        
        no_check = False
        keep_trying = True
        attempts = 0
        while keep_trying:
            if attempts >= 1:
                # print("adding place holder")
                # self.add_place_holder()
                break
            self.xiao.reset_input_buffer()
            sensors = self.xiao.readline()
            # self.get_logger().info(sensors)
            # make sure it's a valid byte that's read
            if len(sensors) != 0:
                # this ensures the right json string format
                if sensors[0] == 32 and sensors[-1] == 10:
                    try:
                        sensor_dict = json.loads(sensors.decode('utf-8'))
                    except:
                        no_check = True
                    # add sensor data
                    if no_check == False:

                        sensor_keys = ('Acc', 'Gyr', 'Quat', 'Voltage', 'Depth')
                        if set(sensor_keys).issubset(sensor_dict):
                            a = np.array(sensor_dict['Acc']).reshape((3,1)) * 9.81
                            self.gyr = np.array(sensor_dict['Gyr']).reshape((3,1))
                            q = np.array(sensor_dict["Quat"])
                            R = quat.quat2mat(q)
                            g_w = np.array([[0], [0], [9.81]])
                            g_local = np.dot(R.T, g_w)
                            self.acc = a - g_local           # acc without gravity 
                            self.quat_vec = q.reshape((4,1))
                            self.voltage = sensor_dict['Voltage'][0]
                            self.depth = sensor_dict['Depth'][0]

                            keep_trying = False
            attempts += 1

def main():
    rclpy.init()
    sensor_node = TurtleSensorsNode()
    try:
        rclpy.spin(sensor_node)
        # rclpy.shutdown()
    except KeyboardInterrupt:
        print("shutdown")
    except Exception as e:
        print("some error occurred")
        # turtle_node.shutdown_motors()
        exec_type, obj, tb = sys.exc_info()
        fname = os.path.split(tb.tb_frame.f_code.co_filename)[1]
        print(exec_type, fname, tb.tb_lineno)
        print(e)
    # turtle_node.save_data()

if __name__ == '__main__':
    main()