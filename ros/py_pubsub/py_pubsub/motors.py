import rclpy
from rclpy.node import Node
from rclpy.executors import Executor, MultiThreadedExecutor

from std_msgs.msg import String
from std_msgs.msg import Float64MultiArray
from dynamixel_sdk import *                    # Uses Dynamixel SDK library
from Dynamixel import *                        # Dynamixel motor class                                  
from dyn_functions import *                    # Dynamixel support functions
from turtle_controller import *                # Controller 
from Constants import *                        # File of constant variables
from Mod import *
from utilities import *
import math
from math import cos, sin
from datetime import datetime
import numpy as np
from matplotlib import pyplot as plt
from utilities import *
import json
import traceback
from queue import Queue
import serial
import socket



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
        # set up Dynamxiel ish
        if portHandlerJoint.openPort():
            log = "Succeeded to open the port"
        else:
            log = "Failed to open the port"
        self.get_logger().info(log)
        # Set port baudrate
        if portHandlerJoint.setBaudRate(BAUDRATE):
            log = "Succeeded to change the baudrate"
        else:
            log = "Failed to change the baudrate"
        self.packetHandlerJoint = PacketHandler(PROTOCOL_VERSION)
        self.get_logger().info(log)
        # Instantiate the motors
        IDs = [1]
        self.nq = len(IDs)
        self.Joints = Mod(packetHandlerJoint, portHandlerJoint, IDs)
        self.Joints.disable_torque()
        self.Joints.set_current_cntrl_mode()
        self.Joints.enable_torque()
        self.q = np.array(self.Joints.get_position()).reshape(-1,1)
        log = "Our initial q: " + str(self.q)
        self.get_logger().info(log)

    def listener_callback(self, msg):
        if msg.data == 'd1':
            self.Joints.send_torque_cmd(self.nq * [20])
            log = 'Running Swimming Trajectory'
        elif msg.data == 'd2':
            log = 'Running Walking Trajectory'
        elif msg.data == 'stop':
            self.Joints.disable_torque()
            log = 'stopping'
        else: 
            log = 'I heard:' + msg.data
        self.get_logger().info(log)

def main(args=None):
    rclpy.init(args=args)

    from_planner = MinimalSubscriber('planner_motors')
    from_master = DynamicSubscriber('master_motors')

    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(from_planner)
    executor.add_node(from_master)

    executor.spin()
    
    rclpy.shutdown()


if __name__ == '__main__':
    main()
