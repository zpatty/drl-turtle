import rclpy
from rclpy.node import Node
from rclpy.executors import Executor, MultiThreadedExecutor
from rcl_interfaces.msg import SetParametersResult
from turtle_interfaces.msg import TurtleTraj
import json
import sys
import os

submodule = os.path.expanduser("~") + "/drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware"
sys.path.append(submodule)

from std_msgs.msg import String
from std_msgs.msg import Float64MultiArray
from dynamixel_sdk import *                    # Uses Dynamixel SDK library
from ros2_ws.src.turtle_hardware.turtle_hardware.turtle_dynamixel.Dynamixel import *                        # Dynamixel motor class                                  
from ros2_ws.src.turtle_hardware.turtle_hardware.turtle_dynamixel.dyn_functions import *                    # Dynamixel support functions
from ros2_ws.src.turtle_hardware.turtle_hardware.turtle_dynamixel.turtle_controller import *                # Controller 
from ros2_ws.src.turtle_hardware.turtle_hardware.turtle_dynamixel.Constants import *                        # File of constant variables
from ros2_ws.src.turtle_hardware.turtle_hardware.turtle_dynamixel.Mod import *
from ros2_ws.src.turtle_hardware.turtle_hardware.turtle_dynamixel.utilities import *
import math
from math import cos, sin
from datetime import datetime
import numpy as np
from matplotlib import pyplot as plt
from ros2_ws.src.turtle_hardware.turtle_hardware.turtle_dynamixel.utilities import *
import json
import traceback
from queue import Queue
import serial



def main(args=None):
    # {"Acc":[  89.36,  29.30,  1024.90 ], "Gyr" :[  0.52, -0.55,  0.07 ], "Mag" :[ -54.45, -10.65,  48.30 ],"Voltage": [ 0.04 ]}
    xiao =  serial.Serial('/dev/ttyACM0', 115200, timeout=3)   
    sensors = xiao.readline()
    json_dict = json.loads(sensors.decode('utf-8'))
    print(json_dict)
    print(f"All them keys: {json_dict.keys()}")
    print(json_dict["Voltage"])
if __name__=='__main__':
    main()