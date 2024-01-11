#!/usr/bin/env python
import rclpy
from rclpy.node import Node
from rclpy.executors import Executor, MultiThreadedExecutor
from rcl_interfaces.msg import SetParametersResult
import sys
sys.path.append("/home/crush/drl-turtle/ros/py_pubsub/py_pubsub")
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

class SeaTurtleControl(Node):
    """
    Create a turtle control object
    """
    def __init__(self):
        """
        Initialize controller 
        """

    def __turtle_pos_callback(self, msg):
        pass
    def __turtle_traj_mode_callback(self, msg):
        pass
    