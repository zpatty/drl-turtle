import rclpy
from rclpy.node import Node
from rclpy.executors import Executor, MultiThreadedExecutor
from rcl_interfaces.msg import SetParametersResult
from turtle_interfaces.msg import TurtleTraj, TurtleSensors, TurtleMotorPos, TurtleIMU
import sys
import os

submodule = os.path.expanduser("~") + "/drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware"
sys.path.append(submodule)

from std_msgs.msg import String
from std_msgs.msg import Float64MultiArray
from dynamixel_sdk import *                    # Uses Dynamixel SDK library
from turtle_dynamixel.Dynamixel import *                        # Dynamixel motor class                                  
from turtle_dynamixel.dyn_functions import *                    # Dynamixel support functions
from turtle_dynamixel.turtle_controller import *                # Controller 
from turtle_dynamixel.Constants import *                        # File of constant variables
from turtle_dynamixel.Mod import *
from turtle_dynamixel.utilities import *
import math
from math import cos, sin
from datetime import datetime
import numpy as np
from matplotlib import pyplot as plt
from turtle_dynamixel.utilities import *
import json
import traceback
from queue import Queue
import serial

# global variable was set in the callback function directly
global mode
mode = 'rest'


q = np.array([1,2,3,4,5,6,7,9,10]).reshape((9,1))
print(q)
print(q.shape)
q = np.insert(q, 7, 8).reshape((10,1))
print(q)
print(q.shape)


q = np.delete(q, [7]).reshape((9,1))
print(f"took out 8th index for turtle: {q.reshape(9,1)}")
print(f"new shape: {q.shape}")
lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
print(lst)
del lst[7]
print(lst)
def np2msg(mat):
    nq = 10
    squeezed = np.reshape(mat, (nq * mat.shape[1]))
    return list(squeezed)

qd_mat = mat2np('/home/ranger/drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware/turtle_trajectory/qd.mat', 'qd')
qd = np2msg(qd_mat)
msg = TurtleSensors()
msg.qd = qd
traj = TurtleTraj()

tvec = mat2np('/home/ranger/drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware/turtle_trajectory/tvec.mat', 'tvec')
traj.tvec = tvec.tolist()[0]
print(list(traj.tvec))
print(len(list(traj.tvec)))


quat_data = np.zeros((4,1))
print(quat_data)
print(f"quat data: {quat_data.shape}")
test = np.array([1,2,3,4]).reshape((4,1))
quat_data = np.append(quat_data, test, axis=1)
print(f"new quat data to package: {quat_data}")
# print(quat_data)


turtle_msg = TurtleSensors()

# extract and flatten numpy arrays
quat_x = quat_data[0, :]
quat_y = quat_data[1, :]
quat_z = quat_data[2, :]
quat_w = quat_data[3, :]

turtle_msg.imu.quat_x = quat_x.tolist()
turtle_msg.imu.quat_y = quat_y.tolist()
turtle_msg.imu.quat_z = quat_z.tolist()
turtle_msg.imu.quat_w = quat_w.tolist()

quat_x = turtle_msg.imu.quat_x
quat_y = turtle_msg.imu.quat_y
quat_z = turtle_msg.imu.quat_z
quat_w = turtle_msg.imu.quat_w

quat_data = np.array([quat_x, quat_y, quat_z, quat_w])
print(f"quat data from opening msg: {quat_data}")
