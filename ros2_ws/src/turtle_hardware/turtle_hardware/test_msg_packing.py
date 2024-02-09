from __future__ import print_function
import os
import sys
import json
import traceback

submodule = os.path.expanduser("~") + "/drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware"
sys.path.append(submodule)

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import Executor, MultiThreadedExecutor
import numpy as np
from matplotlib import pyplot as plt
from turtle_dynamixel.dyn_functions import *                    # Dynamixel support functions


from std_msgs.msg import String
from std_msgs.msg import Float64MultiArray

def np2msg(mat):
    nq = 10
    squeezed = np.reshape(mat, (nq * mat.shape[1]))
    return list(squeezed)

def main(args=None):
    qd_mat = mat2np('/home/ranger/drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware/turtle_trajectory/qd.mat', 'qd')
    dqd_mat = mat2np('/home/ranger/drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware/turtle_trajectory/dqd.mat', 'dqd')
    ddqd_mat = mat2np('/home/ranger/drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware/turtle_trajectory/ddqd.mat', 'ddqd')
    tvec = mat2np('/home/ranger/drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware/turtle_trajectory/tvec.mat', 'tvec')

    print(f"qd_mat shape: {qd_mat.shape}")
    motor1 = qd_mat[0, :]
    print(f"motor1 shape: {motor1.shape}")
    print(f"tvec mat shape: {tvec.shape}\n")
    qd = np2msg(qd_mat)
    dqd = np2msg(dqd_mat)
    ddqd = np2msg(ddqd_mat)
    # print(f"t vec list: {tvec.tolist()}")
    motor1 = motor1.tolist()
    other = tvec.tolist()
    tvec = tvec.tolist()[0]
    # print(f"new tvec: {tvec}")
    print(f"new motor 1: {len(motor1)}")

    readq = np.array(qd).reshape(10, len(tvec))
if __name__ == '__main__':
    main()
