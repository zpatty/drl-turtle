from __future__ import print_function
import os
import sys
import json
import traceback

submodule = os.path.expanduser("~") + "/drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware"
sys.path.append(submodule)

import rclpy
from rclpy.node import Node
from rclpy.executors import Executor, MultiThreadedExecutor
import numpy as np
from matplotlib import pyplot as plt
from ros2_ws.src.turtle_hardware.turtle_hardware.turtle_dynamixel.dyn_functions import *                    # Dynamixel support functions

from turtle_interfaces.msg import TurtleTraj

from std_msgs.msg import String
from std_msgs.msg import Float64MultiArray

import turtle_trajectory

ESC_ASCII_VALUE             = 0x1b
SPACE_ASCII_VALUE           = 0x20
WKEY_ASCII_VALUE            = 0x77
SKEY_ASCII_VALUE            = 0x73
AKEY_ASCII_VALUE            = 0x61
DKEY_ASCII_VALUE            = 0x64
CKEY_ASCII_VALUE            = 0x63
BKEY_ASCII_VALUE            = 0x62      # key to bend the top module
UKEY_ASCII_VALUE            = 0x75      # key to unbend the modules
NKEY_ASCII_VALUE            = 0x6E
IKEY_ASCII_VALUE            = 0x69     
QKEY_ASCII_VALUE            = 0x71 
TKEY_ASCII_VALUE            = 0x74   
RKEY_ASCII_VALUE            = 0x72     
MOD1_VALUE                  = 0x31      # pressing 1 on keyboard
MOD2_VALUE                  = 0x32
MOD3_VALUE                  = 0x33

import termios, fcntl, sys, os
from select import select
fd = sys.stdin.fileno()
old_term = termios.tcgetattr(fd)
new_term = termios.tcgetattr(fd)

global rest_received
global stop_received

rest_received = False
stop_received = False

def getch():
    new_term[3] = (new_term[3] & ~termios.ICANON & ~termios.ECHO)
    termios.tcsetattr(fd, termios.TCSANOW, new_term)
    try:
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_term)
    return ch

def kbhit():
    new_term[3] = (new_term[3] & ~(termios.ICANON | termios.ECHO))
    termios.tcsetattr(fd, termios.TCSANOW, new_term)
    try:
        dr,dw,de = select([sys.stdin], [], [], 0)
        if dr != []:
            return 1
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_term)
        sys.stdout.flush()

    return 0
    
def turtle_state_callback(msg):
    global stop_received
    global rest_received
    print("MESSAGE RECEIVED")
    if msg.data == "rest_received":
        rest_received = True
    elif msg.data == "stop_received":
        stop_received = True
        print("changing val!!!!!!!!!")
        print(f"stop received is: {stop_received}\n")
    else:
        print("NOPE")

def main(args=None):
    rclpy.init(args=args)
    global stop_received
    global rest_received
    node = rclpy.create_node('keyboard_node')
    tomotors = node.create_publisher(String, 'turtle_mode_cmd', 10)
    traj_pub = node.create_publisher(TurtleTraj, 'turtle_traj', 10)
    stop_sub = node.create_subscription(String, 'turtle_state', turtle_state_callback, 10)
    rate = node.create_rate(50)
    msg = String()
    traj = TurtleTraj()
    while rclpy.ok():
        print("\nT: Traj1, W: Traj2, R: Rest D: Custom Traj(or press SPACE to STOP!)")
        key_input = getch()
        if key_input == chr(SPACE_ASCII_VALUE):
            stop_received = False
            msg.data='stop'
            tomotors.publish(msg)
            while stop_received == False:
                rclpy.spin_once(node)
                msg.data='stop'
                tomotors.publish(msg)
                node.get_logger().info(msg.data)
            log = "Stop was sucessfully sent! Closing program...."
            node.get_logger().info(log)
            break           
            
        elif key_input == chr(WKEY_ASCII_VALUE) or key_input == chr(TKEY_ASCII_VALUE):    # print out the length changes
            if key_input == chr(TKEY_ASCII_VALUE):
                msg.data='traj1'
            else:
                msg.data='traj2'
            tomotors.publish(msg)
            node.get_logger().info(msg.data)
        elif key_input == chr(RKEY_ASCII_VALUE):
            msg.data='rest'
            tomotors.publish(msg)
            rest_received = False
            while rest_received == False:
                rclpy.spin_once(node)
                msg.data='rest'
                tomotors.publish(msg)
                node.get_logger().info(msg.data)
            log = "Rest command was sucessfully sent!"
            node.get_logger().info(log)
            
        elif key_input == chr(DKEY_ASCII_VALUE):
            def np2msg(mat):
                nq = 6
                squeezed = np.reshape(mat, (nq * mat.shape[1]))
                return list(squeezed)
            qd_mat = mat2np('/home/zach/drl-turtle/ros2_ws/src/py_pubsub/py_pubsub/turtle_trajectory/qd.mat', 'qd')
            dqd_mat = mat2np('/home/zach/drl-turtle/ros2_ws/src/py_pubsub/py_pubsub/turtle_trajectory/dqd.mat', 'dqd')
            ddqd_mat = mat2np('/home/zach/drl-turtle/ros2_ws/src/py_pubsub/py_pubsub/turtle_trajectory/ddqd.mat', 'ddqd')
            tvec = mat2np('/home/zach/drl-turtle/ros2_ws/src/py_pubsub/py_pubsub/turtle_trajectory/tvec.mat', 'tvec')

            # print(f"tvec mat shape: {tvec.shape}\n")
            traj.qd = np2msg(qd_mat)
            traj.dqd = np2msg(dqd_mat)
            traj.ddqd = np2msg(ddqd_mat)
            # print(f"t vec list: {tvec.tolist()}")

            traj.tvec = tvec.tolist()[0]
            print("sent trajectories...")
            traj_pub.publish(traj)
        

    rclpy.shutdown()

if __name__ == '__main__':
    try:
        main()
    except rclpy.ROSInterruptException:
        pass
