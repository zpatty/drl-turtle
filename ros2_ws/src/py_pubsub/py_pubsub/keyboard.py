from __future__ import print_function
import os
import json
import traceback

import rclpy
from rclpy.node import Node
from rclpy.executors import Executor, MultiThreadedExecutor
import numpy as np
from matplotlib import pyplot as plt
from dyn_functions import *                    # Dynamixel support functions

from turtle_interfaces.msg import TurtleTraj

from std_msgs.msg import String
from std_msgs.msg import Float64MultiArray

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
    
def main(args=None):
    rclpy.init(args=args)

    node = rclpy.create_node('turtle_keyboard_cntrl_node')

    tomotors = node.create_publisher(String, 'master_motors', 10)
    traj_pub = node.create_publisher(TurtleTraj, 'motors_traj', 10)
    msg = String()
    traj = TurtleTraj()
    while rclpy.ok():
        print("\nT: Traj1, W: Traj2, R: Rest D: Custom Traj(or press SPACE to STOP!)")
        key_input = getch()
        if key_input == chr(SPACE_ASCII_VALUE):
            msg.data='stop'
            tomotors.publish(msg)
            node.get_logger().info(msg.data)
            break           
            
        elif key_input == chr(WKEY_ASCII_VALUE) or key_input == chr(TKEY_ASCII_VALUE):    # print out the length changes
            if key_input == chr(TKEY_ASCII_VALUE):
                msg.data='traj1'
            else:
                msg.data='traj2'
        elif key_input == chr(RKEY_ASCII_VALUE):
                msg.data='rest'
        elif key_input == chr(DKEY_ASCII_VALUE):
            def np2msg(mat):
                nq = 6
                squeezed = np.reshape(mat, (nq * mat.shape[1]))
                return list(squeezed)
            qd_mat = mat2np('qd.mat', 'qd')
            dqd_mat = mat2np('dqd.mat', 'dqd')
            ddqd_mat = mat2np('ddqd.mat', 'ddqd')
            tvec = mat2np('tvec.mat', 'tvec')

            traj.qd = np2msg(qd_mat)
            traj.dqd = np2msg(dqd_mat)
            traj.ddqd = np2msg(ddqd_mat)
            traj.tvec = list(tvec)
            print("sent trajectories...")
            traj_pub.publish(traj)
        
        tomotors.publish(msg)
        node.get_logger().info(msg.data)

    rclpy.shutdown()

if __name__ == '__main__':
    try:
        main()
    except rclpy.ROSInterruptException:
        pass