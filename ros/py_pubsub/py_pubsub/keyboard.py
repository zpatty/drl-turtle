from __future__ import print_function
import os
import json
import traceback

import rclpy
from rclpy.node import Node
from rclpy.executors import Executor, MultiThreadedExecutor

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

    node = rclpy.create_node('minimal_publisher')

    tomotors = node.create_publisher(String, 'master_motors', 10)

    msg = String()
    while rclpy.ok():
        print("\nT: Swimming Trajectory, P: Walking Trajectory, (or press SPACE to quit!)")
        key_input = getch()
        if key_input == chr(SPACE_ASCII_VALUE):
            msg.data='stop'
            break           
            
        elif key_input == chr(WKEY_ASCII_VALUE) or key_input == chr(TKEY_ASCII_VALUE):    # print out the length changes
            if key_input == chr(TKEY_ASCII_VALUE):
                msg.data='d1'
            else:
                msg.data='d2'
        tomotors.publish(msg)

    rclpy.shutdown()


if __name__ == '__main__':
    main()

# try: 
#     while 1:
#         print("\nT: Swimming Trajectory, P: Walking Trajectory, (or press SPACE to quit!)")
#         key_input = getch()
#         if key_input == chr(SPACE_ASCII_VALUE):
#             print("we're quitting\n")
#             break           
            
#         elif key_input == chr(WKEY_ASCII_VALUE) or key_input == chr(TKEY_ASCII_VALUE):    # print out the length changes
#             if key_input == chr(TKEY_ASCII_VALUE):
#                 print("\nRunning Swimming Trajectory")
#             else:
#                 print("\nRunning Walking Trajectory")

#     print("[END OF PROGRAM] Disabling torque\n")
#     # Disable Dynamixel Torque
# except Exception:
#     print("[ERROR] Disabling torque\n")
#     traceback.print_exc()