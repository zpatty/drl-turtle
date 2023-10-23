from __future__ import print_function
import rclpy
from rclpy.node import Node
from rclpy.executors import Executor, MultiThreadedExecutor

from std_msgs.msg import String
from std_msgs.msg import Float64MultiArray

import os
import pty
import json
import traceback

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
pid = pty.fork()
if not pid:
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


class MinimalPublisher(Node):

    def __init__(self, topic):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, topic, 10)
        timer_period = 1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0.

    def timer_callback(self):
        msg = String()
        msg.data = "test " + str(self.i)
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: ' +msg.data)
        self.i += 1

class DynamicPublisher(Node):

    def __init__(self, topic):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, topic, 10)
        timer_period = 1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0.

    def timer_callback(self):
        msg = String()
        try: 
            key_input = getch()
            if key_input == chr(SPACE_ASCII_VALUE):
                msg.data = 'stop'
            elif key_input == chr(WKEY_ASCII_VALUE) or key_input == chr(TKEY_ASCII_VALUE):    # print out the length changes
                if key_input == chr(TKEY_ASCII_VALUE):
                    msg.data = 'Running Swimming Trajectory'
                else:
                    msg.data = 'Running Walking Trajectory'
            else:
                msg.data = "test " + str(self.i)
        except Exception:
            print("[ERROR] Disabling torque\n")
            traceback.print_exc()
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: ' +msg.data)
        self.i += 1

def main(args=None):
    print("\nT: Swimming Trajectory, P: Walking Trajectory, (or press SPACE to quit!)")
    rclpy.init(args=args)

    tomotors = DynamicPublisher('master_motors')
    tocv = MinimalPublisher('master_cv')

    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(tomotors)
    executor.add_node(tocv)
    
    executor.spin() 

    rclpy.shutdown()


if __name__ == '__main__':
    main()
