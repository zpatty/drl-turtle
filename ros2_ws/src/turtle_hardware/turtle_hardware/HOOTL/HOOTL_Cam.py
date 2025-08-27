#!/usr/bin/python

import os
import sys
import json
import traceback

# Standard imports
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import json 
import glob
import re

submodule = os.path.expanduser("~") + "/drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware"
sys.path.append(submodule)
from threading import Thread
from std_msgs.msg import String
import rclpy
from rclpy.node import Node
from rclpy.executors import Executor, MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from turtle_interfaces.msg import TurtleCam, TurtleMode
from turtle_dynamixel.dyn_functions import *                    # Dynamixel support functions

from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float64MultiArray

from datetime import datetime

###################################################
# This HOOTL class is the exact version found on turtle robot raspberry pi from May 16th, 2025 wall obstacle avoidance test. 
###################################################

class CamStream():
    """
    Class that just reads from the camera
    """
    def __init__(self, idx=0):
        self.cap = cv2.VideoCapture(4)
        print("opened first cam")
        self.cap1 = cv2.VideoCapture(0)
        print("opened second cam")
        self.ret, self.left = self.cap.read()
        self.ret1, self.right = self.cap1.read()
        self.stopped = False

    def start(self):
        Thread(target=self.get, args=()).start()
        Thread(target=self.get1, args=()).start()
        return self
    def get(self):
        while not self.stopped:
            self.ret, self.left = self.cap.read()
    def get1(self):
        while not self.stopped:
            self.ret1, self.right = self.cap1.read()
    def stop_process(self):
        self.stopped = True
        self.cap.release()
    



class CamNode(Node):

    def __init__(self):
        super().__init__('cam_node')
        self.flag = ''
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

        self.mode_sub = self.create_subscription(
            TurtleMode,
            'turtle_mode',
            self.turtle_mode_callback,
            qos_profile)

        self.cam_publisher = self.create_publisher(TurtleCam, 'frames' , qos_profile)

        self.br = CvBridge()

        self.create_rate(1000)

        self.count = 0
        timer_cb_group = None
        # make timer slower for easier debugging
        self.call_timer = self.create_timer(0.1, self._cam_cb, callback_group=timer_cb_group)
        # where to pull the images
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.image_dir = os.path.join(script_dir, 'video/08_14_2025_10_42_38')
        self.left_folder = os.path.join(self.image_dir, 'left_detect')
        self.right_folder = os.path.join(self.image_dir, 'right_detect')
        self.extension = 'jpg'
        self.left_files = glob.glob(os.path.join(self.left_folder, f"*.{self.extension}"))
        self.right_files = glob.glob(os.path.join(self.right_folder, f"*.{self.extension}"))
        self.left_files.sort(key=self.natural_sort_key)
        self.right_files.sort(key=self.natural_sort_key)

        # Ensure all folders have the same number of files
        self.num_frames = min(len(self.left_files), len(self.right_files))
        print(f"Using {self.num_frames} frames from each folder")
        
        self.left_files = self.left_files[:self.num_frames]
        self.right_files = self.right_files[:self.num_frames]



    def natural_sort_key(self, s):
        """
        Sort strings containing numbers naturally (e.g., frame1.npy, frame2.npy, ..., frame10.npy)
        """
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

    def _cam_cb(self):
        msg = TurtleCam()
        # if self.count < 2:
        if self.count < self.num_frames:
            left = cv2.imread(self.left_files[self.count])
            right = cv2.imread(self.right_files[self.count])
            msg.data[0] = self.br.cv2_to_compressed_imgmsg(left)
            msg.data[1] = self.br.cv2_to_compressed_imgmsg(right)
            self.cam_publisher.publish(msg)
            self.count += 1
        else:
            print("Sent all frames!")
            raise KeyboardInterrupt
    def turtle_mode_callback(self, msg):
        if msg.mode == "kill":
            raise KeyboardInterrupt


def main():
    rclpy.init()
    cam_pub = CamNode()
    try:
        rclpy.spin(cam_pub)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        # rclpy.shutdown()
        print("some error occurred")
        # turtle_node.shutdown_motors()
        exec_type, obj, tb = sys.exc_info()
        fname = os.path.split(tb.tb_frame.f_code.co_filename)[1]
        print(exec_type, fname, tb.tb_lineno)
        print(e)
    print("closing")
    cam_pub.destroy_node()

    # control_node.save_data()
if __name__ == '__main__':
    main()
