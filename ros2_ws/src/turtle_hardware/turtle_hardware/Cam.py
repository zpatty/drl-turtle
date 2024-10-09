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

submodule = os.path.expanduser("~") + "/drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware"
sys.path.append(submodule)
from threading import Thread
from std_msgs.msg import String
import rclpy
from rclpy.node import Node
from rclpy.executors import Executor, MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from turtle_interfaces.msg import TurtleCam
from fuse import fuse_feeds
from turtle_dynamixel.dyn_functions import *                    # Dynamixel support functions

from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float64MultiArray

from datetime import datetime

class CamStream():
    """
    Class that just reads from the camera
    """
    def __init__(self, idx=0):
        self.cap = cv2.VideoCapture(0)
        self.cap1 = cv2.VideoCapture(4)
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

    def __init__(self, stream):
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

        self.stream = stream
        self.cam_publisher = self.create_publisher(TurtleCam, 'frames' , qos_profile)

        self.br = CvBridge()

        self.create_rate(1000)

        self.count = 0
        t = datetime.today().strftime("%m_%d_%Y_%H_%M_%S")
        folder_name =  "video/" + t
        os.makedirs(folder_name + "/right")
        os.makedirs(folder_name + "/left")
        self.output_folder = folder_name
        timer_cb_group = None
        self.call_timer = self.create_timer(0.05, self._cam_cb, callback_group=timer_cb_group)
    def _cam_cb(self):
        msg = TurtleCam()
        msg.data[0] = self.br.cv2_to_compressed_imgmsg(self.stream.left)
        msg.data[1] = self.br.cv2_to_compressed_imgmsg(self.stream.right)
        self.cam_publisher.publish(msg)
        # cv2.imwrite(self.output_folder + "/left/frame%d.jpg" % self.count, self.stream.left)
        # cv2.imwrite(self.output_folder + "/right/frame%d.jpg" % self.count, self.stream.right)
        self.count += 1


def main():
    rclpy.init()
    stream = CamStream().start()
    cam_pub = CamNode(stream)
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
    stream.stop_process()
    cam_pub.destroy_node()

    # control_node.save_data()
if __name__ == '__main__':
    main()
