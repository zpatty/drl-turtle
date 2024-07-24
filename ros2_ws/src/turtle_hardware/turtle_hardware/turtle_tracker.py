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
sys.path.append(os.path.expanduser("~") + "/tortuga_tracker/Yolo-FastestV2")
sys.path.append(submodule)
from threading import Thread
from std_msgs.msg import String
import rclpy
from rclpy.node import Node
from rclpy.executors import Executor, MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from fuse import fuse_feeds
from turtle_dynamixel.dyn_functions import *                    # Dynamixel support functions
from turtle_interfaces.msg import TurtleTraj, TurtleSensors

from std_msgs.msg import String
from std_msgs.msg import Float64MultiArray

from tracking_class import Tracker


global flag
flag = ''
turn_thresh = 150
dive_thresh = 100


class MinimalSubscriber(Node):

    def __init__(self, tracker = Tracker()):
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
        img_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.subscription = self.create_subscription(
            String,
            'turtle_mode_cmd',
            self.keyboard_callback,
            qos_profile)

        self.cam_color = self.create_subscription(
            Image,
            'video_frames_color',
            self.img_callback_color,
            img_profile
            )
        
        self.tracker = tracker
        self.first_detect = True

        self.cam_pub = self.create_publisher(String, 'primitive', buff_profile)

        self.br = CvBridge()

        self.create_rate(100)
        self.subscription  # prevent unused variable warning

    def keyboard_callback(self, msg):
        """
        Callback function that updates the mode the turtle should be in.
        This method is what enables us to set "emergency stops" mid-trajectory. 
        """
        # global mode
        if msg.data == 'stop':
            self.flag = 'stop'

    def img_callback_color(self, data):
        self.get_logger().info('Receiving other video frame')
        current_frame = self.br.imgmsg_to_cv2(data)
        # estimate position here
        print("tracking")
        msg = String()
        DIM=(640, 480)
        if self.first_detect:
            rects = self.tracker.detect_and_track(current_frame)
            if not rects:
                self.first_detect = True
                print("No detection, dwell...\n")
                msg.data = 'dwell'
                self.cam_pub.publish(msg)
            else:
                self.first_detect = False
                self.tracker.tracker.init(current_frame, rects[0])
        
        
        if not self.first_detect:
            centroids = []
            for rect in rects:
                centroids.append(self.tracker.track(current_frame))
            if not centroids:
                self.first_detect = True
                print("No detection, dwell...\n")
                msg.data = 'dwell'
                self.cam_pub.publish(msg)
            else:
                centroid = centroids[0]
                print(centroid)
                if abs(centroid[0] - DIM[0]/2) < turn_thresh and abs(centroid[1] - DIM[1]/2) < dive_thresh:
                    # output straight primitive
                    print("go straight...\n")
                    msg.data = 'straight'
                    self.cam_pub.publish(msg)
                elif centroid[0] > DIM[0] - (DIM[0]/2 - turn_thresh):
                    # turn right
                    print("turn right...\n")
                    msg.data = 'turnrf'
                    self.cam_pub.publish(msg)
                elif centroid[0] < (DIM[0]/2 - turn_thresh):
                    # turn left
                    print("turn left...\n")
                    msg.data = 'turnlf'
                    self.cam_pub.publish(msg)
                elif centroid[1] > DIM[1] - (DIM[1]/2 - dive_thresh):
                    # dive
                    print("dive...\n")
                    msg.data = 'dive'
                    self.cam_pub.publish(msg)
                elif centroid[1] < (DIM[1]/2 - dive_thresh): 
                    # surface
                    print("surface...\n")
                    msg.data = 'surface'
                    self.cam_pub.publish(msg)
                else:
                    # dwell
                    print("dwell...\n")
                    msg.data = 'dwell'
                    self.cam_pub.publish(msg)
def main(args=None):
    global flag
    home_dir = os.path.expanduser("~")

    rclpy.init(args=args)

    minimal_subscriber = MinimalSubscriber()

    # node = rclpy.create_node('cam_node')

    

    print("created subscriber")
    



    while(minimal_subscriber.flag != 'stop'):
        rclpy.spin_once(minimal_subscriber)
    print("closing")
    minimal_subscriber.destroy_node()
    rclpy.shutdown()
    # cap0.release()



if __name__ == '__main__':
    main()
