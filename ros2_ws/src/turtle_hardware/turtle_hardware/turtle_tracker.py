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
from turtle_interfaces.msg import TurtleTraj, TurtleSensors, TurtleCtrl

from std_msgs.msg import String
from std_msgs.msg import Float64MultiArray

import random
from Tracker import Tracker


global flag
flag = ''
turn_thresh = 150
dive_thresh = 100


class TrackerNode(Node):

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

        self.cam_color = self.create_subscription(
            Image,
            'video_frames_color',
            self.img_callback_color,
            img_profile
            )

        self.config_pub = self.create_publisher(
            TurtleCtrl,
            'turtle_ctrl_params',
            qos_profile
        )
        
        self.tracker = tracker
        self.first_detect = True

        self.cam_pub = self.create_publisher(String, 'primitive', buff_profile)

        self.publisher_detect = self.create_publisher(Image, 'video_detect' , qos_profile)

        self.br = CvBridge()

        self.create_rate(100)
        # self.subscription  # prevent unused variable warning
        self.yaw_thresh = 870/2
        self.pitch_thresh = 480/2
        self.last_ctrl = [1.0, 0.0, 0.0, 0.0]

    def img_callback_color(self, data):
        self.get_logger().info('Receiving other video frame')
        current_frame = self.br.imgmsg_to_cv2(data)
        # estimate position here
        # print("tracking")
        msg = TurtleCtrl()
        DIM=(640, 480)
        
        if self.first_detect:
            # rects = self.tracker.detect_and_track(current_frame)
            rects, img, preds, frame = self.tracker.detect_and_track(current_frame)
            # print(preds)
            self.publisher_detect.publish(self.br.cv2_to_imgmsg(frame, encoding="passthrough"))
            if not rects:
                self.first_detect = True
                print("No detection, random walk...\n")
                # if random.uniform(0.0, 1.0) < 0.9:
                    
                # self.cam_pub.publish(msg)
            else:
                self.first_detect = False
                print("Turtle Detected!\n")
                self.tracker.tracker.init(current_frame, rects[0])
        
        
        if not self.first_detect:
            # centroids = []
            # for rect in rects:
            centroids, frame = self.tracker.track(current_frame)
            self.publisher_detect.publish(self.br.cv2_to_imgmsg(frame, encoding="passthrough"))
            if not centroids.any():
                self.first_detect = True
                print("No detection, random walk...\n")
                # msg.data = 'dwell'
                # self.cam_pub.publish(msg)
            else:
                # centroid = centroids[0]
                print(centroids)
                msg.yaw = centroids[0]/870 * 2 - 1
                msg.pitch = 1 - centroids[1]/480 * 2
                msg.roll = 1000.0
                msg.fwd = 1000.0
                self.config_pub.publish(msg)
                
                # if abs(centroid[0] - DIM[0]/2) < turn_thresh and abs(centroid[1] - DIM[1]/2) < dive_thresh:
                #     # output straight primitive
                #     print("go straight...\n")
                #     msg.data = 'straight'
                #     self.cam_pub.publish(msg)
                # elif centroid[0] > DIM[0] - (DIM[0]/2 - turn_thresh):
                #     # turn right
                #     print("turn right...\n")
                #     msg.data = 'turnrf'
                #     self.cam_pub.publish(msg)
                # elif centroid[0] < (DIM[0]/2 - turn_thresh):
                #     # turn left
                #     print("turn left...\n")
                #     msg.data = 'turnlf'
                #     self.cam_pub.publish(msg)
                # elif centroid[1] > DIM[1] - (DIM[1]/2 - dive_thresh):
                #     # dive
                #     print("dive...\n")
                #     msg.data = 'dive'
                #     self.cam_pub.publish(msg)
                # elif centroid[1] < (DIM[1]/2 - dive_thresh): 
                #     # surface
                #     print("surface...\n")
                #     msg.data = 'surface'
                #     self.cam_pub.publish(msg)
                # else:
                #     # dwell
                #     print("dwell...\n")
                #     msg.data = 'dwell'
                #     self.cam_pub.publish(msg)

def main():
    rclpy.init()
    tracker_node = TrackerNode()
    try:
        rclpy.spin(tracker_node)
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
    # control_node.save_data()



if __name__ == '__main__':
    main()
