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
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from turtle_dynamixel.dyn_functions import *                    # Dynamixel support functions

from std_msgs.msg import Float32MultiArray



global flag
flag = ''
turn_thresh = 150
dive_thresh = 100


class TrackerNode(Node):

    def __init__(self):
        super().__init__('tracker_node')
        self.flag = ''
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=2
        )


        self.centroids_pub = self.create_publisher(
            Float32MultiArray,
            'centroids',
            qos_profile
        )
        
        self.first_detect = True

        self.publisher_detect = self.create_publisher(Image, 'video_detect' , qos_profile)

        self.br = CvBridge()

        self.create_rate(100)
        # self.subscription  # prevent unused variable warning
        self.yaw_thresh = 870/2
        self.pitch_thresh = 480/2
        self.last_ctrl = [1.0, 0.0, 0.0, 0.0]

        self.dt = 1.0
        timer_cb_group = None
        self.call_timer = self.create_timer(self.dt, self._dummy_cb, callback_group=timer_cb_group)
        self.see_a_turtle = False
        self.centroid_dummy = [870.0/2, 480.0/2]

    def _dummy_cb(self):
        msg = Float32MultiArray()
        if self.see_a_turtle:
            v = np.random.normal([0.0, 0.0],[50.0, 20.0])
            self.centroid_dummy = np.array(self.centroid_dummy) + v
            if self.centroid_dummy[0] > 870 or self.centroid_dummy[1] > 480 or self.centroid_dummy[0] < 0 or self.centroid_dummy[0] < 0:
                print("Turtle Ran Away")
                self.centroid_dummy = np.random.uniform([0.0, 0.0], [870.0, 480.0])
                self.see_a_turtle = False
                self.centroids_pub.publish(Float32MultiArray(data=[]))
            else:
                self.centroids_pub.publish(Float32MultiArray(data=self.centroid_dummy.tolist()))
        else:
            self.turtle_reappear = np.random.uniform(0.0, 100.0)
            if self.turtle_reappear > 80.0:
                self.see_a_turtle = True
                print("I Like Toytuls!")
            self.centroids_pub.publish(Float32MultiArray(data=[]))
        print(self.centroid_dummy)



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
