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

from turtle_dynamixel.dyn_functions import *                    # Dynamixel support functions
from turtle_interfaces.msg import TurtleTraj, TurtleSensors

from std_msgs.msg import String
from std_msgs.msg import Float64MultiArray

class CamStream():
    """
    Class that just reads from the camera
    """
    def __init__(self, idx=0):
        self.cap = cv2.VideoCapture(0)
        self.ret, self.frame = self.cap.read()
        self.stopped = False
    
    def start(self):
        Thread(target=self.get, args=()).start()
        return self
    def get(self):
        while not self.stopped:
            self.ret, self.frame = self.cap.read()
    def stop_process(self):
        self.stopped = True
        self.cap.release()

def parse_cv_params():
    with open('cv_config.json') as config:
        param = json.load(config)
    print(f"[MESSAGE] Config: {param}\n")    
    # Serializing json
    config_params = json.dumps(param, indent=14)
    return param, config_params

def fixHSVRange(val):
    # Normal H,S,V: (0-360,0-100%,0-100%)
    # OpenCV H,S,V: (0-180,0-255 ,0-255)
    return (180 * val[0] / 360, 255 * val[1] / 100, 255 * val[2] / 100)

global flag
flag = ''

class MinimalSubscriber(Node):

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
        self.subscription = self.create_subscription(
            String,
            'turtle_mode_cmd',
            self.keyboard_callback,
            qos_profile)
        self.cam_pub = self.create_publisher(String, 'primitive', buff_profile)
        self.publisher_ = self.create_publisher(Image, 'video_frames' , qos_profile)
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

def main(args=None):
    global flag
    home_dir = os.path.expanduser("~")

    rclpy.init(args=args)

    minimal_subscriber = MinimalSubscriber()
    stream = CamStream().start()

    # node = rclpy.create_node('cam_node')

    cv_params, __ = parse_cv_params()
    turn_thresh = cv_params["turn_thresh"]
    dive_thresh = cv_params["dive_thresh"]

    print("created subscriber")
    DIM=(640, 480)

    lower_yellow = np.array(cv_params["lower_yellow"])
    upper_yellow = np.array(cv_params["upper_yellow"])

    cap0 = stream.cap
    # cap0 = cv2.VideoCapture(0)
    # cap0.set(3, 1920)
    # cap0.set(4, 1080)
    # cap1 = cv2.VideoCapture(2)
    print("active video feed")
    print("created publisher")
    # big_mask = np.zeros((640, 480, 1))
    msg = String()
    count = 0
    print("entering while loop")
    try:
        while(minimal_subscriber.flag != 'stop'):
            print(minimal_subscriber.flag)
            rclpy.spin_once(minimal_subscriber)
            # ret0, left = cap1.read()
            # ret1, right = cap0.read()
            ret1 = stream.ret
            right = stream.frame
            # if ret1 == True:
            #     print(f"count: {count}")
            #     # print("sending image")
            #     minimal_subscriber.publisher_.publish(minimal_subscriber.br.cv2_to_imgmsg(right, encoding="bgr8"))
            #     count += 1
            # else:
            #     print("ignore frame")
            denoise = 15
            blur = cv2.GaussianBlur(right, (5,5), 1)
            # Converting from BGR to HSV color space
            hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

            lab = cv2.cvtColor(right, cv2.COLOR_BGR2LAB)
            bin_y = cv2.inRange(hsv, fixHSVRange(lower_yellow), fixHSVRange(upper_yellow))
            open_kern = np.ones((10,10), dtype=np.uint8)
            bin_y = cv2.morphologyEx(bin_y, cv2.MORPH_OPEN, open_kern, iterations=2)

            rip_y = right.copy()
            rip_y[bin_y==0] = 0
            mark_y = cv2.addWeighted(right, .4, rip_y, .6, 1)

            mask = bin_y #cv2.bitwise_not(bin_y)
            kernel = np.ones((10,10),np.uint8)

            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            # big_mask = np.append(big_mask, mask, axis=2)

            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cnt_s = sorted(cnts, key=cv2.contourArea)

            if not (len(cnts) == 0):
                cnt = cnt_s[-1]
                (x,y),radius = cv2.minEnclosingCircle(cnt)
                center = (int(x),int(y))
                radius = int(radius)
                cv2.circle(mask,center,radius,10,2)
                centerMask=cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
                minimal_subscriber.publisher_.publish(minimal_subscriber.br.cv2_to_imgmsg(centerMask, encoding="bgr8"))
                # minimal_subscriber.publisher_.publish(minimal_subscriber.br.cv2_to_imgmsg(right, encoding="bgr8"))


                # shesh = '/home/crush/drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware/'
                # print(f"check: {shesh}")
                # cv2.imwrite(shesh + "images/frame%d.jpg" % count, centerMask)
                # print("saved")
                # count += 1
                # cv2.imshow('Mask',centerMask)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
                centroid = center
                if abs(centroid[0] - DIM[0]/2) < turn_thresh and abs(centroid[1] - DIM[1]/2) < dive_thresh:
                    # output straight primitive
                    print("go straight...\n")
                    msg.data = 'straight'
                    minimal_subscriber.cam_pub.publish(msg)
                elif centroid[0] > DIM[0] - (DIM[0]/2 - turn_thresh):
                    # turn right
                    print("turn right...\n")
                    msg.data = 'turnrf'
                    minimal_subscriber.cam_pub.publish(msg)
                elif centroid[0] < (DIM[0]/2 - turn_thresh):
                    # turn left
                    print("turn left...\n")
                    msg.data = 'turnlf'
                    minimal_subscriber.cam_pub.publish(msg)
                elif centroid[1] > DIM[1] - (DIM[1]/2 - dive_thresh):
                    # dive
                    print("dive...\n")
                    msg.data = 'dive'
                    minimal_subscriber.cam_pub.publish(msg)
                elif centroid[1] < (DIM[1]/2 - dive_thresh): 
                    # surface
                    print("surface...\n")
                    msg.data = 'surface'
                    minimal_subscriber.cam_pub.publish(msg)
                else:
                    # dwell
                    print("dwell...\n")
                    msg.data = 'dwell'
                    minimal_subscriber.cam_pub.publish(msg)
            else:
                print("no detection")
                # centerMask=cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
                # cv2.imshow('Mask',centerMask)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
            print(minimal_subscriber.flag)    # cv2.destroyAllWindows()
    except KeyboardInterrupt:
        print("cntrl c input: shutting down...")
        stream.stop_process()
        minimal_subscriber.cam_pub.destroy_node()
        rclpy.shutdown()
    print("closing")
    stream.stop_process()
    minimal_subscriber.destroy_node()
    rclpy.shutdown()
    # cap0.release()

if __name__ == '__main__':
    main()
