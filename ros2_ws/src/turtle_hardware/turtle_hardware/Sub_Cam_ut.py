import os
import sys
import time
import cv2
import rclpy
import numpy as np
import traceback
from datetime import datetime
from cv_bridge import CvBridge
from matplotlib import pyplot as plt

from rclpy.node import Node 
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from std_msgs.msg import String, Float32MultiArray
from sensor_msgs.msg import Image, CompressedImage
from turtle_interfaces.msg import TurtleCam, TurtleMode
from StereoProcessor import StereoProcessor


class CamSubscriber(Node):

    def __init__(self):
        super().__init__('cam_sub_node')
        self.flag = ''
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.frames_sub = self.create_subscription(
            TurtleCam,
            'frames',
            self.img_callback,
            qos_profile
            )

        self.cam_depth = self.create_subscription(
            CompressedImage,
            'video_frames_depth',
            self.img_callback_depth,
            qos_profile
            )
        
        self.stereo_pub = self.create_publisher(
            Float32MultiArray,
            'stereo',
            qos_profile
            )

        self.cam_detect = self.create_subscription(
            CompressedImage,
            'video_detect',
            self.img_callback_detect,
            qos_profile
            )
        
        self.mode_sub = self.create_subscription(
            TurtleMode,
            'turtle_mode',
            self.turtle_mode_callback,
            qos_profile)

        self.create_rate(1000)
        self.br = CvBridge()
        self.frames = []
        self.count = 0
        self.detect_count = 0
        self.depth_count = 0

        self.stereo = StereoProcessor()
        self.first = 1
        self.start_time = time.time()
        self.cam_time = 0
        t = datetime.today().strftime("%m_%d_%Y_%H_%M_%S")
        folder_name =  "video/" + t
        os.makedirs(folder_name + "/left")
        os.makedirs(folder_name + "/right")
        os.makedirs(folder_name + "/detection")
        os.makedirs(folder_name + "/depth")
        self.output_folder = folder_name
        

    def img_callback(self, msg):
        self.cam_time = time.time()
        left = self.br.compressed_imgmsg_to_cv2(msg.data[0])
        right = self.br.compressed_imgmsg_to_cv2(msg.data[1])
        cv2.imwrite(self.output_folder + "/left/frame%d.jpg" % self.count, left)
        cv2.imwrite(self.output_folder + "/right/frame%d.jpg" % self.count, right)
        self.count += 1
        end_time = time.time()
        seconds = end_time - self.start_time
        fps = 1.0 / seconds
        print(f"fps: {fps}\n")
        self.start_time = end_time
        stereo_depth, x, y, norm_disparity = self.stereo.update(left, right)
        if x is None:
            x = 0.0
            y = 0.0
        # print(f"stereo_depth: {stereo_depth} x: {x}, y: {y}\n")
        self.stereo_pub.publish(Float32MultiArray(data=[stereo_depth, x, y]))

    def img_callback_detect(self, data):
        current_frame = self.br.compressed_imgmsg_to_cv2(data)
        cv2.imwrite(self.output_folder + "/detection/frame%d.jpg" % self.count, current_frame)
        self.detect_count += 1
        end_time = time.time()
        seconds = end_time - self.start_time
        fps = 1.0 / seconds
        print("Estimated frames per second : {0}".format(fps))
        self.start_time = end_time

    def img_callback_depth(self, data):
        current_frame = self.br.compressed_imgmsg_to_cv2(data)
        cv2.imwrite(self.output_folder + "/depth/frame%d.jpg" % self.count, current_frame)
    
    def turtle_mode_callback(self, msg):
        if msg.mode == "kill":
            raise KeyboardInterrupt


def main(args=None):
    rclpy.init(args=args)
    print("Cowabunga")
    cam_sub = CamSubscriber()
    try:
        rclpy.spin(cam_sub)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        # rclpy.shutdown()
        print("some error occurred")
        traceback.print_exc()
        # turtle_node.shutdown_motors()
        exec_type, obj, tb = sys.exc_info()
        fname = os.path.split(tb.tb_frame.f_code.co_filename)[1]
        print(exec_type, fname, tb.tb_lineno)
        print(e)
    cv2.destroyAllWindows() 
if __name__ == '__main__':
  main()