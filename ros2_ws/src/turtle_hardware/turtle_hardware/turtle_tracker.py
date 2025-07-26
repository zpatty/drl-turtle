#!/usr/bin/python

import os
import sys
import json
import traceback
import argparse
# Standard imports
import cv2
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
import numpy as np
import json 
import yaml

submodule = os.path.expanduser("~") + "/drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware"
sys.path.append(os.path.expanduser("~") + "/work/tortuga_tracker/Yolo-FastestV2")
sys.path.append(os.path.expanduser("~") + "/work/track_anything")
sys.path.append(os.path.expanduser("~") + "/work/segment-anything")

sys.path.append(submodule)
from threading import Thread
from std_msgs.msg import String
import rclpy
from rclpy.node import Node
from rclpy.executors import Executor, MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

# from fuse import fuse_feeds
from turtle_dynamixel.dyn_functions import *                    # Dynamixel support functions
from turtle_interfaces.msg import TurtleTraj, TurtleSensors, TurtleCtrl, TurtleCam, TurtleMode

from std_msgs.msg import String
from std_msgs.msg import Float32MultiArray

import random
# from Tracker import Tracker
from TrackAny import TrackAny

# from TrackAny_daniel import TrackAny
from StereoProcessor import StereoProcessor


global flag
flag = ''
turn_thresh = 150
dive_thresh = 100


class TrackerNode(Node):

    def __init__(self, args):
        super().__init__('cam_node')

        t = datetime.today().strftime("%m_%d_%Y_%H_%M_%S")
        folder_name =  "video/" + t
        os.makedirs(folder_name + "/left_detect")
        os.makedirs(folder_name + "/right_detect")
        self.output_folder = folder_name
        if args.save_images_to:
            os.makedirs(folder_name + "/track")
            args.save_images_to = folder_name + "/track"

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
            depth=2
        )

        self.frames_sub = self.create_subscription(
            TurtleCam,
            'frames',
            self.img_callback,
            img_profile
            )

        self.config_pub = self.create_publisher(
            Float32MultiArray,
            'centroids',
            qos_profile
        )
        self.mode_sub = self.create_subscription(
            TurtleMode,
            'turtle_mode',
            self.turtle_mode_callback,
            qos_profile)
        
        self.gamepad_sub = self.create_subscription(
            Float32MultiArray,
            'gamepad',
            self.gamepad_callback,
            qos_profile
        )
        
        if args.track_any:
            tracker = TrackAny(args)
            self.clicks_for_retrack = None
        # else:
        #     tracker = Tracker()
        self.track_type = args.track_any
        self.tracker = tracker
        self.first_detect = True
        self.status = None

        self.stereo = StereoProcessor()
        self.publisher_detect = self.create_publisher(CompressedImage, 'video_detect' , qos_profile)
        self.publisher_depth = self.create_publisher(CompressedImage, 'video_frames_depth' , qos_profile)

        self.br = CvBridge()

        self.create_rate(100)
        # self.subscription  # prevent unused variable warning
        self.yaw_thresh = 870/2
        self.pitch_thresh = 480/2
        self.last_ctrl = [1.0, 0.0, 0.0, 0.0]
        self.count = 0

        script_dir = os.path.dirname(os.path.abspath(__file__))
        yaml_path = os.path.join(script_dir, 'rig_params.yaml')

        with open(yaml_path, 'r') as f:
            params = yaml.safe_load(f)
        
        self.dx = params['tx']
        self.dy = params['ty']
        self.scale = params['scale']
        self.theta = np.deg2rad(params['rot_deg'])
        self.affine_matrix = np.array([
            [self.scale * np.cos(self.theta), -self.scale * np.sin(self.theta), self.dx],
            [self.scale * np.sin(self.theta), self.scale * np.cos(self.theta), self.dy]
        ])     

        ##Needed for SIFT stitching
        # self.KL = np.array([[708.3477312219868, 0.0, 260.69187590557686], [0.0, 675.3059166594338, 301.31936629865646], [0.0, 0.0, 1.0]])
        # self.DL = np.array([[-0.39383047117877457], [6.721465255404687], [-35.99917141986595], [61.49579122578909]])
        # self.KR = np.array([[667.0400978057647, 0.0, 334.8109094526051], [0.0, 644.922628956739, 364.07228200370565], [0.0, 0.0, 1.0]])
        # self.DR = np.array([[0.8809516193294453], [-6.609640306922403], [21.549513701823056], [-24.149385093847197]])
        # self.R = np.array([[0.8721459388442752, 0.02130940474841954, -0.4887815162490354], [-0.06589130347707366, 0.9950649291385254, -0.07418977641584823], [0.4847884048567273, 0.09691076342599622, 0.8692459412897253]])
        # self.T = np.array([[-2.085136618149882], [0.1939622251215522], [-0.9258137973647751]])*25.4  

        # self.previous_num_keypoints = 0
        # self.previous_H = None
        # self.homography_sum = np.zeros((3, 3))
        # self.frame_count = 0
        # self.max_frames_to_average = 3 


        
    def turtle_mode_callback(self, msg):
        if msg.mode == "kill":
            raise KeyboardInterrupt

    def img_callback(self, msg):
        # self.get_logger().info('Receiving other video frame')
        left = self.br.compressed_imgmsg_to_cv2(msg.data[0])
        right = self.br.compressed_imgmsg_to_cv2(msg.data[1])

        
        #interactive stitcher
        h, w = left.shape[:2]
        image_size = (w, h)

        corners = np.array([
            [0,0],
            [0,h],
            [w,0],
            [w,h]
        ], dtype=np.float32)
        transformed_corners = cv2.transform(np.array([corners]),self.affine_matrix)[0]
        all_corners = np.vstack(([[0,0],[0,h],[w,0],[w,h]],transformed_corners))

        [xmin, ymin] = np.floor(all_corners.min(axis=0)).astype(int)
        [xmax, ymax] = np.ceil(all_corners.max(axis=0)).astype(int)

        offset = np.array([-xmin, -ymin])
        output_size = (xmax - xmin, ymax - ymin)  # width, height

        affine_with_offset = self.affine_matrix.copy()
        affine_with_offset[:, 2] += offset

        transformed_right = cv2.warpAffine(right, affine_with_offset, output_size)
        
        canvas = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)
        canvas[offset[1]:offset[1]+h, offset[0]:offset[0]+w] = left
        frame = np.maximum(canvas, transformed_right)
        


        # #SIFT stitcher WIP very laggy has issues crashing system
        # h, w = left.shape[:2]
        # image_size = (w, h)

        # l_map1, l_map2 = cv2.fisheye.initUndistortRectifyMap(
        #     K=self.KL, D=self.DL, R=np.eye(3), P=self.KL, size=image_size, m1type=cv2.CV_16SC2
        # )

        # r_map1, r_map2 = cv2.fisheye.initUndistortRectifyMap(
        #     K=self.KR, D=self.DR, R=np.eye(3), P=self.KR, size=image_size, m1type=cv2.CV_16SC2
        # )

        # undistorted_left = cv2.remap(left, l_map1, l_map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        # undistorted_right = cv2.remap(right, r_map1, r_map2, interpolation=cv2.INTER_LINEAR,borderMode= cv2.BORDER_CONSTANT)

        # sift = cv2.SIFT_create()
        # kp1, des1 = sift.detectAndCompute(undistorted_left, None)
        # kp2, des2 = sift.detectAndCompute(undistorted_right, None)

        # bf = cv2.BFMatcher()
        # matches = bf.knnMatch(des1, des2, k=2)

        # good = []
        # for m,n in matches:
        #     if m.distance < 0.75*n.distance:
        #         good.append(m)

        # H_avg = self.previous_H if self.previous_H is not None else np.eye(3)

        # num_keypoints = len(good)
        # if num_keypoints >= self.previous_num_keypoints:
        #     self.previous_num_keypoints = num_keypoints

        #     src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        #     dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        #     H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        #     self.homography_sum += H
        #     print(self.homography_sum)

        #     self.frame_count += 1

        #     if self.frame_count >= self.max_frames_to_average:
        #         H_avg = self.homography_sum / self.frame_count
        #         self.previous_H = H_avg
        #         self.frame_count = 0
        #         self.homography_sum = np.zeros((3, 3))
        #     else:
        #         self.previous_H = H

        # else:
        #     H_avg = self.previous_H if self.previous_H is not None else np.eye(3)

        # h1, w1 = undistorted_left.shape[:2]
        # h2, w2 = undistorted_right.shape[:2]

        # corners_left = np.array([[0,0], [0,h1], [w1,h1], [w1,0]], dtype=np.float32).reshape(-1,1,2)
        # corners_right = np.array([[0,0], [0,h2], [w2,h2], [w2,0]], dtype=np.float32).reshape(-1,1,2)

        # warped_corners_right = cv2.perspectiveTransform(corners_right, H_avg)

        # all_corners = np.concatenate((corners_left, warped_corners_right), axis=0)
        # [x_min, y_min] = np.floor(all_corners.min(axis=0).ravel()).astype(int)
        # [x_max, y_max] = np.ceil(all_corners.max(axis=0).ravel()).astype(int)

        # output_width = x_max - x_min
        # output_height = y_max - y_min

        # translation = np.array([[1, 0, -x_min],
        #                         [0, 1, -y_min],
        #                         [0, 0, 1]])

        # stitched2 = cv2.warpPerspective(undistorted_right, translation @ H_avg, (output_width, output_height))
        # stitched2[-y_min:h1 - y_min, -x_min:w1 - x_min] = undistorted_left
        # frame = stitched2[-y_min:h1-y_min, 0:1000]


        # frame = np.concatenate((left[:,0:230], right), axis=1)
        # estimate position here
        # print("tracking")
        DIM=(640, 480)
        # print(self.output_folder + "/frame%d.jpg" % self.count)
        # cv2.imwrite(self.output_folder + "/frame%d.jpg" % self.count, frame)
        self.count += 1
        cv2.imwrite(self.output_folder + "/left_detect/frame%d.jpg" % self.count, left)
        cv2.imwrite(self.output_folder + "/right_detect/frame%d.jpg" % self.count, right)
        if self.track_type:
            frame = self.track_anything(frame)
        else:
            frame = self.tortuga_track(frame)
                
        self.publisher_detect.publish(self.br.cv2_to_compressed_imgmsg(frame))
        
        # cv2.imwrite(self.output_folder + "/right/frame%d.jpg" % self.count, self.stream.right)
    
    def track_anything(self, frame):
        masks = "bloop"
        print(f"[STATUS] {self.status}\n")

        if self.status != "Tracking":
            bounding_boxes, masks, frame = self.tracker.detect_object(frame)
            self.tracker.clicks_for_retrack = []
            self.tracker.state_for_retrack = 0
            # print(f"resetting clicks for retrack: {self.tracker.clicks_for_retrack} in mode {self.status}\n")
        if masks is not None:
            self.tracker.mission_counter +=1
            self.status, mean_point, masks, clicks_for_retrack = self.tracker.track_object_with_cutie(masks, frame)
            # print(f"[DEBUG] Mean Point: {mean_point}\n")
            self.config_pub.publish(Float32MultiArray(data=mean_point))
        else:
            self.status = None
            self.config_pub.publish(Float32MultiArray(data=[]))
        
        if self.status == 'Failed': 
            print("Object Lost... Redetecting....")
            self.tracker.clicks_for_retrack = None
            self.tracker.clicks_for_retrack = []

        elif self.status == "retrack":
            print("Retracking based in user comman....")
            self.tracker.frame_idx = 0
        elif self.status == "Success":
            print("Success....")
            self.tracker.frame_idx = 0
            self.tracker.clicks_for_retrack = []
            self.tracker.state_for_retrack = 0
            self.tracker.state = 0
        print(f"[STATUS] {self.status}\n")

        return frame
    
    def tortuga_track(self, frame):
        if self.first_detect:
            # rects = self.tracker.detect_and_track(current_frame)
            rects, img, preds, frame = self.tracker.detect_and_track(frame)
            # print(preds)
            if not rects:
                self.first_detect = True
                print("No detection, random walk...\n")
                self.config_pub.publish(Float32MultiArray(data=[]))
                # if random.uniform(0.0, 1.0) < 0.9:
                    
                # self.cam_pub.publish(msg)
            else:
                self.first_detect = False
                print("Turtle Detected!\n")
                self.tracker.tracker.init(frame, rects[0])
                print("Tracker Initialized\n")
        
        
        if not self.first_detect:
            centroids = []
            # for rect in rects:
            centroids, frame = self.tracker.track(frame)
            # centroids = centroids.tolist()
            if not centroids:
                self.first_detect = True
                print("No detection, random walk...\n")
                self.config_pub.publish(Float32MultiArray(data=[]))
                # msg.data = 'dwell'
                # self.cam_pub.publish(msg)
            else:
                # centroid = centroids[0]

                print(centroids)
                # msg.data = centroids
                self.config_pub.publish(Float32MultiArray(data=centroids))
        return frame
    
    def gamepad_callback(self, msg):
        # print(msg)
        if msg.data[-1] == 3:
            self.tracker.button = "stop"
            print("Stop Tracking")
        elif msg.data[-1] == 2:
            self.tracker.button = "track"
                
            # print("Retrack")


def main(args):
    rclpy.init()
    tracker_node = TrackerNode(args)
    try:
        rclpy.spin(tracker_node)
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
    # control_node.save_data()



if __name__ == '__main__':
    # python track_anything.py --height 400 --width 800 --video /dev/video0  --detection_mode click  --re_detection_mode click --plot_visualization --save_images_to outputs/here --is_stream 1

    parser = argparse.ArgumentParser(description='PyTorch + mavsdk -- zero shot detection, tracking, and drone control')
    parser.add_argument('--track_any', default =True, help='Track anything or turtle tracker')
    parser.add_argument('--use_filter', action='store_true',  default =False, help='use_filter')
    parser.add_argument('--plot_visualizations', action='store_true', default =True, help='plot_visualizations')
    parser.add_argument('--height', default=-1, type=int, help='desired_height resulution')
    parser.add_argument('--width', default=-1, type=int, help='desired_width resulution')
    parser.add_argument('--save_images_to', default=True, help='The path to save all semgentation/tracking frames')

    parser.add_argument('--detection_mode', default = "click", help='')
    parser.add_argument('--re_detection_mode', default = "click", help='')
    #parser.add_argument('--num_of_points_to_track', default = 3,  type=int, help='')


    # parser.add_argument('--is_stream', default = 1, type=float, help='realtime_or_not')
    #parser.add_argument('--streaming', default = 0, type=int, help='indicator')
    parser.add_argument('--wait_key', default=1, type=int, help='cv waitkey')

    parser.add_argument('--sam_model_type', default = "vit_h")
    parser.add_argument('--sam_model_path', default = os.path.expanduser("~") + "/work/track_anything/segment-anything/models/sam_vit_h_4b8939.pth")
    parser.add_argument('--num_of_clicks_for_detection', default=2, type = float,  help='')
    args = parser.parse_args()
    main(args)
