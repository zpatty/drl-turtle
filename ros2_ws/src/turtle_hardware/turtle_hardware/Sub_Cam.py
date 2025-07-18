import rclpy
from rclpy.node import Node 
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from std_msgs.msg import String, Float32MultiArray
from sensor_msgs.msg import Image, CompressedImage
from turtle_interfaces.msg import TurtleCam, TurtleMode
import cv2
from cv_bridge import CvBridge
import cv2 
from matplotlib import pyplot as plt
import time
import os
from datetime import datetime
import sys
import numpy as np
from StereoProcessor import StereoProcessor
import traceback
import yaml


class CamSubscriber(Node):

    def __init__(self):
        super().__init__('cam_sub_node')
        self.flag = ''
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=2
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
        script_dir = os.path.dirname(os.path.abspath(__file__))
        t = datetime.today().strftime("%m_%d_%Y_%H_%M_%S")
        folder_name =  os.path.join(script_dir, "video" , t)
        os.makedirs(os.path.join(folder_name, "left"))
        os.makedirs(os.path.join(folder_name, "right"))
        os.makedirs(os.path.join(folder_name, "detection"))
        os.makedirs(os.path.join(folder_name, "depth"))
        self.output_folder = folder_name

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

        

    def img_callback(self, msg):
        #### LEFT RIGHT ####
        # self.get_logger().info('Receiving video frame')
        left = self.br.compressed_imgmsg_to_cv2(msg.data[0])
        right = self.br.compressed_imgmsg_to_cv2(msg.data[1])
        # fused = np.concatenate((left[:,0:230], right), axis=1)
        cv2.imwrite(self.output_folder + "/left/frame%d.jpg" % self.count, left)
        cv2.imwrite(self.output_folder + "/right/frame%d.jpg" % self.count, right)
        self.count += 1
        end_time = time.time()
        seconds = end_time - self.start_time
        fps = 1.0 / seconds
        # print("Estimated frames per second : {0}".format(fps))
        self.start_time = end_time

        h,w = left.shape[:2]
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
        # transformed_right = cv2.warpAffine(right, self.affine_matrix, (left.shape[1] + abs(int(self.dx)), left.shape[0]))
        
        canvas = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)
        canvas[offset[1]:offset[1]+h, offset[0]:offset[0]+w] = left
        stitched = np.maximum(canvas, transformed_right)
        # expanded_left = np.zeros_like(transformed_right)
        # expanded_left[:left.shape[0],:left.shape[1]] = left
        # stitched = np.maximum(expanded_left,transformed_right)

        cv2.imshow("stitched", stitched)
        cv2.imshow("left", left)
        cv2.imshow("right", right)
        cv2.waitKey(1)


        #### DEPTH ####
        stereo_depth, x, y, norm_disparity = self.stereo.update(left, right)
        if x is None:
            x = 0.0
            y = 0.0
        self.stereo_pub.publish(Float32MultiArray(data=[stereo_depth, x, y]))
        # else:
        #     self.stereo_pub.publish(Float32MultiArray(data=[1000000000000.0, 0.0, 0.0]))
        cv2.imwrite(self.output_folder + "/depth/frame%d.jpg" % self.count, norm_disparity)
        # self.depth_count += 1
        if self.first:
            plt.ion()
            self.fig, ax = plt.subplots()
            self.im = ax.imshow(norm_disparity)   
            plt.show()
            self.first = 0
        else:
            self.im.set_data(norm_disparity)
            self.fig.canvas.flush_events()
        


        


    def img_callback_detect(self, data):
        # self.destroy_subscription(self.frames_sub)
        # cv2.destroyWindow("fused")
        # self.get_logger().info('Receiving video frame')
        # print(data.shape)
        current_frame = self.br.compressed_imgmsg_to_cv2(data)
        # shesh = '/home/ranger/drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware/'
        # cv2.imwrite(shesh + "images/frame%d.jpg" % self.count, current_frame)
        cv2.imwrite(self.output_folder + "/detection/frame%d.jpg" % self.count, current_frame)
        self.detect_count += 1
        # print(current_frame.shape)
        end_time = time.time()
        seconds = end_time - self.start_time
        fps = 1.0 / seconds
        print("Estimated frames per second : {0}".format(fps))
        self.start_time = end_time
        cv2.imshow("detection", current_frame)   
        cv2.waitKey(1)

    def img_callback_depth(self, data):
        # self.get_logger().info('Receiving other video frame')
        current_frame = self.br.compressed_imgmsg_to_cv2(data)
        # shesh = '/home/ranger/drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware/'
        # cv2.imwrite(shesh + "images/frame%d.jpg" % self.count, current_frame)
        # self.count += 1
        cv2.imwrite(self.output_folder + "/depth/frame%d.jpg" % self.count, current_frame)
        # self.depth_count += 1
        if self.first:
            plt.ion()
            self.fig, ax = plt.subplots()
            self.fig2, ax2 = plt.subplots()
            self.im = ax.imshow(current_frame)   
            plt.show()
            self.first = 0
        else:
            self.im.set_data(current_frame)
            self.fig.canvas.flush_events()
    
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