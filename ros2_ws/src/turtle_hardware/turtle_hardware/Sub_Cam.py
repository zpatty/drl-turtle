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
from scipy.spatial.transform import Rotation as R_scipy


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
        os.makedirs(os.path.join(folder_name, "flat_left"))
        os.makedirs(os.path.join(folder_name, "flat_right"))
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

        self.KL = np.array([[708.3477312219868, 0.0, 260.69187590557686], [0.0, 675.3059166594338, 301.31936629865646], [0.0, 0.0, 1.0]])
        self.DL = np.array([[-0.39383047117877457], [6.721465255404687], [-35.99917141986595], [61.49579122578909]])
        self.KR = np.array([[667.0400978057647, 0.0, 334.8109094526051], [0.0, 644.922628956739, 364.07228200370565], [0.0, 0.0, 1.0]])
        self.DR = np.array([[0.8809516193294453], [-6.609640306922403], [21.549513701823056], [-24.149385093847197]])
        self.R = np.array([[0.8721459388442752, 0.02130940474841954, -0.4887815162490354], [-0.06589130347707366, 0.9950649291385254, -0.07418977641584823], [0.4847884048567273, 0.09691076342599622, 0.8692459412897253]])
        self.T = np.array([[-2.085136618149882], [0.1939622251215522], [-0.9258137973647751]])*25.4

        self.rotation = R_scipy.from_matrix(self.R)
        euler_angles = self.rotation.as_euler('zyx', degrees=True)

        self.yaw, self.pitch, self.roll = euler_angles

        self.fx = self.KL[0, 0] 
        self.baseline = np.linalg.norm(self.T)
        self.Z = 1000  # assume scene is ~1m away

        self.pixel_shift = int((self.fx * self.baseline) / self.Z)

        self.previous_num_keypoints = 0
        self.previous_H = None
        self.homography_sum = np.zeros((3, 3))
        self.frame_count = 0
        self.max_frames_to_average = 5


    def euler_to_rotation_matrix(self, yaw, pitch, roll):
        r = R_scipy.from_euler('zyx', [yaw, pitch, roll], degrees=True)
        return r.as_matrix()
    
    def get_3d_rotation_homography(self, K, yaw, pitch, roll):
        R_mat = self.euler_to_rotation_matrix(yaw, pitch, roll)
        return K @ R_mat @ np.linalg.inv(K)
    
    def get_rotated_image_bounds(self,image, H):
        h, w = image.shape[:2]
        corners = np.array([
            [0, 0],
            [w, 0],
            [w, h],
            [0, h]
        ], dtype=np.float32).reshape(-1, 1, 2)

        transformed_corners = cv2.perspectiveTransform(corners, H)
        x_coords = transformed_corners[:, 0, 0]
        y_coords = transformed_corners[:, 0, 1]

        min_x, max_x = np.min(x_coords), np.max(x_coords)
        min_y, max_y = np.min(y_coords), np.max(y_coords)

        return int(np.ceil(max_x - min_x)), int(np.ceil(max_y - min_y)), min_x, min_y
    
    def warp_image_3d_fullview(self, image, H):
        h, w = image.shape[:2]
        new_w, new_h, min_x, min_y = self.get_rotated_image_bounds(image, H)

        # Translate so the top-left corner is visible
        translation = np.array([
            [1, 0, -min_x],
            [0, 1, -min_y],
            [0, 0, 1]
        ])

        H_translated = translation @ H

        return cv2.warpPerspective(image, H_translated, (new_w, new_h), flags=cv2.INTER_LINEAR)


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

        h, w = left.shape[:2]
        image_size = (w, h)

        R1, R2, P1, P2, Q = cv2.fisheye.stereoRectify(
            self.KL, self.DL, self.KR, self.DR, image_size, self.R, self.T,
            flags=cv2.fisheye.CALIB_USE_INTRINSIC_GUESS, balance=1.0, newImageSize=image_size
        )

        l_map1, l_map2 = cv2.fisheye.initUndistortRectifyMap(
            K=self.KL, D=self.DL, R=np.eye(3), P=self.KL, size=image_size, m1type=cv2.CV_16SC2
        )

        r_map1, r_map2 = cv2.fisheye.initUndistortRectifyMap(
            K=self.KR, D=self.DR, R=np.eye(3), P=self.KR, size=image_size, m1type=cv2.CV_16SC2
        )

        # #Does not work rn tried to turn cameras virtually to be parallel in order to stitch them together using intrinsics
        # flat_left = cv2.remap(left, l_map1, l_map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        # flat_right = cv2.remap(right, r_map1, r_map2, interpolation=cv2.INTER_LINEAR,borderMode= cv2.BORDER_CONSTANT)

        # cv2.imshow("flat_left.jpg", flat_left)
        # cv2.imshow("flat_right.jpg", flat_right)

        # yaw_half = self.yaw / 2.0
        # pitch_half = self.pitch / 2.0
        # roll_half = self.roll / 2.0

        # H_left = self.get_3d_rotation_homography(self.KL, -yaw_half, -pitch_half/2, -roll_half)
        # H_right = self.get_3d_rotation_homography(self.KR, yaw_half, pitch_half/2, roll_half)

        # rectified_left = self.warp_image_3d_fullview(flat_left, H_left)
        # rectified_right = self.warp_image_3d_fullview(flat_right, H_right)
        # cv2.imwrite(self.output_folder + "/flat_left/frame%d.jpg" % self.count, rectified_left)
        # cv2.imwrite(self.output_folder + "/flat_right/frame%d.jpg" % self.count, rectified_right)

        # cv2.imshow("rectified_left", rectified_left)
        # cv2.imshow("rectified_right", rectified_right)


        #gave up now trying to use SIFT to stitch the images together
        undistorted_left = cv2.remap(left, l_map1, l_map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        undistorted_right = cv2.remap(right, r_map1, r_map2, interpolation=cv2.INTER_LINEAR,borderMode= cv2.BORDER_CONSTANT)

        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(undistorted_left, None)
        kp2, des2 = sift.detectAndCompute(undistorted_right, None)

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        good = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append(m)

        H_avg = self.previous_H if self.previous_H is not None else np.eye(3)

        num_keypoints = len(good)
        if num_keypoints >= self.previous_num_keypoints:
            self.previous_num_keypoints = num_keypoints

            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
            self.homography_sum += H
            print(self.homography_sum)

            self.frame_count += 1

            if self.frame_count >= self.max_frames_to_average:
                H_avg = self.homography_sum / self.frame_count
                self.previous_H = H_avg
                self.frame_count = 0
                self.homography_sum = np.zeros((3, 3))
            else:
                self.previous_H = H

        else:
            H_avg = self.previous_H if self.previous_H is not None else np.eye(3)

        h1, w1 = undistorted_left.shape[:2]
        h2, w2 = undistorted_right.shape[:2]

        corners_left = np.array([[0,0], [0,h1], [w1,h1], [w1,0]], dtype=np.float32).reshape(-1,1,2)
        corners_right = np.array([[0,0], [0,h2], [w2,h2], [w2,0]], dtype=np.float32).reshape(-1,1,2)

        warped_corners_right = cv2.perspectiveTransform(corners_right, H_avg)

        all_corners = np.concatenate((corners_left, warped_corners_right), axis=0)
        [x_min, y_min] = np.floor(all_corners.min(axis=0).ravel()).astype(int)
        [x_max, y_max] = np.ceil(all_corners.max(axis=0).ravel()).astype(int)

        output_width = x_max - x_min
        output_height = y_max - y_min

        translation = np.array([[1, 0, -x_min],
                                [0, 1, -y_min],
                                [0, 0, 1]])

        stitched2 = cv2.warpPerspective(undistorted_right, translation @ H_avg, (output_width, output_height))
        stitched2[-y_min:h1 - y_min, -x_min:w1 - x_min] = undistorted_left
        cv2.imshow("stitched2", stitched2)
        cropped_stitch = stitched2[-y_min:h1-y_min, 0:x_max-x_min]
        cv2.imshow("cropped_stitch", cropped_stitch)




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
        output_size = (xmax - xmin, ymax - ymin)

        affine_with_offset = self.affine_matrix.copy()
        affine_with_offset[:, 2] += offset

        transformed_right = cv2.warpAffine(right, affine_with_offset, output_size)
        
        canvas = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)
        canvas[offset[1]:offset[1]+h, offset[0]:offset[0]+w] = left
        stitched = np.maximum(canvas, transformed_right)

        cv2.imshow("stitched", stitched)
        # cv2.imshow("left", left)
        # cv2.imshow("right", right)
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