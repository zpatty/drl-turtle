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

from fuse import fuse_feeds
from turtle_dynamixel.dyn_functions import *                    # Dynamixel support functions
from turtle_interfaces.msg import TurtleTraj, TurtleSensors, TurtleCtrl

from std_msgs.msg import String
from std_msgs.msg import Float64MultiArray

from datetime import datetime

class CamStream():
    """
    Class that just reads from the camera
    """
    def __init__(self, idx=0):
        self.cap = cv2.VideoCapture(0)
        self.cap1 = cv2.VideoCapture(4)
        self.ret, self.frame = self.cap.read()
        self.ret1, self.frame1 = self.cap1.read()
        self.stopped = False

        cv_params, __ = self.parse_cv_params()
        self.turn_thresh = cv_params["turn_thresh"]
        self.dive_thresh = cv_params["dive_thresh"]

        self.lower_yellow = np.array(cv_params["lower_yellow"])
        self.upper_yellow = np.array(cv_params["upper_yellow"])

        DIM=(640, 480)
        KL=np.array([[418.862685263043, 0.0, 344.7127322430985], [0.0, 418.25718294545146, 244.86612342606873], [0.0, 0.0, 1.0]])
        DL=np.array([[-0.10698601862156384], [0.1536626773154175], [-0.25203748540815346], [0.134699123195767]])
        KR=np.array([[412.8730413534633, 0.0, 334.94298327120686], [0.0, 413.2522575868915, 245.1860564579], [0.0, 0.0, 1.0]])
        DR=np.array([[0.003736892852052395], [-0.331577509789992], [0.5990981643072193], [-0.3837158104256219]])
        R=np.array([[0.8484938703183661, -0.053646440984050164, -0.5264790702410727], [0.060883267910572095, 0.9981384545889453, -0.00358513030742383], [0.5256913350253065, -0.029011805192654422, 0.8501805310866477]])
        T=np.array([[-2.178632057371688], [-0.03710693058315735], [-0.6477466090945703]])*25.4

        R1,R2,P1,P2,self.Q = cv2.fisheye.stereoRectify(KL,DL,KR,DR,DIM,R,T, cv2.fisheye.CALIB_ZERO_DISPARITY)
        print(self.Q)
        self.L_undist_map=cv2.fisheye.initUndistortRectifyMap(KL,DL,np.identity(3),KL,DIM,cv2.CV_32FC1)
        self.R_undist_map=cv2.fisheye.initUndistortRectifyMap(KR,DR,np.identity(3),KR,DIM,cv2.CV_32FC1)
        self.left1, self.left2 = cv2.fisheye.initUndistortRectifyMap(KL, DL, R1, P1, DIM, cv2.CV_32FC1)
        self.right1, self.right2 = cv2.fisheye.initUndistortRectifyMap(KR,DR, R2, P2, DIM, cv2.CV_32FC1)
        self.stereo = cv2.StereoBM.create(numDisparities=cv_params["numDisparities"], blockSize=cv_params["blockSize"])
        self.stereo.setMinDisparity(cv_params["MinDisparity"])
        self.stereo.setTextureThreshold(cv_params["TextureThreshold"])

        #post filtering parameters: prevent false matches, help filter at boundaries
        self.stereo.setSpeckleRange(cv_params["SpeckleRange"])
        self.stereo.setSpeckleWindowSize(cv_params["SpeckleWindowSize"])
        self.stereo.setUniquenessRatio(cv_params["UniquenessRatio"])

        self.stereo.setDisp12MaxDiff(cv_params["Disp12MaxDiff"])


    def start(self):
        Thread(target=self.get, args=()).start()
        Thread(target=self.get1, args=()).start()
        return self
    def get(self):
        while not self.stopped:
            self.ret, self.frame = self.cap.read()
    def get1(self):
        while not self.stopped:
            self.ret1, self.frame1 = self.cap1.read()
    def stop_process(self):
        self.stopped = True
        self.cap.release()
    def stereo_params_update(self):
        if self._last_update != os.stat('cv_config.json').st_mtime:
            cv_params, __ = self.parse_cv_params()
            self.stereo.setNumDisparities(cv_params["numDisparities"])
            self.stereo.setBlockSize(cv_params["blockSize"])
            self.stereo.setMinDisparity(cv_params["MinDisparity"])
            self.stereo.setTextureThreshold(cv_params["TextureThreshold"])

            #post filtering parameters: prevent false matches, help filter at boundaries
            self.stereo.setSpeckleRange(cv_params["SpeckleRange"])
            self.stereo.setSpeckleWindowSize(cv_params["SpeckleWindowSize"])
            self.stereo.setUniquenessRatio(cv_params["UniquenessRatio"])

            self.stereo.setDisp12MaxDiff(cv_params["Disp12MaxDiff"])
    def parse_cv_params(self):
        with open('cv_config.json') as config:
            param = json.load(config)
            self._last_update = os.fstat(config.fileno()).st_mtime
        print(f"[MESSAGE] Config: {param}\n")    
        # Serializing json
        config_params = json.dumps(param, indent=14)
        return param, config_params
    def fuse_feeds(self):
        self.left = self.frame
        self.right = self.frame1
        self.fused = np.concatenate((self.left[:,0:230], self.right), axis=1)
        return self.fused
    def stereo_update(self):
        fixedLeft = cv2.remap(self.left, self.left1, self.left2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        fixedRight = cv2.remap(self.right, self.right1, self.right2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        grayLeft = cv2.cvtColor(fixedLeft, cv2.COLOR_BGR2GRAY)
        grayRight = cv2.cvtColor(fixedRight, cv2.COLOR_BGR2GRAY)
        disparity = self.stereo.compute(grayLeft,grayRight)
        denoise = 5
        noise=cv2.erode(disparity,np.ones((denoise,denoise)))
        noise=cv2.dilate(noise,np.ones((denoise,denoise)))
        disparity = cv2.medianBlur(noise, ksize=5)
        # norm_disparity = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        invalid_pixels = disparity < 0.0001
        disparity[invalid_pixels] = 0
        norm_disparity = np.array((disparity/16.0 - self.stereo.getMinDisparity())/self.stereo.getNumDisparities(), dtype='f')
        self.norm_disparity = norm_disparity
        points3D = cv2.reprojectImageTo3D(np.array(disparity/16.0/1000.0, dtype='f'), self.Q, handleMissingValues=True)
        depth = self.Q[2,3]/self.Q[3,2]/np.array(disparity/16.0, dtype='f')/1000
        x_bounds = [158,613]
        y_bounds = [30,450]
        depth_window = depth[30:450,158:613]
        finite_depth = depth_window[np.isfinite(depth_window)]
        stop_mean = np.median(finite_depth)
        h_thresh = 80
        w_thresh = 100
        depth[np.isinf(depth)] = np.median(finite_depth)

        
        depth_thresh = 0.5 # Threshold for SAFE distance (in cm)

        # Mask to segment regions with depth less than threshold
        mask = cv2.inRange(depth,0.1,depth_thresh)

        # Check if a significantly large obstacle is present and filter out smaller noisy regions
        if np.sum(mask)/255.0 > 0.01*mask.shape[0]*mask.shape[1]:
            
            # Contour detection 
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cnts = sorted(contours, key=cv2.contourArea, reverse=True)
            
            # Check if detected contour is significantly large (to avoid multiple tiny regions)
            if cv2.contourArea(cnts[0]) > 0.01*mask.shape[0]*mask.shape[1]:
                
                x,y,w,h = cv2.boundingRect(cnts[0])
                x_center = int(x + w/2)
                y_center = int(y + h/2)
                # finding average depth of region represented by the largest contour 
                mask2 = np.zeros_like(mask)
                cv2.drawContours(mask2, cnts, 0, (255), -1)
                cv2.drawContours(norm_disparity, cnts, 0, (255), -1)
                # Calculating the average depth of the object closer than the safe distance
                depth_mean, _ = cv2.meanStdDev(depth, mask=mask2)
                    
                # Display warning text
                cv2.putText(norm_disparity, "WARNING !", (x+5,y-40), 1, 2, (0,0,255), 2, 2)
                cv2.putText(norm_disparity, "Object at", (x+5,y), 1, 2, (100,10,25), 2, 2)
                cv2.putText(norm_disparity, "%.2f m"%depth_mean, (x+5,y+40), 1, 2, (100,10,25), 2, 2)
                
                # STEER AWAY HERE
                # print(depth_thresh/depth_mean[0,0])
                return - 1.0 + 2.0 * 1 / (1 + np.exp((depth_mean[0,0] - depth_thresh)/depth_thresh))
            else:
                cv2.putText(norm_disparity, "SAFE!", (100,100),1,3,(0,255,0),2,3)
                return 1.0




def fixHSVRange(val):
    # Normal H,S,V: (0-360,0-100%,0-100%)
    # OpenCV H,S,V: (0-180,0-255 ,0-255)
    return (180 * val[0] / 360, 255 * val[1] / 100, 255 * val[2] / 100)

global flag
flag = ''

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
        self.subscription = self.create_subscription(
            String,
            'turtle_mode_cmd',
            self.keyboard_callback,
            qos_profile)
        
        self.config_pub = self.create_publisher(
            TurtleCtrl,
            'turtle_ctrl_params',
            qos_profile
        )
        self.stream = stream
        self.cam_pub = self.create_publisher(String, 'primitive', buff_profile)
        self.publisher_ = self.create_publisher(Image, 'video_frames' , qos_profile)
        # self.publisher_1 = self.create_publisher(Image, 'video_frames_1' , qos_profile)
        self.publisher_color = self.create_publisher(Image, 'video_frames_color' , qos_profile)
        # self.publisher_color_1 = self.create_publisher(Image, 'video_frames_color_1' , qos_profile)
        self.publisher_depth = self.create_publisher(Image, 'video_frames_depth' , qos_profile)

        self.br = CvBridge()

        self.create_rate(100)
        self.subscription  # prevent unused variable warning

        self.count = 0
        t = datetime.today().strftime("%m_%d_%Y_%H_%M_%S")
        folder_name =  "video/" + t
        os.makedirs(folder_name + "/fused")
        os.makedirs(folder_name + "/right")
        os.makedirs(folder_name + "/left")
        os.makedirs(folder_name + "/depth")
        self.output_folder = folder_name

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

    stream = CamStream().start()
    cam_pub = CamNode(stream)
    

    print("created subscriber")
    DIM=(640, 480)

    msg = TurtleCtrl()
    count = 0
    print("entering while loop")
    try:
        while(cam_pub.flag != 'stop'):
            stream.stereo_params_update()
            print(cam_pub.flag)
            rclpy.spin_once(cam_pub)

            stream.fuse_feeds()
            cam_pub.publisher_color.publish(cam_pub.br.cv2_to_imgmsg(stream.fused, encoding="bgr8"))
            msg = TurtleCtrl()
            msg.fwd = stream.stereo_update()
            msg.pitch = 1000.0
            msg.yaw = 1000.0
            msg.roll = 1000.0
            print(msg)
            cam_pub.config_pub.publish(msg)
            # print(depth)
            cam_pub.publisher_depth.publish(cam_pub.br.cv2_to_imgmsg(stream.norm_disparity, encoding="passthrough"))
            cv2.imwrite(cam_pub.output_folder + "/left/frame%d.jpg" % cam_pub.count, stream.left)
            cv2.imwrite(cam_pub.output_folder + "/right/frame%d.jpg" % cam_pub.count, stream.right)
            cv2.imwrite(cam_pub.output_folder + "/fused/frame%d.jpg" % cam_pub.count, stream.fused)
            cv2.imwrite(cam_pub.output_folder + "/depth/frame%d.jpg" % cam_pub.count, stream.norm_disparity)
            cam_pub.count += 1



    except KeyboardInterrupt:
        print("cntrl c input: shutting down...")
        stream.stop_process()
        cam_pub.destroy_node()
        # rclpy.shutdown()
    # except:
    #     print("general error: shutting down...")
    #     stream.stop_process()
    #     minimal_subscriber.destroy_node()
    #     rclpy.shutdown()
    print("closing")
    stream.stop_process()
    cam_pub.destroy_node()
    # rclpy.shutdown()
    # cap0.release()

if __name__ == '__main__':
    main()
