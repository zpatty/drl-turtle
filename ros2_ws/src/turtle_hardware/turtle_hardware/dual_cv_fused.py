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
from turtle_interfaces.msg import TurtleTraj, TurtleSensors

from std_msgs.msg import String
from std_msgs.msg import Float64MultiArray

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

        # KL=np.array([[417.15751114066205, 0.0, 336.595336628034], [0.0, 416.8576501537559, 241.5489118345027], [0.0, 0.0, 1.0]])
        # DL=np.array([[-0.06815812211170555], [-0.016732544509364528], [0.029182156593969097], [-0.017701284426359723]])
        DIM=(640, 480)
        # KR=np.array([[416.3903560278583, 0.0, 343.1831889045121], [0.0, 415.88140111385025, 241.99492603370734], [0.0, 0.0, 1.0]])
        # DR=np.array([[-0.06197454939758593], [-0.031440749408005376], [0.04248811930174599], [-0.02113466201121944]])
        # T = np.array([[65.1823933534524,  -4.73724842509345,   -20.8527190447127]])
        # R = np.array([[0.773850568208457,    0.0947576135973355,  0.626239804506858],
        #             [-0.142154340481768, 0.989504617178404,   0.0259375416108133],
        #             [-0.617209398474815,  -0.109094487706562,  0.779198916314955]])
        # R=np.array([[0.8468346068067919, 0.09196014378063505, 0.5238458558299675], [-0.09612880246461031, 0.9951817136007952, -0.01930311506739112], [-0.5230969337045535, -0.03401012893874326, 0.8515943336345446]])
        # T=np.array([[2.555600192469948], [-0.10876708731459717], [-0.8815627957762043]])
        KL=np.array([[415.3865625264921, 0.0, 340.2089884295033], [0.0, 414.97465263535884, 242.5868155566783], [0.0, 0.0, 1.0]])
        self.cam_left_params = KL
        DL=np.array([[-0.04736548487859318], [-0.0610307853579702], [0.07719884723889463], [-0.03916741540522904]])
        KR=np.array([[416.7150603771628, 0.0, 334.41222209893743], [0.0, 415.87631065929355, 243.07735361304927], [0.0, 0.0, 1.0]])
        DR=np.array([[-0.05766925342572732], [-0.03805959476299588], [0.01818581757623005], [0.0050609238522335565]])
        R=np.array([[0.8468488902168668, -0.09608976792441304, -0.5230809819126417], [0.09196308292756036, 0.9951841743019882, -0.03393008395698958], [0.5238222489708887, -0.019370485773520248, 0.8516073248651506]])
        T=np.array([[-2.6360517899826723], [-0.15711102688565556], [-0.5894243721990178]])*22.0
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

        # self.stereo.setMinDisparity(0)
        # self.stereo.setTextureThreshold(0)

        # #post filtering parameters: prevent false matches, help filter at boundaries
        # self.stereo.setSpeckleRange(2)
        # self.stereo.setSpeckleWindowSize(5)
        # self.stereo.setUniquenessRatio(2)

        # self.stereo.setDisp12MaxDiff(2)
    
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
    def stereo_update(self):
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
        # self.publisher_1 = self.create_publisher(Image, 'video_frames_1' , qos_profile)
        self.publisher_color = self.create_publisher(Image, 'video_frames_color' , qos_profile)
        # self.publisher_color_1 = self.create_publisher(Image, 'video_frames_color_1' , qos_profile)
        self.publisher_depth = self.create_publisher(Image, 'video_frames_depth' , qos_profile)

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

    

    print("created subscriber")
    DIM=(640, 480)

    

    cap0 = stream.cap
    cap1 = stream.cap1
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
            stream.stereo_update()
            print(minimal_subscriber.flag)
            rclpy.spin_once(minimal_subscriber)
            # ret0, left = cap1.read()
            # ret1, right = cap0.read()
            ret1 = stream.ret
            ret0 = stream.ret1
            left = stream.frame
            right = stream.frame1
            fused = fuse_feeds(left, right)
            minimal_subscriber.publisher_color.publish(minimal_subscriber.br.cv2_to_imgmsg(fused, encoding="bgr8"))
            # minimal_subscriber.publisher_color_1.publish(minimal_subscriber.br.cv2_to_imgmsg(left, encoding="bgr8"))

                # just undistorted, no stereo
            # fixedLeft = cv2.remap(left, stream.L_undist_map[0], stream.L_undist_map[1], cv2.INTER_LINEAR)
            # fixedRight = cv2.remap(right, stream.R_undist_map[0], stream.R_undist_map[1], cv2.INTER_LINEAR)
            # cv.imshow("fixedLeft", fixedLeft)
            # cv.imshow("fixedRight", fixedRight)
            #stereo
            fixedLeft = cv2.remap(left, stream.left1, stream.left2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            fixedRight = cv2.remap(right, stream.right1, stream.right2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            # cv.imshow("fixedLeft", fixedLeft)
            # cv.imshow("fixedRight", fixedRight)

            grayLeft = cv2.cvtColor(fixedLeft, cv2.COLOR_BGR2GRAY)
            grayRight = cv2.cvtColor(fixedRight, cv2.COLOR_BGR2GRAY)
            disparity = stream.stereo.compute(grayLeft,grayRight)
            denoise = 5
            noise=cv2.erode(disparity,np.ones((denoise,denoise)))
            noise=cv2.dilate(noise,np.ones((denoise,denoise)))
            disparity = cv2.medianBlur(noise, ksize=5)
            # norm_disparity = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            invalid_pixels = disparity < 0.0001
            disparity[invalid_pixels] = 0
            norm_disparity = np.array((disparity/16.0 - stream.stereo.getMinDisparity())/stream.stereo.getNumDisparities(), dtype='f')
            points3D = cv2.reprojectImageTo3D(np.array(disparity/16.0/1000.0, dtype='f'), stream.Q, handleMissingValues=True)
            depth = stream.Q[2,3]/stream.Q[3,2]/np.array(disparity/16.0, dtype='f')/1000
            x_bounds = [158,613]
            y_bounds = [30,450]
            depth_window = depth[30:450,158:613]
            finite_depth = depth_window[np.isfinite(depth_window)]
            stop_mean = np.median(finite_depth)
            h_thresh = 80
            w_thresh = 100
            # middle_window = depth_window[h : depth_window.shape[1] - h, w : depth_window.shape[0] - w]
            # left_window = depth_window[:, 0:int(depth_window.shape[1]/2)]
            # right_window = depth_window[:, int(depth_window.shape[1]/2):]
            # top_window = depth_window[0:int(depth_window.shape[0]/2), :]
            # bottom_window = depth_window[int(depth_window.shape[0]/2):, :]

            depth[np.isinf(depth)] = np.median(finite_depth)

            # colors = np.reshape(cv2.cvtColor(fixedLeft, cv2.COLOR_BGR2RGB), (-1,3))

            # projected_points,_ = cv2.projectPoints(np.reshape(points3D, (-1,3)), np.identity(3), np.array([0., 0., 0.]), \
            #               stream.cam_left_params, np.array([0., 0., 0., 0.]))

            # blank_img = np.zeros(norm_disparity.shape, 'uint8')

            # for i, pt in enumerate(projected_points):
            #     if np.isfinite(pt).all():
            #         # print(i)
            #         # print(pt.all())
            #         # print(np.isfinite(pt).all())
            #         pt_x = int(pt[0][0])
            #         pt_y = int(pt[0][1])
            #         if pt_x > 0 and pt_y > 0:
            #             # use the BGR format to match the original image type
            #             col = (int(colors[i, 2]), int(colors[i, 1]), int(colors[i, 0]))
            #             cv2.circle(blank_img, (pt_x, pt_y), 1, col)

            
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

                    if x_center > w_thresh + x_bounds[0] and x_center < x_bounds[1] - w_thresh and y_center > h_thresh + y_bounds[0] and y_center < y_bounds[1] - h_thresh:
                        # dwell
                        print("backwards...\n")
                        msg.data = 'backwards'
                        minimal_subscriber.cam_pub.publish(msg)
                    elif x_center < w_thresh + x_bounds[0]:
                        print("back right...\n")
                        msg.data = 'backr'
                        minimal_subscriber.cam_pub.publish(msg)
                    elif x_center > x_bounds[1] - w_thresh:
                        print("back left...\n")
                        msg.data = 'backl'
                        minimal_subscriber.cam_pub.publish(msg)
                    elif y_center < h_thresh + y_bounds[0]:
                        print("back down...\n")
                        msg.data = 'backd'
                        minimal_subscriber.cam_pub.publish(msg)
                    elif y_center > y_bounds[1] - h_thresh:
                        print("back up...\n")
                        msg.data = 'backu'
                        minimal_subscriber.cam_pub.publish(msg)
                    print((x_center,y_center))
                    stop = True
                else:
                    stop = False
             
            else:
              cv2.putText(norm_disparity, "SAFE!", (100,100),1,3,(0,255,0),2,3)
              stop = False
            print(stop_mean)
            # print(depth)
            minimal_subscriber.publisher_depth.publish(minimal_subscriber.br.cv2_to_imgmsg(norm_disparity, encoding="passthrough"))
            # minimal_subscriber.publisher_depth.publish(minimal_subscriber.br.cv2_to_imgmsg(depth[30:450,158:613], encoding="passthrough"))
            # local_max = disparity.max()
            # local_min = disparity.min()
            # disparity_visual = (disparity-local_min)*(1.0/(local_max-local_min))
            # cv.imshow("depth", norm_disparity)


            denoise = 15
            blur = cv2.GaussianBlur(fused, (5,5), 1)
            # Converting from BGR to HSV color space
            hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

            lab = cv2.cvtColor(fused, cv2.COLOR_BGR2LAB)
            bin_y = cv2.inRange(hsv, fixHSVRange(stream.lower_yellow), fixHSVRange(stream.upper_yellow))
            open_kern = np.ones((10,10), dtype=np.uint8)
            bin_y = cv2.morphologyEx(bin_y, cv2.MORPH_OPEN, open_kern, iterations=2)

            rip_y = fused.copy()
            rip_y[bin_y==0] = 0
            mark_y = cv2.addWeighted(fused, .4, rip_y, .6, 1)

            mask = bin_y #cv2.bitwise_not(bin_y)
            kernel = np.ones((10,10),np.uint8)

            mask_right = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            disp_mask_r=cv2.cvtColor(mask_right,cv2.COLOR_GRAY2BGR)
            minimal_subscriber.publisher_.publish(minimal_subscriber.br.cv2_to_imgmsg(disp_mask_r, encoding="bgr8"))

            # denoise = 15
            # blur = cv2.GaussianBlur(left, (5,5), 1)
            # # Converting from BGR to HSV color space
            # hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

            # lab = cv2.cvtColor(left, cv2.COLOR_BGR2LAB)
            # bin_y = cv2.inRange(hsv, fixHSVRange(lower_yellow), fixHSVRange(upper_yellow))
            # open_kern = np.ones((10,10), dtype=np.uint8)
            # bin_y = cv2.morphologyEx(bin_y, cv2.MORPH_OPEN, open_kern, iterations=2)

            # rip_y = left.copy()
            # rip_y[bin_y==0] = 0
            # mark_y = cv2.addWeighted(left, .4, rip_y, .6, 1)

            # mask = bin_y #cv2.bitwise_not(bin_y)
            # kernel = np.ones((10,10),np.uint8)

            # mask_left = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            # disp_mask_l=cv2.cvtColor(mask_left,cv2.COLOR_GRAY2BGR)
            # minimal_subscriber.publisher_1.publish(minimal_subscriber.br.cv2_to_imgmsg(disp_mask_l, encoding="bgr8"))
            # big_mask = np.append(big_mask, mask, axis=2)

            cnts, _ = cv2.findContours(mask_right, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cnt_s = sorted(cnts, key=cv2.contourArea)

            if not stop:
                if not (len(cnts) == 0):
                # if stop_mean < 0.1:
                #     # dwell
                #     print("backwards...\n")
                #     msg.data = 'backwards'
                #     minimal_subscriber.cam_pub.publish(msg)
                # elif np.median(left_window[np.isfinite(left_window)]) < 0.5:
                #     print("back right...\n")
                #     msg.data = 'backr'
                #     minimal_subscriber.cam_pub.publish(msg)
                # elif np.median(right_window[np.isfinite(right_window)]) < 0.5:
                #     print("back left...\n")
                #     msg.data = 'backl'
                #     minimal_subscriber.cam_pub.publish(msg)
                # elif np.median(top_window[np.isfinite(top_window)]) < 0.5:
                #     print("back down...\n")
                #     msg.data = 'backd'
                #     minimal_subscriber.cam_pub.publish(msg)
                # elif np.median(bottom_window[np.isfinite(bottom_window)]) < 0.5:
                #     print("back up...\n")
                #     msg.data = 'backu'
                #     minimal_subscriber.cam_pub.publish(msg)
                # else:
                    cnt = cnt_s[-1]
                    (x,y),radius = cv2.minEnclosingCircle(cnt)
                    center = (int(x),int(y))
                    radius = int(radius)
                    cv2.circle(mask,center,radius,10,2)
                    centerMask=cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
                    # minimal_subscriber.publisher_.publish(minimal_subscriber.br.cv2_to_imgmsg(centerMask, encoding="bgr8"))
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
                    if abs(centroid[0] - DIM[0]/2) < stream.turn_thresh and abs(centroid[1] - DIM[1]/2) < stream.dive_thresh:
                        # output straight primitive
                        print("go straight...\n")
                        msg.data = 'straight'
                        minimal_subscriber.cam_pub.publish(msg)
                    elif centroid[0] > DIM[0] - (DIM[0]/2 - stream.turn_thresh):
                        # turn right
                        print("turn right...\n")
                        msg.data = 'turnrf'
                        minimal_subscriber.cam_pub.publish(msg)
                    elif centroid[0] < (DIM[0]/2 - stream.turn_thresh):
                        # turn left
                        print("turn left...\n")
                        msg.data = 'turnlf'
                        minimal_subscriber.cam_pub.publish(msg)
                    elif centroid[1] > DIM[1] - (DIM[1]/2 - stream.dive_thresh):
                        # dive
                        print("dive...\n")
                        msg.data = 'dive'
                        minimal_subscriber.cam_pub.publish(msg)
                    elif centroid[1] < (DIM[1]/2 - stream.dive_thresh): 
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
            else:
                print("obstacle detected")
    except KeyboardInterrupt:
        print("cntrl c input: shutting down...")
        stream.stop_process()
        minimal_subscriber.destroy_node()
        rclpy.shutdown()
    # except:
    #     print("general error: shutting down...")
    #     stream.stop_process()
    #     minimal_subscriber.destroy_node()
    #     rclpy.shutdown()
    print("closing")
    stream.stop_process()
    minimal_subscriber.destroy_node()
    rclpy.shutdown()
    # cap0.release()

if __name__ == '__main__':
    main()
