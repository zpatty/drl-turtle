import cv2 as cv
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.executors import Executor, MultiThreadedExecutor
from sensor_msgs.msg import Image

from cv_bridge import CvBridge, CvBridgeError

class Image_Publisher(Node):
    def __init__(self):
        super().__init__("image_tutorial")
        self.add_msg_to_info_logger("initializing node")
        self.image_publisher_ = self.create_publisher(Image, 'image', 10)
        self.bridge = CvBridge()

        
    def publish(self, cv2_image):
        try:
             msg = self.bridge.cv2_to_imgmsg(cv2_image, "bgr8")
             self.image_publisher_.publish(msg)
        except CvBridgeError as e:
             self.add_msg_to_info_logger(e)
        self.add_msg_to_info_logger("sending_image")
        

    def add_msg_to_info_logger(self, msg):
        self.get_logger().info(msg)

DIM=(1920, 1080)
KR=np.array([[936.3090354816636, 0.0, 1011.6603031360017], [0.0, 936.2302136924422, 542.9121240301195], [0.0, 0.0, 1.0]])
DR=np.array([[-0.0697808462865659], [-0.0031580486418187653], [0.0024539254744271204], [-0.0012931220378302362]])
KL=np.array([[935.5251123185054, 0.0, 990.8427164459354], [0.0, 936.1743386189506, 536.9829760404886], [0.0, 0.0, 1.0]])
DL=np.array([[-0.08446254512511926], [0.036252030276062254], [-0.03991208556882756], [0.01427235965589032]])
R=np.array([[0.1227003194381537, 0.9854418738780312, -0.11768153983327023], [-0.8864336535536629, 0.05549700973154362, -0.45951655003749525], [-0.44629587658556735, 0.16069970478694565, 0.8803383414483936]])
T=np.array([[-1.0727790320133055], [-2.814903957297424], [-0.654113161470161]])

R1,R2,P1,P2,Q = cv.fisheye.stereoRectify(KL,DL,KR,DR,DIM,R,T,cv.fisheye.CALIB_ZERO_DISPARITY)

#undistortion
L_undist_map=cv.fisheye.initUndistortRectifyMap(KL,DL,np.identity(3),KL,DIM,cv.CV_32FC1)
R_undist_map=cv.fisheye.initUndistortRectifyMap(KR,DR,np.identity(3),KR,DIM,cv.CV_32FC1)

#reprojectto3d
left1, left2 = cv.fisheye.initUndistortRectifyMap(KL, DL, R1, P1, DIM, cv.CV_32FC1)
right1, right2 = cv.fisheye.initUndistortRectifyMap(KR,DR, R2, P2, DIM, cv.CV_32FC1)

stereo = cv.StereoBM_create(numDisparities=48, blockSize=29)
#prefiltering parameters: indended to normalize brightness & enhance texture
#stereo.setPreFilterSize(5)
#^must be odd & btwn 5 & 255
stereo.setPreFilterCap(63)
#^must be btwn 1 and 63
#stereo.setPreFilterType(2)

#stereo correspondence parameters: find matches between camera views
stereo.setMinDisparity(-50)
stereo.setTextureThreshold(1)

#post filtering parameters: prevent false matches, help filter at boundaries
stereo.setSpeckleRange(50)
stereo.setSpeckleWindowSize(80)
stereo.setUniquenessRatio(9)

stereo.setDisp12MaxDiff(20)

def main(args=None):

    rclpy.init(args=args)

    image_publisher = Image_Publisher()
    image_publisher.add_msg_to_info_logger("initializing camera")

    cap0 = cv.VideoCapture(0)
    cap1 = cv.VideoCapture(1)
    cv.namedWindow("main", cv.WINDOW_NORMAL)

    while True:
        ret0, left = cap1.read()
        ret1, right = cap0.read()

        #just undistorted, no stereo
        # fixedLeft = cv.remap(left, L_undist_map[0], L_undist_map[1], cv.INTER_LINEAR)
        # fixedRight = cv.remap(right, R_undist_map[0], R_undist_map[1], cv.INTER_LINEAR)

        #stereo
        fixedLeft = cv.remap(left, left1, left2, cv.INTER_LINEAR)
        fixedRight = cv.remap(right, right1, right2, cv.INTER_LINEAR)

        grayLeft = cv.cvtColor(fixedLeft, cv.COLOR_BGR2GRAY)
        grayRight = cv.cvtColor(fixedRight, cv.COLOR_BGR2GRAY)
        disparity = stereo.compute(grayLeft,grayRight)
        local_max = disparity.max()
        local_min = disparity.min()
        disparity_visual = (disparity-local_min)*(1.0/(local_max-local_min))
        cv.imshow("depth", disparity_visual)

        image_publisher.publish(disparity_visual/255)

        # cv.imshow('left', fixedLeft)
        # cv.imshow('right', fixedRight)
        # cv.imshow('depth', depth)
        #cv.imshow('stereo', fixed)

        k = cv.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break

    image_publisher.destroy_node()
    rclpy.shutdown()    

    cap0.release()
    cap1.release()
    cv.destroyAllWindows()