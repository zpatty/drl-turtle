import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import rclpy.parameter

from rclpy.parameter_event_handler import ParameterEventHandler
from cv_bridge import CvBridge

import os, sys
submodule = os.path.expanduser("~") + "/drl-turtle/ros2_ws/install/turtle_hardware/lib/python3.12/site-packages/turtle_hardware/"
sys.path.append(submodule)
import transforms3d.quaternions as quat
import transforms3d.euler as euler
from turtle_interfaces.msg import TurtleTraj, TurtleSensors, TurtleCtrl, TurtleMode, TurtleState, TurtleCam
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import String, Float32MultiArray
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
import numpy as np
import serial
import time
import json
from StereoProcessor import StereoProcessor

np.set_printoptions(precision=2, suppress=True)  # neat printing


def clip(value, lower=-1.0, upper=1.0):
    return lower if value < lower else upper if value > upper else value

def cayley(phi):
    return 1/np.sqrt(1 + np.linalg.norm(phi)**2) * np.insert(phi, 0, 1.0)

def c_map(theta):
    theta = theta + np.pi
    if theta >= 2 * np.pi:
        theta = theta - 2 * np.pi
    elif theta < 0.0:
        theta = 2 * np.pi - theta

class TurtlePlanner(Node):

# class TurtleRobot(Node, gym.Env):
    """
    Takes input from the cameras, vision-based tracker, sensors (IMU, depth), teleop controller,
    and proprioception. Publishes the desired control parameters
    """

    def __init__(self, params=None):
        super().__init__('turtle_planner_node')

        # ros2 qos profile
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        timer_cb_group = None
        self.call_timer = self.create_timer(0.05, self._timer_cb, callback_group=timer_cb_group)

        # SUBSCRIBERS AND PUBLISHER
        # Frames from camera
        self.cam_subscription = self.create_subscription(
            TurtleCam,
            'frames',
            self.img_callback,
            qos_profile
            )
        
        # IMU, depth, etc
        self.sensors_sub = self.create_subscription(
            TurtleSensors,
            'turtle_imu',
            self.sensors_callback,
            qos_profile
        )

        # Centroids from tracker
        self.centroids_sub = self.create_subscription(
            Float32MultiArray,
            'centroids',
            self.tracker_callback,
            qos_profile
        )

        # 4DOF control params
        self.config_pub = self.create_publisher(
            Float32MultiArray,
            'turtle_4dof',
            qos_profile
        )

        # Depth frames
        self.publisher_depth = self.create_publisher(
            CompressedImage, 
            'video_frames_depth', 
            qos_profile
        )

        self.create_rate(1000)
        self.depth = 0.0

        self.acc = np.zeros((3,1))
        self.gyr = np.zeros((3,1))
        self.quat = [1.0, 0.0, 0.0, 0.0]
        self.off_flag = False
     

        self.print_sensors = True
        self.pitch_d = 0.0
        self.yaw_d = 0.0
        self.qd = np.array([1.0, 0.0, 0.0, 0.0])

        self.centroids = []
        self.pilot = "auto"

        self.stereo = StereoProcessor()
        self.br = CvBridge()
        self.depth = None
        self.x = None
        self.y = None
        # t = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        # self.folder_name =  "data/" + t
        # os.makedirs(self.folder_name)
        


    def _timer_cb(self):
        # 
        if self.pilot == "auto":
            if not self.centroids:
                if np.random.uniform(0.0, 1.0) > 0.99:
                    # v = cayley(np.random.normal([0.0, 0.0, 0.0],[0.5, 0.00000000000000000000005, 0.05]))
                    euler_v = np.random.normal([0.0, 0.0, 0.0], [0.5, 0.0, 0.05]).tolist()
                    v = euler.euler2quat(*tuple(euler_v))
                    # self.pitch_d = self.pitch_d + v[0]
                    # if self.pitch_d > np.pi:
                    #     self.pitch_d = - self.pitch_d + np.pi
                    # self.yaw_d = self.yaw_d + v[1]
                    self.qd = quat.qmult(self.qd, v)
                    # print("CHANGED")
                err = quat.qmult(self.qd, quat.qinverse(self.quat))
                u_euler = euler.quat2euler(err)
                pitch_d = euler.quat2euler(self.qd)
                # u = [1.0, 0.0, - u_euler[2], 0, - u_euler[2]]
                u = [1.0, u_euler[1], - u_euler[2], clip(2.0 * u_euler[0]), - u_euler[2]]
                # u = [1.0, , 0.0, 0.0, 0.0]
                print(f"[DEBUG] euler: ", np.array(euler.quat2euler(self.quat)), "desired: ", np.array(pitch_d), "\n")
            else:
                err = quat.qmult([1.0, 0.0, 0.0, 0.0], quat.qinverse(self.quat))
                u_euler = euler.quat2euler(err)
                u_yaw = self.centroids[0]/870 * 2 - 1
                u_pitch = 1 - self.centroids[1]/480 * 2
                u = [1.0, u_euler[1], u_pitch, u_yaw, u_pitch]
                print(f"[DEBUG] euler: ", np.array(euler.quat2euler(self.quat)), "\n")
        elif self.pilot == "remote":
            v = self.remote_v
            u_pitch = v[0]
            u_yaw = v[1]
            u_fwd = v[2]
            u_roll = v[3]
            u = [u_fwd, u_roll, u_pitch, u_yaw, u_pitch]

        if self.depth:
            u_euler = euler.quat2euler(self.quat)
            u_yaw = - (self.x/870 * 2 - 1)
            u_pitch = - (1 - self.y/480) * 2
            u_fwd = 1.0 - 2.0 * 1 / (1 + np.exp(((self.depth - 0.5) - 0.1)/0.1))
            u[0] = u_fwd
            # u[1] = u_euler[1]
            u[2:] = [u_pitch, u_yaw, u_pitch]
            print(f"[DEBUG] depth: ", self.depth, "\n")
        # print("[DEBUG] quat: ", np.array(self.quat), "desired: ", self.qd, "\n")
        
        print(f"[DEBUG] u: {np.array(u)}\n")
        self.config_pub.publish(Float32MultiArray(data=u))
        



    def np2msg(self, mat):
        """
        flattens nd numpy array into a lst for ros pkg messaging
        """
        nq = 10
        squeezed = np.reshape(mat, (nq * mat.shape[1]))
        return squeezed.tolist()
    
    def sensors_callback(self, msg):
        """
        Callback function that takes in list of squeezed arrays
        msg: [quat acc gyr voltage t_0 q dq ddq u qd t]
        """    
        self.quat = msg.imu.quat.tolist()
        
    
    def tracker_callback(self, msg):
        self.centroids = msg.data
        
    def img_callback(self, msg):
        left = self.br.compressed_imgmsg_to_cv2(msg.data[0])
        right = self.br.compressed_imgmsg_to_cv2(msg.data[1])
        self.depth, self.x, self.y, depth_frame = self.stereo.stereo_update(left, right)

        self.publisher_depth.publish(self.br.cv2_to_compressed_imgmsg(depth_frame))

def main():
    
    rclpy.init()
    planner = TurtlePlanner()
    try:
        rclpy.spin(planner)
        # rclpy.shutdown()
    except KeyboardInterrupt:
        print("shutdown")
    except Exception as e:
        print("some error occurred")
        # turtle_node.shutdown_motors()
        exec_type, obj, tb = sys.exc_info()
        fname = os.path.split(tb.tb_frame.f_code.co_filename)[1]
        print(exec_type, fname, tb.tb_lineno)
        print(e)
    # turtle_node.save_data()

if __name__ == '__main__':
    main()