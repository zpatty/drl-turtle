import os, sys
import traceback
import rclpy

from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.parameter_event_handler import ParameterEventHandler
from cv_bridge import CvBridge

import transforms3d.quaternions as quat
import transforms3d.euler as euler

from turtle_interfaces.msg import TurtleTraj, TurtleSensors, TurtleCtrl, TurtleMode, TurtleState, TurtleCam
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import String, Float32MultiArray

import numpy as np
import time
import scipy
submodule = os.path.expanduser("~") + "/drl-turtle/ros2_ws/install/turtle_hardware/lib/python3.12/site-packages/turtle_hardware/"
sys.path.append(submodule)
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
        
        # IMU, depth, etc
        self.sensors_sub = self.create_subscription(
            TurtleSensors,
            'turtle_imu',
            self.sensors_callback,
            qos_profile
        )

        self.stereo_sub = self.create_subscription(
            Float32MultiArray,
            'stereo',
            self.stereo_callback,
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

        # gamepad
        self.gamepad_sub = self.create_subscription(
            Float32MultiArray,
            'gamepad',
            self.gamepad_callback,
            qos_profile
        )

        # Depth frames
        self.publisher_depth = self.create_publisher(
            CompressedImage, 
            'video_frames_depth', 
            qos_profile
        )

        self.mode_sub = self.create_subscription(
            TurtleMode,
            'turtle_mode',
            self.turtle_mode_callback,
            qos_profile)
        
        # continously publishes the current motor position      
        self.desired_sub = self.create_subscription(
            TurtleCtrl,
            'turtle_ctrl_params',
            self.turtle_desired_callback,
            qos_profile
        )

        self.create_rate(1000)

        self.acc = np.zeros((3,1))
        self.gyr = np.zeros((3,1))
        self.quat = [1.0, 0.0, 0.0, 0.0]
        self.quat_first = True
        self.quat_init = [0.0, 1.0, 0.0, 0.0]
        self.off_flag = False
     

        self.print_sensors = True
        self.pitch_d = 0.0
        self.yaw_d = 0.0
        self.qd = quat.axangle2quat([0.0, 0.0, 0.0], np.pi/2)
        self.last_yaw = 0.0
        self.yaw_accumulator = 0.0
        self.rand_yaw = 0.0
        self.Ht = np.block([[0,0,0], [np.eye(3,3)]]).T
        self.centroids = []
        self.centroids_hist = []
        self.pilot = "track"
        self.remote_v = [0.0, 0.0, 0.0, 0.0]
        self.altitude_d = 20.0
        self.depth_d = 0.0
        self.altitude = [0.0]
        self.alt_confidence = 0.0
        self.depth_sensor = 0.0
        self.euler_convention = 'szyx'
        self.omega = np.array([0, 0, 0])
        
        self.br = CvBridge()
        self.stereo_depth = None
        self.x = None
        self.y = None
        self.last_time = time.time()      
        mode_msg = TurtleMode()
        self.mode_pub = self.create_publisher(
        TurtleMode,
        'turtle_mode',
        qos_profile)
        print('Position')
        mode_msg.mode = "p"
        self.mode = mode_msg.mode
        self.mode_pub.publish(mode_msg)

    def _timer_cb(self):
        
        if self.stereo_depth  is not None:
            if self.stereo_depth < 13:
                x_bounds = [157,425]
                y_bounds = [30,450]
                u_euler = euler.quat2euler(self.quat)
                if (self.x - 157) / (425 - 157) > 0.5:
                    u_yaw = -1.0
                else:
                    u_yaw = 1.0
                u[3] += u_yaw
        
        print(f"[DEBUG] \n mode: ", self.pilot, "\n ctrl: ", np.array(u), "\n quat: ", np.array(self.quat), "\n alt: ", filtered_alt[0], "\n depth: ", 
                  self.depth_sensor, "\n desired depth", self.depth_d, "\n yaw", heading, "\n yaw desired", 
                  self.yaw_d,"\n centroids: ", self.centroids,"\n depth: ", [self.stereo_depth, self.x, self.y], "\n")
        # u = np.clip(u, -1.0, 1.0)
        # print(f"u: {u}")
        # self.config_pub.publish(Float32MultiArray(data=u))
        # print("published!!!!")
        
    def np2msg(self, mat):
        """
        flattens nd numpy array into a lst for ros pkg messaging
        """
        nq = 10
        squeezed = np.reshape(mat, (nq * mat.shape[1]))
        return squeezed.tolist()
    
    def turtle_desired_callback(self, msg):
        self.yaw_d = msg.yaw
        self.depth_d = msg.pitch
        self.altitude_d = msg.fwd
    
    def tracker_callback(self, msg):
        self.centroids = msg.data
        print(msg.data)
                
    def img_callback(self, msg):
        left = self.br.compressed_imgmsg_to_cv2(msg.data[0])
        right = self.br.compressed_imgmsg_to_cv2(msg.data[1])
        self.depth, self.x, self.y, depth_frame = self.stereo.stereo_update(left, right)

        self.publisher_depth.publish(self.br.cv2_to_compressed_imgmsg(depth_frame))

    def turtle_mode_callback(self, msg):
        if msg.mode == "kill":
            raise KeyboardInterrupt

def main():
    
    rclpy.init()
    planner = TurtlePlanner()
    try:
        rclpy.spin(planner)
    except KeyboardInterrupt:
        print("shutdown")
    except Exception as e:
        print("some error occurred")
        traceback.print_exc()
        exec_type, obj, tb = sys.exc_info()
        fname = os.path.split(tb.tb_frame.f_code.co_filename)[1]
        print(exec_type, fname, tb.tb_lineno)
        print(e)
    rclpy.shutdown()
    print("planner node shutdown")

if __name__ == '__main__':
    main()