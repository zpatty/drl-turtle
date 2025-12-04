import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import rclpy.parameter
import traceback
from rclpy.parameter_event_handler import ParameterEventHandler
from cv_bridge import CvBridge
from matplotlib import pyplot as plt

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
import scipy
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
        # self.cam_subscription = self.create_subscription(
        #     TurtleCam,
        #     'frames',
        #     self.img_callback,
        #     qos_profile
        #     )
        
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
        
        self.mode_pub = self.create_publisher(
            TurtleMode,
            'turtle_mode',
            qos_profile)


        # continously publishes the current motor position      
        self.desired_pub = self.create_publisher(
            TurtleCtrl,
            'turtle_ctrl_params',
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
        self.yaw_d = 2.2

        self.qd = quat.axangle2quat([0.0, 0.0, 0.0], np.pi/2)
        self.last_yaw = 0.0
        self.yaw_accumulator = 0.0
        self.rand_yaw = 0.0
        self.Ht = np.block([[0,0,0], [np.eye(3,3)]]).T
        self.centroids = []
        self.centroids_hist = []
        self.pilot = "depth"
        self.remote_v = [0.0, 0.0, 0.0, 0.0]
        self.altitude_d = 20.0
        # self.depth_d = 0.5
        self.depth_d = 1.0
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
        self.turn_command = "left"
        self.first = True
        self.plot_depth = []
        self.flag_turn = False
        self.u_last = [1, 0, 0, 0, 0]
        self.experiment_time = time.time()
        self.flag_command = "null"
        mode_msg = TurtleMode()
        time.sleep(2)
        print('Position')
        mode_msg.mode = "p"
        self.mode = mode_msg.mode
        self.mode_pub.publish(mode_msg)
        self.time_elapsed = 0
    def _timer_cb(self):
        # 
        self.time = time.time() - self.last_time
        # self.time_elapsed = time.time() - self.experiment_time
        # print(self.time_elapsed)
        # YAW pitch roll 
        q_eul = euler.quat2euler(self.quat,self.euler_convention)
        # this is definitely yaw
        heading = q_eul[0]
        print(f"heading: {heading}")
        filtered_alt = scipy.signal.medfilt(self.altitude)
        # if self.time_elapsed <= 120:
            
        if self.pilot == "depth":
            
            print(f"[DEBUG] stereo depth: {self.stereo_depth}")
            x_bounds = [157,425]        # trial 1 and trial 2 may 16th with depth < 2.5 with OG intrinsics from Zach
            # x_bounds = [162,455]        # trial 3 and 4 may 16th with depth < 2.5 and OG intrinsics from Zach
            # x_bounds = [185, 450]
            y_bounds = [30,450]
            msg_d = TurtleCtrl()
            if self.stereo_depth  is not None:
                # we are about to hit an obstacle and are not in turn state
                if self.stereo_depth < 3.5 and not self.flag_turn:
                    u_euler = euler.quat2euler(self.quat)                    
                    if (self.x - x_bounds[0]) / (x_bounds[1] - x_bounds[0]) > 0.5:
                        print(f"[DEBUG] yaw left")
                        print(f"stereo depth: {self.stereo_depth}, x: {self.x}, y: {self.y}")

                        u_yaw = -1.0
                        u_roll = -1.0
                        self.yaw_d = np.arctan2(np.sin(heading + np.deg2rad(120)), np.cos(heading + np.deg2rad(120)))
                        # if heading + np.pi/2 > np.pi:
                            # self.yaw_d = - heading  + np.pi/2
                    else:
                        print(f"[DEBUG] yaw right")
                        print(f"stereo depth: {self.stereo_depth}, x: {self.x}, y: {self.y}")
                        u_yaw = 1.0
                        u_roll = 1.0
                        self.yaw_d = np.arctan2(np.sin(heading - np.deg2rad(120)), np.cos(heading - np.deg2rad(120)))
                        # if heading - np.pi/2 < -np.pi:
                        #     self.yaw_d = - heading  - np.pi/2            
            if self.stereo_depth is None:
                if len(self.plot_depth) < 100:
                    self.plot_depth.append(np.nan)
                else:
                    self.plot_depth.pop(0)
                    self.plot_depth.append(np.nan)
            else:
                if len(self.plot_depth) < 100:
                    self.plot_depth.append(self.stereo_depth)
                else:
                    self.plot_depth.pop(0)
                    self.plot_depth.append(self.stereo_depth)
            
            msg_d.yaw = self.yaw_d
            msg_d.pitch = self.depth_d
            self.desired_pub.publish(msg_d)
            # if self.time > 6000000.0:
            if self.time > 500.0:
                print("------------going to surface!!!---------------")
                self.depth_d = 0.0
            if self.time > 600.0:
                print("------------------ending program!!-----------------")
                mode_msg = TurtleMode()
                mode_msg.mode = "kill"
                self.mode = mode_msg.mode
                self.mode_pub.publish(mode_msg)
                raise KeyboardInterrupt

            depth_err = self.depth_sensor - self.depth_d
            # (yaw, pitch, roll)
            qd_dive = euler.euler2quat(*tuple([self.yaw_d, np.clip(- 4.0 * depth_err, -np.pi/2, np.pi/2), 0.0]),self.euler_convention)
            # print(f"qd_dive: {qd_dive}")
            q_inv = quat.qinverse(self.quat)
            # err = 2.0 * quat.qmult(qd_dive, q_inv)
            err = 2.0 * quat.qmult(q_inv, qd_dive)

            # Add this check to ensure we get the "short path" rotation:
            if err[0] < 0:  # If scalar part is negative
                err = -err   # Flip the entire quaternion
            # print(f"err: {err}")
            # [x, y, z]
            w = np.clip(1.0*(np.array(err[1:]) - 0.1*np.array(self.omega)), -1.0, 1.0)
            # print(f"w: {w}")
            if not self.flag_turn:
                u = [1.0, - w[0], - 5.0*depth_err,  -w[2], -5.0*depth_err - 0.25]
                print(f"no flag u: {u}")
            else:
                u = self.u_last

            if self.stereo_depth  is not None:
                if self.stereo_depth < 3.0 and not self.flag_turn:
                    self.flag_turn = True
                    u_euler = euler.quat2euler(self.quat)
                    if (self.x - x_bounds[0]) / (x_bounds[1] - x_bounds[0]) > 0.5:
                        print("[DEBUG] turn left ")
                        print(f"stereo depth: {self.stereo_depth}, x: {self.x}, y: {self.y}")

                        self.flag_command = "left"
                        u_yaw = -1.0
                        u_roll = -1.0
                        u[3] = u_yaw
                        u[1] = u_roll
                        u[4] = -1.0
                        # u[0] = 0.75
                        
                    else:
                        print("[DEBUG] turn right")
                        print(f"stereo depth: {self.stereo_depth}, x: {self.x}, y: {self.y}")

                        self.flag_command = "right"
                        u_yaw = 1.0
                        u_roll = 1.0
                        u[3] = u_yaw
                        u[1] = u_roll
                        u[4] = -1.0
                        # u[0] = 0.75
                    
                    self.u_last = u
                if self.stereo_depth > 3.0 and self.flag_turn:
                    self.flag_turn = False
                    self.yaw_d = heading
            # print(f"current quat norm: {np.linalg.norm(self.quat)}")
            # print(f"desired quat norm: {np.linalg.norm(qd_dive)}")
            # print(f"yaw desired: {self.yaw_d}")
            # print(f"yaw error should be: {self.yaw_d - heading}")  # Expected: -1.0 - (-1.573) = +0.573
            # print(f"quaternion yaw error: {err[3]}")  # Should have same sign as yaw error
            # print(f"current angles: {euler.quat2euler(self.quat, 'szyx')}")
            # print(f"desired angles: {euler.quat2euler(qd_dive, 'szyx')}")
            # print(f"Current yaw: {heading}, Desired yaw: {self.yaw_d}")
            # print(f"flag command: {self.flag_command}")
            
        elif self.pilot == "altitude":
            q_inv = quat.qinverse(self.quat)
            alt_err = filtered_alt[0] - self.altitude_d
            qd_alt = quat.axangle2quat([0.0, 0.0, 1.0], np.clip(1.0 * alt_err, -np.pi/6, np.pi/6)) 
            qd_alt = euler.euler2quat(*tuple([np.clip(- 4.0 * alt_err, -np.pi/2, np.pi/2), 0.0, self.yaw_d]),self.euler_convention)
            err = 2.0 * quat.qmult(qd_alt, q_inv)
            w = - 1.0*(np.array(err[1:]) - 0.1*np.array(self.omega))
            # u = [1.0, - 0.2* w[1], w[2], - 0.2*w[0], w[2]]
            u = [1.0, - w[1], w[2], - w[0], w[2]]
            # print(f"[DEBUG] \n mode: ", self.pilot, "\n ctrl: ", np.array(u), "\n quat: ", np.array(self.quat), "\n alt: ", filtered_alt[0], "\n depth: ", 
            #       self.depth_sensor, "\n desired depth", self.depth_d, "\n yaw", heading, "\n yaw desired", self.yaw_d,"\n")

        elif self.pilot == "remote":
            v = self.remote_v
            # print(f"[DEBUG] w: ", np.array(w), "desired: ", np.array(self.quat),"\n")
            u_pitch = v[1]
            u_yaw = v[2]
            u_fwd = v[0]
            u_roll = v[3]
            u = [u_fwd, u_roll, u_pitch, u_yaw, u_pitch]
            # print(f"[DEBUG] \n mode: ", self.pilot, "\n ctrl: ", np.array(u), "\n quat: ", np.array(self.quat), "\n alt: ", filtered_alt[0], "\n depth: ", 
            #       self.depth_sensor, "\n desired depth", self.depth_d, "\n yaw", heading, "\n yaw desired", self.yaw_d,"\n")

        elif self.pilot == "DR":
            v = self.remote_v
            depth_err = self.depth_sensor - self.depth_d
            # in base coordinates
            qd_dive = euler.euler2quat(*tuple([np.clip(- 10.0 * depth_err, -np.pi/2, np.pi/2), 0.0, 0.0]),self.euler_convention)
            q_inv = quat.qinverse(self.quat)
            err = 2.0 * quat.qmult(qd_dive, q_inv)
            w = - np.clip(1.0*(np.array(err[1:]) - 0.1*np.array(self.omega)), -1.0, 1.0)
            u = [1.0, - 0.2* w[1], w[2], - 0.2*w[0], w[2]]
            u_pitch = v[1]
            u_yaw = v[2]
            u_fwd = v[0]
            u_roll = v[3]
            if abs(v[1]) > 0.05:
                u = np.clip([u_fwd, u_roll - 0.2* w[1], u_pitch - w[2],  - w[0], u_pitch - 1.0 * depth_err],-1,1)
            else:
                u = np.clip([u_fwd, u_roll - 0.2* w[1], - w[2], - w[0], - 1.0 * depth_err],-1,1)
            if abs(v[2]) > 0.05:
                u[3] = u_yaw
        
        elif self.pilot == "track":
            u_fwd = 1.0
            if np.size(self.centroids) == 2:
                # print(self.centroids)
                u_yaw = self.centroids[1]/870 * 2 - 1
                u_pitch = 1 - self.centroids[0]/480 * 2
                self.centroids_hist.append(self.centroids)
            elif self.centroids_hist != []:
                c = self.centroids_hist[-1] 
                u_pitch = 1 - c[0]/480 * 2
                u_yaw = c[1]/870 * 2 - 1
            else:
                u_pitch = 0
                if np.random.uniform(0.0, 1.0) > 0.97:
                    self.rand_yaw = np.random.uniform(-1.0, 1.0)
                u_yaw = self.rand_yaw
            u = [u_fwd, 0.0, - u_pitch, u_yaw, - u_pitch]

        # for emily debugging
        # print(f"[DEBUG] \n mode: ", self.pilot, "\n ctrl: ", np.array(u), "\n quat: ", np.array(self.quat), "\n alt: ", filtered_alt[0], "\n depth: ", 
        #           self.depth_sensor, "\n desired depth", self.depth_d, "\n yaw", heading, "\n yaw desired", 
        #           self.yaw_d,"\n centroids: ", self.centroids,"\n depth: ", [self.stereo_depth, self.x, self.y], "\n")
    
        u = np.clip(u, -1.0, 1.0)
        self.config_pub.publish(Float32MultiArray(data=u))
        # else:
        #     print("time has elapsed")
        #     u = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        #     self.config_pub.publish(Float32MultiArray(data=u))
        #     mode_msg.mode = "kill"
        #     self.mode = mode_msg.mode
        #     self.mode_pub.publish(mode_msg)
        #     raise KeyboardInterrupt

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
        self.depth_sensor = msg.depth
        # self.altitude = msg.altitude
        if len(self.altitude) < 20:
            self.altitude.append(msg.altitude)
        else:
            self.altitude.pop(0)
            self.altitude.append(msg.altitude)
        self.alt_confidence = msg.alt_confidence
        self.omega = msg.imu.gyr
        if self.quat_first:
            self.quat_init = self.quat
        # print(f"[DEBUG] altitude: ", self.altitude, "confidence: ", self.alt_confidence, "\n")
    
    def stereo_callback(self, msg):
        self.stereo_depth = msg.data[0]
        # print(msg.data[0])
        self.x = msg.data[1]
        self.y = msg.data[2]

    def turtle_desired_callback(self, msg):
        self.yaw_d = msg.yaw
        self.depth_d = msg.pitch
        self.altitude_d = msg.fwd
    
    def tracker_callback(self, msg):
        self.centroids = msg.data
        # print(msg.data)
        
    def update_plot(self):
        if self.first:
            plt.ion()
            self.fig, self.ax = plt.subplots()
            self.myplot, = self.ax.plot(self.plot_depth)
            plt.show()
            self.first = False
        else:
            # Remove the previous line
            self.myplot.remove()
            
            # Replot with the new data
            self.myplot, = self.ax.plot(self.plot_depth)
            
            # Adjust the axis limits
            self.ax.relim()
            self.ax.autoscale_view()
            
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def gamepad_callback(self, msg):
        if self.pilot != "track":
            if msg.data[-1] == 0:
                self.pilot = "remote"
            elif msg.data[-1] == 1:
                self.pilot = "depth"
            elif msg.data[-1] == 2:
                self.pilot = "altitude"
            elif msg.data[-1] == 3:
                self.pilot = "DR"
            elif msg.data[-1] == 4:
                self.pilot = "track"
        else:
            if msg.data[-1] == 0:
                self.pilot = "remote"

        self.remote_v = msg.data[:-1]
        
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
        # rclpy.shutdown()
    except KeyboardInterrupt:
        print("shutdown")
    except Exception as e:
        print("some error occurred")
        traceback.print_exc()
        # turtle_node.shutdown_motors()
        exec_type, obj, tb = sys.exc_info()
        fname = os.path.split(tb.tb_frame.f_code.co_filename)[1]
        print(exec_type, fname, tb.tb_lineno)
        print(e)
    # turtle_node.save_data()

if __name__ == '__main__':
    main()
