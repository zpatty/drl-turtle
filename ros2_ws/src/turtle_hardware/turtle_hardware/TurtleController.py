import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import rclpy.parameter
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup

import sys
import os
from rclpy.parameter_event_handler import ParameterEventHandler

from turtle_interfaces.msg import TurtleTraj, TurtleCtrl, TurtleMode, TurtleState
from std_msgs.msg import String, Float32MultiArray

import numpy as np

from statistics import mode as md

import time

from datetime import datetime
# submodule = os.path.expanduser("~") + "/drl-turtle/ros2_ws/install/turtle_hardware/lib/python3.12/site-packages/turtle_hardware/"
# sys.path.append(submodule)
# # submodule = os.path.expanduser("~") + "/colcon_venv/venv/lib/python3.12/site-packages/"
# # sys.path.append(submodule)
# submodule = os.path.expanduser("~") + "/git-repos/orbitally-stable-motion-primitives-for-turtle-locomotion/src/"
# sys.path.append(submodule)


from turtle_dynamixel.torque_control import torque_control  
# from turtle_dynamixel import torque_control

from mjc_turtle_robot.controllers.joint_space_controllers import cornelia_joint_space_control_factory, cornelia_joint_space_motion_primitive_control_factory
from mjc_turtle_robot.controllers.task_space_controller import green_sea_turtle_task_space_control_factory
from mjc_turtle_robot.controllers.joint_space_controllers import navigation_joint_space_motion_primitive_control_factory

class TurtleController(Node):

# class TurtleRobot(Node, gym.Env):
    """
    This node is responsible for continously reading sensor data and receiving commands from the keyboard node
    to execute specific trajectoreies or handle emergency stops. It also is responsible for sending motor pos commands to the RL node
    for training and deployment purposes.
    TODO: migrate dynamixel motors into this class
    TLDR; this is the node that handles all turtle hardware things
    """

    def __init__(self, params=None):
        super().__init__('turtle_ctrl_node')
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

        self.dt = 0.001
        timer_cb_group = None
        self.call_timer = self.create_timer(self.dt, self._ctrl_cb, callback_group=timer_cb_group)

        self.handler = ParameterEventHandler(self)

        
        # subscribes to keyboard setting different turtle modes 
        self.ctrl_sub = self.create_subscription(
            TurtleCtrl,
            'turtle_ctrl_params',
            self.turtle_script_callback,
            qos_profile)
        
        self.ctrl_sub = self.create_subscription(
            Float32MultiArray,
            'turtle_4dof',
            self.turtle_ctrl_callback,
            qos_profile)
        
        self.mode_sub = self.create_subscription(
            TurtleMode,
            'turtle_mode',
            self.turtle_mode_callback,
            qos_profile)
        
        # for case when trajectory mode, receives trajectory msg
        self.sensors_sub = self.create_subscription(
            TurtleState,
            'turtle_sensors',
            self.sensors_callback,
            qos_profile
        )
        # # for case camera mode, receives motion primitive
        # self.cam_sub = self.create_subscription(
        #     String,
        #     'primitive',
        #     self.primitive_callback,
        #     buff_profile
        # )

        # continously publishes the current motor position      
        self.u_pub = self.create_publisher(
            TurtleTraj,
            'turtle_u',
            qos_profile
        )
        # self.mode_cmd_sub       # prevent unused variable warning
        self.create_rate(1000)
        self.mode = 'rest'      # initialize motors mode to rest state
        self.traj = self.joint_space_traj
        self.traj_str = 'js'
        self.primitives = ['dwell']
        self.voltage = 12.0
        self.n_axis = 3


        self.nq = 10
        
        # set thresholds for motor angles 
        self.epsilon = 0.1
        self.min_threshold, _ = self.convert_motors_to_q(np.array([1.60, 3.0, 2.4, 2.43, 1.2, 1.7, 1.45, 1.2, 3.0, 2.3]), np.zeros((10,)))
        self.max_threshold, _ = self.convert_motors_to_q(np.array([3.45, 5.0, 4.2, 4.5, 4.15, 3.8, 3.2, 4.0, 4.0, 4.7]), np.zeros((10,)))
        # for PD control
        self.Kp = np.diag([0.8, 0.4, 0.4, 0.8, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4])*2
        self.Kp = 0.1 * np.eye(10)
        self.KD = 0.1 * np.eye(10)

        # these should be messages instead of parameters
        self.A = np.pi / 180 * np.array([40, 40, 40]).reshape(-1,1)
        self.C = np.pi / 180 * np.array([0, 10, 10])
        self.yaw = 0.0
        self.pitch = 0.0
        self.fwd = 1.0
        self.roll = 0.0
        self.freq_offset = 0.3
        self.T = 2
        self.w = 2 * np.pi / self.T

        self.u = [0]*10
        self.t0 = time.time()
        self.t = self.t0

        self.q = np.zeros((self.nq, 1))
        self.dq = np.zeros((self.nq, 1))

        self.qd = np.zeros((self.nq, 1))
        self.dqd = np.zeros((self.nq, 1))
        self.ddqd = np.zeros((self.nq, 1))

        self.q_off = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.sw = 1.0
        self.kpv=[1] * 6
        self.use_learned_motion_primitive = False
        self.d_pinv = 2e-3

        self.kp_th = 5e0 * np.ones(2)  # Proportional gain for twist
        self.w_th = 1e-3  # weight for the twist tracking in the Jacobian product
        self.kp_s = 20

        self.construct_js_control()
        self.nav_u = None

        

    def _ctrl_cb(self):
        self.t = time.time() - self.t0
        self.qd, self.dqd, self.dqdd = self.traj(self.t, self.q, self.nav_u)
        # qd_clipped = np.clip(self.qd, self.min_threshold, self.max_threshold)
        # print(self.qd)
        # print(qd_clipped)
        print(f"[DEBUG] q: {(np.squeeze(self.q))}\n")
        # print(f"[DEBUG] qd: {(np.squeeze(self.qd))}\n")
        u = self.get_u(self.qd.reshape(-1,1), self.dqd.reshape(-1,1), self.dqdd.reshape(-1,1))
        print(f"[DEBUG] u: {u}\n")
        self.publish_u(u)
        # self.get_logger().info(f'Time to compute control: {time.time() - self.t - self.t0}\n')

    def turtle_mode_callback(self,msg):
        if self.traj_str != msg.traj or self.mode != msg.mode:
            self.t0 = time.time()
        self.mode = msg.mode
        print(self.mode)
        match msg.traj:
            case "sin":
                self.traj = self.sinusoidal_traj
            case "js":
                self.construct_js_control()
                self.traj = self.joint_space_traj
            case "ts":
                self.construct_ts_control()
                self.traj = self.task_space_traj
            case "nav":
                self.construct_nav_control()
                self.traj = self.nav_traj
                self.nav_u = np.array([self.fwd, self.roll, self.pitch, self.yaw])
            case _:
                self.mode = "rest"
        self.traj_str = msg.traj
        print("here")

        

    
    def turtle_script_callback(self, msg):

        print(msg)
        kp = msg.kp
        kd = msg.kd
        if len(kd) == 10:
            self.KD = np.diag(kd)
        elif len(kd) == 1:
            self.KD = kd[0] * np.eye(10)

        if len(kp) == 10:
            self.Kp = np.diag(kp)
        elif len(kd) == 1:
            self.Kp = kp[0] * np.eye(10)

            

        # these should be messages instead of parameters
        self.A = np.array(msg.amplitude) * np.pi/180
        self.C = np.array(msg.center) * np.pi/180
        self.yaw = msg.yaw
        self.pitch = msg.pitch
        self.fwd = msg.fwd
        self.roll = msg.roll
        self.freq_offset = msg.frequency_offset
        self.T = msg.period
        self.w = 2 * np.pi / self.T
        
        self.kpv = msg.kpv
        self.d_pinv = msg.d_pinv
        self.use_learned_motion_primitive = msg.learned
        self.kp_s = msg.kp_s
        self.w_th = msg.w_th
        self.kp_th = msg.kp_th
        self.q_off = msg.offset
        self.sw = msg.sw

        match self.traj_str:
            case "sin":
                self.traj = self.sinusoidal_traj
            case "js":
                self.construct_js_control()
                self.traj = self.joint_space_traj
            case "ts":
                self.construct_ts_control()
                self.traj = self.task_space_traj
            case "nav":
                # self.construct_nav_control()
                self.traj = self.nav_traj
                # self.nav_u = np.array([self.fwd, self.roll, self.pitch, self.yaw])
            case _:
                self.mode = "rest"
        # print(self.traj_str)
        if self.traj_str == 'nav':
            self.nav_u = np.array([self.fwd, self.roll, self.pitch, self.yaw])
        else:
            self.nav_u = None
    
    def turtle_script_callback(self, msg):
        self.nav_u = msg.data

    def sensors_callback(self, msg):
        """
        Callback function that takes in list of squeezed arrays
        msg: [quat acc gyr voltage t_0 q dq ddq u qd t]
        """    
        # self.get_logger().info(f"Received update to q")
        self.q = np.array(msg.q).reshape(-1,1)
        self.dq = np.array(msg.dq).reshape(-1,1)
        self.quat = np.array(msg.imu.quat)

    def publish_u(self, u):
        cmd_msg = TurtleTraj()
        cmd_msg.u = u
        cmd_msg.qd = self.qd
        cmd_msg.dqd = self.dqd
        self.u_pub.publish(cmd_msg)
        

    def get_u(self, qd, dqd, ddqd):
        # qd = np.zeros((10,1))
        # dqd = np.zeros((10,1))
        match self.mode:
            case 'rest':
                u = [0]*10
            case 'v':
                u = dqd
                u[6:] = -0.1 * (self.q[6:])
            case 'p':
                
                u = (self.Kp.dot((qd - self.q)) + self.KD.dot((dqd - self.dq))) * 150
                # u = torque_control(self.q,self.dq,qd,dqd,ddqd,self.Kp,self.KD)
            case _:
                u = [0]*10

        return u
    
    
    def sinusoidal_traj(self, t, q, u):
        # q, _, _ = convert_q_to_motors(q, q, q * 0.0)
        T = 2 * np.pi / self.w
        offset = np.pi * (1 - self.freq_offset/(1 - self.freq_offset))
        t = np.mod(t, T)
        if t < self.freq_offset * T:
            q_des1 = self.C[0] + self.A[0] * np.cos(self.w * t / (2 * self.freq_offset))
            q_des2 = self.C[1] + self.A[1] * np.cos(self.w * t / (2 * self.freq_offset))
            q_des3 = self.C[2] + self.A[2] * np.cos(self.w * t / (2 * self.freq_offset))
            qd_des1 = - self.w / (2 * self.freq_offset) * self.A[0] * np.sin(self.w * t / (2 * self.freq_offset))
            qd_des2 = - self.w / (2 * self.freq_offset) * self.A[1] * np.sin(self.w * t / (2 * self.freq_offset))
            qd_des3 = - self.w / (2 * self.freq_offset) * self.A[2] * np.sin(self.w * t / (2 * self.freq_offset))
            qdd_des1 = - (self.w / (2 * self.freq_offset))**2 * self.A[0] * np.cos(self.w * t / (2 * self.freq_offset))
            qdd_des2 = - (self.w / (2 * self.freq_offset))**2 * self.A[1] * np.cos(self.w * t / (2 * self.freq_offset))
            qdd_des3 = - (self.w / (2 * self.freq_offset))**2 * self.A[2] * np.cos(self.w * t / (2 * self.freq_offset))
        else:
            q_des1 = self.C[0] + self.A[0] * np.cos(self.w * t / (2 * (1 - self.freq_offset)) + offset)
            q_des2 = self.C[1] + self.A[1] * np.cos(self.w * t / (2 * (1 - self.freq_offset)) + offset)
            q_des3 = self.C[2] + self.A[2] * np.cos(self.w * t / (2 * (1 - self.freq_offset)) + offset)
            qd_des1 = - self.w / (2 * (1 - self.freq_offset)) + self.A[0] * np.sin(self.w * t / (2 * (1 - self.freq_offset)) + offset)
            qd_des2 = - self.w / (2 * (1 - self.freq_offset)) + self.A[1] * np.sin(self.w * t / (2 * (1 - self.freq_offset)) + offset)
            qd_des3 = - self.w / (2 * (1 - self.freq_offset)) + self.A[2] * np.sin(self.w * t / (2 * (1 - self.freq_offset)) + offset)
            qdd_des1 = - (self.w / (2 * (1 - self.freq_offset)))**2 + self.A[0] * np.cos(self.w * t / (2 * (1 - self.freq_offset)) + offset)
            qdd_des2 = - (self.w / (2 * (1 - self.freq_offset)))**2 + self.A[1] * np.cos(self.w * t / (2 * (1 - self.freq_offset)) + offset)
            qdd_des3 = - (self.w / (2 * (1 - self.freq_offset)))**2 + self.A[2] * np.cos(self.w * t / (2 * (1 - self.freq_offset)) + offset)
        pitch = (self.pitch + 1.0) / 2
        yaw = (self.yaw + 1.0) / 2
        qd_rear = (pitch - 0.5) / 0.5 * np.pi / 3
        q_right_des = np.array([q_des1, q_des2, q_des3]).reshape(-1,1)
        qd_right_des = np.array([qd_des1, qd_des2, qd_des3]).reshape(-1,1)
        qdd_right_des = np.array([qdd_des1, qdd_des2, qdd_des3]).reshape(-1,1)
        q_left_des = np.array([q_des1, - q_des2, - q_des3]).reshape(-1,1)
        qd_left_des = np.array([qd_des1, - qd_des2, - qd_des3]).reshape(-1,1)
        qdd_left_des = np.array([qdd_des1, - qdd_des2, - qdd_des3]).reshape(-1,1)
        if self.yaw > 0.5:
            q_right_des *= (1.0 - yaw) / 0.5
            qd_right_des *= (1.0 - yaw) / 0.5
            qdd_right_des *= (1.0 - yaw) / 0.5
        elif self.yaw < 0.5:
            q_left_des *= yaw / 0.5
            qd_left_des *= yaw / 0.5
            qdd_left_des *= yaw / 0.5
        q_des = np.vstack((q_right_des, q_left_des, np.array([0.0, qd_rear, 0.0, -qd_rear]).reshape(-1,1)))
        qd_des = np.vstack((qd_right_des, qd_left_des, np.zeros((4,1))))
        qdd_des = np.vstack((qdd_right_des, qdd_left_des, np.zeros((4,1))))
        # qd, dqd, ddqd = convert_q_to_motors(np.squeeze(q_des), np.squeeze(qd_des), np.squeeze(qdd_des))
        return np.squeeze(q_des), np.squeeze(qd_des), np.squeeze(qdd_des)

    def task_space_traj(self, t, q, u):
        dqd, aux = self.task_space_control_fn(t, np.squeeze(q[:6]))
        # _, dqd, ddqd = convert_q_to_motors(dqd, dqd, dqd * 0.0)
        dqd = np.hstack((dqd, np.zeros((4,))))
        # print(f"[DEBUG] q: {q * 180 / np.pi}\n")
        # print(f"[DEBUG] dqd: {dqd * 180 / np.pi}\n")
        qd = np.squeeze(q) + dqd * self.dt

        return qd, dqd, dqd * 0.0
    
    def joint_space_traj(self, t, q, u):
        dqd, aux = self.joint_space_control_fn(t, np.squeeze(q[:6]))
        # _, dqd, ddqd = convert_q_to_motors(dqd, dqd, dqd * 0.0)
        dqd = np.hstack((dqd, np.zeros((4,))))
        # print(f"[DEBUG] q: {q * 180 / np.pi}\n")
        # print(f"[DEBUG] dqd: {dqd * 180 / np.pi}\n")
        
        qd = np.squeeze(q) + dqd * self.dt

        return qd, dqd, dqd * 0.0
    
    def nav_traj(self, t, q, u):
        dqd, aux = self.nav_control_fn(t, np.squeeze(q[:6]), u)
        # _, dqd, ddqd = convert_q_to_motors(dqd, dqd, dqd * 0.0)
        dqd = np.hstack((dqd, np.zeros((4,))))
        # print(f"[DEBUG] q: {q * 180 / np.pi}\n")
        # print(f"[DEBUG] dqd: {dqd * 180 / np.pi}\n")
        
        qd = np.squeeze(q) + dqd * self.dt

        return qd, dqd, dqd * 0.0
    
    def construct_nav_control(self):
        nav_control_fn = navigation_joint_space_motion_primitive_control_factory(
            sw=self.sw, q_off=self.q_off, phase_sync_kp=self.kp_s, limit_cycle_kp=self.kpv[0], sliding_mode_params = dict(sigma=0.05, plateau_width=0.05)
        )
        self.nav_control_fn = nav_control_fn
    
    def construct_js_control(self):
        if self.use_learned_motion_primitive:
            joint_space_control_fn = cornelia_joint_space_motion_primitive_control_factory(
                q_off=self.q_off, sw=self.sw, phase_sync_kp=self.kp_s, limit_cycle_kp=self.kpv[0], sliding_mode_params = dict(sigma=0.05, plateau_width=0.05)
            )
        else:
            joint_space_control_fn = cornelia_joint_space_control_factory(kp=self.kpv, q_off=self.q_off, sw=self.sw)
        self.joint_space_control_fn = joint_space_control_fn
        
    def construct_ts_control(self):
        sf = 1.0  # spatial scaling factor

        if self.w_th == 0.0:
            self.w_th = None

        x_off = np.array(
            [0.34808895, -0.08884996, 0.00209812]
        )
        task_space_control_fn = green_sea_turtle_task_space_control_factory(
            x_off=x_off,
            sf=sf,
            sw=self.sw,
            kp=np.array(self.kpv),
            kp_th=np.array(self.kp_th),
            w_th=self.w_th,
            pinv_damping=self.d_pinv,
            use_learned_motion_primitive=self.use_learned_motion_primitive,
            use_straight_flipper=True,
            phase_sync_kp=self.kp_s,
            limit_cycle_kp=self.Kp[0,0],
        )
        self.task_space_control_fn = task_space_control_fn
    
    def convert_motors_to_q(self, q, dq):
        q = np.array(q).reshape(10,)  - np.pi
        qd = np.array(dq).reshape(10,)
        q_new = np.hstack((-q[3:6], [q[0], -q[1], -q[2]], q[6:]))
        qd_new = np.hstack((-qd[3:6], [qd[0], -qd[1], -qd[2]], qd[6:]))
        return q_new, qd_new

    
    
def main():
    rclpy.init()
    control_node = TurtleController()
    try:
        rclpy.spin(control_node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        # rclpy.shutdown()
        print("some error occurred")
        # turtle_node.shutdown_motors()
        exec_type, obj, tb = sys.exc_info()
        fname = os.path.split(tb.tb_frame.f_code.co_filename)[1]
        print(exec_type, fname, tb.tb_lineno)
        print(e)
    # control_node.save_data()
    # control_node.construct_ts_control()
    control_node.traj(control_node.t, control_node.q, control_node.nav_u)
    control_node.publish_u([0]*10)
if __name__ == '__main__':
    main()