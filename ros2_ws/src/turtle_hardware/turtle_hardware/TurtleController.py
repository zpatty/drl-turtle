import sys
import os
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.parameter_event_handler import ParameterEventHandler

from turtle_interfaces.msg import TurtleTraj, TurtleCtrl, TurtleMode, TurtleState
from std_msgs.msg import String, Float32MultiArray

from statistics import mode as md
from datetime import datetime
import numpy as np
from turtle_dynamixel.torque_control import torque_control  
from turtle_ctrl.turtle_ctrl_factory import cornelia_joint_space_trajectory_tracking_control_factory
from turtle_ctrl.template_model_oracles import reverse_stroke_joint_oracle_factory

class TurtleController(Node):
    """
    This node is responsible for sending control commands to the turtle motors based on the received parameters and mode.
    It subscribes to various topics to receive control parameters, turtle states, and mode commands.
    The node continuously computes the control signals based on the current state and desired trajectory,
    and publishes the computed control signals to the motors.
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
        self.ctrl_params_sub = self.create_subscription(
            TurtleCtrl,
            'turtle_ctrl_params',
            self.turtle_ctrl_params_callback,
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

        # continously publishes the current motor position      
        self.u_pub = self.create_publisher(
            TurtleTraj,
            'turtle_u',
            qos_profile
        )
        self.create_rate(1000)
        self.mode = 'rest'      # initialize motors mode to rest state
        self.traj = self.new_controller
        self.traj_str = 'nav'
        self.voltage = 12.0
        self.n_axis = 3
        self.nq = 10
        
        # set thresholds for motor angles 
        self.epsilon = 0.1
        self.min_threshold, _ = self.convert_motors_to_q(np.array([1.60, 3.0, 2.4, 2.43, 1.2, 1.7, 1.45, 1.2, 3.0, 2.3]), np.zeros((10,)))
        self.max_threshold, _ = self.convert_motors_to_q(np.array([3.45, 5.0, 4.2, 4.5, 4.15, 3.8, 3.2, 4.0, 4.0, 4.7]), np.zeros((10,)))
        # for PD control
        self.Kp = np.diag([0.8, 0.6, 0.4, 0.8, 0.6, 0.4, 0.4, 0.4, 0.4, 0.4])*2
        # self.Kp = 0.1 * np.eye(10)
        self.KD = 0.2 * np.eye(10)

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
        self.sw = 1.8
        self.kpv=[1] * 6
        self.use_learned_motion_primitive = False
        self.d_pinv = 2e-3

        self.kp_th = 5e0 * np.ones(2)   # Proportional gain for twist
        self.w_th = 1e-3                # weight for the twist tracking in the Jacobian product
        self.kp_s = 20

        self.joint_space_control_fn = cornelia_joint_space_trajectory_tracking_control_factory(kp=[2.0]*6, sw=self.sw)
        self.q_ra_fn, self.q_d_ra_fn, q_dd_ra_fn = reverse_stroke_joint_oracle_factory(s=1, sw=self.sw)
        self.q_la_fn, self.q_d_la_fn, q_dd_la_fn = reverse_stroke_joint_oracle_factory(s=-1, sw=self.sw)
        self.nav_u = [0.0, 0.0, 0.0, 0.0]
        self.rear_pitch = 0.0
        self.time_last = self.t0
        self.t_new = (4.3 / self.sw) * 0.32

    def _ctrl_cb(self):
        self.t = time.time() - self.t0
        self.qd, self.dqd, self.dqdd = self.traj(self.t, self.q, self.nav_u)
        self.time_last = self.t
        # qd_clipped = np.clip(self.qd, self.min_threshold, self.max_threshold)
        # print(self.qd)
        # print(qd_clipped)
        print(f"[DEBUG] q: {(np.squeeze(self.q))}\n")
        # print(f"[DEBUG] qd: {(np.squeeze(self.qd))}\n")
        # print(f"[DEBUG] dqd: {(np.squeeze(self.dqd))}\n")
        u = self.get_u(self.qd.reshape(-1,1), self.dqd.reshape(-1,1), self.dqdd.reshape(-1,1))
        print(f"[DEBUG] u: {u}\n")
        self.publish_u(u)
        # print(f"[DEBUG] navu: {self.nav_u}\n")
        # self.get_logger().info(f'Time to compute control: {time.time() - self.t - self.t0}\n')

    def turtle_mode_callback(self,msg):
        if self.traj_str != msg.traj or self.mode != msg.mode:
            self.t0 = time.time()
        self.mode = msg.mode
        if msg.mode == "kill":
            raise KeyboardInterrupt
        # print(self.mode)
        match msg.traj:
            case "js":
                self.construct_js_control()
                self.traj = self.joint_space_traj
                self.traj_str = msg.traj
            case "ts":
                self.construct_ts_control()
                self.traj = self.task_space_traj
                self.traj_str = msg.traj
            case "nav":
                self.construct_nav_control()
                self.traj = self.nav_traj
                self.traj_str = msg.traj
                # self.nav_u = np.array([self.fwd, self.roll, self.pitch, self.yaw])
            case _:
                self.mode = self.mode
                self.traj_str = self.traj_str
        
        # print(self.mode)

    def turtle_ctrl_callback(self, msg):
        self.nav_u = np.array(msg.data[:-1])
        # print(f"[DEBUG] navu: {self.nav_u}\n")
        self.rear_pitch = msg.data[-1]
    
    def turtle_ctrl_params_callback(self, msg):
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
                u_rear = -700 * self.rear_pitch
                # print(u_rear)
                # apply the control signal
                k_r = 10.0
                u[7] =  - k_r * (self.q[7] - 10 * np.pi / 180 * np.cos(4.0 * np.pi  * self.t) - np.clip(u_rear, -40.0, 40.0) * np.pi / 180)
                u[9] = - k_r * (self.q[9] + 10 * np.pi / 180 * np.cos(4.0 * np.pi  * self.t) + np.clip(u_rear, -40.0, 40.0) * np.pi / 180)

            case 'p':
                
                u = (self.Kp.dot((qd - self.q)) + self.KD.dot((dqd - self.dq))) * 150
                u_rear = - 100 * self.rear_pitch
                # print(u_rear)
                # apply the control signal
                k_r = 3.0
                u[7] =  - k_r * (self.q[7] - 5 * np.pi / 180 * np.cos(4.0 * np.pi  * self.t) - np.clip(u_rear, -70.0, 70.0) * np.pi / 180) * 150
                u[9] = - k_r * (self.q[9] + 5 * np.pi / 180 * np.cos(4.0 * np.pi  * self.t) + np.clip(u_rear, -70.0, 70.0) * np.pi / 180) * 150
                # u = torque_control(self.q,self.dq,qd,dqd,ddqd,self.Kp,self.KD)
            case _:
                u = [0]*10

        return u
    
    def joint_space_traj(self, t, q, u):
        dqd, aux = self.joint_space_control_fn(t, np.squeeze(q[:6]))
        # _, dqd, ddqd = convert_q_to_motors(dqd, dqd, dqd * 0.0)
        dqd = np.hstack((dqd, np.zeros((4,))))
        # print(f"[DEBUG] q: {q * 180 / np.pi}\n")
        # print(f"[DEBUG] dqd: {dqd * 180 / np.pi}\n")
        
        qd = np.squeeze(q) + dqd * self.dt

        return qd, dqd, dqd * 0.0
    
    def convert_motors_to_q(self, q, dq):
        q = np.array(q).reshape(10,)  - np.pi
        qd = np.array(dq).reshape(10,)
        q_new = np.hstack((-q[3:6], [q[0], -q[1], -q[2]], q[6:]))
        qd_new = np.hstack((-qd[3:6], [qd[0], -qd[1], -qd[2]], qd[6:]))
        return q_new, qd_new

    def new_controller(self, t, q, u):
        self.t_new = self.sw * (t - self.time_last) * np.abs(u[0]) + self.t_new
        # if self.u[0] < 0.05:
        #     self.t_new = (4.3 / self.sw) * 0.32
        # t_new = self.data.time
        if u[0] > 0.0:
            q_d_des, aux = self.joint_space_control_fn(
                t=self.t_new,
                q=np.squeeze(q[:6]),
                # u=self.u
            )
            # q_arm_des = self.data.qpos[7: 7 + 6] + q_d_des * (self.data.time - self.time_last_ctrl)
            q_arm_des = aux["q_des"]
        # print(q_arm_des)
        else:
            q_ra, q_d_ra = self.q_ra_fn(self.t_new), self.q_d_ra_fn(self.t_new)
            q_la, q_d_la = self.q_la_fn(self.t_new), self.q_d_la_fn(self.t_new)
            q_arm_des = np.concatenate([q_ra, q_la], axis=-1)
            
            q_d_des = np.concatenate([q_d_ra, q_d_la], axis=-1)

        q_arm_des[2] += u[2] * 60 * np.pi / 180.0
        q_arm_des[5] += - u[2] * 60 * np.pi / 180.0
        # q_arm_des[0] += u[2] * 25 * np.pi / 180.0
        # q_arm_des[3] += - u[2] * 25 * np.pi / 180.0

        # q_arm_des[1] += self.u[2] * np.pi/2
        # q_arm_des[4] += - self.u[2] * np.pi/2
        
        if u[3] < 0.0:
            q_arm_des[3:] *= (u[3] + 1.0)
        else:
            q_arm_des[:3] *= - (u[3] - 1.0)

        # ROLL
        q_arm_des[2] -= - np.pi/4 * self.u[1]
        q_arm_des[5] += np.pi/4 * self.u[1]

        q_arm_des[1] += np.pi/4 * self.u[1]
        q_arm_des[4] += np.pi/4 * self.u[1]

        q_arm_des += np.diag([0.5, 0.25, 0, 0.5, 0.25, 0]) @  np.abs(q_arm_des) * np.sign(q_arm_des) * 0.25
        
        q_arm_des = np.hstack((q_arm_des, np.zeros((4,))))
        q_d_des = np.hstack((q_d_des, np.zeros((4,))))
        return q_arm_des, q_d_des, q_d_des*0
        
    
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
    try:
        control_node.publish_u([0]*10)
    except:
        print("Kill command failed, check connection.")
if __name__ == '__main__':
    main()