import argparse
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np
import os
from pathlib import Path
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import rclpy.parameter
import sys
from rclpy.parameter_event_handler import ParameterEventHandler
import transforms3d.quaternions as quat
import transforms3d.euler as euler

from std_msgs.msg import String, Float32MultiArray
from turtle_interfaces.msg import TurtleTraj, TurtleSensors, TurtleCtrl, TurtleMode, TurtleState

from turtle_ctrl.turtle_ctrl_factory import cornelia_joint_space_trajectory_tracking_control_factory
from turtle_ctrl.template_model_oracles import reverse_stroke_joint_oracle_factory



class Simulator(Node):
    def __init__(self, params=None):
        super().__init__('turtle_sim_node')
        # Configure MuJoCo to use the EGL rendering backend (requires GPU)
        os.environ["MUJOCO_GL"] = "egl"

        # robot settings
        base_constraint = "free"  # "sliding" or "free"
        self.base_qpos_dim = 7
        use_flexible_flipper = False

        flippers_type_str = "_flexible_flippers" if use_flexible_flipper else "_straight_flippers"
        model_name = f"turtle_{base_constraint}_base_control_t{flippers_type_str}"
        # model_name = f"turtle_free_base_control_t_shell"
        # TODO: properly implement the flexible flipper model
        assert use_flexible_flipper is False, "Flexible flipper model not yet (properly) implemented"
        indices = np.array([7, 8, 9, 16, 17, 18])
        flex_indices = np.array(list(range(4,10)) + list(range(13,19)))
        bend_stiffness = 1
        twist_stiffness = 0.5
        stiffness = np.array([bend_stiffness, twist_stiffness] * 6)

        # control frequency
        self.f_ctrl = 50.0  # Hz
        # control signals for forward/reverse, roll, pitch, and yaw
        self.u = np.array([1.0, 0.0, 0.0, 0.0])
        self.pitch_d = 0.0
        # motion primitive settings
        limit_cycle_kp = 1e0
        # sliding_mode_params = dict(sigma=0.05, plateau_width=0.05)
        sliding_mode_params = None
        synchronize_flippers = True
        phase_sync_method = "sine"  # "sine", "normalized_error", or "master_slaves"
        phase_sync_kp = 1e0 if synchronize_flippers else 0.0
        
        self.last_time = 0.0
        # reference trajectory settings
        self.sw = 1.8  # s
        self.t_new = (4.3 / self.sw) * 0.32
        q_off = np.zeros((6,))
        # joint space control function
        # self.joint_space_control_fn = navigation_joint_space_motion_primitive_control_factory(
        #     sw=sw,
        #     q_off=q_off,
        #     limit_cycle_kp=limit_cycle_kp,
        #     phase_sync_method=phase_sync_method,
        #     phase_sync_kp=phase_sync_kp,
        #     sliding_mode_params=sliding_mode_params,
        # )
        self.joint_space_control_fn = cornelia_joint_space_trajectory_tracking_control_factory(kp=[2.0]*6, sw=self.sw)
        self.q_ra_fn, self.q_d_ra_fn, q_dd_ra_fn = reverse_stroke_joint_oracle_factory(s=1, sw=self.sw)
        self.q_la_fn, self.q_d_la_fn, q_dd_la_fn = reverse_stroke_joint_oracle_factory(s=-1, sw=self.sw)
        model_path = Path("mujoco_models") / (str(model_name) + str(".xml"))
        # Load the model and data
        self.model = mujoco.MjModel.from_xml_path(str(model_path.absolute()))
        if use_flexible_flipper:
            self.model.jnt_stiffness[flex_indices] = stiffness
        self.data = mujoco.MjData(self.model)
        self.sim_ts = dict(
            ts=[],
            base_pos=[],
            base_vel=[],
            base_acc=[],
            base_force=[],
            base_torque=[],
            q=[],
            qvel=[],
            ctrl=[],
            actuator_force=[],
            qfrc_fluid=[],
            q_des=[],
        )
        self.time_last_ctrl = 0.0
        self.q_des = np.zeros((self.data.qpos.shape[0] - 7,))
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()
        # self.data.qpos[3:7] = quat.axangle2quat([1.0, 0.0, 0.0], np.pi/3)
 
        self.altitude = 20.0
        self.altitude_d = 15.0

        self.dwell_time = 0.1
        self.dwell = 2.0
        
        # ros2 qos profile
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.create_rate(1000)

        # 4DOF control params
        self.ctrl_sub = self.create_subscription(
            Float32MultiArray,
            'turtle_4dof',
            self.ctrl_callback,
            qos_profile
        )

        self.sensors_pub = self.create_publisher(
            TurtleSensors,
            'turtle_imu',
            qos_profile
        )

        self.mode_sub = self.create_subscription(
            TurtleMode,
            'turtle_mode',
            self.turtle_mode_callback,
            qos_profile)
        
        # continously reads from all the sensors and publishes data at end of trajectory
        self.state_pub = self.create_publisher(
            TurtleState,
            'turtle_sensors',
            qos_profile
        )

        timer_cb_group = None
        self.call_timer = self.create_timer(0.001, self._timer_cb, callback_group=timer_cb_group)

    def ctrl_callback(self, msg):
        self.u = msg.data[:4]
        self.u[2] =-  self.u[2]
        self.pitch_d = - msg.data[4]

    def _timer_cb(self):
        if self.viewer.is_running():
            step_start = time.time()

            if (self.data.time - self.time_last_ctrl) >= (1.0 / self.f_ctrl):
                # evaluate the controller
                # self.u = np.random.normal([0.95, 0.0, 0.0, 0.0],[0.000001, 0.001, 0.2, 0.2])
                # self.u = [1.0, 0.0, 1.0, 0.7]
                
                tn = self.sw * self.t_new  # speed-up the trajectory by a factor of sw
                tn = tn % 4.3  # repeat the trajectory every (normalized) 4.3 seconds
                # if tn > 4.2 and self.dwell_time < self.dwell:
                #     self.t_new = self.t_new
                #     self.dwell_time += self.data.time - self.time_last_ctrl
                # else:
                #     self.dwell_time = 0.0
                self.t_new = self.sw * (self.data.time - self.time_last_ctrl) * np.abs(self.u[0]) + self.t_new
                
                # self.u[2] = - self.u[2]
                # print(self.dwell_time)
                
                # if self.u[0] < 0.05:
                #     self.t_new = (4.3 / self.sw) * 0.32
                # t_new = self.data.time
                # self.u = np.array([1.0, 0.0, 0.0, 0.0, 1.0])
                if self.u[0] > 0.0:
                    q_d_des, aux = self.joint_space_control_fn(
                        t=self.t_new,
                        q=self.data.qpos[7: 7 + 6],
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
                
                # PITCH
                q_arm_des[2] += self.u[2] * 35 * np.pi / 180.0
                q_arm_des[5] += - self.u[2] * 35 * np.pi / 180.0

                q_arm_des[1] += - self.u[2] * np.pi/9
                q_arm_des[4] += self.u[2] * np.pi/9

                # YAW
                if self.u[3] < 0.0:
                    q_arm_des[3:] *= (self.u[3] + 1.0)
                    # if self.u[3] == -1.0:
                    #     q_ra, q_d_ra = self.q_ra_fn(self.t_new), self.q_d_ra_fn(self.t_new)
                    #     q_la, q_d_la = self.q_la_fn(self.t_new), self.q_d_la_fn(self.t_new)
                    #     q_back = np.concatenate([q_ra, q_la], axis=-1)
                    #     dq_back = np.concatenate([q_d_ra, q_d_la], axis=-1)
                    #     q_arm_des[3:] = q_back[3:]
                    #     q_d_des[3:] = dq_back[3:]
                else:
                    q_arm_des[:3] *= - (self.u[3] - 1.0)
                    # if self.u[3] == 1.0:
                    #     q_ra, q_d_ra = self.q_ra_fn(self.t_new), self.q_d_ra_fn(self.t_new)
                    #     q_la, q_d_la = self.q_la_fn(self.t_new), self.q_d_la_fn(self.t_new)
                    #     q_back = np.concatenate([q_ra, q_la], axis=-1)
                    #     dq_back = np.concatenate([q_d_ra, q_d_la], axis=-1)
                    #     q_arm_des[:3] = q_back[:3]
                    #     q_d_des[:3] = dq_back[:3]

                # x = 2 / (1 + np.exp(-(self.u[1])**2)) - 1
                # x = self.u[1]
                # if self.u[1] < 0.0:
                #     q_arm_des[3:] *= (x + 1.0)
                #     q_arm_des[1:3] *= (x + 1.0)
                # else:
                #     q_arm_des[:3] *= - (x - 1.0)
                #     q_arm_des[4:] *= - (x - 1.0)
                
                # ROLL
                q_arm_des[2] -= - np.pi/4 * self.u[1]
                q_arm_des[5] += np.pi/4 * self.u[1]

                q_arm_des[1] += np.pi/4 * self.u[1]
                q_arm_des[4] += np.pi/4 * self.u[1]


                # q_arm_des[0] *= 1.5 - abs(self.u[4])
                # q_arm_des[3] *= 1.5 - abs(self.u[4])
                # q_arm_des[1] *= 1 + abs(self.u[4])
                # q_arm_des[4] *= 1 + abs(self.u[4])
                # q_arm_des[5] *= -1
                # q_arm_des[2] *= -1
                

                # q_arm_des[1] *= abs(self.u[4])
                # q_arm_des[4] *= abs(self.u[4])
                

                    # # q_arm_des[1] *= - 0.0
                    # q_arm_des[4] = 0.0 + np.pi/3
                    # q_arm_des[1] = 0.0 - np.pi/3
                    # # # q_arm_des[0] *= (self.u[1] - 0.5)
                    # # q_arm_des[4] *= - (self.u[1] - 1.0)
                    # q_arm_des[5] += np.pi/3
                    # q_arm_des[2] += - np.pi/3
                    # q_arm_des[0] = - q_arm_des[3]
                err = q_arm_des + 0.0*np.diag([0.5, 0.25, 0, 0.5, 0.25, 0]) @  np.abs(q_arm_des) * np.sign(q_arm_des) - self.data.qpos[7:13]
                # apply the control signal
                # print(err)
                # print(self.u)
                # print(q_arm_des[5])
                fluid = self.data.qfrc_fluid.copy()
                kp = 1.0 * np.diag([2, 1, 5, 2, 1, 5])
                u = kp @ err + 0.01 * (q_d_des - self.data.qvel[6:12]) 

                self.data.ctrl[0:6] = u
                
                q = self.data.qpos[7 : 7 + 10]
                dq =  self.data.qvel[6:16]
                q_rear1 = q[7]
                q_rear2 = q[9]
                pitch_err = self.pitch_d
                # print(pitch_err)
                u_rear = 100 * (pitch_err)
                tn = self.sw * self.t_new  # speed-up the trajectory by a factor of sw
                tn = tn % 4.3
        
                roll_wr = 1.461
                # print(q_arm_des[0])
                # apply the control signal
                k_r = 1.0
                qd_rear = (10 * np.pi / 180 * np.cos(2.0 * np.pi  * self.data.time) + np.clip(u_rear, -60.0, 60.0) * np.pi / 180)
                # print(qd_rear)
                qd1_rear = qd_rear
                # if self.u[3] == 1.0:
                #     qd1_rear = 60 * np.pi / 180
                    
                self.data.ctrl[7] =  - k_r * (q_rear1 - qd1_rear) - 0.01 * dq[7]
                qd2_rear = qd_rear
                # if self.u[3] == -1.0:
                #     qd2_rear = 60 * np.pi / 180
                    
                self.data.ctrl[9] = - k_r * (q_rear2 + (qd2_rear)) - 0.01 * dq[9]
                self.data.ctrl[8] = -1.0*(q[8])  - 0.05 * dq[8]
                self.data.ctrl[6] = -1.0*q[6] - 0.05 * dq[6]
                # # update the last control time
                self.time_last_ctrl = self.data.time

                self.q = self.data.qpos[7: 7 + 10].tolist()
                self.dq = self.data.qvel[6: 6 + 10].tolist()
                self.tau = self.data.ctrl[:10].tolist()
                self.qd = [q_arm_des.tolist(), 0.0, qd_rear.tolist(), 0.0, - qd_rear.tolist()]
                self.dqd = [q_d_des.tolist(), 0.0, 0.0, 0.0, 0.0]
                self.publish_turtle_data()
            turtle_msg = TurtleSensors()
            Ht = np.block([[0,0,0], [np.eye(3,3)]]).T
            self.quat = self.data.qpos[3:7].tolist()
            qd = quat.axangle2quat([0.0, 0.0, 1.0], np.pi/6)
            self.quat = self.quat[-1:] + self.quat[:-1]
            q_inv = quat.qinverse(self.quat)
            err = quat.qmult(qd, q_inv)
            err = quat.qmult(quat.qinverse(qd), self.quat)
            # if err[0] < 0.0:
            #     err = quat.qinverse(err)
            w = 2.0 * Ht @ quat.qmult(err, q_inv)
            # self.data.qpos[0:3] = np.zeros((3,))
            # # self.data.qvel[3:6] =  - 1.0 * ((skew(err[1:]) + err[0] * np.eye(3,3)) + (1-err[0]) * np.eye(3,3)) @ err[1:] - 0.1 * self.data.qvel[3:6]
            # self.data.qvel[3:6] = - 5.0*np.array(err[1:]) - 1.0*self.data.qvel[3:6]
            # print(f"[DEBUG] error: ", err, "w: ", w,"\n")
            turtle_msg.imu.quat = self.quat
            turtle_msg.altitude = self.altitude
            turtle_msg.depth = - self.data.qpos[2]
            self.gyr = self.data.qvel[3:6]
            
            # print(np.round(quat, 2))
            # pitch = data.qpos[5]
            turtle_msg.imu.gyr = self.gyr
            # turtle_msg.imu.acc = self.acc
            # turtle_msg.depth = self.depth
            # publish msg 
            self.sensors_pub.publish(turtle_msg)
            
            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(self.model, self.data)

            self.sim_ts["ts"].append(self.data.time)
            # extract the sensor data
            self.sim_ts["base_pos"].append(self.data.sensordata[:3].copy())
            self.sim_ts["base_vel"].append(self.data.sensordata[3:6].copy())
            self.sim_ts["base_acc"].append(self.data.sensordata[6:9].copy())
            self.sim_ts["base_force"].append(self.data.sensordata[9:12].copy())
            self.sim_ts["base_torque"].append(self.data.sensordata[12:15].copy())
            self.sim_ts["q"].append(self.data.qpos.copy())
            self.sim_ts["qvel"].append(self.data.qvel.copy())
            self.sim_ts["ctrl"].append(self.data.ctrl.copy())
            self.sim_ts["actuator_force"].append(self.data.actuator_force.copy())
            self.sim_ts["qfrc_fluid"].append(self.data.qfrc_fluid.copy())
            self.sim_ts["q_des"].append(self.q_des.copy())

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            self.viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                # print(time_until_next_step)
                time.sleep(time_until_next_step)
        else: 
            print("sim closed")
            raise SystemExit
        
    def turtle_mode_callback(self, msg):
        if msg.mode == "kill":
            raise KeyboardInterrupt
        
    def publish_turtle_data(self):
        """
        Send data out
        """
        turtle_msg = TurtleState()
        
        self.spoof_altitude()
        # turtle_msg.q = self.q
        # turtle_msg.dq = self.dq
        turtle_msg.q = self.q
        turtle_msg.dq = self.dq
        
        turtle_msg.t = self.time_last_ctrl
        
        turtle_msg.u = self.tau
        # turtle_msg.qd = self.qd

        # turtle_msg.dqd = self.dqd.tolist()
        turtle_msg.imu.quat = self.quat
        turtle_msg.depth = self.data.qpos[2]
        turtle_msg.altitude = self.altitude
        # # angular velocity
        # print("acc msg")
        # # linear acceleration

        # publish msg 
        self.state_pub.publish(turtle_msg)
    
    def spoof_altitude(self):
        depth = - self.data.qpos[2] + 20.0
        self.altitude = depth
        if np.random.uniform(0.0, 1.0) > 1.0:
            self.altitude = self.altitude + np.random.normal(0.0, 10.0) * np.random.normal(0.0, 1.0) 
        if self.data.time > 2.0 and self.data.time < 10.0:
            self.altitude = self.altitude - self.data.time*0.5
        vec, theta = quat.quat2axangle(self.quat)
        # print(theta*180/np.pi)
        self.altitude = self.altitude / np.cos(theta)
            

def skew(v):
    """
    Returns the skew-symmetric matrix of a 3-dimensional vector.
    
    Parameters:
    v (array-like): A 3-element vector.
    
    Returns:
    numpy.ndarray: A 3x3 skew-symmetric matrix.
    """
    v = np.asarray(v)
    if v.size != 3:
        raise ValueError("Input vector must have exactly 3 elements.")
        
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

def main():
    

    rclpy.init()
    sim = Simulator()
    try:
        rclpy.spin(sim)
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
    raise SystemExit

if __name__ == '__main__':
    main()
