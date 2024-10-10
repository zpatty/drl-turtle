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
from std_msgs.msg import String, Float32MultiArray
from turtle_interfaces.msg import TurtleTraj, TurtleSensors, TurtleCtrl, TurtleMode, TurtleState

from crush_mujoco_sim.analysis import compute_cost_of_transport, generate_control_plots
from turtle_robot_motion_control.controllers.joint_space_controllers import (
    navigation_joint_space_motion_primitive_control_factory
)



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
        model_name = f"turtle_{base_constraint}_base_control_vel{flippers_type_str}"

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

        # reference trajectory settings
        sw = 1.3  # s
        q_off = np.zeros((6,))
        # joint space control function
        self.joint_space_control_fn = navigation_joint_space_motion_primitive_control_factory(
            sw=sw,
            q_off=q_off,
            limit_cycle_kp=limit_cycle_kp,
            phase_sync_method=phase_sync_method,
            phase_sync_kp=phase_sync_kp,
            sliding_mode_params=sliding_mode_params,
        )
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

        timer_cb_group = None
        self.call_timer = self.create_timer(0.001, self._timer_cb, callback_group=timer_cb_group)

    def ctrl_callback(self, msg):
        self.u = msg.data[:4]
        self.pitch_d = msg.data[4]

    def _timer_cb(self):
        if self.viewer.is_running():
            step_start = time.time()

            if (self.data.time - self.time_last_ctrl) >= (1.0 / self.f_ctrl):
                # evaluate the controller
                # self.u = np.random.normal([0.95, 0.0, 0.0, 0.0],[0.000001, 0.001, 0.2, 0.2])
                # self.u = [1.0, 0.0, 1.0, 0.7]
                q_d_des, aux = self.joint_space_control_fn(
                    t=self.data.time,
                    q=self.data.qpos[7: 7 + 6],
                    u=self.u
                )
                q_arm_des = self.data.qpos[7: 7 + 6] + q_d_des * (self.data.time - self.time_last_ctrl)
                err = q_arm_des + 0.0 * np.diag([0.5, 0.25, 0, 0.5, 0.25, 0]) @  np.abs(q_arm_des) * np.sign(q_arm_des) - self.data.qpos[7:13]
                # apply the control signal
                print(err)
                fluid = self.data.qfrc_fluid.copy()
                kp = np.diag([2, 1, 5, 2, 1, 5])
                u = kp @ err + 0.1 * kp @ (q_d_des - self.data.qvel[7:13])

                self.data.ctrl[0:6] = q_d_des
                
                q = self.data.qpos[7 : 7 + 10]
                q_rear1 = q[7]
                q_rear2 = q[9]
                pitch_err = self.pitch_d
                # print(pitch_err)
                u_rear = 700 * pitch_err
                # print(u_rear)
                # apply the control signal
                k_r = 10.0
                self.data.ctrl[7] =  - k_r * (q_rear1 - 10 * np.pi / 180 * np.cos(4.0 * np.pi  * self.data.time) - np.clip(u_rear, -40.0, 40.0) * np.pi / 180)
                self.data.ctrl[9] = - k_r * (q_rear2 + 10 * np.pi / 180 * np.cos(4.0 * np.pi  * self.data.time) + np.clip(u_rear, -40.0, 40.0) * np.pi / 180)

                # update the last control time
                self.time_last_ctrl = self.data.time
            turtle_msg = TurtleSensors()
            quat = self.data.qpos[3:7].tolist()
            quat = quat[-1:] + quat[:-1]
            turtle_msg.imu.quat = quat
            # print(np.round(quat, 2))
            # pitch = data.qpos[5]
            # turtle_msg.imu.gyr = self.gyr
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

if __name__ == '__main__':
    main()
