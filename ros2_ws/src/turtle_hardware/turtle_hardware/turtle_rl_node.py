import rclpy
from rclpy.node import Node
from rclpy.executors import Executor, MultiThreadedExecutor
from rcl_interfaces.msg import SetParametersResult
from turtle_interfaces.msg import TurtleTraj, TurtleSensors, TurtleMotorPos
from EPHE import EPHE
from DualCPG import DualCPG
from AukeCPG import AukeCPG
import sys
import os
import transforms3d.quaternions as quat
import numpy as np
import pickle
import torch


submodule = os.path.expanduser("~") + "/drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware"
sys.path.append(submodule)


from std_msgs.msg import String
from std_msgs.msg import Float64MultiArray
from dynamixel_sdk import *                                     # Uses Dynamixel SDK library
from turtle_dynamixel.Dynamixel import *                        # Dynamixel motor class                                  
from turtle_dynamixel.dyn_functions import *                    # Dynamixel support functions
from turtle_dynamixel.turtle_controller import *                # Controller 
from turtle_dynamixel.Constants import *                        # File of constant variables
from turtle_dynamixel.Mod import *
from turtle_dynamixel.utilities import *
import math
from math import cos, sin
from datetime import datetime
import numpy as np
from matplotlib import pyplot as plt
from turtle_dynamixel.utilities import *
import json
import traceback
from queue import Queue
import serial

if os.name == 'nt':
    import msvcrt
    def getch():
        return msvcrt.getch().decode()
    def kbhit():
        return msvcrt.kbhit()
else:
    import termios, fcntl, sys, os
    from select import select
    fd = sys.stdin.fileno()
    old_term = termios.tcgetattr(fd)
    new_term = termios.tcgetattr(fd)

    def getch():
        new_term[3] = (new_term[3] & ~termios.ICANON & ~termios.ECHO)
        termios.tcsetattr(fd, termios.TCSANOW, new_term)
        try:
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_term)
        return ch

    def kbhit():
        new_term[3] = (new_term[3] & ~(termios.ICANON | termios.ECHO))
        termios.tcsetattr(fd, termios.TCSANOW, new_term)
        try:
            dr,dw,de = select([sys.stdin], [], [], 0)
            if dr != []:
                return 1
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_term)
            sys.stdout.flush()

        return 0

os.system('sudo /home/crush/drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware/latency_write.sh')

# global variable was set in the callback function directly
global mode
mode = 'rest'

class TurtleRobot(Node):
    """
    This node is responsible for continously reading sensor data and receiving commands from the keyboard node
    to execute specific trajectoreies or handle emergency stops. It also is responsible for sending motor pos commands to the RL node
    for training and deployment purposes.
    TODO: migrate dynamixel motors into this class
    TLDR; this is the node that handles all turtle hardware things
    """

    def __init__(self, topic, max_episode_length=60, M=40, max_episodes=20, weights=[1, 1, 1, 1, 0.001]):
        super().__init__('turtle_node')
        # subscribes to keyboard setting different turtle modes 
        self.mode_cmd_sub = self.create_subscription(
            String,
            topic,
            self.turtle_mode_callback,
            10)
        # for case when trajectory mode, receives trajectory msg
        self.motor_traj_sub = self.create_subscription(
            TurtleTraj,
            'turtle_traj',
            self.trajectory_callback,
            10
        )
        # continously reads from all the sensors and publishes data at end of trajectory
        self.sensors_pub = self.create_publisher(
            TurtleSensors,
            'turtle_sensors',
            10
        )
        # sends acknowledgement packet to keyboard node
        self.cmd_received_pub = self.create_publisher(
            String,
            'turtle_state',
            10
        )
        # continously publishes the current motor position      
        self.motor_pos_pub = self.create_publisher(
            String,
            'turtle_motor_pos',
            10
        )
        self.mode_cmd_sub       # prevent unused variable warning
        self.create_rate(100)
        self.mode = 'rest'      # initialize motors mode to rest state
        self.voltage = 12.0
        self.n_axis = 3
        self.qs = np.zeros((10,1))
        self.dqs = np.zeros((10,1))
        self.tvec = np.zeros((1,1))
        self.acc_data = np.zeros((self.n_axis,1))
        self.gyr_data = np.zeros((self.n_axis,1))
        self.quat_data = np.zeros((4,1))
        self.tau_data = np.zeros((10,1))
        self.voltage_data = np.zeros((1,1))
        if portHandlerJoint.openPort():
            print("[MOTORS STATUS] Suceeded to open port")
        else:
            print("[ERROR] Failed to open port")
        if portHandlerJoint.setBaudRate(BAUDRATE):
            print("[MOTORS STATUS] Suceeded to open port")
        else:
            print("[ERROR] Failed to change baudrate")
        self.IDs = [1,2,3,4,5,6,7, 8, 9,10]
        self.nq = 10
        self.Joints = Mod(packetHandlerJoint, portHandlerJoint, self.IDs)
        self.Joints.disable_torque()
        self.Joints.set_current_cntrl_mode()
        self.Joints.enable_torque()
        self.q = np.array(self.Joints.get_position()).reshape(-1,1)
        self.xiao = serial.Serial('/dev/ttyACM0', 115200, timeout=3)   
        self.xiao.reset_input_buffer()
        self.input_history = np.zeros((self.nq,10))
        
        # set thresholds for motor angles 
        self.epsilon = 0.1
        self.min_threshold = np.array([1.60, 3.0, 2.4, 2.43, 1.2, 1.7, 1.45, 1.2, 3.0, 2.3])
        self.max_threshold = np.array([3.45, 5.0, 4.2, 4.5, 4.15, 3.8, 3.2, 4.0, 4.0, 4.7])
        # orientation at rest
        self.quat_data[:, -1] = [1, 1, 1, 1]
        self.orientation = np.array([0.679, 0.0, 0.0, -0.733])      # w, x, y, z
        self.a_weight, self.x_weight, self.y_weight, self.z_weight, self.tau_weight = weights

        # for PD control
        self.Kp = np.diag([0.5, 0.1, 0.1, 0.6, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1])*4
        self.KD = 0.1

    def turtle_mode_callback(self, msg):
        """
        Callback function that updates the mode the turtle should be in.
        This method is what enables us to set "emergency stops" mid-trajectory. 
        """
        # global mode
        # print("\n\n\n\n\n\ CALLLLBACKKKKKKKKK \n\n\n\n\n")
        if msg.data == 'traj1':
            self.mode = 'traj1'
        elif msg.data == 'train':
            self.mode = 'train'
        elif msg.data == 'stop':
            self.mode = 'stop'
        elif msg.data == 'teacher':
            self.mode = 'teacher'
        elif msg.data == 'Auke':
            self.mode = 'Auke'
        else:
            self.mode = 'rest'
    def reset(self):
        print("resetting turtle...\n")
        self.Joints.disable_torque()
        self.Joints.enable_torque()
        observation = "hi"
        # reset the sensor variables for next trajectory recording

        self.qs = np.zeros((10,1))
        self.dqs = np.zeros((10,1))
        self.tvec = np.zeros((1,1))
        self.acc_data = np.zeros((self.n_axis,1))
        self.gyr_data = np.zeros((self.n_axis,1))
        self.quat_data = np.zeros((4,1))
        self.tau_data = np.zeros((10,1))
        self.voltage_data = np.zeros((1,1))
        return observation, "reset"
    def quatL(self,q):
        qs = q[0]
        if qs == 0:
            print("f QS: {qs}\n\n\n\n\n")
        qv = q[1:]
        o = qs*np.identity(3) + self.skew(qv)
        # qv = qv.reshape((3,1))
        first_row = np.hstack((qs, -qv.T)).reshape(((1,4)))
        second_row = np.hstack((qv.reshape((3,1)), qs*np.identity(3) + self.skew(qv)))
        # return np.array([[qs, -qv.T],[qv, qs*np.identity(3) + self.skew(qv)]])
        return np.vstack((first_row, second_row))

    def inv_q(self,q):
        return np.diag([1, -1, -1, -1]) @ q

    def inv_cayley(self,q):
        return q[1:]/q[0]

    def skew(self,x):
        return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])
    
    # def _get_reward(self, tau):

    #     acc = self.acc_data[:, -1]
    #     del_q = self.inv_cayley(self.quatL(self.inv_q(self.quat_data[:, -1])) @ self.quat_data[:, -2])
    #     R = self.a_weight*acc[1]**2 - self.a_weight*acc[0]**2 - self.a_weight*acc[2]**2 - del_q.T @ del_q -self.tau_weight* np.linalg.norm(tau)**2

    #     # reward = 0
    #     # acc = self.acc_data[:, -1]
    #     # reward = np.linalg.norm(acc)
    #     # if acc[1] >0:
    #     #     reward += self.a_weight* acc[1]/reward
    #     # print(f"acc reward: {reward}")
    #     # orientation_cost = -np.linalg.norm(self.quat_data[1:, -1] - self.orientation[1:])
    #     # print(f"o cost: {orientation_cost}")
    #     # tau_cost = -self.tau_weight* np.linalg.norm(tau)**2
    #     # # forward thrust correlates to positive reading on the y-axis of the accelerometer
    #     # reward += orientation_cost
    #     # reward += tau_cost
    #     # print(R)
    #     return R
    def _get_reward(self, tau):
        reward = 0
        acc = self.acc_data[:, -1]
        reward = np.linalg.norm(acc)
        if acc[1] >0:
            reward += self.a_weight* acc[1]/reward
        # print(f"acc reward: {reward}")
        orientation_cost = -np.linalg.norm(self.quat_data[1:, -1] - self.orientation[1:])
        # print(f"o cost: {orientation_cost}")
        tau_cost = -self.tau_weight* np.linalg.norm(tau)**2
        # forward thrust correlates to positive reading on the y-axis of the accelerometer
        reward += orientation_cost
        reward += tau_cost
        return reward
    
    def _get_reward_weighted(self, tau):
        reward = 0
        return reward
    
    def step(self, action, PD=False):
        """
        action is tau which we pass into the turtle
        OR its position in radians when PD flag is set to True
        """
        # action = np.zeros((10,1))
        # action[7] = 1000
        # inputt = grab_arm_current(action, min_torque, max_torque)

        # action = action * 100000
        # print(f"act: {action}")
        # for a in range(len(action)):
        #     if action[a] < 0:
        #         action[a] = action[a] - 50
        #     elif action[a] > 0:
        #         action[a] = action[a] + 50
        #     else:
        #         action[a] = action[a]
        # print(f"CPG output scaled: {action}\n")
        keep_trying = True
        while keep_trying:
            try:
                observation = np.array(self.Joints.get_position())
                v = np.array(self.Joints.get_velocity())
                keep_trying = False
            except:
                print("failed to read from motors")
        if PD:
            # print(f"action shape: {action.shape}")
            # position control 
            qd = np.where(action < self.max_threshold - self.epsilon, action, self.max_threshold - self.epsilon)
            qd = np.where(qd > self.min_threshold + self.epsilon, qd, self.min_threshold + self.epsilon)
            dqd = np.zeros((self.nq,1))
            ddqd = np.zeros((self.nq,1))
            # print(f"dqd shape: {dqd.shape}")
            tau = turtle_controller(observation.reshape((10,1)),v.reshape((10,1)),qd.reshape((10,1)),dqd,ddqd,self.Kp,self.KD)
            avg_tau = np.mean(abs(tau))
            clipped = grab_arm_current(tau, min_torque, max_torque)
        else:
            # print(f"action: {action}")
            action = 10e3 * action
            print(f"action: {action}")

            inputt = grab_arm_current(action, min_torque, max_torque)
            # print(f"observation: {observation}\n")
            clipped = np.where(observation < self.max_threshold - self.epsilon, inputt, np.minimum(np.zeros(len(inputt)), inputt))
            # print(f"clipped1: {clipped}")
            clipped = np.where(observation > self.min_threshold + self.epsilon, clipped, np.maximum(np.zeros(len(inputt)), clipped)).astype('int')
            # print(f"clipped: {clipped}")
            avg_tau = np.mean(abs(clipped))
        print(f"torque: {clipped}")
        self.Joints.send_torque_cmd(clipped)
        self.read_sensors()
        reward = self._get_reward(avg_tau)
        terminated = False
        truncated = False
        # return velocity and clipped tau (or radians) passed into motors
        info = [v, clipped]
        return observation, reward, terminated, truncated, info

    def trajectory_callback(self, msg):
        """
        Callback function that takes in list of squeezed arrays
        msg: [qd, dqd, ddqd, tvec]
        """    
        n = len(msg.tvec)
        self.qds = np.array(msg.qd).reshape(10,n)
        self.tvec = np.array(msg.tvec).reshape(1,n)
        if len(msg.dqd) < 1:
            self.mode = 'position_control_mode'
        else:
            self.dqds = np.array(msg.dqd).reshape(10,n)
            self.ddqds = np.array(msg.ddqd).reshape(10,n)
            self.mode = 'traj_input'

    def teacher_callback(self, msg):
        """
        Has the robot turtle follow the teacher trajectory
        
        """
        n = len(msg.tvec)
        self.qds = np.array(msg.qd).reshape(10,n)
        self.tvec = np.array(msg.tvec).reshape(1,n)
        self.dqds = np.array(msg.dqd).reshape(10,n)
        self.ddqds = np.array(msg.ddqd).reshape(10,n)
        self.mode = 'teacher_traj_input'

    def np2msg(self, mat):
        """
        flattens nd numpy array into a lst for ros pkg messaging
        """
        nq = 10
        squeezed = np.reshape(mat, (nq * mat.shape[1]))
        return squeezed.tolist()
    
    def publish_turtle_data(self, q_data=[], dq_data=[], tau_data=[], timestamps=[], t_0=0.0, ddq_data=[]):
        """
        At the end of a trajectory (i.e when we set the turtle into a rest state or stop state), turtle node needs to 
        package all the sensor and motor data collected during that trajectory and publish it to the turtle_sensors node
        That way, it can be the local machine and not the turtle machine carrying all the data
        TODO: perhaps look into having turtle save data locally on raspi as well just in case?
        """
        turtle_msg = TurtleSensors()
        
        # # extract and flatten numpy arrays
        # print("quat")
        quat_x = self.quat_data[0, :]
        quat_y = self.quat_data[1, :]
        quat_z = self.quat_data[2, :]
        quat_w = self.quat_data[3, :]

        # print("acc")
        acc_x = self.acc_data[0, :]
        acc_y = self.acc_data[1, :]
        acc_z = self.acc_data[2, :]

        # print("gyr")
        gyr_x = self.gyr_data[0, :]
        gyr_y = self.gyr_data[1, :]
        gyr_z = self.gyr_data[2, :]

        # print("wuat msg")
        # # quaternion 
        # print(f"quat x: {quat_x}\n")
        quatx = quat_x.tolist()
        # print(f"quat x: {quatx}\n")
        # print(f"element type: {type(quatx[0])}")
        turtle_msg.imu.quat_x = quat_x.tolist()
        turtle_msg.imu.quat_y = quat_y.tolist()
        turtle_msg.imu.quat_z = quat_z.tolist()
        turtle_msg.imu.quat_w = quat_w.tolist()
        # print("gyr msg")
        # # angular velocity
        turtle_msg.imu.gyr_x = gyr_x.tolist()
        turtle_msg.imu.gyr_y = gyr_y.tolist()
        turtle_msg.imu.gyr_z = gyr_z.tolist()
        # print("acc msg")
        # # linear acceleration
        turtle_msg.imu.acc_x = acc_x.tolist()
        turtle_msg.imu.acc_y = acc_y.tolist()
        turtle_msg.imu.acc_z = acc_z.tolist()
        
        # print("volt msg")
        # print(f"voltage data: {self.voltage_data.tolist()}")
        # # voltage
        # volt_data = self.voltage_data.tolist()
        # print(f"voltage data: {volt_data}")
        # turtle_msg.voltage = volt_data  

        # timestamps and t_0
        if len(timestamps) > 0:
            turtle_msg.timestamps = timestamps.tolist()
        turtle_msg.t_0 = t_0

        # motor positions, desired positions and tau data
        if len(q_data) > 0:
            turtle_msg.q = self.np2msg(q_data) 
        if len(q_data) > 0:
            turtle_msg.dq = self.np2msg(dq_data)
        if len(ddq_data) > 0:
            turtle_msg.ddq = self.np2msg(ddq_data)
        qd_squeezed = self.np2msg(self.qds)
        turtle_msg.qd = qd_squeezed
        if len(tau_data) > 0:
            turtle_msg.tau = self.np2msg(tau_data)

        # publish msg 
        self.sensors_pub.publish(turtle_msg)
        print(f"published....")
        # reset the sensor variables for next trajectory recording
        self.qds = np.zeros((10,1))
        self.dqds = np.zeros((10,1))
        self.ddqds = np.zeros((10,1))
        self.tvec = np.zeros((1,1))
        self.acc_data = np.zeros((self.n_axis,1))
        self.gyr_data = np.zeros((self.n_axis,1))
        self.mag_data = np.zeros((self.n_axis,1))
        self.voltage_data = np.zeros((1,1))

# { "Quat": [ 0.008, 0.015, -0.766, 0.642 ],"Acc": [-264.00, -118.00, 8414.00 ],"Gyr": [-17.00, 10.00, 1.00 ],"Voltage": [ 11.77 ]}
    def read_voltage(self):
        no_check = False
        sensors = self.xiao.readline()
        # print(sensors)
        if len(sensors) != 0:
        # this ensures the right json string format
            if sensors[0] == 32 and sensors[-1] == 10:
                try:
                    sensor_dict = json.loads(sensors.decode('utf-8'))
                    # print(sensor_dict)
                except:
                    no_check = True
                # add sensor data
                if no_check == False:
                    sensor_keys = {'Voltage'}
                    if set(sensor_keys).issubset(sensor_dict):
                        volt = sensor_dict['Voltage'][0]
                        self.voltage = volt   
    def read_sensors(self):
        """
        Appends current sensor reading to the turtle node's sensor data structs
        """
        def add_place_holder():
            acc = self.acc_data[:, -1].reshape((3,1))
            gyr = self.gyr_data[:, -1].reshape((3,1))
            quat_vec = self.quat_data[:, -1].reshape((4,1))
            self.acc_data = np.append(self.acc_data, acc, axis=1)
            self.gyr_data = np.append(self.gyr_data, gyr, axis=1)
            self.quat_data = np.append(self.quat_data, quat_vec, axis=1)
            self.voltage_data = np.append(self.voltage_data, self.voltage_data[-1])
        
        no_check = False
        self.xiao.reset_input_buffer()
        sensors = self.xiao.readline()
        # print(sensors)
        # make sure it's a valid byte that's read
        if len(sensors) != 0:
            # this ensures the right json string format
            if sensors[0] == 32 and sensors[-1] == 10:
                try:
                    sensor_dict = json.loads(sensors.decode('utf-8'))
                except:
                    no_check = True
                # add sensor data
                if no_check == False:
                    sensor_keys = ('Acc', 'Gyr', 'Quat', 'Voltage')
                    if set(sensor_keys).issubset(sensor_dict):
                        a = np.array(sensor_dict['Acc']).reshape((3,1)) * 9.81
                        gyr = np.array(sensor_dict['Gyr']).reshape((3,1))
                        q = np.array(sensor_dict["Quat"])
                        R = quat.quat2mat(q)
                        g_w = np.array([[0], [0], [9.81]])
                        g_local = np.dot(R.T, g_w)
                        acc = a - g_local           # acc without gravity 
                        quat_vec = q.reshape((4,1))
                        volt = sensor_dict['Voltage'][0]
                        self.acc_data = np.append(self.acc_data, acc, axis=1)
                        self.gyr_data = np.append(self.gyr_data, gyr, axis=1)
                        self.quat_data = np.append(self.quat_data, quat_vec, axis=1)
                        self.voltage_data = np.append(self.voltage_data, volt)
                        self.voltage = volt
                else:
                    add_place_holder()
            else:
                add_place_holder()
        else:
            add_place_holder()

# THIS IS TURTLE CODE
def main(args=None):
    rclpy.init(args=args)
    trial = 6
    threshold = 11.5
    # the weights applied to reward function terms acc, dx, dy, dz, tau
    weights = [10, 1, 1, 1, 0.001]
    turtle_node = TurtleRobot('turtle_mode_cmd', weights=weights)
    q = np.array(turtle_node.Joints.get_position()).reshape(-1,1)
    print(f"Our initial q: " + str(q))

    # create folders 
    trial_folder = f'Auke_CPG_exp_{trial}'
    best_param_fname = trial_folder + f'/best_params_ephe_{trial}.pth'
    os.makedirs(trial_folder, exist_ok=True)

    # Bo Chen paper says robot converged in 20 episodes
    max_episodes = 20
    dt = 0.01
    max_episode_length = 800    # 200 * 0.01 = 2 seconds

    # initial params for training
    num_params = 21
    M = 10
    K = 3
    num_mods = 10

    # mu = np.random.rand((num_params))
    # mu[num_mods + 1:] = mu[num_mods + 1] + 5000
    # mu[0] = np.random.random(1) * 0.5


    # to allow higher torques at the beginning of training
    # tau_init = np.random.uniform(low=0.04, high=0.1, size=1)[0]     # lower tau gives you higher amplitude
    # params = np.random.uniform(low=0.05, high=1, size=num_params)
    # params[0] = tau_init
    params = np.random.rand((num_params)) 
    # data structs for plotting
    param_data = np.zeros((num_params, M, max_episodes))
    mu_data = np.zeros((num_params, max_episodes))
    sigma_data = np.zeros((num_params, max_episodes))
    reward_data = np.zeros((max_episode_length, M, max_episodes))
    cpg_data = np.zeros((10, max_episode_length, M, max_episodes))
    time_data = np.zeros((max_episode_length, M, max_episodes))
    q_data = np.zeros((10, max_episode_length, M, max_episodes))
    dq_data = np.zeros((10, max_episode_length, M, max_episodes))
    tau_data = np.zeros((10, max_episode_length, M, max_episodes))
    voltage_data = np.zeros((max_episode_length, M, max_episodes))
    acc_data = np.zeros((3, max_episode_length, M, max_episodes))
    gyr_data = np.zeros((3, max_episode_length, M, max_episodes))
    quat_data = np.zeros((4, max_episode_length, M, max_episodes))

    best_params = np.zeros((num_params))
    best_reward = 0

    # Specify the file path where you want to save the JSON file
    file_path = trial_folder + "/config.json"

    try: 
        while rclpy.ok():
            rclpy.spin_once(turtle_node)
            print(f"turtle mode: {turtle_node.mode}\n")
            if turtle_node.voltage < threshold:
                print("[WARNING] volt reading too low--closing program....")
                turtle_node.Joints.disable_torque()
                break
            if turtle_node.mode == 'stop':
                print("ending entire program...")
                print("disabling torques entirely...")
                turtle_node.Joints.send_torque_cmd([0] *len(turtle_node.IDs))
                turtle_node.Joints.disable_torque()
                cmd_msg = String()
                cmd_msg.data = "stop_received"
                turtle_node.cmd_received_pub.publish(cmd_msg)
                print("sent stop received msg")
                break
            elif turtle_node.mode == 'rest':
                rclpy.spin_once(turtle_node)
                turtle_node.read_voltage()
                if turtle_node.voltage < threshold:
                    turtle_node.Joints.disable_torque()
                    print("THRESHOLD MET TURN OFFFF")
                    break
                # print(turtle_node.Joints.get_position())
                # time.sleep(0.5)
                # turtle_node.Joints.send_torque_cmd([0] *len(turtle_node.IDs))
                turtle_node.Joints.disable_torque()
                print(turtle_node.Joints.get_position())
                cmd_msg = String()
                cmd_msg.data = 'rest_received'
                turtle_node.cmd_received_pub.publish(cmd_msg)
                print(f"current voltage: {turtle_node.voltage}\n")
            elif turtle_node.mode == 'Auke':
                # Auke CPG model that outputs position in radians
                phi=0.0
                w=0.5
                a_r=20
                a_x=20
                mu = np.random.rand((num_params)) * 5
                sigma = np.random.rand((num_params)) + 0.3

                config_log = {
                    "mu_init": list(mu),
                    "sigma_init": list(sigma),
                    "M": M,
                    "K": K,
                    "num episodes": max_episodes,
                    "max_episode_length": max_episode_length,
                    "phi": phi,
                    "w": w, 
                    "a_r": a_r,
                    "a_x": a_x,
                    "weights": weights,
                    "dt": dt
                }
                # Save the dictionary as a JSON file
                with open(file_path, 'w') as json_file:
                    json.dump(config_log, json_file)

                print(f"intial mu: {mu}\n")
                print(f"initial sigma: {sigma}\n")
                print(f"M: {M}\n")
                print(f"K: {K}\n")

                print("Auke CPG training time\n")
                cpg = AukeCPG(num_params=num_params, num_mods=num_mods, phi=phi, w=w, a_r=a_r, a_x=a_x, dt=dt)
                ephe = EPHE(
                    
                    # We are looking for solutions whose lengths are equal
                    # to the number of parameters required by the policy:
                    solution_length=mu.shape[0],
                    
                    # Population size: the number of trajectories we run with given mu and sigma 
                    popsize=M,
                    
                    # Initial mean of the search distribution:
                    center_init=mu,
                    
                    # Initial standard deviation of the search distribution:
                    stdev_init=sigma,

                    # dtype is expected as float32 when using the policy objects
                    dtype='float32', 

                    K=K
                )
                # get to training
                turtle_node.Joints.enable_torque()
    ############################ EPISODE #################################################
                for episode in range(max_episodes):
                    # data folder for this episode
                    rclpy.spin_once(turtle_node)
                    if turtle_node.mode == 'rest' or turtle_node.mode == 'stop' or turtle_node.voltage < threshold:
                        turtle_node.Joints.send_torque_cmd([0] *len(turtle_node.IDs))
                        turtle_node.Joints.disable_torque()
                        print("saving data...")
                        # save the best params
                        torch.save(best_params, best_param_fname)
                        # save data structs to matlab 
                        scipy.io.savemat(trial_folder + "/data.mat", {'mu_data': mu_data,'sigma_data': sigma_data,
                                            'param_data': param_data, 'reward_data': reward_data, 'q_data': q_data, 'dq_data': dq_data,
                                            'tau_data': tau_data, 'voltage_data': voltage_data, 'acc_data': acc_data, 'quat_data': quat_data, 
                                            'gyr_data': gyr_data, 'time_data': time_data, 'cpg_data': cpg_data})
                        break
                    print(f"--------------------episode: {episode} of {max_episodes}------------------")
                    lst_params = np.zeros((num_params, M))
                    solutions = ephe.ask()     
                    param_data[:, :, episode] = solutions.T
                    # rewards from M rollouts   
                    R = np.zeros(M)
    ###################### run your M rollouts########################################################################3
                    for m in range(M):
                        print(f"--------------------episode {episode} of {max_episodes}, rollout: {m} of {M}--------------------")
                        rclpy.spin_once(turtle_node)
                        if turtle_node.mode == 'rest' or turtle_node.mode == 'stop' or turtle_node.voltage < threshold:
                            print("breaking out of episode----------------------------------------------------------------------")
                            turtle_node.Joints.send_torque_cmd([0] *len(turtle_node.IDs))
                            turtle_node.Joints.disable_torque()
                            break
                        lst_params[:, m] = solutions[m]
                        cpg.set_parameters(solutions[m])
                        cumulative_reward = 0.0

                        # reset the environment after every rollout
                        timestamps = np.zeros(max_episode_length)
                        t = 0                       # to ensure we only go max_episode length
                        print("getting rollout")
                        cpg_actions = cpg.get_rollout(max_episode_length)
                        # cpg_actions = cpg.get_coupled_rollout(max_episode_length)
                        print("got rollout")
                        cpg_data[:, :, m, episode] = cpg_actions
                        observation, __ = turtle_node.reset()
                        t_0 = time.time()
    ############################ ROLL OUT ###############################################################################
                        while True:
                            rclpy.spin_once(turtle_node)
                            if turtle_node.mode == 'rest' or turtle_node.mode == 'stop' or turtle_node.voltage < threshold:
                                print("breaking out of rollout----------------------------------------------------------------------")
                                turtle_node.Joints.send_torque_cmd([0] *len(turtle_node.IDs))
                                turtle_node.Joints.disable_torque()
                                break
                            action = cpg_actions[:, t]
                            # save the raw cpg output
                            timestamps[t] = time.time()-t_0 
                            observation, reward, terminated, truncated, info = turtle_node.step(action, PD=True)
                            v, clipped = info
                            done = truncated or terminated
                            reward_data[t, m, episode] = reward
                            tau_data[:, t, m, episode] = clipped
                            q_data[:, t, m, episode] = observation
                            dq_data[:, t, m, episode] = v
                            cumulative_reward += reward
                            t += 1
                            if t >= max_episode_length:
                                turtle_node.Joints.disable_torque()
                                print("\n\n")
                                print(f"---------------rollout reward: {cumulative_reward}\n\n\n\n")
                                break
                        try:
                            # record data from rollout
                            time_data[:, m, episode] = timestamps
                            acc_data[:, :, m, episode] = turtle_node.acc_data[:, 1:]
                            gyr_data[:, :, m, episode] = turtle_node.gyr_data[:, 1:]
                            quat_data[:, :, m, episode] = turtle_node.quat_data[:, 1:]
                            voltage_data[:, m, episode] = turtle_node.voltage_data[1:]
                        except:
                            print(f"stopped mid rollout-- saving everything but this rollout data")
                        # save to folder after each rollout
                        scipy.io.savemat(trial_folder + "/data.mat", {'mu_data': mu_data,'sigma_data': sigma_data,
                            'param_data': param_data, 'reward_data': reward_data, 'q_data': q_data, 'dq_data': dq_data,
                            'tau_data': tau_data, 'voltage_data': voltage_data, 'acc_data': acc_data, 'quat_data': quat_data, 
                            'gyr_data': gyr_data, 'time_data': time_data, 'cpg_data': cpg_data})

                        # replaces set_params_and_run function for now
                        if cumulative_reward < 0:
                            fitness = 0
                        else:
                            fitness = cumulative_reward
                        if fitness > best_reward:
                            print(f"params with best fitness: {solutions[m]} with reward {fitness}\n")
                            best_reward = fitness
                            best_params = solutions[m]  
                        
                        # update reward array for updating mu and sigma
                        R[m] = fitness
                    print("--------------------- Episode:", episode, "  median score:", np.median(R), "------------------")
                    print(f"all rewards: {R}\n")
                    # get indices of K best rewards
                    best_inds = np.argsort(-R)[:K]
                    k_params = lst_params[:, best_inds]
                    print(f"k params: {k_params}")
                    k_rewards = R[best_inds]
                    print(f"k rewards: {k_rewards}")
                    # We inform our ephe solver of the fitnesses we received, so that the population gets updated accordingly.
                    # BUT we only update if all k rewards are positive
                    update = True
                    for g in k_rewards:
                        if g <= 0:
                            update = False
                    if update:
                        ephe.update(k_rewards=k_rewards, k_params=k_params)
                        print(f"---------------------new mu: {ephe.center()}---------------------------\n")
                        print(f"--------------------new sigma: {ephe.sigma()}--------------------------\n")
                    # save mus and sigmas
                    mu_data[:, episode] = ephe.center()
                    sigma_data[:, episode] = ephe.sigma()
                    best_mu = ephe.center()
                    best_sigma = ephe.sigma()
                turtle_node.Joints.disable_torque()
                print(f"best mu: {best_mu}\n")
                print(f"best sigma: {best_sigma}\n")
                print(f"best params: {best_params} got reward of {best_reward}\n")
                print("------------------------Saving data------------------------\n")
                print("saving data....")
                # save the best params from this episode
                torch.save(best_params, best_param_fname)
                # save data structs to matlab for this episode
                scipy.io.savemat(trial_folder + "/data.mat", {'mu_data': mu_data,'sigma_data': sigma_data,
                                            'param_data': param_data, 'reward_data': reward_data, 'q_data': q_data, 'dq_data': dq_data,
                                            'tau_data': tau_data, 'voltage_data': voltage_data, 'acc_data': acc_data, 'quat_data': quat_data, 
                                            'gyr_data': gyr_data, 'time_data': time_data, 'cpg_data': cpg_data})
                break
            elif turtle_node.mode == 'train':
                # Bo Cheng CPG implementation
                mu = np.random.uniform(low=0, high=2*np.pi, size=num_params)
                mu[1 + num_mods:] = np.random.uniform(low=10, high=20)
                mu[0] = 1.5
                sigma = np.random.rand((num_params)) + 0.3

                # CPG intrinsic weights
                alpha = 0.5
                omega = 0.5
                config_log = {
                    "mu_init": list(mu),
                    "sigma_init": list(sigma),
                    "M": M,
                    "K": K,
                    "num episodes": max_episodes,
                    "max_episode_length": max_episode_length,
                    "alpha": alpha,
                    "omega": omega, 
                    "weights": weights,
                    "dt": dt
                }

                # Save the dictionary as a JSON file
                with open(file_path, 'w') as json_file:
                    json.dump(config_log, json_file)
                
                print("TRAIN MODE")
                cpg = DualCPG(num_params=num_params, num_mods=num_mods, alpha=alpha, omega=omega, dt=dt)
                cpg.set_parameters(params=params)
                print(f"--------------starting params: {cpg.get_params()}-----------")
                ephe = EPHE(
                    # We are looking for solutions whose lengths are equal
                    # to the number of parameters required by the policy:
                    solution_length=mu.shape[0],
                    
                    # Population size: the number of trajectories we run with given mu and sigma 
                    popsize=M,
                    
                    # Initial mean of the search distribution:
                    center_init=mu,
                    
                    # Initial standard deviation of the search distribution:
                    stdev_init=sigma,

                    # dtype is expected as float32 when using the policy objects
                    dtype='float32', 

                    K=K
                )
                # get to training
                turtle_node.Joints.enable_torque()
    ############################ EPISODE #################################################
                for episode in range(max_episodes):
                    # data folder for this episode
                    rclpy.spin_once(turtle_node)
                    if turtle_node.mode == 'rest' or turtle_node.mode == 'stop' or turtle_node.voltage < threshold:
                        turtle_node.Joints.send_torque_cmd([0] *len(turtle_node.IDs))
                        turtle_node.Joints.disable_torque()
                        print("saving data...")
                        # save the best params
                        torch.save(best_params, best_param_fname)
                        # save data structs to matlab 
                        scipy.io.savemat(trial_folder + "/data.mat", {'mu_data': mu_data,'sigma_data': sigma_data,
                                            'param_data': param_data, 'reward_data': reward_data, 'q_data': q_data, 'dq_data': dq_data,
                                            'tau_data': tau_data, 'voltage_data': voltage_data, 'acc_data': acc_data, 'quat_data': quat_data, 
                                            'gyr_data': gyr_data, 'time_data': time_data, 'cpg_data': cpg_data})
                        break
                    print(f"--------------------episode: {episode} of {max_episodes}------------------")
                    lst_params = np.zeros((num_params, M))
                    solutions = ephe.ask()     
                    param_data[:, :, episode] = solutions.T
                    # rewards from M rollouts   
                    R = np.zeros(M)
    ###################### run your M rollouts########################################################################3
                    for m in range(M):
                        print(f"--------------------episode {episode} of {max_episodes}, rollout: {m} of {M}--------------------")
                        rclpy.spin_once(turtle_node)
                        if turtle_node.mode == 'rest' or turtle_node.mode == 'stop' or turtle_node.voltage < threshold:
                            print("breaking out of episode----------------------------------------------------------------------")
                            turtle_node.Joints.send_torque_cmd([0] *len(turtle_node.IDs))
                            turtle_node.Joints.disable_torque()
                            break
                        lst_params[:, m] = solutions[m]
                        cpg.set_parameters(solutions[m])
                        cumulative_reward = 0.0
                        timestamps = np.zeros(max_episode_length)
                        t = 0                       # to ensure we only go max_episode length
                        print("getting rollout...\n")
                        cpg_actions = cpg.get_rollout(max_episode_length)
                        cpg_data[:, :, m, episode] = cpg_actions
                        print(f"starting rollout...\n")
                        # reset the environment after every rollout
                        observation, __ = turtle_node.reset()
                        t_0 = time.time()
    ############################ ROLL OUT ###############################################################################
                        while True:
                            rclpy.spin_once(turtle_node)
                            if turtle_node.mode == 'rest' or turtle_node.mode == 'stop' or turtle_node.voltage < threshold:
                                print("breaking out of rollout----------------------------------------------------------------------")
                                turtle_node.Joints.send_torque_cmd([0] *len(turtle_node.IDs))
                                turtle_node.Joints.disable_torque()
                                break
                            action = cpg_actions[:, t]
                            timestamps[t] = time.time()-t_0 
                            observation, reward, terminated, truncated, info = turtle_node.step(action)
                            v, clipped = info
                            done = truncated or terminated
                            reward_data[t, m, episode] = reward
                            tau_data[:, t, m, episode] = clipped
                            q_data[:, t, m, episode] = observation
                            dq_data[:, t, m, episode] = v
                            cumulative_reward += reward
                            t += 1
                            if t >= max_episode_length:
                                turtle_node.Joints.disable_torque()
                                print(f"---------------rollout reward: {cumulative_reward}\n\n\n\n")
                                break
                        try:
                            # record data from rollout
                            time_data[:, m, episode] = timestamps
                            acc_data[:, :, m, episode] = turtle_node.acc_data[:, 1:]
                            gyr_data[:, :, m, episode] = turtle_node.gyr_data[:, 1:]
                            quat_data[:, :, m, episode] = turtle_node.quat_data[:, 1:]
                            voltage_data[:, m, episode] = turtle_node.voltage_data[1:]
                        except:
                            print(f"stopped mid rollout--saving everything but this rollout")

                        # save to folder after each rollout
                        scipy.io.savemat(trial_folder + "/data.mat", {'mu_data': mu_data,'sigma_data': sigma_data,
                            'param_data': param_data, 'reward_data': reward_data, 'q_data': q_data, 'dq_data': dq_data,
                            'tau_data': tau_data, 'voltage_data': voltage_data, 'acc_data': acc_data, 'quat_data': quat_data, 
                            'gyr_data': gyr_data, 'time_data': time_data, 'cpg_data': cpg_data})

                        # replaces set_params_and_run function for now
                        if cumulative_reward < 0:
                            fitness = 0
                        else:
                            fitness = cumulative_reward
                        if fitness > best_reward:
                            print(f"params with best fitness: {solutions[m]} with reward {fitness}\n")
                            best_reward = fitness
                            best_params = solutions[m]  
                        
                        # update the data structs for this rollout
                        R[m] = fitness
                    print("--------------------- Episode:", episode, "  median score:", np.median(R), "------------------")
                    print(f"all rewards: {R}\n")
                    # get indices of K best rewards
                    best_inds = np.argsort(-R)[:K]
                    k_params = lst_params[:, best_inds]
                    print(f"k params: {k_params}")
                    k_rewards = R[best_inds]
                    print(f"k rewards: {k_rewards}")
                    # We inform our ephe solver of the fitnesses we received, so that the population gets updated accordingly.
                    update = True
                    for g in k_rewards:
                        if g <= 0:
                            update = False
                    if update:
                        ephe.update(k_rewards=k_rewards, k_params=k_params)
                        print(f"---------------------new mu: {ephe.center()}---------------------------\n")
                        print(f"--------------------new sigma: {ephe.sigma()}--------------------------\n")
                    # save mus and sigmas
                    mu_data[:, episode] = ephe.center()
                    sigma_data[:, episode] = ephe.sigma()
                    best_mu = ephe.center()
                    best_sigma = ephe.sigma()
                turtle_node.Joints.disable_torque()
                print(f"best mu: {best_mu}\n")
                print(f"best sigma: {best_sigma}\n")
                print(f"best params: {best_params} got reward of {best_reward}\n")
                print("------------------------Saving data------------------------\n")
                # save the best params from this episode
                torch.save(best_params, best_param_fname)
                # save data structs to matlab for this episode
                scipy.io.savemat(trial_folder + "/data.mat", {'mu_data': mu_data,'sigma_data': sigma_data,
                                            'param_data': param_data, 'reward_data': reward_data, 'q_data': q_data, 'dq_data': dq_data,
                                            'tau_data': tau_data, 'voltage_data': voltage_data, 'acc_data': acc_data, 'quat_data': quat_data, 
                                            'gyr_data': gyr_data, 'time_data': time_data, 'cpg_data': cpg_data})
                break
            else:
                print("wrong command received....")
    except Exception as e:
        print("some error occurred")
        turtle_node.Joints.send_torque_cmd([0] * len(turtle_node.IDs))
        turtle_node.Joints.disable_torque()
        exec_type, obj, tb = sys.exc_info()
        fname = os.path.split(tb.tb_frame.f_code.co_filename)[1]
        print(exec_type, fname, tb.tb_lineno)
        print(e)
    rclpy.shutdown()

if __name__ == '__main__':
    main()

