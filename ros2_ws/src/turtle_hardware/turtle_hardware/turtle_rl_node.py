import rclpy
from rclpy.node import Node
from rclpy.executors import Executor, MultiThreadedExecutor
from rcl_interfaces.msg import SetParametersResult
from turtle_interfaces.msg import TurtleTraj, TurtleSensors, TurtleMotorPos
from EPHE import EPHE
from DualCPG import DualCPG
import sys
import os
import transforms3d.quaternions as quat
import numpy as np
import pickle
import torch

import gymnasium as gym

submodule = os.path.expanduser("~") + "/drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware"
sys.path.append(submodule)


from std_msgs.msg import String
from std_msgs.msg import Float64MultiArray
from dynamixel_sdk import *                    # Uses Dynamixel SDK library
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

    def __init__(self, topic, weights=[1, 1, 1, 1, 1]):
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
        self.qds = np.zeros((10,1))
        self.dqds = np.zeros((10,1))
        self.ddqds = np.zeros((10,1))
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
        self.xiao =  serial.Serial('/dev/ttyACM0', 115200, timeout=3)   
        self.xiao.reset_input_buffer()
        self.input_history = np.zeros((self.nq,10))
        
        # set thresholds for motor angles 
        self.epsilon = 0.2

        # TODO: fix limits on turtle
#         observation: [2.0202527  4.07732093 3.7045636  4.37184525 2.87161203 3.66928204
#  3.22135965 3.37168977 4.06658307 4.96549581]
        
#         Our initial q: [[ 2.31324303]
#  [ 3.85335974]
#  [ 3.29652471]
#  [ 3.88403935]
#  [ 1.38211669]
#  [-0.32673791]
#  [ 1.52631088]
#  [ 1.52937885]
#  [ 4.06044715]
#  [ 4.95782591]]

        
        self.min_threshold = np.array([1.60, 3.0, 2.4, 2.43, 1.2, 1.56, 1.45, 1.2, 3.0, 2.3])
        self.max_threshold = np.array([3.45, 5.0, 4.2, 4.5, 4.15, 3.6, 3.2, 4.0, 4.0, 5.3])

#         clipped: [ 36. 113.   0.  46. 220.   0.   0. 222.  43.   0.]
# clipped: (10,)
# observation: [1.85611675 4.56666081 4.83357346 4.30128213 3.72143739 3.61099077
#  3.22902956 3.99295199 3.1063111  5.30450556]



        # orientation at rest
        self.orientation = [0.679, 0.0, -0.005, -0.733]
        self.a_weight, self.x_weight, self.y_weight, self.z_weight, self.tau_weight = weights
    def turtle_mode_callback(self, msg):
        """
        Callback function that updates the mode the turtle should be in.
        This method is what enables us to set "emergency stops" mid-trajectory. 
        """
        # global mode
        if msg.data == 'traj1':
            self.mode = 'traj1'
        elif msg.data == 'train':
            self.mode = 'train'
        elif msg.data == 'stop':
            self.mode = 'stop'
        elif msg.data == 'teacher':
            self.mode = 'teacher'
        else:
            self.mode = 'rest'
    def reset(self):
        print(f"resetting turtle.../n")
        self.Joints.disable_torque()
        self.Joints.enable_torque()
        observation = "hi"
        # reset the sensor variables for next trajectory recording
        self.qds = np.zeros((10,1))
        self.dqds = np.zeros((10,1))
        self.ddqds = np.zeros((10,1))
        self.tvec = np.zeros((1,1))
        self.acc_data = np.zeros((self.n_axis,1))
        self.gyr_data = np.zeros((self.n_axis,1))
        self.mag_data = np.zeros((self.n_axis,1))
        self.voltage_data = np.zeros((1,1))
        return observation, "reset"
    
    def step(self, action):
        """
        action is tau which we pass into the turtle
        """
        keep_trying = True
        action = (action * 1000000) + 10
        # print(f"CPG output scaled: {action}\n")

        inputt = grab_arm_current(action, min_torque, max_torque)
        # print(q)
        while keep_trying:
            try:
                observation = np.array(self.Joints.get_position())
                keep_trying = False
            except:
                print("okie")
        print(f"observation: {observation}\n")
        clipped = np.where(observation < self.max_threshold, inputt, np.minimum(np.zeros(len(inputt)), inputt))
        # print(f"clipped: {clipped}")
        clipped = np.where(observation > self.min_threshold, clipped, np.maximum(np.zeros(len(inputt)), clipped)).astype('int')
        print(f"clipped: {clipped}")
        avg_tau = np.mean(abs(clipped))
        self.Joints.send_torque_cmd(clipped)
        # if observation > self.max_threshold:
        #     self.Joints.disable_torque()
        # if observation < self.min_threshold:
        #     self.Joints.disable_torque()
        
        # print(f"tau data shape: {self.tau_data.shape}")
        # print(f"clipped shape: {clipped.shape}")
        self.tau_data=np.append(self.tau_data, clipped.reshape((10,1)), axis=1) 
        self.read_sensors()
        # get latest acc data 
        acc = self.acc_data[:, -1]
        # print(f"acc: {acc}\n")
        terminated = False
        # grab current quat data
        w, x, y, z = self.quat_data[:, -1]
        # Convert quaternion to Euler angles
        dx = abs(self.orientation[0]- x)
        dy = abs(self.orientation[1]- y)
        dz = abs(self.orientation[2]- z)
        # print(f"dx, dy, dz: {[dx, dy, dz]}\n")
        # forward thrust correlates to positive reading on the y-axis of the accelerometer
        reward = acc[1] - dx - dy - dz - avg_tau
        if reward < 0:
            reward = 0
        # print(f"reward: {reward}\n")
        if reward > 1000:
            terminated = True
        truncated = False
        info = reward
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
        # # voltage
        # print(f"voltage data: {self.voltage_data.tolist()}")
        volt_data = self.voltage_data.tolist()
        # print(f"voltage data: {volt_data}")
        turtle_msg.voltage = volt_data  

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
            acc = np.array([-100.0, -100.0, -100.0]).reshape((3,1))
            gyr = np.array([-100.0, -100.0, -100.0]).reshape((3,1))
            quat_vec = np.array([-100.0, -100.0, -100.0, -100.0]).reshape((4,1))
            volt = -1
            self.acc_data = np.append(self.acc_data, acc, axis=1)
            self.gyr_data = np.append(self.gyr_data, gyr, axis=1)
            self.quat_data = np.append(self.quat_data, quat_vec, axis=1)
            self.voltage_data = np.append(self.voltage_data, volt)
        
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

# THIS IS TURTLE CODE
def main(args=None):

    # TODO: save q, dqs, ddqds, 
    rclpy.init(args=args)
    trial = 1
    threshold = 11.1
    # the weights applied to reward function terms acc, dx, dy, dz, tau
    weights = [1, 1, 1, 1, 1]
    turtle_node = TurtleRobot('turtle_mode_cmd', weights=weights)
    q = np.array(turtle_node.Joints.get_position()).reshape(-1,1)
    print(f"Our initial q: " + str(q))
    try: 
        while rclpy.ok():
            rclpy.spin_once(turtle_node)
            if turtle_node.voltage < threshold:
                print("[WARNING] volt reading too low--closing program....")
                turtle_node.Joints.disable_torque()
                break
            if turtle_node.mode == 'stop':
                print("ending entire program...")
                print("disabling torques entirely...")
                turtle_node.Joints.send_torque_cmd([0] *len(turtle_node.IDs))
                turtle_node.Joints.disable_torque()
                turtle_node.publish_turtle_data(t_0=0.0, q_data=[], dq_data=[], tau_data=[], timestamps=[]) 

                cmd_msg = String()
                cmd_msg.data = "stop_received"
                turtle_node.cmd_received_pub.publish(cmd_msg)
                print("sent stop received msg")
                break
            elif turtle_node.mode == 'rest':
                turtle_node.read_voltage()
                if turtle_node.voltage < threshold:
                    turtle_node.Joints.disable_torque()
                    print("THRESHOLD MET TURN OFFFF")
                    break
                # print(turtle_node.Joints.get_position())
                # time.sleep(0.5)
                turtle_node.Joints.send_torque_cmd([0] *len(turtle_node.IDs))
                turtle_node.Joints.disable_torque()
                cmd_msg = String()
                cmd_msg.data = 'rest_received'
                turtle_node.cmd_received_pub.publish(cmd_msg)
                print(f"current voltage: {turtle_node.voltage}\n")
                # turtle_node.read_sensors()
                # print(f"acc: {turtle_node.acc_data[:, -1]}\n")
                # print(f"quat: {turtle_node.quat_data[:, -1]}\n")
            elif turtle_node.mode == 'train1':
                print("Isjpeert")
            elif turtle_node.mode == 'traj_input':
                # NOTE: each trajectory starts with a two second offset period for turtle to properly 
                # get to the first desired q state (TODO: maybe set it to 1 second offset?)

                # Load desired trajectories from motors node
                print("traj input")
                qd_mat = turtle_node.qds
                dqd_mat = turtle_node.dqds
                ddqd_mat = turtle_node.ddqds
                tvec = turtle_node.tvec     
                # print(f"qd mat: {qd_mat.shape}\n")
                # print(f"qd mat first elemetn: {qd_mat[:, 1]}\n")
                print(f"full thing is: {tvec.shape}\n")
                print(f"shape qd_mat: {qd_mat.shape}\n")
                # print(f"shape of tvec: {tvec.sh}")
                first_time = True
                first_loop = True
                input_history = np.zeros((turtle_node.nq,10))
                q_data = np.zeros((turtle_node.nq,1))
                dq_data = np.zeros((turtle_node.nq,1))
                tau_data = np.zeros((turtle_node.nq,1))
                timestamps = np.zeros((1,1))
                print(f"[MODE] TRAJECTORY\n")
                turtle_node.Joints.disable_torque()
                turtle_node.Joints.set_current_cntrl_mode()
                turtle_node.Joints.enable_torque()
                Kp = np.diag([0.5, 0.1, 0.1, 0.6, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1])*4
                # Kp = np.diag([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.07, 0.07, 0.5, 0.2])*4
                # print(Kp)

                KD = 0.1
                t_begin = time.time()
                # zero =  np.zeros((self.nq,1))
                t_old = time.time()
                # our loop's "starting" time
                t_0 = time.time()
                while 1:
                    if turtle_node.voltage < threshold:
                        print("voltage too low--powering off...")
                        turtle_node.Joints.disable_torque()
                        print("saving data....")
                        turtle_node.publish_turtle_data(t_0=t_0, q_data=q_data, dq_data=dq_data, tau_data=tau_data, timestamps=timestamps) 
                        break
                    rclpy.spin_once(turtle_node)
                    # print("traj 1...")
                    if turtle_node.mode == 'rest' or turtle_node.mode == 'stop':
                        turtle_node.Joints.send_torque_cmd([0] *len(turtle_node.IDs))
                        turtle_node.Joints.disable_torque()
                        print("saving data...")
                        turtle_node.publish_turtle_data(t_0=t_0, q_data=q_data, dq_data=dq_data, tau_data=tau_data, timestamps=timestamps) 
                        first_time = True
                        break
                    q = np.array(turtle_node.Joints.get_position()).reshape(-1,1)
                    # print(f"q: {q}\n")
                    # q = np.insert(q, 7, place_holder).reshape((10,1))

                    if first_loop:
                        n = get_qindex((time.time() - t_0), tvec)
                    else:
                        # print("done with first loop")
                        offset = t_0 - 2
                        n = get_qindex((time.time() - offset), tvec)

                    # print(f"n: {n}\n")
                    if n == len(tvec[0]) - 1:
                        # print(f"time: {(time.time() - offset)}\n")
                        first_loop = False
                        t_0 = time.time()
                    
                    qd = np.array(qd_mat[:, n]).reshape(-1,1)
                    dqd = np.array(dqd_mat[:, n]).reshape(-1,1)
                    ddqd = np.array(ddqd_mat[:, n]).reshape(-1,1)
                    # # print(f"[DEBUG] qdata: {q_data}\n")
                    # print(f"[DEBUG] qd: {qd}\n")
                    q_data=np.append(q_data, q, axis = 1) 
                    # # At the first iteration velocity is 0  
                    
                    if first_time:
                        # dq = np.zeros((nq,1))
                        dq = np.array(turtle_node.Joints.get_velocity()).reshape(-1,1)
                        dq_data=np.append(dq_data, dq, axis = 1) 
                        q_old = q
                        first_time = False
                    else:
                        t = time.time()
                        dt = t - t_old
                    #     # print(f"[DEBUG] dt: {dt}\n")  
                        t_old = t
                        # dq = diff(q, q_old, dt)
                        dq = np.array(turtle_node.Joints.get_velocity()).reshape(-1,1)
                        dq_data=np.append(dq_data, dq, axis = 1) 
                        q_old = q
                    # print(f"dq from mod: {np.array(Joints.get_velocity()).reshape(-1,1)}\n")
                    # print(f"calculated dq: {dq}")
                    
                    timestamps = np.append(timestamps, (time.time()-t_begin)) 
                    tau = turtle_controller(q,dq,qd,dqd,ddqd,Kp,KD)
                    tau_data=np.append(tau_data, tau, axis=1) 
                    
                    input_history = np.append(input_history[:,1:], tau,axis=1)

                    input_mean = np.mean(input_history, axis = 1)

                    inputt = grab_arm_current(input_mean, min_torque, max_torque)
                    # del inputt[7]
                    # print(f"[DEBUG] inputt: {inputt}\n")
                    # print(f"voltage: {turtle_node.voltage}\n")
                    # inputt = [0]*10
                    turtle_node.Joints.send_torque_cmd(inputt)
                    turtle_node.read_sensors()
                turtle_node.Joints.disable_torque()
            
            elif turtle_node.mode == 'train':
                print("TRAIN MODE")
                # for saving data
                trial_folder = f'CPG_exp_{trial}'
                best_param_fname = trial_folder + f'/best_params_ephe_{trial}.pth'
                os.makedirs(trial_folder, exist_ok=True)

                # initial params for training
                num_params = 21
                M = 40
                K = 5
                mu = np.random.rand((num_params)) * 10
                params = np.random.rand((num_params)) * 10
                tau_shift = 0.2
                B_shift = 0.5
                E_shift = 0.5
                sigma = np.random.rand((num_params)) 
                # shifty_shift = [tau_shift, 
                #                 B_shift, B_shift, B_shift + 0.3,
                #                 B_shift + 0.3, B_shift, B_shift,
                #                 E_shift, E_shift, E_shift,
                #                 E_shift, E_shift, E_shift]
                print(f"intial mu: {mu}\n")
                print(f"initial sigma: {sigma}\n")
                print(f"M: {M}\n")
                print(f"K: {K}\n")
                num_mods = 10
                alpha = 0.5
                omega = 0.5
                cpg = DualCPG(num_params=num_params, num_mods=num_mods, alpha=alpha, omega=omega)
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

                # Bo Chen paper says robot converged in 20 episodes
                max_episodes = 2
                max_episode_length = 80     # 60 * 0.05 = ~3 seconds

                config_log = {
                    "mu_init": list(mu),
                    "sigma_init": list(sigma),
                    "params": list(params),
                    "M": M,
                    "K": K,
                    "num episodes": max_episodes,
                    "max_episode_length": max_episode_length,
                    "alpha": alpha,
                    "omega": omega, 
                    "weights": weights
                }

                # data structs for plotting
                param_data = np.zeros((num_params, M, max_episodes))
                mu_data = np.zeros((num_params, max_episodes))
                sigma_data = np.zeros((num_params, max_episodes))
                reward_data = np.zeros((M, max_episodes))
                best_params = np.zeros((num_params))
                best_reward = 0

                # Specify the file path where you want to save the JSON file
                file_path = trial_folder + "/config.json"

                # Save the dictionary as a JSON file
                with open(file_path, 'w') as json_file:
                    json.dump(config_log, json_file)

                t_0 = time.time()
                timestamps = np.zeros((1,1))

                # get to training
                turtle_node.Joints.enable_torque()
                for episode in range(max_episodes):
                    folder_name = trial_folder + f"/CPG_episode_{episode}"
                    rclpy.spin_once(turtle_node)
                    if turtle_node.voltage < threshold:
                        print("voltage too low--powering off...")
                        turtle_node.Joints.disable_torque()
                        print("saving data....")
                        # save the best params
                        torch.save(best_params, best_param_fname)
                        # save data structs to matlab 
                        scipy.io.savemat(trial_folder + "/data.mat", {'mu_data': mu_data,'sigma_data': sigma_data, 'param_data': param_data, 'reward_data': reward_data})
                        turtle_node.publish_turtle_data(t_0=t_0, q_data=turtle_node.qds, dq_data=turtle_node.dqds, tau_data=turtle_node.tau_data, timestamps=timestamps) 
                        break
                    if turtle_node.mode == 'rest' or turtle_node.mode == 'stop':
                        turtle_node.Joints.send_torque_cmd([0] *len(turtle_node.IDs))
                        turtle_node.Joints.disable_torque()
                        print("saving data...")
                        # save the best params
                        torch.save(best_params, best_param_fname)
                        # save data structs to matlab 
                        scipy.io.savemat(trial_folder + "/data.mat", {'mu_data': mu_data,'sigma_data': sigma_data, 'param_data': param_data, 'reward_data': reward_data})
                        turtle_node.publish_turtle_data(t_0=t_0, q_data=turtle_node.qds, dq_data=turtle_node.dqds, tau_data=turtle_node.tau_data, timestamps=timestamps) 

                        trial += 1
                        break
                    print(f"--------------------episode: {episode}------------------")
                    lst_params = np.zeros((num_params, M))
                    solutions = ephe.ask()          
                    R = np.zeros(M)
                    for i in range(M):
                        rclpy.spin_once(turtle_node)
                        if turtle_node.mode == 'rest' or turtle_node.mode == 'stop':
                            print("breaking out of episode----------------------------------------------------------------------")
                            turtle_node.Joints.send_torque_cmd([0] *len(turtle_node.IDs))
                            turtle_node.Joints.disable_torque()
                            trial += 1
                            break
                        subplot=True
                        lst_params[:, i] = solutions[i]

                        # need to just implement method here so that we can actually call stoof
                        
                        cpg.set_parameters(solutions[i])
                        cumulative_reward = 0.0
                        observation, __ = turtle_node.reset()
                        t = 0
                        first_time = True
                        total_actions = np.zeros((num_mods, 1))
                        while True:
                            rclpy.spin_once(turtle_node)
                            if turtle_node.mode == 'rest' or turtle_node.mode == 'stop':
                                print("breaking out of trajectory----------------------------------------------------------------------")
                                turtle_node.Joints.send_torque_cmd([0] *len(turtle_node.IDs))
                                turtle_node.Joints.disable_torque()
                                break
                            dt = 0.002
                            action = cpg.get_action(dt)
                            # print(f"action shape: {action.shape}")
                            total_actions = np.append(total_actions, action.reshape((10,1)), axis=1)
                            # print(f"params: {self.params}\n")
                            # print(f"action: {action}\n")
                            timestamps = np.append(timestamps, (time.time()-t_0)) 
                            observation, reward, terminated, truncated, info = turtle_node.step(action)
                            done = truncated or terminated
                            cumulative_reward += reward
                            t += 1
                            if t > max_episode_length:
                                print(f"we're donesies with {t}")
                                break
                            if done:
                                if truncated:
                                    print("truncccc")
                                if terminated:
                                    print("terminator")
                                break
                        # replaces set_params_and_run function for now
                        fitness = cumulative_reward
                        # fitness, total_actions = cpg.set_params_and_run(env=turtle_node, policy_parameters=solutions[i], max_episode_length=max_episode_length)
                        timesteps = total_actions.shape[1] - 1
                        dt = 0.002
                        t = np.arange(0, timesteps*dt, dt)
                        if fitness > best_reward:
                            print(f"params with best fitness: {solutions[i]} with reward {fitness}\n")
                            best_reward = fitness
                            best_params = solutions[i]  
                        R[i] = fitness
                    print("--------------------- Episode:", episode, "  median score:", np.median(R), "------------------")
                    print(f"all rewards: {R}\n")
                    # get indices of K best rewards
                    best_inds = np.argsort(-R)[:K]
                    k_params = lst_params[:, best_inds]
                    print(f"k params: {k_params}")
                    k_rewards = R[best_inds]
                    print(f"k rewards: {k_rewards}")
                    # We inform our ephe solver of the fitnesses we received,
                    # so that the population gets updated accordingly.
                    ephe.update(k_rewards=k_rewards, k_params=k_params)
                    print(f"new mu: {ephe.center()}\n")
                    print(f"new sigma: {ephe.sigma()}\n")
                    # save param data
                    param_data[:, :, episode] = lst_params
                    # save mus and sigmas
                    mu_data[:, episode] = ephe.center()
                    sigma_data[:, episode] = ephe.sigma()
                    reward_data[:, episode] = R
                best_mu = ephe.center()
                best_sigma = ephe.sigma()
                print(f"best mu: {best_mu}\n")
                print(f"best sigma: {best_sigma}\n")
                print(f"best params: {best_params} got reward of {best_reward}\n")
                print("------------------------Saving data------------------------\n")
                # save the best params
                torch.save(best_params, best_param_fname)
                # save data structs to matlab 
                scipy.io.savemat(trial_folder + "/data.mat", {'mu_data': mu_data,'sigma_data': sigma_data, 'param_data': param_data, 'reward_data': reward_data})
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

