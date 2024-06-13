import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from turtle_interfaces.msg import TurtleTraj, TurtleSensors
import transforms3d.quaternions as quat
# import torch
from EPHE import EPHE
from DualCPG import DualCPG
from AukeCPG import AukeCPG
# import gymnasium as gym
# from gymnasium import spaces
import cv2

# submodule = os.path.expanduser("~") + "/drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware"
# sys.path.append(submodule)

from std_msgs.msg import String
from dynamixel_sdk import *                                     # Uses Dynamixel SDK library
from turtle_dynamixel.Dynamixel import *                        # Dynamixel motor class                                  
from turtle_dynamixel.dyn_functions import *                    # Dynamixel support functions
from turtle_dynamixel.turtle_controller import *                # Controller 
from turtle_dynamixel.Constants import *                        # File of constant variables
from turtle_dynamixel.Mod import *
from turtle_dynamixel.utilities import *
from turtle_dynamixel.utilities import *
import numpy as np
import json
import serial
import random
from statistics import mode as md
# print(f"sec import toc: {time.time()-tic}")

# tic = time.time()
os.system('sudo /home/tortuga/drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware/turtle_dynamixel/latency_write.sh')

# global variable was set in the callback function directly
global mode
mode = 'rest'

# print(f"toc: {time.time()-tic}\n")
class TurtleRobot(Node):

# class TurtleRobot(Node, gym.Env):
    """
    This node is responsible for continously reading sensor data and receiving commands from the keyboard node
    to execute specific trajectoreies or handle emergency stops. It also is responsible for sending motor pos commands to the RL node
    for training and deployment purposes.
    TODO: migrate dynamixel motors into this class
    TLDR; this is the node that handles all turtle hardware things
    """

    def __init__(self, topic, params=None):
        super().__init__('turtle_rl_node')
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
        # subscribes to keyboard setting different turtle modes 
        self.mode_cmd_sub = self.create_subscription(
            String,
            topic,
            self.turtle_mode_callback,
            qos_profile)
        # for case when trajectory mode, receives trajectory msg
        self.motor_traj_sub = self.create_subscription(
            TurtleTraj,
            'turtle_traj',
            self.trajectory_callback,
            qos_profile
        )
        # for case camera mode, receives motion primitive
        self.cam_sub = self.create_subscription(
            String,
            'primitive',
            self.primitive_callback,
            buff_profile
        )
        # continously reads from all the sensors and publishes data at end of trajectory
        self.sensors_pub = self.create_publisher(
            TurtleSensors,
            'turtle_sensors',
            qos_profile
        )
        # sends acknowledgement packet to keyboard node
        self.cmd_received_pub = self.create_publisher(
            String,
            'turtle_state',
            qos_profile
        )
        # continously publishes the current motor position      
        self.motor_pos_pub = self.create_publisher(
            String,
            'turtle_motor_pos',
            qos_profile
        )
        self.mode_cmd_sub       # prevent unused variable warning
        self.create_rate(100)
        self.mode = 'rest'      # initialize motors mode to rest state
        self.primitives = ['dwell']
        self.voltage = 12.0
        self.n_axis = 3
        self.qs = np.zeros((10,1))
        self.dqs = np.zeros((10,1))
        self.tvec = np.zeros((1,1))
        if portHandlerJoint.openPort():
            print("[MOTORS STATUS] Suceeded to open port")
        else:
            print("[ERROR] Failed to open port")
        if portHandlerJoint.setBaudRate(BAUDRATE):
            print("[MOTORS STATUS] Suceeded to open port")
        else:
            print("[ERROR] Failed to change baudrate")

        # setup params
        # self.read_params(params=params)
        # self.create_data_structs()

        # dynamixel setup
        self.IDs = [1,2,3,4,5,6,7,8,9,10]
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
        # self.quat_data[:, -1] = [1, 1, 1, 1]
        # self.orientation = np.array([0.679, 0.0, 0.0, -0.733])      # w, x, y, z
        # for PD control
        self.Kp = np.diag([0.6, 0.3, 0.1, 0.6, 0.3, 0.1, 0.4, 0.4, 0.4, 0.4])*4
        self.KD = 0.1
        self.turtle_trajs = ["dive", "straight", "surface", "turnlf", "turnrf"]
        # self.action_space = spaces.Box(low=0, high=30,
        #                                     shape=(self.nq,1), dtype=np.float32)
        # self.observation_space = spaces.Box(low=0, high=30,
        #                                     shape=(self.nq,1), dtype=np.float32)
    def read_params(self, params):
        self.ax_weight = params["ax_weight"]
        self.ay_weight = params["ay_weight"]
        self.az_weight = params["az_weight"]

        self.tau_weight = params["tau_weight"]
        self.quat_weight = params["quat_weight"]


    # def forward(self, state, action) -> Tuple[torch.Tensor, torch.Tensor]:
    #     tensor = torch.cat([state, action], dim=1)
    #     q_value = self.multilayer_perceptron(tensor)
    #     return q_value

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
        elif msg.data == 'Auke':
            self.mode = 'Auke'
        elif msg.data == 'cv':
            self.mode = 'cv'
        elif msg.data == 'PGPE':
            self.mode = 'PGPE'
        elif msg.data == 'SAC':
            self.mode = 'SAC'
        elif msg.data in self.turtle_trajs:
            self.mode = msg.data
        elif msg.data == 'straight':
            self.mode = 'straight'
        elif msg.data == 'planner':
            self.mode = 'planner'
        else:
            self.mode = 'rest'  

    def primitive_callback(self, msg):
        """
        Callback function that updates the desired behavior based on the computer vision module. 
        """
        self.primitives.append(msg.data)
        

    def create_data_structs(self):
        """
        Create initial data structs to hold turtle data
        """
        # data structs for recording each rollout
        self.acc_data = np.zeros((self.n_axis,1))
        self.gyr_data = np.zeros((self.n_axis,1))
        self.quat_data = np.zeros((4,1))
        self.tau_data = np.zeros((10,1))
        self.voltage_data = np.zeros((1,1))

        # to store individual contributions of rewards 
        self.a_rewards = np.zeros((1,1))
        self.x_rewards = np.zeros((1,1))
        self.y_rewards = np.zeros((1,1))
        self.z_rewards = np.zeros((1,1))
        self.tau_rewards = np.zeros((1,1))

    def reset(self):
        print("resetting turtle...\n")
        self.Joints.disable_torque()
        self.Joints.enable_torque()
        q = self.Joints.get_position()
        v = self.Joints.get_velocity()
        acc = self.acc_data[:, -1]
        quat = self.quat_data[:, -1]
        observation = np.concatenate((q, v), axis=0)
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
        print(f"q: {q}")
        return q[1:]/q[0]

    def skew(self,x):
        return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])
    
    def _get_reward(self, tau):
        q = self.quat_data[:, -1]
        acc = self.acc_data[:, -1]
        R = self.ay_weight*acc[1]**2 - self.ax_weight*acc[0]**2 - self.az_weight*acc[2]**2 -self.tau_weight* np.linalg.norm(tau)**2 - self.quat_weight * (1 - np.linalg.norm(self.orientation.T @ q))
        return R
    
    def _get_reward_weighted(self, tau):
        reward = 0
        return reward
    
    def step(self, action, PD=False, mode='None'):
        """
        action is tau which we pass into the turtle
        OR its position in radians when PD flag is set to True
        """
        keep_trying = True
        while keep_trying:
            try:
                observation = np.array(self.Joints.get_position())
                v = np.array(self.Joints.get_velocity())
                keep_trying = False
            except:
                print("failed to read from motors")
        if mode == 'SAC':
            qd = np.where(action < self.max_threshold - self.epsilon, action, self.max_threshold - self.epsilon)
            qd = np.where(qd > self.min_threshold + self.epsilon, qd, self.min_threshold + self.epsilon)
            dqd = np.zeros((self.nq,1))
            ddqd = np.zeros((self.nq,1))
            # print(f"dqd shape: {dqd.shape}")
            tau = turtle_controller(observation.reshape((10,1)),v.reshape((10,1)),qd.reshape((10,1)),dqd,ddqd,self.Kp,self.KD)
            clipped = grab_arm_current(tau, min_torque, max_torque)

            terminated = False
            truncated = False
            info = [v, clipped]
            self.read_sensors()
            reward = self._get_reward(tau)
            obs = self.acc_data[:, -1]
            return [obs, reward, terminated, truncated, info]
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
            # print(f"action: {action}")

            inputt = grab_arm_current(action, min_torque, max_torque)
            # print(f"observation: {observation}\n")
            clipped = np.where(observation < self.max_threshold - self.epsilon, inputt, np.minimum(np.zeros(len(inputt)), inputt))
            # print(f"clipped1: {clipped}")
            clipped = np.where(observation > self.min_threshold + self.epsilon, clipped, np.maximum(np.zeros(len(inputt)), clipped)).astype('int')
            # print(f"clipped: {clipped}")
            avg_tau = np.mean(abs(clipped))
        # print(f"torque: {clipped}")
        # self.Joints.send_torque_cmd(clipped)
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
        self.xiao.reset_input_buffer()
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
    def add_place_holder(self):
        acc = self.acc_data[:, -1].reshape((3,1))
        gyr = self.gyr_data[:, -1].reshape((3,1))
        quat_vec = self.quat_data[:, -1].reshape((4,1))
        self.acc_data = np.append(self.acc_data, acc, axis=1)
        self.gyr_data = np.append(self.gyr_data, gyr, axis=1)
        self.quat_data = np.append(self.quat_data, quat_vec, axis=1)
        self.voltage_data = np.append(self.voltage_data, self.voltage_data[-1])
 
    def read_sensors(self):
        """
        Appends current sensor reading to the turtle node's sensor data structs
        """        
        no_check = False
        keep_trying = True
        attempts = 0
        while keep_trying:
            if attempts >= 3:
                print("adding place holder")
                self.add_place_holder()
                break
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
                            keep_trying = False
            attempts += 1

def parse_learning_params():
    with open('turtle_config/config.json') as config:
        param = json.load(config)
    print(f"[MESSAGE] Config: {param}\n")    
    # Serializing json
    config_params = json.dumps(param, indent=14)
    
    return param, config_params
def parse_cv_params():
    with open('cv_config.json') as config:
        param = json.load(config)
    print(f"[MESSAGE] Config: {param}\n")    
    # Serializing json
    config_params = json.dumps(param, indent=14)
    return param, config_params

def main(args=None):
    rclpy.init(args=args)
    threshold = 11.3
    turtle_node = TurtleRobot('turtle_mode_cmd')
    q = np.array(turtle_node.Joints.get_position()).reshape(-1,1)
    print(f"Our initial q: " + str(q))
    # create folders
    try:
        typed_name =  input("give folder a name: ")
        folder_name = "data/" + typed_name
        os.makedirs(folder_name, exist_ok=False)

    except:
        print("received weird input or folder already exists-- naming folder with timestamp")
        t = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        folder_name =  "data/" + t
        os.makedirs(folder_name)
 

    best_reward = 0
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
                    print(f"VOLTAGE: {turtle_node.voltage}")
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
            elif turtle_node.mode in turtle_node.turtle_trajs:
                rclpy.spin_once(turtle_node)
                num_cycles = 5
                turtle_node.Joints.disable_torque()
                turtle_node.Joints.set_current_cntrl_mode()
                turtle_node.Joints.enable_torque()
                input_history = np.zeros((nq,10))
                q_data = np.zeros((nq,1))
                dq_data = np.zeros((nq,1))
                tau_data = np.zeros((nq,1))
                current_data = np.zeros((nq,1))

                time_data = np.zeros((1,1))

                while True:
                    rclpy.spin_once(turtle_node)
                    if turtle_node.mode == 'rest' or turtle_node.mode == 'stop' or turtle_node.voltage < threshold:
                        turtle_node.Joints.send_torque_cmd([0] *len(turtle_node.IDs))
                        turtle_node.Joints.disable_torque()
                        scipy.io.savemat(folder_name + "/primitive_data.mat", {
                               'q_data': q_data, 'dq_data': dq_data,
                                'tau_data': tau_data, 'time_data': time_data, 'current_data': current_data})

                        break

                    primitive = turtle_node.mode
                    if primitive != "dwell":
                        print(f"---------------------------------------PRIMITIVE: {primitive}\n\n")
                        qd_mat = mat2np(f'/home/tortuga/drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware/turtle_trajectory/{primitive}/qd.mat', 'qd')
                        dqd_mat = mat2np(f'/home/tortuga/drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware/turtle_trajectory/{primitive}/dqd.mat', 'dqd')
                        ddqd_mat = mat2np(f'/home/tortuga/drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware/turtle_trajectory/{primitive}/ddqd.mat', 'ddqd')
                        tvec = mat2np(f'/home/tortuga/drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware/turtle_trajectory/{primitive}/tvec.mat', 'tvec')
                        first_time = True
                        first_loop = True
                        cycle = 0

                        # zero =  np.zeros((self.nq,1))
                        t_old = time.time()
                        # our loop's "starting" time
                        t_0 = time.time()
                        while cycle < num_cycles:
                            if turtle_node.voltage < threshold:
                                print("voltage too low--powering off...")
                                turtle_node.Joints.disable_torque()
                                break
                            rclpy.spin_once(turtle_node)
                            if turtle_node.mode == 'rest' or turtle_node.mode == 'stop':
                                turtle_node.Joints.send_torque_cmd(turtle_node.nq * [0])
                                turtle_node.Joints.disable_torque()
                                first_time = True
                                scipy.io.savemat(folder_name + "/primitive_data.mat", {
                               'q_data': q_data, 'dq_data': dq_data,
                                'tau_data': tau_data, 'time_data': time_data, 'current_data': current_data})
                                break
                            
                            q = np.array(turtle_node.Joints.get_position()).reshape(-1,1)

                            n = get_qindex((time.time() - t_0), tvec)
                            if n > len(tvec[0]) - 2:
                                first_loop = False
                                t_0 = time.time()
                                cycle += 1
                                print(f"-----------------cycle: {cycle}\n\n\n")
                            
                            qd = np.array(qd_mat[:, n]).reshape(-1,1)
                            dqd = np.array(dqd_mat[:, n]).reshape(-1,1)
                            ddqd = np.array(ddqd_mat[:, n]).reshape(-1,1)
                            q_data=np.append(q_data, q, axis = 1) 
                            
                            if first_time:
                                dq = np.array(turtle_node.Joints.get_velocity()).reshape(-1,1)
                                dq_data=np.append(dq_data, dq, axis = 1) 
                                q_old = q
                                first_time = False
                            else:
                                t = time.time()
                                dt = t - t_old
                                t_old = t
                                dq = np.array(turtle_node.Joints.get_velocity()).reshape(-1,1)
                                dq_data=np.append(dq_data, dq, axis = 1) 
                                q_old = q

                            err = q - qd
                            err_dot = dq
                            tau = turtle_controller(q,dq,qd,dqd,ddqd,turtle_node.Kp,turtle_node.KD)   
                            tau_data=np.append(tau_data, tau, axis = 1) 
                         
                            input_history = np.append(input_history[:,1:], tau,axis=1)
                            input_mean = np.mean(input_history, axis = 1)
                            curr = grab_arm_current(input_mean, min_torque, max_torque)
                            turtle_node.Joints.send_torque_cmd(curr)
                            current_data=np.append(current_data, curr, axis = 1) 

            elif turtle_node.mode == 'planner':
                """
                Randomly pick a motion primitive and run it 4-5 times
                """
                print("plan plan plan")
                primitives = ['surface', 'turnrf', 'turnrr', 'straight', 'turnlr']
                # primitives = ['turnrr']
                num_cycles = 1
                turtle_node.Joints.disable_torque()
                turtle_node.Joints.set_current_cntrl_mode()
                turtle_node.Joints.enable_torque()

                while True:
                    rclpy.spin_once(turtle_node)
                    if turtle_node.mode == 'rest' or turtle_node.mode == 'stop' or turtle_node.voltage < threshold:
                        turtle_node.Joints.send_torque_cmd([0] *len(turtle_node.IDs))
                        turtle_node.Joints.disable_torque()
                        break
                    # get mode here
                    # primitive = random.choice(primitives)
                    # print(f"list : {turtle_node.primitives}")
                    try:
                        primitive = md(turtle_node.primitives)
                    except:
                        primitive = "dwell"
                    # print(f"prim: {primitive}")
                    turtle_node.primitives = []
                    if primitive != "dwell":
                        print(f"---------------------------------------PRIMITIVE: {primitive}\n\n")
                        qd_mat = mat2np(f'/home/tortuga/drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware/turtle_trajectory/{primitive}/qd.mat', 'qd')
                        dqd_mat = mat2np(f'/home/tortuga/drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware/turtle_trajectory/{primitive}/dqd.mat', 'dqd')
                        ddqd_mat = mat2np(f'/home/tortuga/drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware/turtle_trajectory/{primitive}/ddqd.mat', 'ddqd')
                        tvec = mat2np(f'/home/tortuga/drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware/turtle_trajectory/{primitive}/tvec.mat', 'tvec')
                        first_time = True
                        first_loop = True
                        input_history = np.zeros((turtle_node.nq,10))
                        q_data = np.zeros((turtle_node.nq,1))
                        tau_data = np.zeros((turtle_node.nq,1))
                        timestamps = np.zeros((1,1))
                        dt_loop = np.zeros((1,1))       # hold dt data 
                        dq_data = np.zeros((turtle_node.nq,1))
                        tau_data = np.zeros((turtle_node.nq,1))
                        timestamps = np.zeros((1,1))
                        dt_loop = np.zeros((1,1))       # hold dt data 
                        cycle = 0

                        # zero =  np.zeros((self.nq,1))
                        t_old = time.time()
                        # our loop's "starting" time
                        t_0 = time.time()
                        while cycle < num_cycles:
                            if turtle_node.voltage < threshold:
                                print("voltage too low--powering off...")
                                turtle_node.Joints.disable_torque()
                                break
                            rclpy.spin_once(turtle_node)
                            # print("traj 1...")
                            if turtle_node.mode == 'rest' or turtle_node.mode == 'stop':
                                turtle_node.Joints.send_torque_cmd(turtle_node.nq * [0])
                                turtle_node.Joints.disable_torque()
                                first_time = True
                                break
                            
                            q = np.array(turtle_node.Joints.get_position()).reshape(-1,1)
                            # if first_loop:
                            n = get_qindex((time.time() - t_0), tvec)
                            # else:
                                # print("done with first loop")
                                # offset = t_0 - 2
                                # n = get_qindex((time.time() - offset), tvec)

                            # print(f"n: {n}\n")
                            if n > len(tvec[0]) - 2:
                                first_loop = False
                                t_0 = time.time()
                                cycle += 1
                                print(f"-----------------cycle: {cycle}\n\n\n")
                            
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
                                # # calculate errors
                            err = q - qd
                            # # print(f"[DEBUG] e: {err}\n")
                            # # print(f"[DEBUG] q: {q * 180/3.14}\n")
                            # # print(f"[DEBUG] qd: {qd * 180/3.14}\n")
                            err_dot = dq

                            tau = turtle_controller(q,dq,qd,dqd,ddqd,turtle_node.Kp,turtle_node.KD)

                            # publish motor data, tau data, 
                            
                            input_history = np.append(input_history[:,1:], tau,axis=1)
                            input_mean = np.mean(input_history, axis = 1)
                            curr = grab_arm_current(input_mean, min_torque, max_torque)
                            turtle_node.Joints.send_torque_cmd(curr)
                    else:
                        turtle_node.Joints.send_torque_cmd([0] *len(turtle_node.IDs))
                turtle_node.Joints.disable_torque()

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

