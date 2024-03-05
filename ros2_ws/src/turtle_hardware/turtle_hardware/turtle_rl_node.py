import rclpy
from rclpy.node import Node
from rclpy.executors import Executor, MultiThreadedExecutor
from rcl_interfaces.msg import SetParametersResult
from turtle_interfaces.msg import TurtleTraj, TurtleSensors, TurtleMotorPos
from EPHE import EPHE
from DualCPG import DualCPG
import sys
import os

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

    def __init__(self, topic):
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
        self.tau_data = np.zeros((1,1))
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
        self.epsilon = 0.1
        self.min_threshold = np.array([1.50, 3.0, 2.4, 2.28, 0.8, 1.56, 1.45, 1.2, 3.0, 2.3])
        self.max_threshold = np.array([3.53, 5.0, 4.8, 4.37, 4.15, 3.6, 3.2, 4.0, 4.0, 5.3])

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
        # self.Joints.disable_torque()
        self.Joints.enable_torque()
    
    def step(self, action):
        """
        action is tau which we pass into the turtle
        """
        # desired = np.array([10, 10, 10])        # forward movement
        desired = 10
        # self.tau_data=np.append(self.tau_data, action, axis=1) 
        # self.input_history = np.append(self.input_history[:,1:], action,axis=1)
        # self.input_mean = np.mean(self.input_history, axis = 1) ")
        print(f"CPG output: {action}\n")
        action = (action * 1000000) + 20
        
        inputt = grab_arm_current(action, min_torque, max_torque)
        print(f"pre action: {inputt}\n")
        # need a control barrier function for joint limits 
        q = self.Joints.get_position()
        # print(q)
        observation = np.array(self.Joints.get_position())
        # print(f"obs: {observation}")
        # print(f"ball: {observation + self.epsilon}")
        clipped = np.where(observation + self.epsilon < self.max_threshold, inputt, 0)
        # print(f"clipped: {clipped}")
        clipped = np.where(observation - self.epsilon > self.min_threshold, clipped, 0)
        # for o in range(len(observation)):
        #     if observation[o]< 
        print(f"clipped: {clipped}")
        self.Joints.send_torque_cmd(clipped)
        # print("sent joint cmd")
        self.read_sensors()
        # get latest acc data 
        # TODO: need a QP between our input and output such that it constrains our output for position constraints 
        acc = self.acc_data[:, -1]
        # print(f"acc data: {acc}")
        
        terminated = False
        # print(f"acc: {acc}")
        print(f"sum: {acc + desired}")
        print(f"acc data: {acc}\n")

        reward = acc[0] - desired
        if reward < 0:
            reward = 0
        print(f"reward: {reward}\n")
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
    
    def publish_turtle_data(self, q_data, dq_data, tau_data, timestamps, t_0, ddq_data=[]):
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
        volt_data = self.voltage_data.tolist()
        # print(f"voltage data: {volt_data}")
        turtle_msg.voltage = volt_data  

        # timestamps and t_0
        turtle_msg.timestamps = timestamps.tolist()
        turtle_msg.t_0 = t_0

        # motor positions, desired positions and tau data
        turtle_msg.q = self.np2msg(q_data) 
        turtle_msg.dq = self.np2msg(dq_data)
        if len(ddq_data) > 0:
            turtle_msg.ddq = self.np2msg(ddq_data)
        qd_squeezed = self.np2msg(self.qds)
        # print(f"checking type: {type(qd_squeezed)}")
        # for i in qd_squeezed:
        #     print(f"check element type: {type(i)}")
        turtle_msg.qd = qd_squeezed
        turtle_msg.tau = self.np2msg(tau_data)

        # publish msg 
        self.sensors_pub.publish(turtle_msg)

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
            quat = np.array([-100.0, -100.0, -100.0, -100.0]).reshape((4,1))
            volt = -1
            self.acc_data = np.append(self.acc_data, acc, axis=1)
            self.gyr_data = np.append(self.gyr_data, gyr, axis=1)
            self.quat_data = np.append(self.quat_data, quat, axis=1)
            self.voltage_data = np.append(self.voltage_data, volt)
        
        no_check = False
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
                        acc = np.array(sensor_dict['Acc']).reshape((3,1))
                        gyr = np.array(sensor_dict['Gyr']).reshape((3,1))
                        quat = np.array(sensor_dict['Quat']).reshape((4,1))
                        volt = sensor_dict['Voltage'][0]
                        self.acc_data = np.append(self.acc_data, acc, axis=1)
                        self.gyr_data = np.append(self.gyr_data, gyr, axis=1)
                        self.quat_data = np.append(self.quat_data, quat, axis=1)
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
    threshold = 11.1
    turtle_node = TurtleRobot('turtle_mode_cmd')
    # set up dynamixel stuff
    place_holder = 3.14
    # q = np.insert(q, 7, place_holder).reshape((10,1))

    # print(f"Our initial q: " + str(q))
    print("going into while loop...")

    num_params = 21
    M = 5
    K = 2
    trial = 1
    trial_folder = f'CPG_exp_{trial}'
    best_param_fname = trial_folder + f'/best_params_ephe_{trial}.pth'
    mu = np.random.rand((num_params)) * 20
    params = np.random.rand((num_params)) * 20
    tau_shift = 0.2
    B_shift = 0.5
    E_shift = 0.5
    sigma = np.random.rand((num_params)) 
    shifty_shift = [tau_shift, 
                    B_shift, B_shift, B_shift + 0.3,
                    B_shift + 0.3, B_shift, B_shift,
                    E_shift, E_shift, E_shift,
                    E_shift, E_shift, E_shift]
    print(f"intial mu: {mu}\n")
    print(f"initial sigma: {sigma}\n")
    print(f"M: {M}\n")
    print(f"K: {K}\n")
    num_mods = 10
    alpha = 0.5
    omega = 0.5
    cpg = DualCPG(num_params=num_params, num_mods=num_mods, alpha=alpha, omega=omega)
    cpg.set_parameters(params=params)

    print(f"starting params: {cpg.get_params()}")
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
    max_episodes = 20
    max_episode_length = 60     # 60 * 0.05 = ~3 seconds

    config_log = {
        "mu_init": list(mu),
        "sigma_init": list(sigma),
        "params": list(params),
        "M": M,
        "K": K,
        "max_episode_length": max_episode_length,
        "alpha": alpha,
        "omega": omega
    }

    # data structs for plotting
    param_data = np.zeros((num_params, M, max_episodes))
    mu_data = np.zeros((num_params, max_episodes))
    sigma_data = np.zeros((num_params, max_episodes))
    reward_data = np.zeros((M, max_episodes))
    os.makedirs(trial_folder, exist_ok=True)

    best_params = np.zeros((num_params))
    best_reward = 0

    # Specify the file path where you want to save the JSON file
    file_path = trial_folder + "/config.json"

    q = np.array(turtle_node.Joints.get_position()).reshape(-1,1)
    for i in range(10):
        print(f"Our initial q: " + str(q))

    # Save the dictionary as a JSON file
    with open(file_path, 'w') as json_file:
        json.dump(config_log, json_file)
    # try: 
    while rclpy.ok():
        if turtle_node.voltage < threshold:
            print("[WARNING] volt reading too low--closing program....")
            # Joints.disable_torque()
            break
        rclpy.spin_once(turtle_node)
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
            first_time = True
            turtle_node.read_voltage()
            if turtle_node.voltage < threshold:
                turtle_node.Joints.disable_torque()
                print("THRESHOLD MET TURN OFFFF")
                break
            turtle_node.Joints.send_torque_cmd([0] *len(turtle_node.IDs))
            turtle_node.Joints.disable_torque()
            cmd_msg = String()
            cmd_msg.data = 'rest_received'
            turtle_node.cmd_received_pub.publish(cmd_msg)
            print(f"current voltage: {turtle_node.voltage}\n")
        elif turtle_node.mode == 'train':
            print("TRAIN MODE")
            turtle_node.Joints.enable_torque()

            for episode in range(max_episodes):
                rclpy.spin_once(turtle_node)
                if turtle_node.voltage < threshold:
                    print("voltage too low--powering off...")
                    turtle_node.Joints.disable_torque()
                    print("saving data....")
                    # turtle_node.publish_turtle_data(t_0=t_0, q_data=q_data, dq_data=dq_data, tau_data=tau_data, timestamps=timestamps) 
                    break
                if turtle_node.mode == 'rest' or turtle_node.mode == 'stop':
                    turtle_node.Joints.send_torque_cmd([0] *len(turtle_node.IDs))
                    turtle_node.Joints.disable_torque()
                    print("saving data...")
                    # turtle_node.publish_turtle_data(t_0=t_0, q_data=q_data, dq_data=dq_data, tau_data=tau_data, timestamps=timestamps) 
                    first_time = True
                    break
                print(f"episode: {episode}")
                lst_params = np.zeros((num_params, M))
                solutions = ephe.ask()          
                R = np.zeros(M)
                folder_name = trial_folder + f"/CPG_episode_{episode}"
                for i in range(M):
                    print(f"-------------------------------------------M : {M}--------------------------------------\n")
                    rclpy.spin_once(turtle_node)
                    subplot=True
                    lst_params[:, i] = solutions[i]
                    # Joints.disable_torque()
                    # Joints.enable_torque()
                    fitness, total_actions = cpg.set_params_and_run(env=turtle_node, policy_parameters=solutions[i], max_episode_length=max_episode_length)
                    timesteps = total_actions.shape[1] - 1
                    dt = 0.05
                    t = np.arange(0, timesteps*dt, dt)
                    if fitness > best_reward:
                        print(f"params with best fitness: {solutions[i]} with reward {fitness}\n")
                        best_reward = fitness
                        best_params = solutions[i]
                    # if subplot:
                    #     # Plotting each row as its own subplot
                    #     fig, axs = plt.subplots(nrows=total_actions.shape[0], ncols=1, figsize=(8, 12))
                    #     for j, ax in enumerate(axs):
                    #         ax.plot(t, total_actions[j, 1:])
                    #         ax.set_title(f"CPG {j+1}")
                    #         ax.set_xlabel("Time")
                    #         ax.set_ylabel("Data")
                    #         ax.grid(True)
                    #     plt.tight_layout()
                    #     # plt.show()
                    #     os.makedirs(folder_name, exist_ok=True)
                    #     plt.savefig(folder_name + f'/CPG_output_run{i}_reward_{fitness}.png')
                    if fitness < 0:
                        R[i] = 0
                    else:
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
                # save mus and sigmas 0.00639871]
                mu_data[:, episode] = ephe.center()
                sigma_data[:, episode] = ephe.sigma()
                reward_data[:, episode] = R
            best_mu = ephe.center()
            best_sigma = ephe.sigma()
            print(f"best mu: {best_mu}\n")
            print(f"best sigma: {best_sigma}\n")
            print(f"best params: {best_params} got reward of {best_reward}\n")
            # save the best params
            torch.save(best_params, best_param_fname)
            # save data structs to matlab 
            scipy.io.savemat(trial_folder + "/data.mat", {'mu_data': mu_data,'sigma_data': sigma_data, 'param_data': param_data, 'reward_data': reward_data})
        else:
            print("wrong command received....")
    # except Exception as e:
    #     print("some error occurred")
    #     turtle_node.Joints.send_torque_cmd([0] * len(turtle_node.IDs))
    #     turtle_node.Joints.disable_torque()
    #     exec_type, obj, tb = sys.exc_info()
    #     fname = os.path.split(tb.tb_frame.f_code.co_filename)[1]
    #     print(exec_type, fname, tb.tb_lineno)
    #     print(e)
    
    rclpy.shutdown()

if __name__ == '__main__':
    main()

