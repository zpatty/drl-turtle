import rclpy
from rclpy.node import Node
from rclpy.executors import Executor, MultiThreadedExecutor
from rcl_interfaces.msg import SetParametersResult
from turtle_interfaces.msg import TurtleTraj, TurtleSensors, TurtleMotorPos
import sys
import os

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

    def turtle_mode_callback(self, msg):
        """
        Callback function that updates the mode the turtle should be in.
        This method is what enables us to set "emergency stops" mid-trajectory. 
        """
        # global mode
        if msg.data == 'traj1':
            self.mode = 'traj1'
        elif msg.data == 'group':
            self.mode = 'group'
        elif msg.data == 'stop':
            self.mode = 'stop'
        elif msg.data == 'teacher':
            self.mode = 'teacher'
        else:
            self.mode = 'rest'
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
    
    def np2msg(self, mat):
        """
        flattens nd numpy array into a lst for ros pkg messaging
        """
        nq = 10
        squeezed = np.reshape(mat, (nq * mat.shape[1]))
        return squeezed.tolist()
    
    def publish_turtle_data(self, q_data, dq_data, tau_data, timestamps, t_0):
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
        # TODO: report dq?
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
def read_voltage(xiao, turtle_node):
    no_check = False
    sensors = xiao.readline()
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
                    turtle_node.voltage = volt   
def read_sensors(xiao, turtle_node):
    """
    Appends current sensor reading to the turtle node's sensor data structs
    """
    def add_place_holder():
        acc = np.array([-100.0, -100.0, -100.0]).reshape((3,1))
        gyr = np.array([-100.0, -100.0, -100.0]).reshape((3,1))
        quat = np.array([-100.0, -100.0, -100.0, -100.0]).reshape((4,1))
        volt = -1
        turtle_node.acc_data = np.append(turtle_node.acc_data, acc, axis=1)
        turtle_node.gyr_data = np.append(turtle_node.gyr_data, gyr, axis=1)
        turtle_node.quat_data = np.append(turtle_node.quat_data, quat, axis=1)
        turtle_node.voltage_data = np.append(turtle_node.voltage_data, volt)
    
    no_check = False
    sensors = xiao.readline()
    print(sensors)
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
                    turtle_node.acc_data = np.append(turtle_node.acc_data, acc, axis=1)
                    turtle_node.gyr_data = np.append(turtle_node.gyr_data, gyr, axis=1)
                    turtle_node.quat_data = np.append(turtle_node.quat_data, quat, axis=1)
                    turtle_node.voltage_data = np.append(turtle_node.voltage_data, volt)
                    turtle_node.voltage = volt
            else:
                add_place_holder()
        else:
            add_place_holder()
    else:
        add_place_holder()

# THIS IS TURTLE CODE
def main(args=None):
    rclpy.init(args=args)
    threshold = 11.3
    turtle_node = TurtleRobot('turtle_mode_cmd')
    xiao =  serial.Serial('/dev/ttyACM0', 115200, timeout=3)   
    # xiao.reset_input_buffer()
    # set up dynamixel stuff
    if portHandlerJoint.openPort():
        print("[MOTORS STATUS] Suceeded to open port")
    else:
        print("[ERROR] Failed to open port")
    if portHandlerJoint.setBaudRate(BAUDRATE):
        print("[MOTORS STATUS] Suceeded to open port")
    else:
        print("[ERROR] Failed to change baudrate")
    IDs = [1,2,3,4,5,6,7, 8, 9,10]
    nq = 10
    Joints = Mod(packetHandlerJoint, portHandlerJoint, IDs)
    Joints.disable_torque()
    Joints.set_current_cntrl_mode()
    Joints.enable_torque()
    q = np.array(Joints.get_position()).reshape(-1,1)
    place_holder = 3.14
    # q = np.insert(q, 7, place_holder).reshape((10,1))

    print(f"Our initial q: " + str(q))
    print("going into while loop...")
    try: 
        while rclpy.ok():
            if turtle_node.voltage < threshold:
                print("[WARNING] volt reading too low--closing program....")
                Joints.disable_torque()
                break
            rclpy.spin_once(turtle_node)
            
            if turtle_node.mode == 'position_control_mode':
                # Load desired position trajectories from MATLAB
                print("POSITION CONTROL")
                Joints.disable_torque()
                Joints.set_extended_pos_mode()
                Joints.set_current_cntrl_back_fins()
                Joints.enable_torque()
                qd_mat = turtle_node.qds
                tvec = turtle_node.tvec   
                t_0 = time.time()
                while 1:
                    if turtle_node.voltage < threshold:
                        print("voltage too low--powering off...")
                        Joints.disable_torque()
                        break
                    rclpy.spin_once(turtle_node)
                    if turtle_node.mode == 'rest' or turtle_node.mode == 'stop':
                        Joints.send_torque_cmd([0] *len(IDs))
                        Joints.disable_torque()
                        first_time = True
                        break
                    else:
                        n = get_qindex((time.time() - t_0), tvec)
                        qd = np.array(qd_mat[:, n]).reshape(-1,1)
                        qd = qd[:6]
                        print(f"qd: {qd}")
                        
                        Joints.send_pos_cmd(np.squeeze(to_motor_steps(qd)))
                        Joints.send_backfins_torque()
            elif turtle_node.mode == 'group':
                # group of motion primitives will run 
                # TODO: save list of motion primitives onto turtle robot for now

                break
            elif turtle_node.mode == 'stop':
                print("ending entire program...")
                print("disabling torques entirely...")
                Joints.send_torque_cmd([0] *len(IDs))
                Joints.disable_torque()
                cmd_msg = String()
                cmd_msg.data = "stop_received"
                turtle_node.cmd_received_pub.publish(cmd_msg)
                print("sent stop received msg")
                # save trajectory data
                # turtle_node.publish_turtle_data(t_0=t_0, q_data=q_data) 
                break
            elif turtle_node.mode == 'rest':
                first_time = True
                read_voltage(xiao, turtle_node)
                if turtle_node.voltage < threshold:
                    Joints.disable_torque()
                    print("THRESHOLD MET TURN OFFFF")
                    break
                # print("rest mode....")
                Joints.send_torque_cmd([0] *len(IDs))
                # Joints.disable_torque()
                cmd_msg = String()
                cmd_msg.data = 'rest_received'
                turtle_node.cmd_received_pub.publish(cmd_msg)
                print(f"current voltage: {turtle_node.voltage}\n")
                # print("sent rest received msg")
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
                input_history = np.zeros((nq,10))
                q_data = np.zeros((nq,1))
                dq_data = np.zeros((nq,1))
                tau_data = np.zeros((nq,1))
                timestamps = np.zeros((1,1))
                print(f"[MODE] TRAJECTORY\n")
                Joints.disable_torque()
                Joints.set_current_cntrl_mode()
                Joints.enable_torque()
                Kp = np.diag([0.5, 0.1, 0.05, 0.5, 0.1, 0.05, 0.1, 0.1, 0.0, 0.0])*3.5
                KD = 0.35
                t_begin = time.time()
                # zero =  np.zeros((self.nq,1))
                t_old = time.time()
                # our loop's "starting" time
                t_0 = time.time()
                while 1:
                    if turtle_node.voltage < threshold:
                        print("voltage too low--powering off...")
                        Joints.disable_torque()
                        print("saving data....")
                        turtle_node.publish_turtle_data(t_0=t_0, q_data=q_data, dq_data=dq_data, tau_data=tau_data, timestamps=timestamps) 
                        break
                    rclpy.spin_once(turtle_node)
                    # print("traj 1...")
                    if turtle_node.mode == 'rest' or turtle_node.mode == 'stop':
                        Joints.send_torque_cmd([0] *len(IDs))
                        Joints.disable_torque()
                        print("saving data...")
                        turtle_node.publish_turtle_data(t_0=t_0, q_data=q_data, dq_data=dq_data, tau_data=tau_data, timestamps=timestamps) 
                        first_time = True
                        break
                    q = np.array(Joints.get_position()).reshape(-1,1)
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
                        dq = np.zeros((nq,1))
                        dq_data=np.append(dq_data, dq, axis = 1) 
                        q_old = q
                        first_time = False
                    else:
                        t = time.time()
                        dt = t - t_old
                    #     # print(f"[DEBUG] dt: {dt}\n")  
                        t_old = t
                        dq = diff(q, q_old, dt)
                        dq_data=np.append(dq_data, dq, axis = 1) 
                        q_old = q
                    timestamps = np.append(timestamps, (time.time()-t_begin)) 
                    tau = turtle_controller(q,dq,qd,dqd,ddqd,Kp,KD)
                    tau_data=np.append(tau_data, tau, axis=1) 
                    
                    input_history = np.append(input_history[:,1:], tau,axis=1)

                    input_mean = np.mean(input_history, axis = 1)

                    inputt = grab_arm_current(input_mean, min_torque, max_torque)
                    # del inputt[7]
                    # print(f"[DEBUG] inputt: {inputt}\n")
                    print(f"voltage: {turtle_node.voltage}\n")
                    # inputt = [0]*10
                    Joints.send_torque_cmd(inputt)
                    read_sensors(xiao=xiao, turtle_node=turtle_node)
                Joints.disable_torque()
            elif turtle_node.mode == 'teacher':
                q_data = np.zeros((nq,1))
                dq_data = np.zeros((nq,1))
                tau_data = np.zeros((nq,1))
                timestamps = np.zeros((1,1))
                Joints.disable_torque()
                Joints.set_current_cntrl_mode()
                Joints.enable_torque()
                t_begin = time.time()
                t_old = time.time()
                # our loop's "starting" time
                t_0 = time.time()
                while 1:
                    if turtle_node.voltage < threshold:
                        print("voltage too low--powering off...")
                        Joints.disable_torque()
                        print("saving data....")
                        turtle_node.publish_turtle_data(t_0=t_0, q_data=q_data, dq_data=dq_data, tau_data=tau_data, timestamps=timestamps) 
                        break
                    rclpy.spin_once(turtle_node)
                    # print("traj 1...")
                    if turtle_node.mode == 'rest' or turtle_node.mode == 'stop':
                        # Joints.send_torque_cmd([0] *len(IDs))
                        Joints.disable_torque()
                        print("saving data...")
                        turtle_node.publish_turtle_data(t_0=t_0, q_data=q_data, dq_data=dq_data, tau_data=tau_data, timestamps=timestamps) 
                        first_time = True
                        break
                    q = np.array(Joints.get_position()).reshape(-1,1)
                    q_data=np.append(q_data, q, axis = 1) 
                    
                    if first_time:
                        dq = np.zeros((nq,1))
                        dq_data=np.append(dq_data, dq, axis = 1) 
                        q_old = q
                        first_time = False
                    else:
                        t = time.time()
                        dt = t - t_old
                        t_old = t
                        dq = diff(q, q_old, dt)
                        dq_data=np.append(dq_data, dq, axis = 1) 
                        q_old = q
                    timestamps = np.append(timestamps, (time.time()-t_begin)) 
                    read_sensors(xiao=xiao, turtle_node=turtle_node)
                Joints.disable_torque()
                
            else:
                print("wrong command received....")
    except Exception as e:
        print("some error occurred ¯\_(ツ)_/¯")
        Joints.send_torque_cmd([0] * len(IDs))
        Joints.disable_torque()
        exec_type, obj, tb = sys.exc_info()
        fname = os.path.split(tb.tb_frame.f_code.co_filename)[1]
        print(exec_type, fname, tb.tb_lineno)
        print(e)
    
    rclpy.shutdown()

if __name__ == '__main__':
    main()

