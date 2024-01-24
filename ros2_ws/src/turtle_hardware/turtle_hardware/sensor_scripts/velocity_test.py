import rclpy
from rclpy.node import Node
from rclpy.executors import Executor, MultiThreadedExecutor
from rcl_interfaces.msg import SetParametersResult
from turtle_interfaces.msg import TurtleTraj
import json
import sys
import os

submodule = os.path.expanduser("~") + "/drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware"
sys.path.append(submodule)

from std_msgs.msg import String
from std_msgs.msg import Float64MultiArray
from dynamixel_sdk import *                                                                                 # Uses Dynamixel SDK library
from ros2_ws.src.turtle_hardware.turtle_hardware.turtle_dynamixel.Dynamixel import *                        # Dynamixel motor class                                  
from ros2_ws.src.turtle_hardware.turtle_hardware.turtle_dynamixel.dyn_functions import *                    # Dynamixel support functions
from ros2_ws.src.turtle_hardware.turtle_hardware.turtle_dynamixel.turtle_controller import *                # Controller 
from ros2_ws.src.turtle_hardware.turtle_hardware.turtle_dynamixel.Constants import *                        # File of constant variables
from ros2_ws.src.turtle_hardware.turtle_hardware.turtle_dynamixel.Mod import *
from ros2_ws.src.turtle_hardware.turtle_hardware.turtle_dynamixel.utilities import *
import math
from math import cos, sin
from datetime import datetime
import numpy as np
from matplotlib import pyplot as plt
from ros2_ws.src.turtle_hardware.turtle_hardware.turtle_dynamixel.utilities import *
import json
import traceback
from queue import Queue
import serial


def save_data(acc_data, gyr_data, mag_data, timestamps):
    t = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    folder_name =  "sensor_data/" + t
    os.makedirs(folder_name, exist_ok=True)
    scipy.io.savemat(folder_name + "/data.mat", {'acc_data': acc_data.T,'gyr_data': gyr_data.T,'mag_data': mag_data.T, 'time_data': timestamps})


def main(args=None):
    """
    Testing euler integration with IMU 
    """
    # {"Acc":[  89.36,  29.30,  1024.90 ], "Gyr" :[  0.52, -0.55,  0.07 ], "Mag" :[ -54.45, -10.65,  48.30 ],"Voltage": [ 0.04 ]}
    n_axis = 3
    acc_data = np.zeros((n_axis, 1))
    gyr_data = np.zeros((n_axis,1))
    mag_data = np.zeros((n_axis,1))
    timestamps = np.zeros((1,1))
    delta_t = 15
    t_0 = time.time()
    time_elapsed = 0
    print("recording...")
    first_time = True
    # while time_elapsed < delta_t:
    xiao =  serial.Serial('/dev/ttyACM0', 115200, timeout=3)   
    vel = np.zeros((n_axis, 1))
    t_old = time.time()
    while True:
        sensors = xiao.readline()
        sensors_decoded = sensors.decode('utf-8')
        if sensors_decoded[0] == "{":
            sensor_dict = json.loads(sensors.decode('utf-8'))
            # add sensor data
            # acc_data = np.append(acc_data, sensor_dict['Acc'])
            if first_time:
                dt = 0
                first_time = False
            else:
                t = time.time()
                dt = t - t_old
                t_old = t
            # euler time
            v_prev = vel
            acc = np.array(sensor_dict['Acc']).reshape((n_axis,1))
            # print(f"accel: {acc}")
            # print(f"vel prev: {v_prev}")
            # print(f"dt: {dt}")
            
            vel = v_prev + acc * dt
        print(f"vel: {vel}")
            

    print("recorded data to folder")
    save_data(acc_data=acc_data, gyr_data=gyr_data, mag_data=mag_data, timestamps=timestamps)
if __name__=='__main__':
    main()